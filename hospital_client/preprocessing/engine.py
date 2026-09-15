import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from hospital_client.preprocessing.config import PreprocessingConfig, OutputMode, SplitConfig, SplitStrategy
from hospital_client.preprocessing.quarantine.manager import QuarantineManager, QualityStatus
from hospital_client.preprocessing.labels.hierarchy import LabelResolver
from hospital_client.preprocessing.image.quality import assess_image_quality
from hospital_client.preprocessing.tabular.quality import TabularQualityAssessor
from hospital_client.preprocessing.splitting.splitter import DatasetSplitter
from hospital_client.preprocessing.splitting.grouping import extract_groups
from hospital_client.preprocessing.splitting.split_validation import (
    CANONICAL_SPLITS,
    SplitValidationStatus,
    load_group_id_map,
    normalize_split_name,
    validate_image_split,
)
from hospital_client.preprocessing.balancing.sampler import ClassBalancer
from hospital_client.preprocessing.image.transforms import BaseTransformer
from hospital_client.preprocessing.image.augmentation import AugmentationRegistrar
from hospital_client.preprocessing.tabular.missing_values import ImputationFitter
from hospital_client.preprocessing.tabular.encoding import EncoderFitter
from hospital_client.preprocessing.output.manifest_builder import ManifestBuilder
from hospital_client.preprocessing.output.materializer import Materializer
from hospital_client.preprocessing.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# Module 4 emits dataset_type as "Image", "CSV", or "Excel" (see
# hospital_client/dataset/models/dataset_profile.py). Hand-built/legacy
# profiles in this codebase's own tests use lowercase "image"/"tabular".
# Both are normalized here into the two categories the engine handles.
_IMAGE_DATASET_TYPES = {"image"}
_TABULAR_DATASET_TYPES = {"tabular", "csv", "excel"}


def _normalize_dataset_type(dataset_type: str) -> str:
    normalized = (dataset_type or "").strip().lower()
    if normalized in _IMAGE_DATASET_TYPES:
        return "image"
    if normalized in _TABULAR_DATASET_TYPES:
        return "tabular"
    return normalized


def _extract_split_names(profile: Dict[str, Any]) -> List[str]:
    """
    Module 4 stores detected split names under profile["splits"]["detected_splits"]
    (see hospital_client/dataset/ingestion/image_ingestor.py). The keys of the
    "splits" object are NOT themselves split names in that structure. Some
    hand-built profiles instead use the split names directly as dict keys
    (e.g. {"train": [...]}); that shape is also supported.
    """
    splits_obj = profile.get("splits", {})
    if not isinstance(splits_obj, dict):
        return []
    if "detected_splits" in splits_obj:
        detected = splits_obj.get("detected_splits") or []
        return [s for s in detected if isinstance(s, str)]
    return list(splits_obj.keys())


class PreprocessingEngine:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.quarantine = QuarantineManager()
        self.label_resolver = LabelResolver(config.classes)
        self.dataset_splitter = DatasetSplitter(config.split)
        self.class_balancer = ClassBalancer(config.classes)
        
        self.image_transformer = BaseTransformer(config.image)
        self.augmentation_registrar = AugmentationRegistrar(config.image)
        
        self.tabular_quality = TabularQualityAssessor(config.tabular)
        self.imputer = ImputationFitter(config.tabular)
        self.encoder = EncoderFitter(config.tabular)
        
        self.manifest_builder = ManifestBuilder(config.dataset.output_path)
        self.materializer = Materializer(config.dataset.output_path, config.dataset.storage_safety)
        self.report_generator = ReportGenerator(config.dataset.output_path)
        
    def run(self, profile_path: str):
        with open(profile_path, "r") as f:
            profile = json.load(f)
            
        dataset_type = profile.get("dataset_type", "unknown")
        normalized_type = _normalize_dataset_type(dataset_type)

        if normalized_type == "image":
            self._process_images(profile)
        elif normalized_type == "tabular":
            self._process_tabular(profile)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
    def _process_images(self, profile: Dict[str, Any]):
        source_path = self.config.dataset.input_path
        if not source_path:
            source_path = profile.get("source_path", "")
            
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path {source_path} not found.")
            
        valid_records = []
        
        # 1. Scanning and Quality Assessment (Incremental)
        # Assuming the profile doesn't list every single file to save space, we scan
        total_estimated_size = 0
        detected_splits = _extract_split_names(profile)
        
        rejected_or_review_records = []
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.startswith(".") or file.endswith(".json"): continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_path)

                # Resolve split/label from the path alone (cheap, no file I/O)
                # for every discovered file - valid or not - so split
                # validation below can account for corrupted/rejected samples
                # within an existing split too (requirement H).
                split, parent_label, subtype = self.label_resolver.resolve_label(rel_path, detected_splits)

                status, reason, meta = assess_image_quality(full_path, self.config.quality)
                meta = dict(meta)
                meta["split"] = split
                self.quarantine.add_sample(full_path, status, reason, meta)

                if status == QualityStatus.VALID:
                    if parent_label is None:
                        parent_label = "unknown"

                    file_size = os.path.getsize(full_path)
                    total_estimated_size += file_size

                    valid_records.append({
                        "source_path": full_path,
                        "rel_path": rel_path,
                        "label": parent_label,
                        "subtype": subtype,
                        "pre_assigned_split": split,
                        "size": file_size
                    })
                else:
                    rejected_or_review_records.append({"rel_path": rel_path, "pre_assigned_split": split})

        # 2. Existing-split inspection and validation (requirements 1-11 of
        # the split-validation task). This never modifies the source dataset
        # and never fabricates group identifiers - see split_validation.py.
        split_validation = validate_image_split(
            records=valid_records,
            rejected_records=rejected_or_review_records,
            detected_splits=detected_splits,
            config=self.config.split,
            dataset_root=source_path,
        )

        # Storage check for materialized mode
        if self.config.dataset.output_mode == OutputMode.MATERIALIZED:
            # Add some overhead buffer
            is_safe, safety_meta = self.materializer.check_storage_safety(total_estimated_size)
            if not is_safe and self.config.dataset.storage_safety.enforce_safety:
                raise RuntimeError(f"Storage safety check failed: {safety_meta}")

        # 3. Split decision - "split exists" is not the same as "split is
        # valid". See split_validation.py for how each status is derived.
        respect = self.config.split.respect_existing_splits
        status = split_validation.status

        if not respect or status == SplitValidationStatus.NO_EXISTING_SPLIT:
            split_source = "generated"
            train_idx, val_idx, test_idx = self._generate_ratio_split(valid_records)

        elif status in (SplitValidationStatus.INVALID, SplitValidationStatus.AMBIGUOUS):
            policy = self.config.split.invalid_split_policy
            if policy == "regenerate":
                split_source = f"regenerated_after_{status.value}"
                train_idx, val_idx, test_idx = self._generate_ratio_split(valid_records)
            else:
                raise ValueError(
                    f"Existing split validation failed with status '{status.value}': "
                    f"{split_validation.reasons}. The existing dataset split was NOT modified "
                    "or regenerated. Set PreprocessingConfig.split.invalid_split_policy='regenerate' "
                    "to explicitly allow Module 5 to discard it and generate a fresh split, or fix "
                    "the dataset's split layout."
                )

        else:  # VALID_COMPLETE or VALID_PARTIAL - preserve exactly.
            split_source = "existing"
            for r in valid_records:
                r["assigned_split"] = normalize_split_name(r["pre_assigned_split"])
            train_idx = [i for i, r in enumerate(valid_records) if r["assigned_split"] == "train"]
            val_idx = [i for i, r in enumerate(valid_records) if r["assigned_split"] == "validation"]
            test_idx = [i for i, r in enumerate(valid_records) if r["assigned_split"] == "test"]

            if status == SplitValidationStatus.VALID_PARTIAL and self.config.split.generate_missing_splits_from_train:
                missing = [s for s in CANONICAL_SPLITS if s not in split_validation.declared_splits]
                if missing:
                    split_source = "existing_partial_generated_from_train"
                    train_idx, val_idx, test_idx = self._carve_missing_splits_from_train(
                        valid_records, train_idx, val_idx, test_idx, missing
                    )

        # 4. Balancing (Train only)
        train_records = [valid_records[i] for i in train_idx]
        y_train = np.array([r["label"] for r in train_records])
        balancing_meta = self.class_balancer.compute_training_weights(y_train)
        
        # 5. Output Generation
        aug_meta = self.augmentation_registrar.generate_augmentation_metadata()

        splits_map = {"train": [], "validation": [], "test": []}
        for rec in valid_records:
            split_name = rec["assigned_split"]

            if self.config.dataset.output_mode == OutputMode.MATERIALIZED:
                out_path = self.materializer.materialize_image(
                    rec["source_path"],
                    os.path.join(split_name, rec["label"], os.path.basename(rec["rel_path"])),
                    self.image_transformer
                )
                rec["processed_path"] = out_path
            else:
                rec["processed_path"] = None

            splits_map[split_name].append(rec)

        # Manifests (even in materialized mode)
        split_assignment_meta = {
            "source": split_source,
            "detected_splits": detected_splits,
            "validation": split_validation.to_dict(),
        }
        for split_name, records in splits_map.items():
            meta = {"balancing": balancing_meta, "augmentations": aug_meta} if split_name == "train" else {}
            meta["split_assignment"] = split_assignment_meta
            self.manifest_builder.build_manifest(split_name, records, meta)

        # 6. Report
        self.report_generator.generate_report(
            self.quarantine.summary(), self.config, splits_map, split_validation=split_validation.to_dict()
        )

    def _resolve_real_groups(self, records: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Returns real patient/group identifiers for GROUPED-strategy image
        splitting, sourced ONLY from an explicit group_id_map_path (a
        {rel_path: group_id} JSON file the caller supplies). Returns None
        when grouping isn't requested at all (any other strategy).

        Never fabricates group identifiers (e.g. np.arange(...)): if GROUPED
        is requested for images without a real, complete mapping, this
        raises rather than silently producing a meaningless per-sample
        "group" that would make GroupShuffleSplit behave like an ungrouped
        split while still being reported as group-aware.
        """
        if self.config.split.strategy != SplitStrategy.GROUPED:
            return None

        group_map = load_group_id_map(self.config.split.group_id_map_path)
        if group_map is None:
            raise ValueError(
                "Grouped splitting was requested (SplitConfig.strategy=GROUPED) for an image "
                "dataset, but no real patient/group identifier is available. Set "
                "SplitConfig.group_id_map_path to a real {rel_path: group_id} mapping "
                "(or --group-id-map via the CLI) - Module 5 does not fabricate group identifiers."
            )

        missing = [r["rel_path"] for r in records if r["rel_path"] not in group_map]
        if missing:
            raise ValueError(
                f"Grouped splitting requires every sample to have a real group id; "
                f"{len(missing)} sample(s) are missing from group_id_map_path "
                f"(e.g. {missing[:5]})."
            )
        return np.array([group_map[r["rel_path"]] for r in records])

    def _generate_ratio_split(self, valid_records: List[Dict[str, Any]]):
        """
        The original ratio-based (stratified/random/grouped) split, used when
        there is no existing split to respect, when respect_existing_splits
        is disabled, or when an invalid/ambiguous existing split is
        explicitly configured to be regenerated.
        """
        y_labels = np.array([r["label"] for r in valid_records])
        groups = self._resolve_real_groups(valid_records)

        train_idx, val_idx, test_idx = self.dataset_splitter.split_indices(y_labels, groups)

        for i in train_idx: valid_records[i]["assigned_split"] = "train"
        for i in val_idx: valid_records[i]["assigned_split"] = "validation"
        for i in test_idx: valid_records[i]["assigned_split"] = "test"
        return train_idx, val_idx, test_idx

    def _carve_missing_splits_from_train(self, valid_records, train_idx, val_idx, test_idx, missing_splits):
        """
        Only used for a VALID_PARTIAL existing split when
        generate_missing_splits_from_train is explicitly enabled. Carves the
        missing split(s) out of the EXISTING train portion only, using the
        configured ratios; existing validation/test records (if any) are
        never touched or rebalanced.
        """
        train_records = [valid_records[i] for i in train_idx]
        if not train_records:
            return train_idx, val_idx, test_idx

        y_train = np.array([r["label"] for r in train_records])
        sub_groups = self._resolve_real_groups(train_records)  # never fabricated; see _resolve_real_groups
        scratch_config = SplitConfig(
            train_ratio=self.config.split.train_ratio,
            validation_ratio=self.config.split.validation_ratio if "validation" in missing_splits else 0.0,
            test_ratio=self.config.split.test_ratio if "test" in missing_splits else 0.0,
            random_seed=self.config.split.random_seed,
            strategy=self.config.split.strategy,
        )
        sub_train_rel, sub_val_rel, sub_test_rel = DatasetSplitter(scratch_config).split_indices(y_train, groups=sub_groups)

        new_train_idx = [train_idx[i] for i in sub_train_rel]
        carved_val_idx = [train_idx[i] for i in sub_val_rel]
        carved_test_idx = [train_idx[i] for i in sub_test_rel]

        for i in new_train_idx: valid_records[i]["assigned_split"] = "train"
        for i in carved_val_idx: valid_records[i]["assigned_split"] = "validation"
        for i in carved_test_idx: valid_records[i]["assigned_split"] = "test"

        return new_train_idx, list(val_idx) + carved_val_idx, list(test_idx) + carved_test_idx
        
    def _resolve_target_column(self, profile: Dict[str, Any], df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Resolves the label column used for stratified splitting.

        Module 4 never chooses a final target column - it only reports
        "candidate_target_columns" (low-cardinality candidates, see
        hospital_client/dataset/ingestion/tabular_ingestor.py). This engine
        must not guess one either. Resolution order:
          1. Explicit override: PreprocessingConfig.tabular.target_column.
          2. A target_column already declared on the profile itself
             (tabular_statistics.target_column, or legacy top-level
             target_column, for hand-built/legacy profiles).
          3. If Module 4 reported candidates but nothing was resolved,
             fail with the candidate list rather than silently picking one.
          4. If there is no target information at all, proceed without
             stratification (recorded in the returned metadata, not silent).
        """
        tabular_stats = profile.get("tabular_statistics", {}) or {}
        candidates = tabular_stats.get("candidate_target_columns", []) or []

        configured = self.config.tabular.target_column
        if configured:
            if configured not in df.columns:
                raise ValueError(f"Configured target_column '{configured}' not found in dataset columns.")
            return configured, {"source": "config_override", "target_column": configured, "candidate_target_columns": candidates}

        declared = tabular_stats.get("target_column") or profile.get("target_column")
        if declared:
            if declared not in df.columns:
                raise ValueError(f"Profile-declared target_column '{declared}' not found in dataset columns.")
            return declared, {"source": "profile_declared", "target_column": declared, "candidate_target_columns": candidates}

        if candidates:
            candidate_names = [c.get("column") for c in candidates if isinstance(c, dict) and c.get("column")]
            raise ValueError(
                "Tabular target column is ambiguous: Module 4 reported candidate columns "
                f"{candidate_names} but none was selected. Set "
                "PreprocessingConfig.tabular.target_column (or --target-column via the CLI) "
                "explicitly to proceed."
            )

        return None, {"source": "none", "target_column": None, "candidate_target_columns": candidates}

    def _process_tabular(self, profile: Dict[str, Any]):
        source_path = self.config.dataset.input_path
        if not source_path:
            source_path = profile.get("source_path", "")
            
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path {source_path} not found.")
            
        # 1. Quality
        if source_path.endswith(".csv"):
            df = pd.read_csv(source_path)
        else:
            df = pd.read_excel(source_path)
            
        is_valid, reasons, meta = self.tabular_quality.assess_quality(df)
        if not is_valid:
            self.quarantine.add_sample(source_path, QualityStatus.REJECTED, ", ".join(reasons), meta)
            raise ValueError(f"Tabular dataset rejected: {reasons}")
            
        self.quarantine.add_sample(source_path, QualityStatus.VALID, "", meta)

        # 2. Splitting
        target_col, target_meta = self._resolve_target_column(profile, df)
        if target_col:
            y_labels = df[target_col].to_numpy()
        else:
            # No target column and no candidates were available at all (see
            # _resolve_target_column) - proceed without stratification rather
            # than blocking, but this is recorded in target_meta below.
            y_labels = np.zeros(len(df))

        groups = None
        if self.config.split.group_column and self.config.split.group_column in df.columns:
            groups = extract_groups(df, self.config.split.group_column)
            
        train_idx, val_idx, test_idx = self.dataset_splitter.split_indices(y_labels, groups)
        
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        df_test = df.iloc[test_idx].copy()
        
        # 3. Fit statistics on train
        impute_meta = self.imputer.fit(df_train)
        encode_meta = self.encoder.fit(df_train)
        
        trans_meta = {"imputation": impute_meta, "encoding": encode_meta, "target_resolution": target_meta}
        
        # 4. Transform and Output
        splits_map = {}
        for split_name, split_df in [("train", df_train), ("validation", df_val), ("test", df_test)]:
            if split_df.empty:
                continue
                
            split_df = self.imputer.transform(split_df)
            split_df = self.encoder.transform(split_df)
            
            records = split_df.to_dict(orient="records")
            splits_map[split_name] = records
            
            if self.config.dataset.output_mode == OutputMode.MATERIALIZED:
                self.materializer.materialize_tabular(split_df, split_name)
                
            self.manifest_builder.build_manifest(split_name, [{"source": source_path, "split_index": i} for i in range(len(records))], trans_meta)
            
        # 5. Report
        self.report_generator.generate_report(self.quarantine.summary(), self.config, splits_map)

