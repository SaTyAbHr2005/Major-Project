import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from hospital_client.preprocessing.config import PreprocessingConfig, OutputMode
from hospital_client.preprocessing.quarantine.manager import QuarantineManager, QualityStatus
from hospital_client.preprocessing.labels.hierarchy import LabelResolver
from hospital_client.preprocessing.image.quality import assess_image_quality
from hospital_client.preprocessing.tabular.quality import TabularQualityAssessor
from hospital_client.preprocessing.splitting.splitter import DatasetSplitter
from hospital_client.preprocessing.splitting.grouping import extract_groups
from hospital_client.preprocessing.balancing.sampler import ClassBalancer
from hospital_client.preprocessing.image.transforms import BaseTransformer
from hospital_client.preprocessing.image.augmentation import AugmentationRegistrar
from hospital_client.preprocessing.tabular.missing_values import ImputationFitter
from hospital_client.preprocessing.tabular.encoding import EncoderFitter
from hospital_client.preprocessing.output.manifest_builder import ManifestBuilder
from hospital_client.preprocessing.output.materializer import Materializer
from hospital_client.preprocessing.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

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
        
        if dataset_type == "image":
            self._process_images(profile)
        elif dataset_type == "tabular":
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
        detected_splits = list(profile.get("splits", {}).keys())
        
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.startswith(".") or file.endswith(".json"): continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, source_path)
                
                status, reason, meta = assess_image_quality(full_path, self.config.quality)
                self.quarantine.add_sample(full_path, status, reason, meta)
                
                if status == QualityStatus.VALID:
                    split, parent_label, subtype = self.label_resolver.resolve_label(rel_path, detected_splits)
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
                    
        # Storage check for materialized mode
        if self.config.dataset.output_mode == OutputMode.MATERIALIZED:
            # Add some overhead buffer
            is_safe, safety_meta = self.materializer.check_storage_safety(total_estimated_size)
            if not is_safe and self.config.dataset.storage_safety.enforce_safety:
                raise RuntimeError(f"Storage safety check failed: {safety_meta}")
                
        # 2. Splitting
        y_labels = np.array([r["label"] for r in valid_records])
        groups = None
        if self.config.split.group_column:
            # For images, if group_column is provided, we'd need a way to map it. 
            # Usually it comes from metadata. Assuming random groups for now if missing.
            groups = np.arange(len(y_labels)) 
            
        train_idx, val_idx, test_idx = self.dataset_splitter.split_indices(y_labels, groups)
        
        # Assign splits
        for i in train_idx: valid_records[i]["assigned_split"] = "train"
        for i in val_idx: valid_records[i]["assigned_split"] = "validation"
        for i in test_idx: valid_records[i]["assigned_split"] = "test"
        
        # 3. Balancing (Train only)
        train_records = [valid_records[i] for i in train_idx]
        y_train = np.array([r["label"] for r in train_records])
        balancing_meta = self.class_balancer.compute_training_weights(y_train)
        
        # 4. Output Generation
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
        for split_name, records in splits_map.items():
            meta = {"balancing": balancing_meta, "augmentations": aug_meta} if split_name == "train" else {}
            self.manifest_builder.build_manifest(split_name, records, meta)
            
        # 5. Report
        self.report_generator.generate_report(self.quarantine.summary(), self.config, splits_map)
        
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
        target_col = profile.get("tabular_statistics", {}).get("target_column")
        if not target_col or target_col not in df.columns:
            # Fallback to random if no target
            y_labels = np.zeros(len(df))
        else:
            y_labels = df[target_col].to_numpy()
            
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
        
        trans_meta = {"imputation": impute_meta, "encoding": encode_meta}
        
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

