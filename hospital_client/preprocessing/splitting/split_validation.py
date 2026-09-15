"""
Validates an existing (Module 4-detected) train/validation/test directory
layout before Module 5 decides whether to preserve it.

A directory being named "train"/"validation"/"test" does not by itself mean
it is a safe, usable split - it must be inspected first. This module answers
that question; hospital_client/preprocessing/engine.py decides what to do
with the answer (preserve / carve missing pieces from train / regenerate /
stop), governed by PreprocessingConfig.split.

Only image datasets are covered here. Module 4's CSV/XLS/XLSX ingestion
represents a single tabular file, not a directory split layout, so there is
no existing-split concept to validate for tabular data (see module5's
CLAUDE.md rules and the accompanying task report for this limitation).
"""
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash

CANONICAL_SPLITS = ("train", "validation", "test")

_SPLIT_NAME_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def normalize_split_name(raw_split_name: str) -> str:
    lowered = raw_split_name.lower()
    return _SPLIT_NAME_ALIASES.get(lowered, lowered)


class SplitValidationStatus(Enum):
    NO_EXISTING_SPLIT = "no_existing_split"
    VALID_COMPLETE = "valid_complete"
    VALID_PARTIAL = "valid_partial"
    INVALID = "invalid"
    AMBIGUOUS = "ambiguous"


@dataclass
class SplitValidationResult:
    status: SplitValidationStatus
    detected_splits: List[str] = field(default_factory=list)
    declared_splits: List[str] = field(default_factory=list)
    split_counts: Dict[str, int] = field(default_factory=dict)
    empty_splits: List[str] = field(default_factory=list)
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    missing_classes: Dict[str, List[str]] = field(default_factory=dict)
    unexpected_classes: Dict[str, List[str]] = field(default_factory=dict)
    class_distribution_warnings: List[str] = field(default_factory=list)
    duplicate_leakage: List[Dict[str, Any]] = field(default_factory=list)
    group_leakage: Dict[str, Any] = field(default_factory=dict)
    invalid_sample_counts: Dict[str, int] = field(default_factory=dict)
    unassigned_sample_count: int = 0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "detected_splits": self.detected_splits,
            "declared_splits": self.declared_splits,
            "split_counts": self.split_counts,
            "empty_splits": self.empty_splits,
            "class_distribution": self.class_distribution,
            "missing_classes": self.missing_classes,
            "unexpected_classes": self.unexpected_classes,
            "class_distribution_warnings": self.class_distribution_warnings,
            "duplicate_leakage": self.duplicate_leakage,
            "group_leakage": self.group_leakage,
            "invalid_sample_counts": self.invalid_sample_counts,
            "unassigned_sample_count": self.unassigned_sample_count,
            "reasons": self.reasons,
        }


def load_group_id_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def validate_image_split(
    records: List[Dict[str, Any]],
    rejected_records: List[Dict[str, Any]],
    detected_splits: List[str],
    config,
    dataset_root: str,
) -> SplitValidationResult:
    """
    records: VALID samples only, each with rel_path, source_path, label,
             pre_assigned_split (raw folder name or None).
    rejected_records: rejected/review samples, each with rel_path and
             pre_assigned_split - used only for per-split invalid-sample
             accounting (requirement H); never re-validated as images here.
    detected_splits: raw split folder names as reported by Module 4
             (profile["splits"]["detected_splits"]).
    config: PreprocessingConfig.split (a SplitConfig instance).
    dataset_root: absolute dataset path (kept for future safe-identifier use;
             not currently required beyond what rel_path already provides).
    """
    result = SplitValidationResult(status=SplitValidationStatus.NO_EXISTING_SPLIT)
    result.detected_splits = list(detected_splits)

    declared_splits = sorted({normalize_split_name(s) for s in detected_splits})
    result.declared_splits = declared_splits

    if not declared_splits:
        return result  # NO_EXISTING_SPLIT

    # --- Ambiguity check: any discovered file (valid OR rejected/review)
    # that isn't inside one of the declared split folders means the layout
    # mixes split-organized and non-split-organized content. This must not
    # be silently resolved by re-splitting everything.
    all_records = list(records) + list(rejected_records)
    unassigned = [r for r in all_records if not r.get("pre_assigned_split")]
    result.unassigned_sample_count = len(unassigned)
    if unassigned:
        result.status = SplitValidationStatus.AMBIGUOUS
        examples = sorted({r["rel_path"] for r in unassigned})[:5]
        result.reasons.append(
            f"{len(unassigned)} discovered file(s) are not inside any recognized split folder "
            f"while split folder(s) {declared_splits} also exist in the same dataset. "
            f"Examples: {examples}"
        )
        return result

    # --- Per-split counts and class distributions (valid samples only).
    by_split = defaultdict(list)
    for r in records:
        by_split[normalize_split_name(r["pre_assigned_split"])].append(r)

    for split_name in declared_splits:
        split_records = by_split.get(split_name, [])
        result.split_counts[split_name] = len(split_records)
        class_counts: Dict[str, int] = defaultdict(int)
        for r in split_records:
            class_counts[r["label"]] += 1
        result.class_distribution[split_name] = dict(class_counts)

    result.empty_splits = [s for s in declared_splits if result.split_counts.get(s, 0) == 0]

    # --- Invalid/rejected sample accounting per split (requirement H).
    invalid_counts: Dict[str, int] = defaultdict(int)
    for r in rejected_records:
        split_name = normalize_split_name(r["pre_assigned_split"]) if r.get("pre_assigned_split") else "unassigned"
        invalid_counts[split_name] += 1
    result.invalid_sample_counts = dict(invalid_counts)

    # --- Class consistency against train as the reference split (requirement D).
    train_classes = set(result.class_distribution.get("train", {}).keys())
    if train_classes:
        for split_name in declared_splits:
            if split_name == "train":
                continue
            split_classes = set(result.class_distribution.get(split_name, {}).keys())
            missing = sorted(train_classes - split_classes)
            unexpected = sorted(split_classes - train_classes)
            if missing:
                result.missing_classes[split_name] = missing
                result.reasons.append(f"Split '{split_name}' is missing class(es) present in train: {missing}")
            if unexpected:
                result.unexpected_classes[split_name] = unexpected
                result.reasons.append(f"Split '{split_name}' has class(es) not present in train: {unexpected}")

    # --- Class distribution severity (requirement E) - informational only,
    # never invalidates a split by itself (real medical data can be naturally
    # imbalanced).
    threshold = getattr(config, "severe_class_imbalance_ratio", None)
    if threshold and threshold > 0:
        for split_name, counts in result.class_distribution.items():
            values = [v for v in counts.values() if v > 0]
            if len(values) >= 2:
                ratio = max(values) / min(values)
                if ratio > threshold:
                    result.class_distribution_warnings.append(
                        f"Split '{split_name}' has a severe class imbalance "
                        f"(max/min class count ratio {ratio:.1f} > threshold {threshold})."
                    )

    # --- Cross-split duplicate leakage (requirement F). Only valid samples
    # are hashed, mirroring Module 4's own image_ingestor.py, which likewise
    # only hashes successfully-opened images.
    hash_to_splits: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for split_name in declared_splits:
        for r in by_split.get(split_name, []):
            file_hash = compute_file_hash(r["source_path"])
            if file_hash:
                hash_to_splits[file_hash][split_name].append(r["rel_path"])

    for splits_map in hash_to_splits.values():
        if len(splits_map) > 1:
            result.duplicate_leakage.append({
                "splits": sorted(splits_map.keys()),
                "count": sum(len(v) for v in splits_map.values()),
                "examples": {k: v[:3] for k, v in splits_map.items()},
            })

    # --- Patient/group leakage (requirement G). Only performed when the
    # caller explicitly supplies a real group-id mapping; a fabricated group
    # array (e.g. np.arange(...)) must never be used to claim this was
    # checked.
    group_map = load_group_id_map(getattr(config, "group_id_map_path", None))
    if group_map is None:
        result.group_leakage = {
            "checked": False,
            "reason": "No reliable patient/group identifier is available for this image dataset; "
                      "patient-level leakage could not be verified.",
        }
    else:
        group_to_splits: Dict[str, set] = defaultdict(set)
        for split_name in declared_splits:
            for r in by_split.get(split_name, []):
                group_id = group_map.get(r["rel_path"])
                if group_id:
                    group_to_splits[group_id].add(split_name)
        leaking = {g: sorted(s) for g, s in group_to_splits.items() if len(s) > 1}
        result.group_leakage = {"checked": True, "leaking_groups": leaking}
        if leaking:
            result.reasons.append(f"Patient/group leakage detected across splits: {leaking}")

    # --- Final status determination. Order matters: the most fundamental
    # structural problems are checked first.
    if "train" not in declared_splits:
        result.status = SplitValidationStatus.INVALID
        result.reasons.append("No 'train' split present among the detected split folders.")
        return result

    if result.empty_splits:
        result.status = SplitValidationStatus.INVALID
        result.reasons.append(f"Declared split(s) with zero usable samples: {result.empty_splits}")
        return result

    # Leakage severity is configurable: "invalid" (default) escalates the
    # split to INVALID; "warning" records the finding (never removes/hides
    # it) without discarding an otherwise-valid existing split. This must be
    # explicitly set to "warning" - it never defaults to silently downgrading
    # a leakage finding.
    leakage_policy = getattr(config, "leakage_policy", "invalid")
    if leakage_policy not in ("invalid", "warning"):
        leakage_policy = "invalid"
    leakage_is_blocking = leakage_policy != "warning"

    if result.duplicate_leakage:
        result.reasons.append(
            f"Exact duplicate samples found across splits (cross-split leakage), leakage_policy='{leakage_policy}'."
        )
        if leakage_is_blocking:
            result.status = SplitValidationStatus.INVALID
            return result

    if result.group_leakage.get("leaking_groups"):
        result.reasons.append(
            f"Patient/group-level leakage found across splits, leakage_policy='{leakage_policy}'."
        )
        if leakage_is_blocking:
            result.status = SplitValidationStatus.INVALID
            return result

    if set(declared_splits) == set(CANONICAL_SPLITS):
        result.status = SplitValidationStatus.VALID_COMPLETE
    else:
        result.status = SplitValidationStatus.VALID_PARTIAL
        result.reasons.append(f"Partial split layout: only {declared_splits} present.")

    return result
