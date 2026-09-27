from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class OutputMode(Enum):
    LAZY = "lazy"
    MATERIALIZED = "materialized"

class SplitStrategy(Enum):
    STRATIFIED = "stratified"
    GROUPED = "grouped"
    RANDOM = "random"

class TabularImputeStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    DROP = "drop"

@dataclass
class StorageSafetyConfig:
    safety_buffer_percent: float = 10.0
    enforce_safety: bool = True

@dataclass
class DatasetConfig:
    input_path: str = ""
    output_path: str = ""
    output_mode: OutputMode = OutputMode.LAZY
    storage_safety: StorageSafetyConfig = field(default_factory=StorageSafetyConfig)

@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    group_column: Optional[str] = None  # tabular only: a dataframe column name, e.g. patient_id
    respect_existing_splits: bool = True
    # Governs both INVALID and AMBIGUOUS existing-split classifications (see
    # splitting/split_validation.py): "error" stops and preserves the dataset
    # as-is; "regenerate" explicitly allows falling back to a fresh
    # ratio-based split. Never silently regenerates - this must be set.
    invalid_split_policy: str = "error"  # "error" | "regenerate"
    # Only consulted for VALID_PARTIAL layouts (e.g. train+test only). When
    # True, missing split(s) are carved out of the existing TRAIN portion
    # only, using the configured ratios; existing validation/test data is
    # never touched. Default False: a partial layout is preserved as-is.
    generate_missing_splits_from_train: bool = False
    # Optional path to a JSON file mapping {rel_path: group_id}, used only to
    # check for real patient/group leakage across an existing image split.
    # Without it, group leakage is explicitly reported as unverifiable
    # rather than fabricated (see split_validation.py) - never set this from
    # a synthetic/fabricated group array.
    group_id_map_path: Optional[str] = None
    # Threshold (max-class-count / min-class-count) above which an existing
    # split's class distribution is flagged as severely imbalanced. This is
    # informational only and never invalidates a split by itself.
    severe_class_imbalance_ratio: float = 5.0
    # Severity for cross-split duplicate/group leakage findings: "invalid"
    # (default) escalates the split to INVALID; "warning" records the
    # finding (visible in split_validation.reasons/duplicate_leakage/
    # group_leakage) without discarding an otherwise-valid existing split.
    # Must be explicitly set to "warning" - never defaults to downgrading.
    leakage_policy: str = "invalid"  # "invalid" | "warning"

@dataclass
class ImageConfig:
    target_size: Optional[tuple[int, int]] = None
    color_mode: Optional[str] = None  # e.g. "RGB", "L"
    normalization: Optional[str] = None
    augmentation_flags: List[str] = field(default_factory=list) # e.g. "h_flip", "rotation"
    
@dataclass
class QualityConfig:
    blur_threshold: Optional[float] = None
    brightness_min: Optional[float] = None
    brightness_max: Optional[float] = None
    reject_unreadable: bool = True
    reject_zero_byte: bool = True

@dataclass
class TabularConfig:
    missing_numeric_strategy: TabularImputeStrategy = TabularImputeStrategy.MEDIAN
    missing_categorical_strategy: TabularImputeStrategy = TabularImputeStrategy.MODE
    missing_constant_value: Any = "unknown"
    outlier_strategy: str = "report" # "report", "drop", "clip"
    numerical_scaling: Optional[str] = None # e.g. "standard", "minmax"
    categorical_encoding: Optional[str] = None # e.g. "onehot", "label"
    excluded_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None  # explicit label column; required when Module 4 only reports candidates

@dataclass
class ClassConfig:
    hierarchy_mapping: Dict[str, str] = field(default_factory=dict)
    imbalance_strategy: str = "none" # "oversample", "undersample", "class_weights", "none"
    preserve_subtypes: bool = True

@dataclass
class PreprocessingConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)

    @classmethod
    def default(cls) -> "PreprocessingConfig":
        return cls()

