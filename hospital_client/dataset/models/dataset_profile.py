
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DatasetProfile:
    dataset_type: str
    source_path: str
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    missing_samples: int = 0

    # Image specific
    image_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Tabular specific
    tabular_statistics: Dict[str, Any] = field(default_factory=dict)

    # General
    splits: Dict[str, Any] = field(default_factory=dict)
    classes: List[str] = field(default_factory=list)
    class_distribution: Dict[str, Any] = field(default_factory=dict)
    
    duplicate_information: Dict[str, Any] = field(default_factory=dict)
    missing_data_information: Dict[str, Any] = field(default_factory=dict)

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    generated_at: str = ""
    
    def to_dict(self) -> dict:
        return {
            "dataset_type": self.dataset_type,
            "source_path": self.source_path,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "missing_samples": self.missing_samples,
            "image_statistics": self.image_statistics,
            "tabular_statistics": self.tabular_statistics,
            "splits": self.splits,
            "classes": self.classes,
            "class_distribution": self.class_distribution,
            "duplicate_information": self.duplicate_information,
            "missing_data_information": self.missing_data_information,
            "warnings": self.warnings,
            "errors": self.errors,
            "generated_at": self.generated_at
        }

