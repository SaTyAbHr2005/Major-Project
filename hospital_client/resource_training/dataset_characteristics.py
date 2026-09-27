"""
Reads Module 5's manifest output to learn the dataset characteristics
Module 8's recommendations depend on (sample counts, class count). Reuses
Module 7's own manifest loader/class-mapping logic rather than
re-implementing dataset inspection - Module 8 never re-reads raw images or
re-derives anything Module 4/5/7 already computed.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict

from hospital_client.training.dataloader import TEST_MANIFEST_NAME, TRAIN_MANIFEST_NAME, VALIDATION_MANIFEST_NAME
from hospital_client.training.dataset import derive_class_mapping, load_manifest


@dataclass
class DatasetCharacteristics:
    dataset_dir: str
    num_train_samples: int
    num_validation_samples: int
    num_test_samples: int
    num_classes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "num_train_samples": self.num_train_samples,
            "num_validation_samples": self.num_validation_samples,
            "num_test_samples": self.num_test_samples,
            "num_classes": self.num_classes,
        }


def inspect_dataset(dataset_dir: str) -> DatasetCharacteristics:
    train_path = os.path.join(dataset_dir, TRAIN_MANIFEST_NAME)
    train_manifest = load_manifest(train_path)
    if train_manifest["total_records"] == 0 or not train_manifest["records"]:
        raise ValueError(f"Training manifest at {train_path} contains zero records.")
    class_mapping = derive_class_mapping(train_manifest)

    def _count(name: str) -> int:
        path = os.path.join(dataset_dir, name)
        if not os.path.exists(path):
            return 0
        manifest = load_manifest(path)
        return manifest.get("total_records", 0)

    return DatasetCharacteristics(
        dataset_dir=dataset_dir,
        num_train_samples=train_manifest["total_records"],
        num_validation_samples=_count(VALIDATION_MANIFEST_NAME),
        num_test_samples=_count(TEST_MANIFEST_NAME),
        num_classes=len(class_mapping),
    )
