"""
Builds train/validation/test DataLoaders directly from Module 5's manifest
output. Training shuffles and applies Module 5's configured augmentation
flags; validation and test never shuffle and never augment (deterministic,
untouched evaluation distributions).

Worker RNG reproducibility: each DataLoader worker process gets a seed
deterministically derived from the configured random_seed (never
hard-coded), so re-running with the same seed reproduces the same
per-worker RNG state. This does NOT make GPU training bit-for-bit
deterministic across runs/hardware (see module7_readme.md) - it only makes
worker-process randomness (e.g. any future per-sample randomness) traceable
to the configured seed instead of the OS's own unseeded entropy.
"""
import os
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from hospital_client.training.config import TrainingConfig
from hospital_client.training.dataset import (
    ManifestImageDataset,
    derive_class_mapping,
    load_manifest,
    validate_manifest_against_class_mapping,
)

TRAIN_MANIFEST_NAME = "train_manifest.json"
VALIDATION_MANIFEST_NAME = "validation_manifest.json"
TEST_MANIFEST_NAME = "test_manifest.json"

_SEED_MODULUS = 2 ** 32


def derive_worker_seed(base_seed: int, worker_id: int) -> int:
    """Pure, deterministic derivation - no hard-coded worker seeds."""
    return (base_seed + worker_id) % _SEED_MODULUS


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """
    worker_init_fn must be a module-level function (not a lambda/closure) so
    it pickles correctly under Windows' "spawn" multiprocessing start method.
    """
    worker_seed = derive_worker_seed(base_seed, worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_loader_kwargs(config: TrainingConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=config.random_seed)
    generator = torch.Generator()
    generator.manual_seed(config.random_seed)
    kwargs["generator"] = generator
    return kwargs


@dataclass
class PreparedData:
    train_loader: DataLoader
    validation_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    class_mapping: Dict[str, int]
    train_manifest_metadata: Dict[str, Any]
    train_label_counts: Dict[str, int]
    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    warnings: List[str] = field(default_factory=list)


def build_dataloaders(config: TrainingConfig) -> PreparedData:
    train_path = os.path.join(config.dataset_dir, TRAIN_MANIFEST_NAME)
    val_path = os.path.join(config.dataset_dir, VALIDATION_MANIFEST_NAME)
    test_path = os.path.join(config.dataset_dir, TEST_MANIFEST_NAME)

    train_manifest = load_manifest(train_path)
    if train_manifest["total_records"] == 0 or not train_manifest["records"]:
        raise ValueError(f"Training manifest at {train_path} contains zero records.")

    class_mapping = derive_class_mapping(train_manifest)
    if len(class_mapping) != config.model_config.num_classes:
        raise ValueError(
            f"Class-count mismatch: training manifest has {len(class_mapping)} class(es) "
            f"{sorted(class_mapping.keys())}, but model_config.num_classes="
            f"{config.model_config.num_classes}. Module 7 does not silently resize the "
            "classifier head or drop classes to force agreement."
        )

    warnings: List[str] = []
    augmentation_flags = (train_manifest.get("metadata", {}).get("augmentations", {}) or {}).get("applied_augmentations")

    input_size = tuple(config.model_config.input_size)
    color_mode = config.model_config.color_mode

    train_dataset = ManifestImageDataset(train_manifest, class_mapping, color_mode, input_size, augmentation_flags)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **_build_loader_kwargs(config),
    )

    validation_loader = None
    validation_sample_count = 0
    if os.path.exists(val_path):
        validation_manifest = load_manifest(val_path)
        if validation_manifest["records"]:
            warnings.extend(validate_manifest_against_class_mapping(validation_manifest, class_mapping, "Validation"))
            validation_dataset = ManifestImageDataset(validation_manifest, class_mapping, color_mode, input_size)
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                **_build_loader_kwargs(config),
            )
            validation_sample_count = len(validation_dataset)
        elif config.require_validation:
            raise ValueError(f"Validation manifest at {val_path} contains zero records but require_validation=True.")
    elif config.require_validation:
        raise ValueError(
            f"Validation manifest not found at {val_path} but require_validation=True. "
            "Set require_validation=False to explicitly train without validation."
        )

    test_loader = None
    test_sample_count = 0
    if config.run_test_evaluation:
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"run_test_evaluation=True but test manifest not found at {test_path}.")
        test_manifest = load_manifest(test_path)
        if not test_manifest["records"]:
            raise ValueError(f"Test manifest at {test_path} contains zero records but run_test_evaluation=True.")
        warnings.extend(validate_manifest_against_class_mapping(test_manifest, class_mapping, "Test"))
        test_dataset = ManifestImageDataset(test_manifest, class_mapping, color_mode, input_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **_build_loader_kwargs(config),
        )
        test_sample_count = len(test_dataset)

    train_label_counts: Dict[str, int] = {}
    for record in train_manifest["records"]:
        train_label_counts[record["label"]] = train_label_counts.get(record["label"], 0) + 1

    return PreparedData(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        class_mapping=class_mapping,
        train_manifest_metadata=train_manifest.get("metadata", {}),
        train_label_counts=train_label_counts,
        train_sample_count=len(train_dataset),
        validation_sample_count=validation_sample_count,
        test_sample_count=test_sample_count,
        warnings=warnings,
    )
