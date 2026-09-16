"""
PyTorch Dataset built directly from Module 5's manifest contract - never
re-discovers the dataset from disk, never re-implements Module 4/5 inspection
or preprocessing. Works identically for lazy manifests (processed_path is
None -> load from source_path) and materialized manifests (processed_path
set -> load the already-processed file), since Module 5 always writes the
same manifest shape in both output modes.

Only image classification manifests are supported (Module 6's model catalog
is image-only); Module 5's tabular CSV/XLS/XLSX output is out of scope here.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms

# The only augmentation flags Module 5's ImageConfig.augmentation_flags
# documents as examples (config.py: "e.g. h_flip, rotation"). An unrecognized
# flag is recorded as a warning and skipped - never silently invented,
# never a hard failure (Module 5 already decided augmentation is desired).
_AUGMENTATION_BUILDERS = {
    "h_flip": lambda: tv_transforms.RandomHorizontalFlip(p=0.5),
    "v_flip": lambda: tv_transforms.RandomVerticalFlip(p=0.5),
    "rotation": lambda: tv_transforms.RandomRotation(degrees=15),
}


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if "records" not in manifest:
        raise ValueError(f"Malformed manifest (missing 'records'): {manifest_path}")
    return manifest


def derive_class_mapping(train_manifest: Dict[str, Any]) -> Dict[str, int]:
    """
    The single source of truth for the class -> index mapping used across
    train/validation/test and the constructed model's output dimension.
    Derived once from the TRAIN manifest (sorted for determinism) - never
    reordered later, never silently invented per-split.
    """
    labels = sorted({r["label"] for r in train_manifest["records"]})
    if not labels:
        raise ValueError("Training manifest contains zero valid records; cannot derive a class mapping.")
    return {label: idx for idx, label in enumerate(labels)}


def validate_manifest_against_class_mapping(manifest: Dict[str, Any], class_mapping: Dict[str, int], split_name: str) -> List[str]:
    """
    Returns non-fatal warnings (e.g. a class present in train but absent
    from this split). Raises if this split contains a label that never
    appeared in training - that is a real mismatch, not a benign gap.
    """
    warnings: List[str] = []
    split_labels = {r["label"] for r in manifest["records"]}
    unknown = split_labels - set(class_mapping.keys())
    if unknown:
        raise ValueError(
            f"{split_name} manifest contains label(s) {sorted(unknown)} never seen in the training "
            f"manifest (known classes: {sorted(class_mapping.keys())}). Module 7 does not invent "
            "or silently reassign labels."
        )
    missing = set(class_mapping.keys()) - split_labels
    if missing:
        warnings.append(f"{split_name} split has zero samples for class(es) {sorted(missing)}.")
    return warnings


def build_transform(color_mode: str, input_size: Tuple[int, int], augmentation_flags: Optional[List[str]] = None):
    """
    augmentation_flags is only non-None for the training split; validation
    and test transforms are always deterministic (no randomness).
    """
    ops = []
    if augmentation_flags:
        for flag in augmentation_flags:
            builder = _AUGMENTATION_BUILDERS.get(flag)
            if builder is not None:
                ops.append(builder())
    ops.append(tv_transforms.Resize(input_size))
    ops.append(tv_transforms.ToTensor())
    return tv_transforms.Compose(ops)


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        manifest: Dict[str, Any],
        class_mapping: Dict[str, int],
        color_mode: str,
        input_size: Tuple[int, int],
        augmentation_flags: Optional[List[str]] = None,
    ):
        self.records = manifest["records"]
        self.class_mapping = class_mapping
        self.color_mode = color_mode
        self.transform = build_transform(color_mode, input_size, augmentation_flags)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        path = record.get("processed_path") or record["source_path"]
        with Image.open(path) as img:
            img = img.convert(self.color_mode)
            tensor = self.transform(img)
        label_idx = self.class_mapping[record["label"]]
        return tensor, label_idx
