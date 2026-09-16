"""
Configurable classification loss. Class weighting is a TRAINING technique
only - it never alters validation/test data or their reported distributions.

class_weighting modes:
  "none"     - plain CrossEntropyLoss.
  "manifest" - reuse Module 5's already-computed class_weights from the
               train manifest's balancing metadata (imbalance_strategy must
               have been enabled in Module 5); raises a clear error if
               Module 5 did not compute them, rather than silently
               falling back to unweighted loss.
  "balanced" - Module 7 computes inverse-frequency weights itself, from the
               TRAINING split's observed label counts only.
"""
from typing import Dict

import torch
import torch.nn as nn

from hospital_client.training.dataloader import PreparedData


def _weights_from_manifest(prepared: PreparedData) -> Dict[str, float]:
    balancing = (prepared.train_manifest_metadata or {}).get("balancing") or {}
    class_weights = balancing.get("class_weights")
    if not class_weights:
        raise ValueError(
            "class_weighting='manifest' was requested, but the training manifest has no "
            "balancing.class_weights (Module 5's imbalance_strategy was likely 'none'). "
            "Re-run Module 5 preprocessing with an imbalance strategy enabled, or use "
            "class_weighting='balanced'/'none' instead."
        )
    return class_weights


def _weights_balanced(prepared: PreparedData) -> Dict[str, float]:
    counts = prepared.train_label_counts
    total = sum(counts.values())
    n_classes = len(counts)
    return {label: total / (n_classes * count) for label, count in counts.items()}


def build_loss(class_weighting: str, prepared: PreparedData, device: torch.device) -> nn.CrossEntropyLoss:
    if class_weighting == "none":
        return nn.CrossEntropyLoss()

    if class_weighting == "manifest":
        weights_by_label = _weights_from_manifest(prepared)
    elif class_weighting == "balanced":
        weights_by_label = _weights_balanced(prepared)
    else:
        raise ValueError(f"Unknown class_weighting {class_weighting!r}")

    # Order weights by class index (prepared.class_mapping is the single source of truth).
    ordered = sorted(prepared.class_mapping.items(), key=lambda kv: kv[1])
    weight_tensor = torch.tensor(
        [float(weights_by_label.get(label, 1.0)) for label, _ in ordered],
        dtype=torch.float32,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weight_tensor)
