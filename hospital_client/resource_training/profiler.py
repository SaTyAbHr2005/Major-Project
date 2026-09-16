"""
Real, local dry-run measurement - replaces pure heuristic guessing with an
actual forward+backward+optimizer-step executed on THIS machine, for the
exact architecture/batch size/precision/device being considered. This is
what lets evaluator.py report a MEASURED memory figure (not just an
estimate) and a MEASURED per-batch timing (used to build the training-time
estimate in estimator.py), directly addressing the "declared approximation"
limitation documented in earlier versions of this module.

Uses Module 6's own build_model() - never a parallel model-construction path
- and mirrors Module 7's own training step exactly (autocast + GradScaler,
same as hospital_client.training.trainer.Trainer._run_epoch) so the measured
numbers reflect what Module 7 will actually do.

This never touches real training data - only randomly generated dummy
tensors shaped like the real input, discarded immediately after measurement.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import psutil
import torch
import torch.nn as nn

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import build_model


@dataclass
class DryRunMeasurement:
    measured_total_mb: float
    measured_seconds_per_batch: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_total_mb": self.measured_total_mb,
            "measured": True,
        }


def _channels_for(color_mode: str) -> int:
    return 3 if color_mode == "RGB" else 1


def measure_real_resource_usage(
    architecture: str, num_classes: int, input_size: Tuple[int, int], color_mode: str,
    batch_size: int, device_type: str, precision: str, optimizer_name: str, learning_rate: float = 1e-3,
) -> DryRunMeasurement:
    """
    Executes exactly one real training step (forward + backward +
    optimizer.step) and measures actual peak memory and wall-clock time.
    Raises RuntimeError (including a genuine CUDA OOM) exactly as a real
    Module 7 run would - the caller (evaluator.py) is responsible for
    catching a CUDA OOM here and trying a smaller batch size.
    """
    device = torch.device(device_type)
    config = ModelConfig(architecture=architecture, num_classes=num_classes, input_size=input_size, color_mode=color_mode, pretrained=False)
    model = build_model(config).to(device)
    model.train()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    channels = _channels_for(color_mode)
    images = torch.randn(batch_size, channels, input_size[0], input_size[1], device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    autocast_enabled = precision == "fp16"
    scaler = torch.amp.GradScaler(device=device_type, enabled=autocast_enabled)

    process = None
    mem_before = 0.0
    if device_type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        process = psutil.Process()
        mem_before = process.memory_info().rss

    import time
    started = time.perf_counter()
    try:
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        if device_type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            mem_after = process.memory_info().rss
            # RSS deltas for a single CPU step can be noisy (allocator reuse,
            # OS paging) - floor at the models's real parameter+gradient
            # footprint so noise never reports an implausibly tiny number.
            param_floor_mb = sum(p.numel() for p in model.parameters()) * 4 * 2 / (1024 ** 2)
            peak_mb = max((mem_after - mem_before) / (1024 ** 2), param_floor_mb)

        return DryRunMeasurement(measured_total_mb=peak_mb, measured_seconds_per_batch=elapsed)
    finally:
        del model, optimizer, images, labels
        if device_type == "cuda":
            torch.cuda.empty_cache()
