"""
Numeric memory and training-time ESTIMATION for Module 8. Every number
produced here is explicitly labeled estimated/measured/configured-limit -
never presented as a guaranteed measurement (module8_readme.md §11).

Memory estimate = measured parameter count (model_profiles.py) x configured
byte/optimizer coefficients + a configured activation-memory heuristic.
Time estimate = a configurable baseline heuristic, superseded by real
measured throughput once HistoricalThroughputStore has an entry for the
same (architecture, device_key, precision) - recorded only after an actual
completed Module 7 run (see runner.py). Nothing here is fabricated history.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hospital_client.resource_training.policy import ResourcePolicy


@dataclass
class MemoryEstimate:
    estimated_param_memory_mb: float
    estimated_optimizer_memory_mb: float
    estimated_activation_memory_mb: float
    estimated_total_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_param_memory_mb": self.estimated_param_memory_mb,
            "estimated_optimizer_memory_mb": self.estimated_optimizer_memory_mb,
            "estimated_activation_memory_mb": self.estimated_activation_memory_mb,
            "estimated_total_mb": self.estimated_total_mb,
        }


def estimate_memory_mb(
    param_count: int, batch_size: int, input_size, precision: str, optimizer: str,
    resource_tier: str, policy: ResourcePolicy,
) -> MemoryEstimate:
    bytes_per_param = policy.bytes_per_param_fp16 if precision == "fp16" else policy.bytes_per_param_fp32
    param_mb = (param_count * bytes_per_param) / (1024 ** 2)

    optimizer_multiplier = policy.optimizer_state_multiplier.get(optimizer, 1.0)
    # Optimizer moment buffers are conservatively kept fp32-equivalent regardless
    # of training precision (matches how AMP optimizers actually behave).
    optimizer_mb = (param_count * policy.bytes_per_param_fp32 * optimizer_multiplier) / (1024 ** 2)

    pixels = float(input_size[0]) * float(input_size[1])
    coefficient = policy.activation_mb_per_batch_pixel.get(resource_tier, policy.activation_mb_per_batch_pixel[
        min(policy.activation_mb_per_batch_pixel, key=lambda k: policy.activation_mb_per_batch_pixel[k])
    ])
    activation_mb = coefficient * batch_size * pixels
    if precision == "fp16":
        activation_mb *= policy.bytes_per_param_fp16 / policy.bytes_per_param_fp32

    total = param_mb + optimizer_mb + activation_mb
    return MemoryEstimate(param_mb, optimizer_mb, activation_mb, total)


@dataclass
class TimeEstimate:
    # Central value retained for epoch-budget arithmetic and backward
    # compatibility - always prefer the _min/_max range for display, since a
    # single point value is never presented as exact (see module8_readme.md).
    estimated_training_time_seconds: float
    estimated_training_time_seconds_min: float
    estimated_training_time_seconds_max: float
    estimated_training_time_minutes: float
    estimated_training_time_minutes_min: float
    estimated_training_time_minutes_max: float
    estimated_training_time_display: str  # e.g. "3.1-4.6 minutes" or "12 seconds" when the range collapses
    estimation_method: str  # "baseline_estimate" | "dry_run_estimate" | "measured_hardware_estimate"
    estimation_confidence: str  # "low" | "medium" | "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_training_time_seconds": self.estimated_training_time_seconds,
            "estimated_training_time_seconds_min": self.estimated_training_time_seconds_min,
            "estimated_training_time_seconds_max": self.estimated_training_time_seconds_max,
            "estimated_training_time_minutes": self.estimated_training_time_minutes,
            "estimated_training_time_minutes_min": self.estimated_training_time_minutes_min,
            "estimated_training_time_minutes_max": self.estimated_training_time_minutes_max,
            "estimated_training_time_display": self.estimated_training_time_display,
            "estimation_method": self.estimation_method,
            "estimation_confidence": self.estimation_confidence,
        }


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
    return f"{secs} second{'s' if secs != 1 else ''}"


def format_duration_range(seconds_min: float, seconds_max: float) -> str:
    """A range is always the honest representation of an estimate - a single
    number implies false precision. Collapses to a single value only when
    the two bounds round to the same display unit/value."""
    seconds_min, seconds_max = max(0.0, seconds_min), max(0.0, seconds_max)
    if seconds_max < seconds_min:
        seconds_min, seconds_max = seconds_max, seconds_min

    if seconds_max < 60:
        lo, hi = round(seconds_min), round(seconds_max)
        return f"{lo} second{'s' if lo != 1 else ''}" if lo == hi else f"{lo}-{hi} seconds"
    if seconds_max < 3600:
        lo, hi = round(seconds_min / 60.0, 1), round(seconds_max / 60.0, 1)
        return f"{hi:.1f} minutes" if lo == hi else f"{lo:.1f}-{hi:.1f} minutes"
    lo, hi = round(seconds_min / 3600.0, 1), round(seconds_max / 3600.0, 1)
    return f"{hi:.1f} hours" if lo == hi else f"{lo:.1f}-{hi:.1f} hours"


class HistoricalThroughputStore:
    """
    A small JSON file recording REAL measured samples/second from completed
    Module 7 runs, keyed by (architecture, device_key, precision). Never
    seeded with fabricated data - entries only ever come from runner.py
    after an actual training run completes.
    """

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, List[float]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _key(architecture: str, device_key: str, precision: str) -> str:
        return f"{architecture}|{device_key}|{precision}"

    def record(self, architecture: str, device_key: str, precision: str, samples_per_second: float) -> None:
        if samples_per_second <= 0:
            return
        key = self._key(architecture, device_key, precision)
        self._data.setdefault(key, []).append(samples_per_second)
        self._save()

    def lookup(self, architecture: str, device_key: str, precision: str) -> Optional[List[float]]:
        return self._data.get(self._key(architecture, device_key, precision))


def _seconds_from_samples_per_second(samples_per_second: float, num_train_samples: int, num_validation_samples: int, epochs: int, policy: ResourcePolicy) -> float:
    train_seconds = (num_train_samples * epochs) / samples_per_second
    val_seconds = (num_validation_samples * epochs) / (samples_per_second * policy.validation_speedup_factor) if num_validation_samples else 0.0
    return train_seconds + val_seconds


def estimate_training_time(
    architecture: str, param_count: int, input_size, num_train_samples: int,
    num_validation_samples: int, batch_size: int, epochs: int, device_type: str,
    precision: str, policy: ResourcePolicy, history: Optional[HistoricalThroughputStore] = None,
    device_key: Optional[str] = None, measured_seconds_per_batch: Optional[float] = None,
) -> TimeEstimate:
    """
    Three tiers, most to least trustworthy:
      1. measured_hardware_estimate - real average throughput from completed
         Module 7 runs (HistoricalThroughputStore). Range uses the real
         observed min/max of historical samples once >=2 exist.
      2. dry_run_estimate - a real local forward+backward+optimizer-step was
         just measured on THIS machine for this exact configuration
         (profiler.py, invoked by evaluator.py) and extrapolated over the
         full dataset/epoch count.
      3. baseline_estimate - no real measurement available at all; a
         configurable heuristic scaled by measured parameter count and image
         resolution. Widest uncertainty range of the three.
    Every tier adds a fixed per-run overhead (policy.fixed_overhead_seconds)
    to account for model construction/DataLoader startup/checkpoint I/O that
    a per-batch/per-epoch formula alone does not capture, and every tier
    reports an explicit range rather than a single falsely-precise number.
    """
    device_key = device_key or device_type
    history_samples = history.lookup(architecture, device_key, precision) if history else None

    if history_samples:
        mean_sps = sum(history_samples) / len(history_samples)
        seconds = _seconds_from_samples_per_second(mean_sps, num_train_samples, num_validation_samples, epochs, policy)
        method = "measured_hardware_estimate"
        confidence = "high" if len(history_samples) >= policy.high_confidence_history_samples else "medium"
        if len(history_samples) >= 2:
            sps_min, sps_max = min(history_samples), max(history_samples)
            # Slower measured throughput -> longer time (the max bound); faster -> the min bound.
            seconds_max = _seconds_from_samples_per_second(sps_min, num_train_samples, num_validation_samples, epochs, policy)
            seconds_min = _seconds_from_samples_per_second(sps_max, num_train_samples, num_validation_samples, epochs, policy)
        else:
            seconds_min = seconds * policy.measured_single_sample_range_low
            seconds_max = seconds * policy.measured_single_sample_range_high

    elif measured_seconds_per_batch is not None and measured_seconds_per_batch > 0:
        num_batches = math.ceil(max(num_train_samples, 1) / max(batch_size, 1))
        val_batches = math.ceil(num_validation_samples / max(batch_size, 1)) if num_validation_samples else 0
        epoch_seconds = num_batches * measured_seconds_per_batch + (val_batches * measured_seconds_per_batch / policy.validation_speedup_factor)
        seconds = epoch_seconds * epochs
        method = "dry_run_estimate"
        confidence = "medium"
        seconds_min = seconds * policy.dry_run_range_low
        seconds_max = seconds * policy.dry_run_range_high

    else:
        relative_param_cost = max(param_count, 1) / policy.baseline_reference_param_count
        pixels = float(input_size[0]) * float(input_size[1])
        relative_pixel_cost = max(pixels, 1.0) / policy.baseline_reference_pixels
        base_throughput = policy.baseline_reference_throughput.get(device_type, policy.baseline_reference_throughput["cpu"])
        speedup = policy.fp16_speedup_factor if (precision == "fp16" and device_type == "cuda") else 1.0
        samples_per_second = max((base_throughput / (relative_param_cost * relative_pixel_cost)) * speedup, 0.01)
        seconds = _seconds_from_samples_per_second(samples_per_second, num_train_samples, num_validation_samples, epochs, policy)
        method = "baseline_estimate"
        confidence = "low"
        seconds_min = seconds * policy.baseline_range_low
        seconds_max = seconds * policy.baseline_range_high

    overhead = policy.fixed_overhead_seconds
    seconds += overhead
    seconds_min += overhead
    seconds_max += overhead

    return TimeEstimate(
        estimated_training_time_seconds=seconds,
        estimated_training_time_seconds_min=seconds_min,
        estimated_training_time_seconds_max=seconds_max,
        estimated_training_time_minutes=seconds / 60.0,
        estimated_training_time_minutes_min=seconds_min / 60.0,
        estimated_training_time_minutes_max=seconds_max / 60.0,
        estimated_training_time_display=format_duration_range(seconds_min, seconds_max),
        estimation_method=method,
        estimation_confidence=confidence,
    )
