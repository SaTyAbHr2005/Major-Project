"""
Centralized, configurable resource policy for Module 8 (Resource-Aware Training).

Every hardware threshold, safety margin, batch-size candidate, epoch bound,
time budget, precision rule, worker choice, monitoring interval, and
adaptation limit lives here - nothing else in this package hard-codes an
arbitrary hardware decision. Follows the dataclass + to_dict()/from_dict()
convention already used by Modules 5/6/7.

All numeric coefficients below (activation-memory-per-pixel, baseline
throughput, etc.) are DECLARED APPROXIMATIONS the project can recalibrate -
never claimed as measured universal truths. Only `hospital_client.
resource_training.model_profiles` (real parameter counts via Module 6) and
`estimator.HistoricalThroughputStore` (real measured throughput after an
actual run) are measured/factual inputs.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hospital_client.model_management.registry import ResourceTier


@dataclass
class ResourcePolicy:
    # --- Safety margins: fraction of AVAILABLE (not total) memory usable ---
    vram_safety_margin: float = 0.80
    ram_safety_margin: float = 0.70
    storage_safety_margin: float = 0.90
    # Fixed CUDA-context / driver overhead reserved before any model math (MB).
    cuda_context_overhead_mb: float = 400.0

    # --- Batch size ---
    batch_size_candidates: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])

    # --- Epochs / training budget ---
    epoch_min: int = 3
    epoch_default: int = 10
    epoch_max: int = 40
    # Ceiling on total estimated training time (minutes) per resource tier -
    # NOT "more hardware -> more epochs"; a slow/expensive model on strong
    # hardware can still be capped by this budget.
    time_budget_minutes_by_tier: Dict[str, float] = field(default_factory=lambda: {
        ResourceTier.VERY_LOW_END.value: 20.0,
        ResourceTier.LOW_END.value: 30.0,
        ResourceTier.MEDIUM_END.value: 45.0,
        ResourceTier.HIGH_END.value: 60.0,
    })

    # --- Workers ---
    worker_candidates: List[int] = field(default_factory=lambda: [0, 2, 4])
    # Reserve this many logical CPU cores for the main process/OS before
    # offering workers > 0.
    reserved_cpu_cores: int = 2

    # --- Precision policy ---
    # fp16 is only ever offered on CUDA (Module 7 already enforces this at
    # validation time - this is a recommendation-time mirror of that rule).
    prefer_fp16_on_cuda: bool = True
    fp16_speedup_factor: float = 1.4  # heuristic: relative throughput multiplier, not measured

    # --- Memory-estimate coefficients (approximations, not measurements) ---
    bytes_per_param_fp32: float = 4.0
    bytes_per_param_fp16: float = 2.0
    # Adam keeps 2 extra moment buffers (fp32) per trainable parameter;
    # plain/momentum SGD keeps at most 1. Conservative constants.
    optimizer_state_multiplier: Dict[str, float] = field(default_factory=lambda: {"adam": 3.0, "sgd": 1.0})
    # Estimated activation-memory MB per (batch_size * pixel) unit, by the
    # architecture's own project-defined resource tier (heavier families get
    # a larger configurable coefficient). This is the main approximation in
    # the VRAM estimate and is the first thing to recalibrate against real
    # profiling data.
    activation_mb_per_batch_pixel: Dict[str, float] = field(default_factory=lambda: {
        ResourceTier.VERY_LOW_END.value: 4.0e-4,
        ResourceTier.LOW_END.value: 7.0e-4,
        ResourceTier.MEDIUM_END.value: 1.2e-3,
        ResourceTier.HIGH_END.value: 2.2e-3,
    })

    # --- Time-estimation baseline (heuristic, recalibratable) ---
    # Reference: samples/second for a nominal small model (~11M params,
    # 224x224) on this device type at batch=1, fp32. Everything else scales
    # relative to this via measured param count and pixel count.
    baseline_reference_throughput: Dict[str, float] = field(default_factory=lambda: {"cpu": 6.0, "cuda": 70.0})
    baseline_reference_param_count: float = 11_000_000.0
    baseline_reference_pixels: float = 224.0 * 224.0
    # Confidence thresholds: number of historical samples needed to call an
    # estimate "high" confidence instead of "medium".
    high_confidence_history_samples: int = 3

    # --- Hardware capability tier thresholds ---
    # Classifies the DETECTED MACHINE (not a model) into the same project-
    # defined ResourceTier labels Module 6 uses for models, purely to size
    # the epoch/time budget (§9) - never used to pick a model directly.
    # Sorted ascending; a machine's free VRAM (CUDA) or available RAM (CPU)
    # is compared against these floors to find its tier.
    vram_tier_thresholds_mb: Dict[str, float] = field(default_factory=lambda: {
        ResourceTier.VERY_LOW_END.value: 0.0,
        ResourceTier.LOW_END.value: 3000.0,
        ResourceTier.MEDIUM_END.value: 6000.0,
        ResourceTier.HIGH_END.value: 12000.0,
    })
    ram_tier_thresholds_mb: Dict[str, float] = field(default_factory=lambda: {
        ResourceTier.VERY_LOW_END.value: 0.0,
        ResourceTier.LOW_END.value: 6000.0,
        ResourceTier.MEDIUM_END.value: 16000.0,
        ResourceTier.HIGH_END.value: 32000.0,
    })
    default_optimizer: str = "adam"

    # --- Recommendation selection ---
    # Weight applied to normalized estimated time when scoring the balanced
    # "recommended" pick (higher = more time-averse).
    time_penalty_weight: float = 0.5

    # --- Monitoring ---
    monitor_interval_seconds: float = 1.0

    # --- Runtime adaptation ---
    max_adaptation_attempts: int = 2
    batch_size_reduction_factor: float = 0.5

    # --- Network check (best-effort, non-blocking; never required) ---
    network_check_host: str = "8.8.8.8"
    network_check_port: int = 53
    network_check_timeout_seconds: float = 0.5

    # --- GPU utilization probe (nvidia-smi CLI, no new dependency) ---
    gpu_utilization_probe_timeout_seconds: float = 0.5

    # --- Real dry-run measurement (evaluator.py + profiler.py) ---
    # A single real forward+backward+optimizer-step is executed on THIS
    # machine, for the exact candidate batch size being considered, before it
    # is accepted - converting the memory check from a pure heuristic into a
    # real measurement. The heuristic coefficients above are still used first
    # as a cheap pre-filter so obviously-infeasible candidates never need a
    # real dry run at all.
    enable_dry_run_measurement: bool = True

    # --- Time-estimate uncertainty ranges (never a single false-precise number) ---
    # Multiplicative band applied around the central estimate. Widest for the
    # pure heuristic (no real measurement at all), narrower once a real local
    # dry-run measurement exists, narrower still (and based on REAL observed
    # variance once >=2 samples exist) once real full-run history exists.
    baseline_range_low: float = 0.6
    baseline_range_high: float = 1.8
    dry_run_range_low: float = 0.85
    dry_run_range_high: float = 1.35
    measured_single_sample_range_low: float = 0.85
    measured_single_sample_range_high: float = 1.2
    # Forward-only (validation/test) batches are cheaper than a full
    # forward+backward+optimizer-step training batch.
    validation_speedup_factor: float = 1.5
    # Fixed per-run overhead (model construction, DataLoader startup,
    # checkpoint I/O) NOT captured by a single dry-run batch's timing - added
    # once per run to both bounds of the range. Calibrated against a real
    # measured run on this project's development machine (see
    # module8_readme.md §11): a 25-sample/2-epoch run had ~31s actual duration
    # against a ~2s pure per-batch extrapolation, i.e. ~29s of fixed overhead
    # for a run this small. Larger/longer runs amortize this the same way
    # Module 7 itself amortizes checkpointing - the constant is deliberately
    # conservative (not tuned per-dataset-size) and documented as approximate.
    fixed_overhead_seconds: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vram_safety_margin": self.vram_safety_margin,
            "ram_safety_margin": self.ram_safety_margin,
            "storage_safety_margin": self.storage_safety_margin,
            "cuda_context_overhead_mb": self.cuda_context_overhead_mb,
            "batch_size_candidates": list(self.batch_size_candidates),
            "epoch_min": self.epoch_min,
            "epoch_default": self.epoch_default,
            "epoch_max": self.epoch_max,
            "time_budget_minutes_by_tier": dict(self.time_budget_minutes_by_tier),
            "vram_tier_thresholds_mb": dict(self.vram_tier_thresholds_mb),
            "ram_tier_thresholds_mb": dict(self.ram_tier_thresholds_mb),
            "default_optimizer": self.default_optimizer,
            "worker_candidates": list(self.worker_candidates),
            "reserved_cpu_cores": self.reserved_cpu_cores,
            "prefer_fp16_on_cuda": self.prefer_fp16_on_cuda,
            "fp16_speedup_factor": self.fp16_speedup_factor,
            "bytes_per_param_fp32": self.bytes_per_param_fp32,
            "bytes_per_param_fp16": self.bytes_per_param_fp16,
            "optimizer_state_multiplier": dict(self.optimizer_state_multiplier),
            "activation_mb_per_batch_pixel": dict(self.activation_mb_per_batch_pixel),
            "baseline_reference_throughput": dict(self.baseline_reference_throughput),
            "baseline_reference_param_count": self.baseline_reference_param_count,
            "baseline_reference_pixels": self.baseline_reference_pixels,
            "high_confidence_history_samples": self.high_confidence_history_samples,
            "time_penalty_weight": self.time_penalty_weight,
            "monitor_interval_seconds": self.monitor_interval_seconds,
            "max_adaptation_attempts": self.max_adaptation_attempts,
            "batch_size_reduction_factor": self.batch_size_reduction_factor,
            "network_check_host": self.network_check_host,
            "network_check_port": self.network_check_port,
            "network_check_timeout_seconds": self.network_check_timeout_seconds,
            "gpu_utilization_probe_timeout_seconds": self.gpu_utilization_probe_timeout_seconds,
            "enable_dry_run_measurement": self.enable_dry_run_measurement,
            "baseline_range_low": self.baseline_range_low,
            "baseline_range_high": self.baseline_range_high,
            "dry_run_range_low": self.dry_run_range_low,
            "dry_run_range_high": self.dry_run_range_high,
            "measured_single_sample_range_low": self.measured_single_sample_range_low,
            "measured_single_sample_range_high": self.measured_single_sample_range_high,
            "validation_speedup_factor": self.validation_speedup_factor,
            "fixed_overhead_seconds": self.fixed_overhead_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourcePolicy":
        defaults = cls()
        kwargs = {}
        for key in defaults.to_dict():
            if key in data:
                kwargs[key] = data[key]
        return cls(**kwargs)


def default_policy() -> ResourcePolicy:
    return ResourcePolicy()
