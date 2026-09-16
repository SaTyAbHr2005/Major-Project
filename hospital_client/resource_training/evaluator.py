"""
ResourceEvaluator: converts a ResourceProfile + ResourcePolicy + dataset
characteristics into a per-model feasibility assessment for all 8 of
Module 6's architectures. Never reduces hardware to a bare "low/medium/high"
label for decision-making - actual measurements (VRAM MB, RAM MB, cores,
measured parameter counts) drive every decision; a coarse tier label is only
ever derived afterwards, for time-budget sizing and human-readable display.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from hospital_client.model_management.registry import ARCHITECTURE_CATALOG, ResourceTier
from hospital_client.resource_training.dataset_characteristics import DatasetCharacteristics
from hospital_client.resource_training.estimator import (
    HistoricalThroughputStore, estimate_memory_mb, estimate_training_time,
)
from hospital_client.resource_training.model_profiles import measure_model_profile
from hospital_client.resource_training.policy import ResourcePolicy
from hospital_client.resource_training.profiler import measure_real_resource_usage
from hospital_client.resource_training.resource_profile import ResourceProfile

_TIER_ORDER = [ResourceTier.VERY_LOW_END.value, ResourceTier.LOW_END.value, ResourceTier.MEDIUM_END.value, ResourceTier.HIGH_END.value]


def select_device(profile: ResourceProfile) -> Tuple[str, Optional[str]]:
    """Whole-machine device choice: CUDA if usable, else CPU. Never per-model."""
    if profile.gpu.get("cuda_available"):
        return "cuda", None
    reason = profile.gpu.get("fallback_reason") or "CUDA is not available on this machine."
    return "cpu", reason


def classify_hardware_tier(profile: ResourceProfile, device_type: str, policy: ResourcePolicy) -> str:
    """
    Sizes the epoch/time budget only - NEVER used to pick a model directly
    (model choice always comes from the actual per-model feasibility check).
    """
    if device_type == "cuda":
        available = profile.gpu.get("free_vram_mb")
        thresholds = policy.vram_tier_thresholds_mb
    else:
        available = profile.ram.get("available_mb")
        thresholds = policy.ram_tier_thresholds_mb

    if available is None:
        return ResourceTier.VERY_LOW_END.value

    tier = ResourceTier.VERY_LOW_END.value
    for name in _TIER_ORDER:
        if available >= thresholds.get(name, 0.0):
            tier = name
    return tier


def _usable_memory_mb(profile: ResourceProfile, device_type: str, policy: ResourcePolicy) -> Optional[float]:
    if device_type == "cuda":
        free = profile.gpu.get("free_vram_mb")
        if free is None:
            return None
        return max(0.0, free - policy.cuda_context_overhead_mb) * policy.vram_safety_margin
    available = profile.ram.get("available_mb")
    if available is None:
        return None
    return available * policy.ram_safety_margin


def _choose_workers(profile: ResourceProfile, policy: ResourcePolicy) -> int:
    logical_cores = profile.cpu.get("logical_cores")
    if not logical_cores:
        return 0
    budget = logical_cores - policy.reserved_cpu_cores
    candidates = sorted(w for w in policy.worker_candidates if w <= budget)
    return candidates[-1] if candidates else 0


@dataclass
class ModelAssessment:
    architecture: str
    display_name: str
    resource_tier: str
    param_count: int
    feasible: bool
    safe: bool
    device: str
    precision: str
    batch_size: Optional[int]
    batch_size_reason: str
    epochs: Optional[int]
    epoch_policy_reason: str
    num_workers: int
    memory_estimate: Dict[str, Any]
    available_memory_mb: Optional[float]
    time_estimate: Optional[Dict[str, Any]]
    reason: str
    status_label: str = "Unassigned"  # set by recommender.py relative to the 3 picks
    role: Optional[str] = None        # "recommended" | "high_capacity" | "fast" | None, set by recommender.py

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "display_name": self.display_name,
            "resource_tier": self.resource_tier,
            "param_count": self.param_count,
            "feasible": self.feasible,
            "safe": self.safe,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "batch_size_reason": self.batch_size_reason,
            "epochs": self.epochs,
            "epoch_policy_reason": self.epoch_policy_reason,
            "num_workers": self.num_workers,
            "memory_estimate": self.memory_estimate,
            "available_memory_mb": self.available_memory_mb,
            "time_estimate": self.time_estimate,
            "reason": self.reason,
            "status_label": self.status_label,
            "role": self.role,
        }


class ResourceEvaluator:
    def __init__(self, profile: ResourceProfile, policy: ResourcePolicy, history: Optional[HistoricalThroughputStore] = None):
        self.profile = profile
        self.policy = policy
        self.history = history
        self.device_type, self.device_fallback_reason = select_device(profile)
        self.hardware_tier = classify_hardware_tier(profile, self.device_type, policy)

    def evaluate_model(self, architecture: str, dataset: DatasetCharacteristics) -> ModelAssessment:
        policy = self.policy
        measured = measure_model_profile(architecture, dataset.num_classes)
        precision = "fp16" if (self.device_type == "cuda" and policy.prefer_fp16_on_cuda) else "fp32"
        optimizer = policy.default_optimizer
        usable_mb = _usable_memory_mb(self.profile, self.device_type, policy)
        available_mb = (
            self.profile.gpu.get("free_vram_mb") if self.device_type == "cuda" else self.profile.ram.get("available_mb")
        )

        if usable_mb is None or usable_mb <= 0:
            return ModelAssessment(
                architecture=architecture, display_name=measured.display_name, resource_tier=measured.resource_tier,
                param_count=measured.param_count, feasible=False, safe=False, device=self.device_type,
                precision=precision, batch_size=None, batch_size_reason="No usable memory could be determined.",
                epochs=None, epoch_policy_reason="Not evaluated - no usable memory.", num_workers=0,
                memory_estimate={}, available_memory_mb=available_mb, time_estimate=None,
                reason=f"{measured.display_name} is unsafe: {self.device_type.upper()} memory could not be safely determined on this machine.",
            )

        chosen_batch_size = None
        chosen_memory_dict = None
        memory_measured = False
        dry_run_seconds_per_batch = None
        largest_heuristic_estimate = None

        for bs in sorted(policy.batch_size_candidates, reverse=True):
            heuristic = estimate_memory_mb(measured.param_count, bs, measured.default_input_size, precision, optimizer, measured.resource_tier, policy)
            if largest_heuristic_estimate is None:
                largest_heuristic_estimate = heuristic
            if heuristic.estimated_total_mb > usable_mb:
                continue  # heuristic pre-filter: obviously too big, skip the real dry run entirely

            if not policy.enable_dry_run_measurement:
                chosen_batch_size, chosen_memory_dict = bs, heuristic.to_dict()
                break

            try:
                dry_run = measure_real_resource_usage(
                    architecture, dataset.num_classes, measured.default_input_size, "RGB",
                    bs, self.device_type, precision, optimizer,
                )
            except RuntimeError as e:
                if self.device_type == "cuda" and "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue  # real measurement disagreed with the heuristic - try a smaller candidate
                raise

            if dry_run.measured_total_mb <= usable_mb:
                chosen_batch_size = bs
                chosen_memory_dict = dry_run.to_dict()
                memory_measured = True
                dry_run_seconds_per_batch = dry_run.measured_seconds_per_batch
                break
            # Real measurement exceeded the budget even though the heuristic
            # pre-filter allowed it through - try the next smaller candidate.

        if chosen_batch_size is None:
            smallest = min(policy.batch_size_candidates)
            estimate = largest_heuristic_estimate or estimate_memory_mb(measured.param_count, smallest, measured.default_input_size, precision, optimizer, measured.resource_tier, policy)
            reason = (
                f"{measured.display_name} is unsafe on this machine: even batch_size={smallest} is estimated to "
                f"require ~{estimate.estimated_total_mb:.0f}MB, exceeding the safe budget of ~{usable_mb:.0f}MB "
                f"(of ~{available_mb:.0f}MB available)."
            )
            return ModelAssessment(
                architecture=architecture, display_name=measured.display_name, resource_tier=measured.resource_tier,
                param_count=measured.param_count, feasible=False, safe=False, device=self.device_type,
                precision=precision, batch_size=None,
                batch_size_reason=f"No candidate batch size fits within the ~{usable_mb:.0f}MB safety budget.",
                epochs=None, epoch_policy_reason="Not evaluated - infeasible on memory.", num_workers=0,
                memory_estimate=estimate.to_dict(), available_memory_mb=available_mb, time_estimate=None,
                reason=reason,
            )

        num_workers = _choose_workers(self.profile, policy)

        one_epoch = estimate_training_time(
            architecture, measured.param_count, measured.default_input_size, dataset.num_train_samples,
            dataset.num_validation_samples, chosen_batch_size, 1, self.device_type, precision, policy,
            self.history, device_key=self._device_key(), measured_seconds_per_batch=dry_run_seconds_per_batch,
        )
        budget_minutes = policy.time_budget_minutes_by_tier.get(self.hardware_tier, policy.time_budget_minutes_by_tier[ResourceTier.MEDIUM_END.value])
        budget_seconds = budget_minutes * 60.0
        per_epoch_seconds = max(one_epoch.estimated_training_time_seconds, 0.001)
        max_epochs_in_budget = max(1, int(budget_seconds // per_epoch_seconds))
        epochs = max(policy.epoch_min, min(policy.epoch_default, max_epochs_in_budget))
        epochs = max(policy.epoch_min, min(policy.epoch_max, epochs))

        epoch_reason = (
            f"epochs={epochs} selected to stay within the {self.hardware_tier.replace('_', ' ')} time budget of "
            f"{budget_minutes:.0f} minutes at an estimated ~{per_epoch_seconds:.1f}s/epoch (bounded to "
            f"[{policy.epoch_min}, {policy.epoch_max}])."
        )

        full_estimate = estimate_training_time(
            architecture, measured.param_count, measured.default_input_size, dataset.num_train_samples,
            dataset.num_validation_samples, chosen_batch_size, epochs, self.device_type, precision, policy,
            self.history, device_key=self._device_key(), measured_seconds_per_batch=dry_run_seconds_per_batch,
        )

        memory_kind = "measured (real local dry run)" if memory_measured else "estimated"
        reason = (
            f"{measured.display_name} fits within the detected {self.device_type.upper()} budget: {memory_kind} "
            f"~{chosen_memory_dict.get('estimated_total_mb', 0.0):.0f}MB of a safe ~{usable_mb:.0f}MB (of "
            f"~{available_mb:.0f}MB available), batch_size={chosen_batch_size}, estimated time "
            f"~{full_estimate.estimated_training_time_display}."
        )

        return ModelAssessment(
            architecture=architecture, display_name=measured.display_name, resource_tier=measured.resource_tier,
            param_count=measured.param_count, feasible=True, safe=True, device=self.device_type,
            precision=precision, batch_size=chosen_batch_size,
            batch_size_reason=(
                f"Largest candidate batch size ({chosen_batch_size}) confirmed by a real local dry run to fit "
                "the safety budget." if memory_measured else
                f"Largest candidate batch size ({chosen_batch_size}) whose estimated memory fits the safety budget."
            ),
            epochs=epochs, epoch_policy_reason=epoch_reason, num_workers=num_workers,
            memory_estimate=chosen_memory_dict, available_memory_mb=available_mb,
            time_estimate=full_estimate.to_dict(), reason=reason,
        )

    def _device_key(self) -> str:
        if self.device_type == "cuda":
            return self.profile.gpu.get("gpu_name") or "cuda"
        return "cpu"

    def evaluate_all(self, dataset: DatasetCharacteristics) -> Dict[str, ModelAssessment]:
        return {arch: self.evaluate_model(arch, dataset) for arch in ARCHITECTURE_CATALOG}
