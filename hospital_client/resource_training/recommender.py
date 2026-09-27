"""
Generates up to three dynamically-computed training-configuration
recommendations from a ResourceEvaluator's per-model assessments, as a
ladder around what THIS machine trains comfortably:

  Fast        - one step lighter (the fastest model of the nearest lighter
                tier), or the Recommended model with fewer epochs
  Recommended - the heaviest model tier that still trains the full default
                epochs within the machine's (dataset-scaled) time budget
  High-Cap.   - one step heavier (the largest model of the nearest heavier
                tier), or the Recommended model with more epochs; labelled
                "Not Recommended" with its costs when it does not fit the budget

plus a role/status label for every one of Module 6's 8 architectures (so
EfficientNet-B0 - and every other model - always remains visible, per the
project's add-on requirement, regardless of whether it was picked).

These are RESOURCE/TRAINING-TIME alternatives, not accuracy rankings - see
module8_readme.md. No text generated here claims accuracy superiority;
wording is always built from the actual computed numbers for this machine
and this dataset, never a canned string.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from hospital_client.resource_training.dataset_characteristics import DatasetCharacteristics
from hospital_client.resource_training.estimator import HistoricalThroughputStore, format_budget, format_duration_range
from hospital_client.resource_training.evaluator import ModelAssessment, ResourceEvaluator
from hospital_client.resource_training.policy import ResourcePolicy
from hospital_client.resource_training.resource_profile import ResourceProfile

_TIER_ORDINAL = {"very_low_end": 0, "low_end": 1, "medium_end": 2, "high_end": 3}

RECOMMENDED = "recommended"
HIGH_CAPACITY = "high_capacity"
FAST = "fast"

_LABELS = {RECOMMENDED: "Recommended", HIGH_CAPACITY: "High-Capacity / More Time", FAST: "Fast / Lower Resource"}
_NOT_RECOMMENDED_LABEL = "High-Capacity / Not Recommended"
_FEWER_EPOCHS_LABEL = "Fast / Fewer Epochs"
_MORE_EPOCHS_LABELS = ("More Epochs / More Time", "More Epochs / Not Recommended")


@dataclass
class RecommendedConfig:
    recommendation_type: str
    label: str
    architecture: str
    display_name: str
    device: str
    precision: str
    batch_size: int
    epochs: int
    num_workers: int
    estimated_training_time_seconds: float
    estimated_training_time_seconds_min: float
    estimated_training_time_seconds_max: float
    estimated_training_time_minutes: float
    estimated_training_time_minutes_min: float
    estimated_training_time_minutes_max: float
    estimated_training_time_display: str
    estimation_method: str
    estimation_confidence: str
    resource_summary: str
    reason: str
    tradeoff: str
    safe: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_type": self.recommendation_type,
            "label": self.label,
            "model": self.architecture,
            "display_name": self.display_name,
            "device": self.device,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_workers": self.num_workers,
            "estimated_training_time_seconds": self.estimated_training_time_seconds,
            "estimated_training_time_seconds_min": self.estimated_training_time_seconds_min,
            "estimated_training_time_seconds_max": self.estimated_training_time_seconds_max,
            "estimated_training_time_minutes": self.estimated_training_time_minutes,
            "estimated_training_time_minutes_min": self.estimated_training_time_minutes_min,
            "estimated_training_time_minutes_max": self.estimated_training_time_minutes_max,
            "estimated_training_time_display": self.estimated_training_time_display,
            "estimation_method": self.estimation_method,
            "estimation_confidence": self.estimation_confidence,
            "resource_summary": self.resource_summary,
            "reason": self.reason,
            "tradeoff": self.tradeoff,
            "safe": self.safe,
        }


@dataclass
class RecommendationSet:
    recommendations: List[RecommendedConfig]
    all_model_assessments: Dict[str, ModelAssessment]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "all_model_assessments": {k: v.to_dict() for k, v in self.all_model_assessments.items()},
            "notes": self.notes,
        }


def _seconds(a: ModelAssessment) -> float:
    return (a.time_estimate or {}).get("estimated_training_time_seconds", 0.0)


def _time_display(a: ModelAssessment) -> str:
    return (a.time_estimate or {}).get("estimated_training_time_display", "an unknown duration")


def _tier(a: ModelAssessment) -> int:
    return _TIER_ORDINAL.get(a.resource_tier, 2)


def _tier_name(a: ModelAssessment) -> str:
    return a.resource_tier.replace("_", " ")


def _per_epoch_ratio(a: ModelAssessment, b: ModelAssessment) -> Optional[float]:
    """How many times longer one epoch of `a` takes than one epoch of `b`."""
    if not (a.epochs and b.epochs and _seconds(a) > 0 and _seconds(b) > 0):
        return None
    return (_seconds(a) / a.epochs) / (_seconds(b) / b.epochs)


def _memory_pct(a: ModelAssessment) -> Optional[float]:
    total = (a.memory_estimate or {}).get("estimated_total_mb")
    if total is None or not a.available_memory_mb:
        return None
    return 100.0 * total / a.available_memory_mb


def _memory_phrase(a: ModelAssessment) -> str:
    pct = _memory_pct(a)
    kind = "VRAM" if a.device == "cuda" else "RAM"
    return f"~{pct:.0f}% of available {kind}" if pct is not None else f"an unknown share of available {kind}"


def _is_comfortable(a: ModelAssessment, policy: ResourcePolicy) -> bool:
    """
    Feasible AND trains the full default epoch count within this machine's
    time budget. The evaluator already cuts epochs to fit the budget, so a
    model left with fewer than the default epochs is one this machine can run
    but only by training it less.
    """
    return a.feasible and (a.epochs or 0) >= min(policy.epoch_default, policy.epoch_max)


def _with_epochs(a: ModelAssessment, epochs: int, policy: ResourcePolicy) -> ModelAssessment:
    """
    The same model/batch/memory at a different epoch count. The estimator is
    linear in epochs plus one fixed per-run overhead, so the range scales exactly.
    """
    t = dict(a.time_estimate or {})
    overhead, k = policy.fixed_overhead_seconds, epochs / a.epochs
    for key in ("estimated_training_time_seconds", "estimated_training_time_seconds_min", "estimated_training_time_seconds_max"):
        if key in t:
            t[key] = overhead + (t[key] - overhead) * k
            t[key.replace("seconds", "minutes")] = t[key] / 60.0
    t["estimated_training_time_display"] = format_duration_range(
        t.get("estimated_training_time_seconds_min", 0.0), t.get("estimated_training_time_seconds_max", 0.0)
    )
    return replace(a, epochs=epochs, time_estimate=t)


def _resource_summary(a: ModelAssessment) -> str:
    mem = a.memory_estimate or {}
    total = mem.get("estimated_total_mb")
    if total is None or a.available_memory_mb is None:
        return "Resource usage could not be estimated."
    kind = "VRAM" if a.device == "cuda" else "RAM"
    basis = "measured (real local dry run)" if mem.get("measured") else "estimated"
    return (
        f"{basis.capitalize()} peak {kind}: approximately {total / 1024.0:.2f} GB of "
        f"{a.available_memory_mb / 1024.0:.2f} GB available (~{_memory_pct(a):.0f}%)."
    )


def _config_from_assessment(recommendation_type: str, a: ModelAssessment, reason: str, tradeoff: str, label: Optional[str] = None) -> RecommendedConfig:
    t = a.time_estimate or {}
    return RecommendedConfig(
        recommendation_type=recommendation_type, label=label or _LABELS[recommendation_type],
        architecture=a.architecture, display_name=a.display_name, device=a.device, precision=a.precision,
        batch_size=a.batch_size, epochs=a.epochs, num_workers=a.num_workers,
        estimated_training_time_seconds=t.get("estimated_training_time_seconds", 0.0),
        estimated_training_time_seconds_min=t.get("estimated_training_time_seconds_min", 0.0),
        estimated_training_time_seconds_max=t.get("estimated_training_time_seconds_max", 0.0),
        estimated_training_time_minutes=t.get("estimated_training_time_minutes", 0.0),
        estimated_training_time_minutes_min=t.get("estimated_training_time_minutes_min", 0.0),
        estimated_training_time_minutes_max=t.get("estimated_training_time_minutes_max", 0.0),
        estimated_training_time_display=t.get("estimated_training_time_display", "unknown"),
        estimation_method=t.get("estimation_method", "baseline_estimate"),
        estimation_confidence=t.get("estimation_confidence", "low"),
        resource_summary=_resource_summary(a), reason=reason, tradeoff=tradeoff, safe=a.safe,
    )


# ---------------------------------------------------------------------------
# Generated text (always built from this machine's numbers)
# ---------------------------------------------------------------------------

def _recommended_reason(a: ModelAssessment, comfortable: bool, budget: str) -> str:
    if comfortable:
        return (
            f"Recommended: {a.display_name} is from the heaviest model tier ({_tier_name(a)}) that still trains "
            f"the full {a.epochs} epochs within this machine's {budget} time budget - estimated {_time_display(a)}, "
            f"using {_memory_phrase(a)}."
        )
    return (
        f"Recommended: no architecture trains the full default epochs within this machine's {budget} time "
        f"budget, so {a.display_name} was chosen because it fits the most epochs at the lowest cost - estimated "
        f"{_time_display(a)} for {a.epochs} epochs, using {_memory_phrase(a)}."
    )


def _recommended_tradeoff(a: ModelAssessment) -> str:
    return (
        "Balances model capacity against training time and resource usage; not a claim of highest achievable "
        "accuracy. Actual experimental performance must be measured on the target dataset."
    )


def _fast_reason(a: ModelAssessment, rec: ModelAssessment, variant: bool) -> str:
    if variant:
        return (
            f"Fast option: the Recommended model ({a.display_name}) with fewer epochs ({a.epochs} instead of "
            f"{rec.epochs}), estimated {_time_display(a)} instead of {_time_display(rec)}. No lighter architecture "
            f"that trains faster is feasible on this machine."
        )
    ratio = _per_epoch_ratio(rec, a)
    speed = f" - each epoch ~{ratio:.1f}x faster than Recommended" if ratio else ""
    return (
        f"Fast option, one step lighter than Recommended: {a.display_name} ({_tier_name(a)} tier) is estimated at "
        f"{_time_display(a)} for {a.epochs} epochs{speed}, using {_memory_phrase(a)}."
    )


def _fast_tradeoff(variant: bool) -> str:
    if variant:
        return (
            "Same model capacity with a shorter training budget; fewer epochs may give lower experimental "
            "performance - actual performance must be measured on the target dataset."
        )
    return (
        "Lower model capacity and a shorter training time than Recommended; may provide lower experimental "
        "performance, but actual performance must be measured on the target dataset."
    )


def _high_reason(a: ModelAssessment, rec: ModelAssessment, variant: bool, not_recommended: bool, budget: str) -> str:
    if variant:
        text = (
            f"The Recommended model ({a.display_name}) trained for more epochs ({a.epochs} instead of "
            f"{rec.epochs}), estimated {_time_display(a)}. No heavier architecture fits this machine"
        )
        if not_recommended:
            return text + f"; it exceeds the {budget} time budget, so it is not recommended as the default."
        return text + "; it still fits the time budget but takes longer."

    text = (
        f"{a.display_name} ({a.param_count:,} parameters, {_tier_name(a)} tier) is one step heavier than "
        f"Recommended: it uses {_memory_phrase(a)} and is estimated at {_time_display(a)}"
    )
    ratio = _per_epoch_ratio(a, rec)
    if ratio:
        # Per epoch: totals are both capped by the same time budget, so they hide the real cost difference.
        text += f" (each epoch takes ~{ratio:.1f}x as long as the Recommended option)"
    if not_recommended:
        return text + (
            f". It can run, but it is not recommended as the default: only {a.epochs} epochs fit in the "
            f"{budget} time budget, so it spends more time and resources while training for fewer epochs."
        )
    return text + ". It fits the time budget but places a heavier load on this machine than the Recommended option."


def _high_tradeoff(variant: bool) -> str:
    if variant:
        return (
            "Longer training for the same model; more epochs are not a guarantee of higher accuracy - actual "
            "performance must be evaluated on the target dataset."
        )
    return (
        "Higher computational load and longer estimated training time than the recommended option. Higher model "
        "capacity is not a guarantee of higher accuracy - actual performance must be evaluated on the target dataset."
    )


def _balance_score(a: ModelAssessment, min_p: float, max_p: float, min_t: float, max_t: float, weight: float) -> float:
    norm_capacity = 0.5 if max_p == min_p else (a.param_count - min_p) / (max_p - min_p)
    norm_time = 0.5 if max_t == min_t else (_seconds(a) - min_t) / (max_t - min_t)
    return norm_capacity - weight * norm_time


def _label_unpicked(a: ModelAssessment, policy: ResourcePolicy, budget: str) -> None:
    if not a.feasible:
        a.status_label = "Unsafe"
        return
    if not _is_comfortable(a, policy):
        a.status_label = "High Load"
        a.reason = (
            f"{a.display_name} can run on this machine but is not recommended: it uses {_memory_phrase(a)}, is "
            f"estimated at {_time_display(a)}, and only {a.epochs} epochs fit in the {budget} time budget."
        )
    else:
        a.status_label = "Suitable"
        a.reason = (
            f"{a.display_name} is a suitable feasible alternative on this machine (estimated {_time_display(a)}, "
            f"using {_memory_phrase(a)}), but was not selected as the Recommended, High-Capacity, or Fast option "
            f"for this run."
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _nearest_tier(candidates: List[ModelAssessment], below: bool) -> List[ModelAssessment]:
    """The candidates from the tier closest to Recommended's (highest tier below it, or lowest above it)."""
    if not candidates:
        return []
    target = max(map(_tier, candidates)) if below else min(map(_tier, candidates))
    return [a for a in candidates if _tier(a) == target]


def generate_recommendations(
    profile: ResourceProfile, policy: ResourcePolicy, dataset: DatasetCharacteristics,
    history: Optional[HistoricalThroughputStore] = None,
) -> RecommendationSet:
    evaluator = ResourceEvaluator(profile, policy, history)
    assessments = evaluator.evaluate_all(dataset)
    return select_recommendations(assessments, evaluator.hardware_tier, policy)


def select_recommendations(
    assessments: Dict[str, ModelAssessment], hardware_tier: str, policy: ResourcePolicy,
) -> RecommendationSet:
    """
    The pure selection algorithm, separated from hardware
    detection/evaluation so it can be exercised directly (e.g. in tests)
    against hand-constructed ModelAssessment inputs.
    """
    feasible = {k: v for k, v in assessments.items() if v.feasible}
    notes: List[str] = []

    if not feasible:
        notes.append(
            "No feasible configuration could be generated for the detected hardware: every one of Module 6's 8 "
            "architectures exceeded the configured safety margins. See all_model_assessments for the reason each "
            "architecture was rejected."
        )
        for assessment in assessments.values():
            assessment.status_label = "Unsafe"
        return RecommendationSet([], assessments, notes)

    # The evaluator stores the dataset-scaled budget on every feasible assessment;
    # hand-built assessments without one fall back to the unscaled tier budget.
    budget_seconds = next((a.time_budget_seconds for a in feasible.values() if a.time_budget_seconds), None) or (
        policy.time_budget_minutes_by_tier.get(hardware_tier, policy.time_budget_minutes_by_tier["medium_end"]) * 60.0
    )
    budget = format_budget(budget_seconds)

    param_counts = [a.param_count for a in feasible.values()]
    times = [_seconds(a) for a in feasible.values()]
    min_p, max_p, min_t, max_t = min(param_counts), max(param_counts), min(times), max(times)

    def balance(a: ModelAssessment) -> float:
        return _balance_score(a, min_p, max_p, min_t, max_t, policy.time_penalty_weight)

    # 1. Recommended: the heaviest tier trained comfortably; balance breaks ties within the tier.
    comfortable = [a for a in feasible.values() if _is_comfortable(a, policy)]
    if comfortable:
        rec = max(comfortable, key=lambda a: (_tier(a), balance(a)))
    else:
        # Nothing fits: a capacity-weighted pick would favour the heaviest (slowest) model, the
        # opposite of what an overloaded machine needs - take the most epochs, then the fastest.
        rec = max(feasible.values(), key=lambda a: (a.epochs or 0, -_seconds(a)))
        notes.append(
            f"No architecture trains the default {policy.epoch_default} epochs within this machine's {budget} time "
            f"budget; Recommended falls back to the model that fits the most epochs, fastest first."
        )
    others = [a for a in feasible.values() if a.architecture != rec.architecture]

    # 2. Fast: one step lighter - nearest lighter tier, else a faster model of the same tier, else fewer epochs.
    lighter = _nearest_tier([a for a in others if _tier(a) < _tier(rec) and _seconds(a) < _seconds(rec)], below=True) or [
        a for a in others if _tier(a) == _tier(rec) and _seconds(a) < _seconds(rec)
    ]
    fast, fast_variant = None, False
    if lighter:
        fast = min(lighter, key=_seconds)
    elif rec.epochs and rec.epochs > policy.epoch_min:
        fast, fast_variant = _with_epochs(rec, max(policy.epoch_min, rec.epochs // 2), policy), True
        notes.append("No lighter architecture trains faster on this machine; Fast is the Recommended model with fewer epochs.")
    else:
        notes.append("No faster alternative exists: no lighter architecture is faster and Recommended is already at the minimum epochs.")

    # 3. High-Capacity: one step heavier - nearest heavier tier, else a larger model of the same tier, else more epochs.
    heavier = _nearest_tier([a for a in others if _tier(a) > _tier(rec)], below=False) or [
        a for a in others if _tier(a) == _tier(rec) and a.param_count > rec.param_count
    ]
    high, high_variant = None, False
    if heavier:
        high = max(heavier, key=lambda a: a.param_count)
    elif rec.epochs and rec.epochs < policy.epoch_max:
        high, high_variant = _with_epochs(rec, min(policy.epoch_max, rec.epochs * 2), policy), True
        notes.append("No heavier architecture fits this machine; High-Capacity is the Recommended model with more epochs.")
    else:
        notes.append("No heavier alternative exists: no larger architecture fits and Recommended is already at the maximum epochs.")

    recommendations = [
        _config_from_assessment(RECOMMENDED, rec, _recommended_reason(rec, bool(comfortable), budget), _recommended_tradeoff(rec)),
    ]
    high_not_recommended = False
    if high is not None:
        high_not_recommended = not _is_comfortable(high, policy) or _seconds(high) > budget_seconds
        label = _MORE_EPOCHS_LABELS[high_not_recommended] if high_variant else (_NOT_RECOMMENDED_LABEL if high_not_recommended else None)
        recommendations.append(_config_from_assessment(
            HIGH_CAPACITY, high, _high_reason(high, rec, high_variant, high_not_recommended, budget), _high_tradeoff(high_variant), label=label,
        ))
    if fast is not None:
        recommendations.append(_config_from_assessment(
            FAST, fast, _fast_reason(fast, rec, fast_variant), _fast_tradeoff(fast_variant), label=_FEWER_EPOCHS_LABEL if fast_variant else None,
        ))

    # Per-architecture labels. Epoch variants reuse the Recommended architecture, which keeps its own label.
    picked = {rec.architecture: (RECOMMENDED, "Recommended")}
    if high is not None and not high_variant:
        picked[high.architecture] = (HIGH_CAPACITY, "Not Recommended" if high_not_recommended else "High-Capacity")
    if fast is not None and not fast_variant:
        picked[fast.architecture] = (FAST, "Fast")
    for arch, assessment in assessments.items():
        if arch in picked:
            assessment.role, assessment.status_label = picked[arch]
        else:
            _label_unpicked(assessment, policy, budget)

    return RecommendationSet(recommendations, assessments, notes)
