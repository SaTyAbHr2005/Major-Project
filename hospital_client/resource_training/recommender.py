"""
Generates up to three dynamically-computed training-configuration
recommendations (Recommended / High-Capacity-More-Time / Fast-Lower-
Resource) from a ResourceEvaluator's per-model assessments, plus a
role/status label for every one of Module 6's 8 architectures (so
EfficientNet-B0 - and every other model - always remains visible, per the
project's add-on requirement, regardless of whether it was picked).

These are RESOURCE/TRAINING-TIME alternatives, not accuracy rankings - see
module8_readme.md. No text generated here claims accuracy superiority;
wording is always built from the actual computed numbers for this machine
and this dataset, never a canned string.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hospital_client.resource_training.dataset_characteristics import DatasetCharacteristics
from hospital_client.resource_training.estimator import HistoricalThroughputStore
from hospital_client.resource_training.evaluator import ModelAssessment, ResourceEvaluator
from hospital_client.resource_training.policy import ResourcePolicy
from hospital_client.resource_training.resource_profile import ResourceProfile

_TIER_ORDINAL = {"very_low_end": 0, "low_end": 1, "medium_end": 2, "high_end": 3}

RECOMMENDED = "recommended"
HIGH_CAPACITY = "high_capacity"
FAST = "fast"

_LABELS = {RECOMMENDED: "Recommended", HIGH_CAPACITY: "High-Capacity / More Time", FAST: "Fast / Lower Resource"}


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


def _resource_summary(a: ModelAssessment) -> str:
    mem = a.memory_estimate or {}
    total = mem.get("estimated_total_mb")
    if total is None or a.available_memory_mb is None:
        return "Resource usage could not be estimated."
    kind = "VRAM" if a.device == "cuda" else "RAM"
    basis = "measured (real local dry run)" if mem.get("measured") else "estimated"
    return (
        f"{basis.capitalize()} peak {kind}: approximately {total / 1024.0:.2f} GB of "
        f"{a.available_memory_mb / 1024.0:.2f} GB available."
    )


def _config_from_assessment(recommendation_type: str, a: ModelAssessment, reason: str, tradeoff: str) -> RecommendedConfig:
    t = a.time_estimate or {}
    return RecommendedConfig(
        recommendation_type=recommendation_type, label=_LABELS[recommendation_type],
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


def _recommended_reason(a: ModelAssessment) -> str:
    t = a.time_estimate or {}
    return (
        f"Recommended balanced configuration: {a.display_name} provides a balanced model capacity and training "
        f"budget for the detected hardware, keeping estimated resource usage comfortably within the configured "
        f"safety margin and an estimated training time of approximately {t.get('estimated_training_time_display', 'an unknown duration')}."
    )


def _recommended_tradeoff(a: ModelAssessment) -> str:
    return (
        "Balances model capacity against training time and resource usage; not a claim of highest achievable "
        "accuracy. Actual experimental performance must be measured on the target dataset."
    )


def _high_capacity_reason(a: ModelAssessment) -> str:
    t = a.time_estimate or {}
    mem = a.memory_estimate or {}
    return (
        f"Uses {a.display_name} ({a.param_count:,} parameters), the highest-capacity architecture that still safely "
        f"fits the detected hardware. Estimated to use ~{mem.get('estimated_total_mb', 0.0) / 1024.0:.2f} GB and take "
        f"approximately {t.get('estimated_training_time_display', 'an unknown duration')} - more than the recommended option."
    )


def _high_capacity_tradeoff(a: ModelAssessment) -> str:
    return (
        "Higher computational load and longer estimated training time than the recommended option. Higher model "
        "capacity is not a guarantee of higher accuracy - actual performance must be evaluated on the target dataset."
    )


def _fast_reason(a: ModelAssessment) -> str:
    t = a.time_estimate or {}
    return (
        f"Fastest feasible option: {a.display_name} minimizes estimated training time (~"
        f"{t.get('estimated_training_time_display', 'an unknown duration')}) and resource usage among the safely "
        f"feasible architectures on this machine."
    )


def _fast_tradeoff(a: ModelAssessment) -> str:
    return (
        "Lower model capacity and a shorter training budget than the other options; may provide lower experimental "
        "performance, but actual performance must be measured on the target dataset."
    )


def _balance_score(a: ModelAssessment, min_p: float, max_p: float, min_t: float, max_t: float, weight: float) -> float:
    norm_capacity = 0.5 if max_p == min_p else (a.param_count - min_p) / (max_p - min_p)
    seconds = (a.time_estimate or {}).get("estimated_training_time_seconds", 0.0)
    norm_time = 0.5 if max_t == min_t else (seconds - min_t) / (max_t - min_t)
    return norm_capacity - weight * norm_time


def _label_unpicked(a: ModelAssessment, feasible: Dict[str, ModelAssessment]) -> None:
    if not a.feasible:
        a.status_label = "Unsafe"
        return
    times = sorted((x.time_estimate or {}).get("estimated_training_time_seconds", 0.0) for x in feasible.values())
    median = times[len(times) // 2]
    seconds = (a.time_estimate or {}).get("estimated_training_time_seconds", 0.0)
    if seconds > median:
        a.status_label = "High Load"
        a.reason = (
            f"{a.display_name} is technically feasible under the current policy, but it is not the primary "
            f"recommendation because it would place a relatively high load on the available resources and take "
            f"approximately {(a.time_estimate or {}).get('estimated_training_time_display', 'longer')} compared with "
            f"the selected alternatives."
        )
    else:
        a.status_label = "Suitable"
        a.reason = (
            f"{a.display_name} is a suitable feasible alternative on this machine (estimated "
            f"{(a.time_estimate or {}).get('estimated_training_time_display', 'an unknown duration')}), but was not "
            f"selected as the Recommended, High-Capacity, or Fast option for this run."
        )


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

    param_counts = [a.param_count for a in feasible.values()]
    times = [(a.time_estimate or {}).get("estimated_training_time_seconds", 0.0) for a in feasible.values()]
    min_p, max_p, min_t, max_t = min(param_counts), max(param_counts), min(times), max(times)

    fast_pick = min(feasible.values(), key=lambda a: (a.time_estimate or {}).get("estimated_training_time_seconds", 0.0))
    high_pick = max(feasible.values(), key=lambda a: a.param_count)

    recommendations: List[RecommendedConfig] = []
    picked_roles: Dict[str, str] = {}

    if len(feasible) >= 3:
        remaining = {k: v for k, v in feasible.items() if k not in (fast_pick.architecture, high_pick.architecture)}
        pool = remaining if remaining else feasible
        # "Balanced" is relative to THIS machine's own capability tier (not a
        # fixed param-count ranking) - a weak machine's balanced pick should
        # trend toward a lighter architecture, a strong machine's toward a
        # heavier one (spec §7). Distance to the machine's tier wins first;
        # the capacity/time balance score only breaks ties within that.
        target_ordinal = _TIER_ORDINAL.get(hardware_tier, 2)
        recommended_pick = min(
            pool.values(),
            key=lambda a: (
                abs(_TIER_ORDINAL.get(a.resource_tier, 2) - target_ordinal),
                -_balance_score(a, min_p, max_p, min_t, max_t, policy.time_penalty_weight),
            ),
        )
        recommendations.append(_config_from_assessment(RECOMMENDED, recommended_pick, _recommended_reason(recommended_pick), _recommended_tradeoff(recommended_pick)))
        recommendations.append(_config_from_assessment(HIGH_CAPACITY, high_pick, _high_capacity_reason(high_pick), _high_capacity_tradeoff(high_pick)))
        recommendations.append(_config_from_assessment(FAST, fast_pick, _fast_reason(fast_pick), _fast_tradeoff(fast_pick)))
        picked_roles = {recommended_pick.architecture: RECOMMENDED, high_pick.architecture: HIGH_CAPACITY, fast_pick.architecture: FAST}

    elif len(feasible) == 2:
        recommendations.append(_config_from_assessment(FAST, fast_pick, _fast_reason(fast_pick), _fast_tradeoff(fast_pick)))
        recommendations.append(_config_from_assessment(HIGH_CAPACITY, high_pick, _high_capacity_reason(high_pick), _high_capacity_tradeoff(high_pick)))
        picked_roles = {fast_pick.architecture: FAST, high_pick.architecture: HIGH_CAPACITY}
        notes.append(
            "Only 2 feasible architectures were found on this hardware; a distinct third (Recommended) option was "
            "not generated because it would not meaningfully differ from Fast or High-Capacity. See "
            "all_model_assessments for the full per-architecture evaluation."
        )

    else:  # exactly 1 feasible
        only = next(iter(feasible.values()))
        recommendations.append(_config_from_assessment(RECOMMENDED, only, _recommended_reason(only), _recommended_tradeoff(only)))
        picked_roles = {only.architecture: RECOMMENDED}
        notes.append(
            "Only 1 feasible architecture was found on this hardware; High-Capacity and Fast alternatives were not "
            "generated because no other architecture safely fits the detected resources within the configured "
            "safety margins."
        )

    for arch, assessment in assessments.items():
        if arch in picked_roles:
            role = picked_roles[arch]
            assessment.role = role
            assessment.status_label = _LABELS[role].split(" /")[0] if role != RECOMMENDED else "Recommended"
        else:
            _label_unpicked(assessment, feasible)

    return RecommendationSet(recommendations, assessments, notes)
