"""
Tests the recommendation-selection algorithm. Uses two levels:

1. `select_recommendations()` directly against hand-crafted ModelAssessment
   inputs - deterministic, exercises exact edge cases (ties, 1/2/3 feasible
   models, EfficientNet-B0 in every role) without depending on the real
   memory/time heuristics.
2. `generate_recommendations()` against the simulated hardware-tier fixtures
   (conftest.py) - exercises the real evaluator end-to-end and confirms
   recommendations genuinely change across hardware profiles.
"""
import pytest

from hospital_client.model_management.registry import ARCHITECTURE_CATALOG
from hospital_client.resource_training.evaluator import ModelAssessment
from hospital_client.resource_training.policy import default_policy
from hospital_client.resource_training.recommender import (
    FAST, HIGH_CAPACITY, RECOMMENDED, generate_recommendations, select_recommendations,
)

BANNED_PHRASES = [
    "best accuracy", "highest accuracy", "guaranteed", "clinically superior",
    "medically better", "medically superior", "more epochs will produce better accuracy",
    "more gpu power guarantees", "this model has lower accuracy",
]


def _assessment(architecture, resource_tier, param_count, seconds, feasible=True, vram_mb=1000.0, available_mb=4000.0):
    return ModelAssessment(
        architecture=architecture, display_name=ARCHITECTURE_CATALOG[architecture].display_name,
        resource_tier=resource_tier, param_count=param_count, feasible=feasible, safe=feasible,
        device="cuda", precision="fp16", batch_size=16 if feasible else None,
        batch_size_reason="test", epochs=10 if feasible else None, epoch_policy_reason="test",
        num_workers=2, memory_estimate={"estimated_total_mb": vram_mb} if feasible else {},
        available_memory_mb=available_mb,
        time_estimate={
            "estimated_training_time_seconds": seconds, "estimated_training_time_minutes": seconds / 60.0,
            "estimated_training_time_display": f"{seconds:.0f} seconds", "estimation_method": "baseline_estimate",
            "estimation_confidence": "low",
        } if feasible else None,
        reason="test reason" if feasible else "unsafe: too little memory",
    )


def _all_eight(overrides=None):
    """A full 8-architecture assessment set with plausible tier-correlated numbers, all feasible."""
    base = {
        "mobilevit_xxs": ("very_low_end", 1_300_000, 20),
        "mobilenet_v3_small": ("very_low_end", 2_500_000, 25),
        "resnet18": ("low_end", 11_700_000, 60),
        "mobilenet_v2": ("low_end", 3_500_000, 45),
        "efficientnet_b0": ("medium_end", 5_300_000, 70),
        "resnet50": ("medium_end", 25_600_000, 140),
        "vit_b16": ("high_end", 86_000_000, 400),
        "efficientnet_b4": ("high_end", 19_000_000, 260),
    }
    if overrides:
        base.update(overrides)
    return {arch: _assessment(arch, tier, params, secs) for arch, (tier, params, secs) in base.items()}


# ---------------------------------------------------------------------
# select_recommendations(): pure algorithm, hand-crafted inputs
# ---------------------------------------------------------------------

def test_three_distinct_recommendations_when_all_eight_feasible():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "medium_end", policy)
    assert len(result.recommendations) == 3
    types = {r.recommendation_type for r in result.recommendations}
    assert types == {RECOMMENDED, HIGH_CAPACITY, FAST}
    architectures = {r.architecture for r in result.recommendations}
    assert len(architectures) == 3  # not the same model three times


def test_high_capacity_is_the_largest_feasible_model():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "high_end", policy)
    high = next(r for r in result.recommendations if r.recommendation_type == HIGH_CAPACITY)
    assert high.architecture == "vit_b16"


def test_fast_is_the_minimal_time_feasible_model():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "very_low_end", policy)
    fast = next(r for r in result.recommendations if r.recommendation_type == FAST)
    assert fast.architecture == "mobilevit_xxs"


def test_recommended_tracks_the_machine_tier_not_a_fixed_ranking():
    policy = default_policy()
    weak = select_recommendations(_all_eight(), "very_low_end", policy)
    strong = select_recommendations(_all_eight(), "high_end", policy)
    weak_rec = next(r for r in weak.recommendations if r.recommendation_type == RECOMMENDED)
    strong_rec = next(r for r in strong.recommendations if r.recommendation_type == RECOMMENDED)
    assert weak_rec.architecture != strong_rec.architecture


def test_efficientnet_b0_present_in_all_model_assessments_even_when_not_picked():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "high_end", policy)
    assert "efficientnet_b0" in result.all_model_assessments
    effnet = result.all_model_assessments["efficientnet_b0"]
    assert effnet.status_label != "Unassigned"
    assert effnet.reason


def test_efficientnet_b0_can_be_recommended_when_numerically_favorable():
    policy = default_policy()
    # Craft numbers so efficientnet_b0 is the clear balanced winner at medium tier:
    # cheaper in time than resnet50 while still being medium-tier capacity.
    overrides = {"efficientnet_b0": ("medium_end", 5_300_000, 30), "resnet50": ("medium_end", 25_600_000, 300)}
    result = select_recommendations(_all_eight(overrides), "medium_end", policy)
    recommended = next(r for r in result.recommendations if r.recommendation_type == RECOMMENDED)
    assert recommended.architecture == "efficientnet_b0"
    assert result.all_model_assessments["efficientnet_b0"].status_label == "Recommended"


def test_efficientnet_b0_can_be_high_capacity_when_it_is_the_strongest_feasible_model():
    policy = default_policy()
    # Craft a scenario where every other architecture has fewer parameters
    # than EfficientNet-B0 - it must then legitimately win High-Capacity.
    overrides = {
        "mobilevit_xxs": ("very_low_end", 1_000_000, 10), "mobilenet_v3_small": ("very_low_end", 2_000_000, 15),
        "resnet18": ("low_end", 3_000_000, 20), "mobilenet_v2": ("low_end", 3_200_000, 22),
        "efficientnet_b0": ("medium_end", 5_300_000, 70), "resnet50": ("very_low_end", 3_500_000, 25),
        "vit_b16": ("very_low_end", 3_800_000, 28), "efficientnet_b4": ("very_low_end", 4_000_000, 30),
    }
    result = select_recommendations(_all_eight(overrides), "medium_end", policy)
    high = next(r for r in result.recommendations if r.recommendation_type == HIGH_CAPACITY)
    assert high.architecture == "efficientnet_b0"


def test_efficientnet_b0_marked_unsafe_when_infeasible():
    assessments = _all_eight()
    assessments["efficientnet_b0"] = _assessment("efficientnet_b0", "medium_end", 5_300_000, 0, feasible=False)
    policy = default_policy()
    result = select_recommendations(assessments, "medium_end", policy)
    assert result.all_model_assessments["efficientnet_b0"].status_label == "Unsafe"
    assert not any(r.architecture == "efficientnet_b0" for r in result.recommendations)


def test_efficientnet_b0_labeled_high_load_or_suitable_when_feasible_but_unpicked():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "high_end", policy)  # high_end picks vit_b16/efficientnet_b4/resnet50-ish
    effnet = result.all_model_assessments["efficientnet_b0"]
    if effnet.role is None:
        assert effnet.status_label in ("High Load", "Suitable")


def test_two_feasible_models_returns_two_recommendations_with_explanatory_note():
    policy = default_policy()
    assessments = {
        "resnet18": _assessment("resnet18", "low_end", 11_700_000, 60),
        "vit_b16": _assessment("vit_b16", "high_end", 86_000_000, 400),
    }
    for arch in ARCHITECTURE_CATALOG:
        if arch not in assessments:
            assessments[arch] = _assessment(arch, "high_end", 1, 1, feasible=False)
    result = select_recommendations(assessments, "medium_end", policy)
    assert len(result.recommendations) == 2
    assert result.notes


def test_one_feasible_model_returns_one_recommendation_with_explanatory_note():
    policy = default_policy()
    assessments = {arch: _assessment(arch, "high_end", 1, 1, feasible=False) for arch in ARCHITECTURE_CATALOG}
    assessments["resnet18"] = _assessment("resnet18", "low_end", 11_700_000, 60)
    result = select_recommendations(assessments, "medium_end", policy)
    assert len(result.recommendations) == 1
    assert result.recommendations[0].architecture == "resnet18"
    assert result.notes


def test_zero_feasible_models_returns_empty_list_never_invents_one():
    policy = default_policy()
    assessments = {arch: _assessment(arch, "high_end", 1, 1, feasible=False) for arch in ARCHITECTURE_CATALOG}
    result = select_recommendations(assessments, "medium_end", policy)
    assert result.recommendations == []
    assert result.notes
    assert all(a.status_label == "Unsafe" for a in result.all_model_assessments.values())


def test_no_unsafe_recommendation_is_ever_returned():
    policy = default_policy()
    assessments = _all_eight()
    assessments["vit_b16"] = _assessment("vit_b16", "high_end", 86_000_000, 400, feasible=False)
    result = select_recommendations(assessments, "high_end", policy)
    assert all(r.safe for r in result.recommendations)
    assert not any(r.architecture == "vit_b16" for r in result.recommendations)


@pytest.mark.parametrize("tier", ["very_low_end", "low_end", "medium_end", "high_end"])
def test_every_recommendation_has_reason_tradeoff_and_numeric_time(tier):
    policy = default_policy()
    result = select_recommendations(_all_eight(), tier, policy)
    for r in result.recommendations:
        assert r.reason
        assert r.tradeoff
        assert isinstance(r.estimated_training_time_minutes, float)
        assert r.estimated_training_time_minutes >= 0


def test_no_banned_accuracy_phrases_anywhere_in_generated_text():
    policy = default_policy()
    for tier in ["very_low_end", "low_end", "medium_end", "high_end"]:
        result = select_recommendations(_all_eight(), tier, policy)
        haystacks = [r.reason.lower() + " " + r.tradeoff.lower() for r in result.recommendations]
        haystacks += [a.reason.lower() for a in result.all_model_assessments.values()]
        for text in haystacks:
            for phrase in BANNED_PHRASES:
                assert phrase not in text, f"banned phrase {phrase!r} found in: {text}"


# ---------------------------------------------------------------------
# generate_recommendations(): full pipeline against simulated hardware
# ---------------------------------------------------------------------

def test_recommendations_change_across_simulated_hardware_tiers(all_simulated_profiles, policy, small_dataset_characteristics):
    picks = {}
    for name, profile in all_simulated_profiles.items():
        rec_set = generate_recommendations(profile, policy, small_dataset_characteristics)
        recommended = next((r for r in rec_set.recommendations if r.recommendation_type == RECOMMENDED), None)
        picks[name] = recommended.architecture if recommended else None
    assert picks["very_low"] != picks["high"]


def test_full_pipeline_efficientnet_b0_always_present(all_simulated_profiles, policy, small_dataset_characteristics):
    for profile in all_simulated_profiles.values():
        rec_set = generate_recommendations(profile, policy, small_dataset_characteristics)
        assert "efficientnet_b0" in rec_set.all_model_assessments


def test_full_pipeline_no_unsafe_recommendation(all_simulated_profiles, policy, small_dataset_characteristics):
    for profile in all_simulated_profiles.values():
        rec_set = generate_recommendations(profile, policy, small_dataset_characteristics)
        assert all(r.safe for r in rec_set.recommendations)
