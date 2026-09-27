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


def _assessment(architecture, resource_tier, param_count, seconds, feasible=True, vram_mb=1000.0, available_mb=4000.0, epochs=10):
    return ModelAssessment(
        architecture=architecture, display_name=ARCHITECTURE_CATALOG[architecture].display_name,
        resource_tier=resource_tier, param_count=param_count, feasible=feasible, safe=feasible,
        device="cuda", precision="fp16", batch_size=16 if feasible else None,
        batch_size_reason="test", epochs=epochs if feasible else None, epoch_policy_reason="test",
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
    # Optional 4th tuple element: epochs that fit the machine's time budget (default 10 = the policy default).
    return {arch: _assessment(arch, *spec) if len(spec) == 3 else _assessment(arch, *spec[:3], epochs=spec[3]) for arch, spec in base.items()}


def _rtx3050_like():
    """High-end models fit in memory but only get the minimum 3 epochs in the time budget."""
    return _all_eight({
        "vit_b16": ("high_end", 86_000_000, 1800, 3),
        "efficientnet_b4": ("high_end", 19_000_000, 1800, 3),
    })


# ---------------------------------------------------------------------
# select_recommendations(): pure algorithm, hand-crafted inputs
# ---------------------------------------------------------------------

def test_three_distinct_recommendations_when_all_eight_feasible():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "medium_end", policy)
    assert len(result.recommendations) == 3
    types = {r.recommendation_type for r in result.recommendations}
    assert types == {RECOMMENDED, HIGH_CAPACITY, FAST}
    configurations = {(r.architecture, r.epochs) for r in result.recommendations}
    assert len(configurations) == 3  # never the same configuration twice


def test_high_capacity_is_the_largest_feasible_model():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "high_end", policy)
    high = next(r for r in result.recommendations if r.recommendation_type == HIGH_CAPACITY)
    assert high.architecture == "vit_b16"


def test_fast_is_one_tier_below_recommended_not_the_absolute_fastest():
    result = select_recommendations(_rtx3050_like(), "low_end", default_policy())
    picks = {r.recommendation_type: r for r in result.recommendations}
    tiers = {k: result.all_model_assessments[r.architecture].resource_tier for k, r in picks.items()}
    assert tiers[RECOMMENDED] == "medium_end"
    assert tiers[FAST] == "low_end"  # not very_low_end, even though mobilevit_xxs is the fastest overall
    assert picks[FAST].architecture == "mobilenet_v2"  # the faster of the two low-end models


def test_top_of_the_ladder_offers_recommended_model_with_more_epochs():
    # Everything trains comfortably: Recommended is the largest model, so the
    # heavier option is the same model with more epochs rather than nothing.
    result = select_recommendations(_all_eight(), "high_end", default_policy())
    picks = {r.recommendation_type: r for r in result.recommendations}
    assert picks[HIGH_CAPACITY].architecture == picks[RECOMMENDED].architecture
    assert picks[HIGH_CAPACITY].epochs == 20 and picks[RECOMMENDED].epochs == 10
    assert picks[HIGH_CAPACITY].label.startswith("More Epochs")
    assert picks[HIGH_CAPACITY].estimated_training_time_seconds > picks[RECOMMENDED].estimated_training_time_seconds
    assert any("more epochs" in n for n in result.notes)


def test_bottom_of_the_ladder_offers_recommended_model_with_fewer_epochs():
    # Only very-low-end models train comfortably and Recommended is the fastest of them.
    inputs = _all_eight({
        "mobilevit_xxs": ("very_low_end", 1_300_000, 200), "mobilenet_v3_small": ("very_low_end", 2_500_000, 150),
        "resnet18": ("low_end", 11_700_000, 900, 3), "mobilenet_v2": ("low_end", 3_500_000, 900, 3),
        "efficientnet_b0": ("medium_end", 5_300_000, 900, 3), "resnet50": ("medium_end", 25_600_000, 900, 3),
        "vit_b16": ("high_end", 86_000_000, 900, 3), "efficientnet_b4": ("high_end", 19_000_000, 900, 3),
    })
    result = select_recommendations(inputs, "very_low_end", default_policy())
    picks = {r.recommendation_type: r for r in result.recommendations}
    assert picks[RECOMMENDED].architecture == "mobilenet_v3_small"  # heavier of the two, and also the fastest
    assert picks[FAST].architecture == "mobilenet_v3_small"
    assert picks[FAST].epochs == 5 and picks[FAST].label == "Fast / Fewer Epochs"
    assert picks[FAST].estimated_training_time_seconds < picks[RECOMMENDED].estimated_training_time_seconds


def test_epoch_variant_time_scales_linearly_plus_fixed_overhead():
    policy = default_policy()
    result = select_recommendations(_all_eight(), "high_end", policy)
    picks = {r.recommendation_type: r for r in result.recommendations}
    oh = policy.fixed_overhead_seconds
    rec, more = picks[RECOMMENDED], picks[HIGH_CAPACITY]
    assert more.estimated_training_time_seconds == pytest.approx(oh + (rec.estimated_training_time_seconds - oh) * 2)


def test_recommended_tracks_what_the_machine_trains_comfortably():
    policy = default_policy()
    # Weak machine: only very-low/low-end models get the full default epochs.
    weak_inputs = _all_eight({
        "efficientnet_b0": ("medium_end", 5_300_000, 900, 3), "resnet50": ("medium_end", 25_600_000, 1200, 3),
        "vit_b16": ("high_end", 86_000_000, 1800, 3), "efficientnet_b4": ("high_end", 19_000_000, 1800, 3),
    })
    weak = select_recommendations(weak_inputs, "very_low_end", policy)
    strong = select_recommendations(_all_eight(), "high_end", policy)  # everything comfortable
    weak_rec = next(r for r in weak.recommendations if r.recommendation_type == RECOMMENDED)
    strong_rec = next(r for r in strong.recommendations if r.recommendation_type == RECOMMENDED)
    assert weak.all_model_assessments[weak_rec.architecture].resource_tier == "low_end"
    assert strong.all_model_assessments[strong_rec.architecture].resource_tier == "high_end"


def test_rtx3050_like_machine_recommends_a_medium_tier_model():
    result = select_recommendations(_rtx3050_like(), "low_end", default_policy())
    picks = {r.recommendation_type: r for r in result.recommendations}
    assert result.all_model_assessments[picks[RECOMMENDED].architecture].resource_tier == "medium_end"
    assert result.all_model_assessments[picks[FAST].architecture].resource_tier in ("very_low_end", "low_end")
    assert picks[HIGH_CAPACITY].architecture == "vit_b16"


def test_high_capacity_that_does_not_fit_the_time_budget_is_marked_not_recommended_with_costs():
    result = select_recommendations(_rtx3050_like(), "low_end", default_policy())
    high = next(r for r in result.recommendations if r.recommendation_type == HIGH_CAPACITY)
    assert high.label == "High-Capacity / Not Recommended"
    assert result.all_model_assessments["vit_b16"].status_label == "Not Recommended"
    assert "25% of available VRAM" in high.reason  # 1000MB of 4000MB
    assert "only 3 epochs" in high.reason
    assert "as long as the Recommended option" in high.reason


def test_high_capacity_that_fits_comfortably_keeps_the_normal_label():
    # A time-averse policy makes EfficientNet-B4 the Recommended high-end model, so the
    # larger ViT-B/16 of the same tier is the (still comfortable) heavier option.
    policy = default_policy()
    policy.time_penalty_weight = 3.0
    result = select_recommendations(_all_eight(), "high_end", policy)
    picks = {r.recommendation_type: r for r in result.recommendations}
    assert picks[RECOMMENDED].architecture == "efficientnet_b4"
    assert picks[HIGH_CAPACITY].architecture == "vit_b16"
    assert picks[HIGH_CAPACITY].label == "High-Capacity / More Time"
    assert result.all_model_assessments["vit_b16"].status_label == "High-Capacity"


def test_unpicked_model_short_of_the_time_budget_is_high_load_with_costs():
    result = select_recommendations(_rtx3050_like(), "low_end", default_policy())
    b4 = result.all_model_assessments["efficientnet_b4"]
    assert b4.status_label == "High Load"
    assert "% of available VRAM" in b4.reason and "only 3 epochs" in b4.reason


def test_recommended_falls_back_to_balance_with_note_when_nothing_is_comfortable():
    slow = {arch: (tier, params, secs, 3) for arch, (tier, params, secs) in {
        "mobilevit_xxs": ("very_low_end", 1_300_000, 20), "mobilenet_v3_small": ("very_low_end", 2_500_000, 25),
        "resnet18": ("low_end", 11_700_000, 60), "mobilenet_v2": ("low_end", 3_500_000, 45),
        "efficientnet_b0": ("medium_end", 5_300_000, 70), "resnet50": ("medium_end", 25_600_000, 140),
        "vit_b16": ("high_end", 86_000_000, 400), "efficientnet_b4": ("high_end", 19_000_000, 260),
    }.items()}
    result = select_recommendations(_all_eight(slow), "very_low_end", default_policy())
    rec = next(r for r in result.recommendations if r.recommendation_type == RECOMMENDED)
    assert rec.architecture == "mobilevit_xxs"  # the fastest - never the heaviest when the machine is overloaded
    assert any("falls back" in n for n in result.notes)


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
    # cheaper in time than resnet50 while still being medium-tier capacity, and
    # the high-end models don't fit the default epochs in the time budget.
    overrides = {
        "efficientnet_b0": ("medium_end", 5_300_000, 30), "resnet50": ("medium_end", 25_600_000, 1500, 3),
        "efficientnet_b4": ("high_end", 19_000_000, 1800, 3), "vit_b16": ("high_end", 86_000_000, 1800, 3),
    }
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


def test_two_feasible_models_fill_the_ladder_with_an_epoch_variant_and_a_note():
    policy = default_policy()
    assessments = {
        "resnet18": _assessment("resnet18", "low_end", 11_700_000, 60),
        "vit_b16": _assessment("vit_b16", "high_end", 86_000_000, 400),
    }
    for arch in ARCHITECTURE_CATALOG:
        if arch not in assessments:
            assessments[arch] = _assessment(arch, "high_end", 1, 1, feasible=False)
    result = select_recommendations(assessments, "medium_end", policy)
    picks = {r.recommendation_type: r for r in result.recommendations}
    assert (picks[RECOMMENDED].architecture, picks[FAST].architecture) == ("vit_b16", "resnet18")
    assert picks[HIGH_CAPACITY].architecture == "vit_b16" and picks[HIGH_CAPACITY].epochs > picks[RECOMMENDED].epochs
    assert result.notes


def test_one_feasible_model_is_offered_at_three_epoch_budgets_with_notes():
    policy = default_policy()
    assessments = {arch: _assessment(arch, "high_end", 1, 1, feasible=False) for arch in ARCHITECTURE_CATALOG}
    assessments["resnet18"] = _assessment("resnet18", "low_end", 11_700_000, 60)
    result = select_recommendations(assessments, "medium_end", policy)
    assert {r.architecture for r in result.recommendations} == {"resnet18"}
    assert sorted(r.epochs for r in result.recommendations) == [5, 10, 20]
    assert len(result.notes) == 2


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


def _dataset(num_train):
    from hospital_client.resource_training.dataset_characteristics import DatasetCharacteristics
    return DatasetCharacteristics(dataset_dir="synthetic", num_train_samples=num_train,
                                  num_validation_samples=num_train // 8, num_test_samples=0, num_classes=2)


def _picks(rec_set):
    return {r.recommendation_type: r for r in rec_set.recommendations}


def test_full_pipeline_ladder_order_holds_on_every_simulated_machine(all_simulated_profiles, policy):
    # Fast is never a heavier tier than Recommended, High-Capacity never a lighter one - on every machine.
    for name, profile in all_simulated_profiles.items():
        rec_set = generate_recommendations(profile, policy, _dataset(20_000))
        tier = {k: rec_set.all_model_assessments[r.architecture].resource_tier for k, r in _picks(rec_set).items()}
        order = {"very_low_end": 0, "low_end": 1, "medium_end": 2, "high_end": 3}
        if FAST in tier:
            assert order[tier[FAST]] <= order[tier[RECOMMENDED]], name
        if HIGH_CAPACITY in tier:
            assert order[tier[HIGH_CAPACITY]] >= order[tier[RECOMMENDED]], name


def test_full_pipeline_huge_dataset_does_not_push_recommendation_to_a_lighter_model(all_simulated_profiles, policy):
    # 100x more images makes every model ~100x slower; the time budget scales with the
    # dataset, so the same models are chosen and only the reported time grows.
    for name, profile in all_simulated_profiles.items():
        small = _picks(generate_recommendations(profile, policy, _dataset(20_000)))
        huge = _picks(generate_recommendations(profile, policy, _dataset(2_000_000)))
        assert {k: r.architecture for k, r in small.items()} == {k: r.architecture for k, r in huge.items()}, name
        assert huge[RECOMMENDED].estimated_training_time_seconds > 50 * small[RECOMMENDED].estimated_training_time_seconds


def test_full_pipeline_efficientnet_b0_always_present(all_simulated_profiles, policy, small_dataset_characteristics):
    for profile in all_simulated_profiles.values():
        rec_set = generate_recommendations(profile, policy, small_dataset_characteristics)
        assert "efficientnet_b0" in rec_set.all_model_assessments


def test_full_pipeline_no_unsafe_recommendation(all_simulated_profiles, policy, small_dataset_characteristics):
    for profile in all_simulated_profiles.values():
        rec_set = generate_recommendations(profile, policy, small_dataset_characteristics)
        assert all(r.safe for r in rec_set.recommendations)
