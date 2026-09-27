from hospital_client.model_management.registry import ARCHITECTURE_CATALOG
from hospital_client.resource_training.evaluator import ResourceEvaluator, classify_hardware_tier, select_device


def test_select_device_cuda_available(low_profile):
    device, reason = select_device(low_profile)
    assert device == "cuda"
    assert reason is None


def test_select_device_cuda_unavailable(very_low_profile):
    device, reason = select_device(very_low_profile)
    assert device == "cpu"
    assert reason


def test_evaluate_all_covers_every_registry_architecture(medium_profile, policy, small_dataset_characteristics):
    evaluator = ResourceEvaluator(medium_profile, policy)
    assessments = evaluator.evaluate_all(small_dataset_characteristics)
    assert set(assessments.keys()) == set(ARCHITECTURE_CATALOG.keys())


def test_evaluation_is_deterministic(medium_profile, policy, small_dataset_characteristics):
    # The DECISION logic (model/batch/epoch/precision selection) must be
    # deterministic given the same inputs. Real dry-run wall-clock timing is
    # intentionally excluded from this check - it is a genuine measurement
    # and is expected to vary slightly run to run (see module8_readme.md).
    policy.enable_dry_run_measurement = False
    evaluator = ResourceEvaluator(medium_profile, policy)
    first = evaluator.evaluate_model("resnet18", small_dataset_characteristics)
    second = evaluator.evaluate_model("resnet18", small_dataset_characteristics)
    assert first.to_dict() == second.to_dict()


def test_vit_infeasible_or_smaller_batch_on_very_low_vs_feasible_full_batch_on_high(
    very_low_profile, high_profile, policy, small_dataset_characteristics,
):
    very_low = ResourceEvaluator(very_low_profile, policy).evaluate_model("vit_b16", small_dataset_characteristics)
    high = ResourceEvaluator(high_profile, policy).evaluate_model("vit_b16", small_dataset_characteristics)
    assert high.feasible is True
    if very_low.feasible:
        # If it fits at all on the very weak machine, it must be far more constrained.
        assert very_low.batch_size <= high.batch_size
    # Either way, weak hardware must never get a LARGER batch than strong hardware.


def test_feasibility_never_forced_when_memory_absent(policy, small_dataset_characteristics, very_low_profile):
    broken = very_low_profile
    broken.ram["available_mb"] = None
    evaluator = ResourceEvaluator(broken, policy)
    assessment = evaluator.evaluate_model("resnet18", small_dataset_characteristics)
    assert assessment.feasible is False
    assert assessment.safe is False


def test_batch_size_never_exceeds_configured_candidates(medium_profile, policy, small_dataset_characteristics):
    evaluator = ResourceEvaluator(medium_profile, policy)
    for arch in ARCHITECTURE_CATALOG:
        assessment = evaluator.evaluate_model(arch, small_dataset_characteristics)
        if assessment.batch_size is not None:
            assert assessment.batch_size in policy.batch_size_candidates


def test_epochs_bounded_by_policy(medium_profile, policy, small_dataset_characteristics):
    evaluator = ResourceEvaluator(medium_profile, policy)
    for arch in ARCHITECTURE_CATALOG:
        assessment = evaluator.evaluate_model(arch, small_dataset_characteristics)
        if assessment.epochs is not None:
            assert policy.epoch_min <= assessment.epochs <= policy.epoch_max


def test_precision_is_fp32_on_cpu_never_fp16(very_low_profile, policy, small_dataset_characteristics):
    evaluator = ResourceEvaluator(very_low_profile, policy)
    for arch in ARCHITECTURE_CATALOG:
        assessment = evaluator.evaluate_model(arch, small_dataset_characteristics)
        assert assessment.precision == "fp32"


def test_classify_hardware_tier_orders_correctly(very_low_profile, low_profile, medium_profile, high_profile, policy):
    order = ["very_low_end", "low_end", "medium_end", "high_end"]
    tiers = [
        classify_hardware_tier(very_low_profile, "cpu", policy),
        classify_hardware_tier(low_profile, "cuda", policy),
        classify_hardware_tier(medium_profile, "cuda", policy),
        classify_hardware_tier(high_profile, "cuda", policy),
    ]
    assert [order.index(t) for t in tiers] == sorted(order.index(t) for t in tiers)


def test_unsafe_model_carries_a_numeric_reason(low_profile, policy, small_dataset_characteristics):
    # Force an absurdly tiny safety margin so even small models become unsafe.
    policy.vram_safety_margin = 0.0001
    evaluator = ResourceEvaluator(low_profile, policy)
    assessment = evaluator.evaluate_model("vit_b16", small_dataset_characteristics)
    assert assessment.feasible is False
    assert "MB" in assessment.reason


def test_real_dry_run_measurement_produces_a_measured_memory_and_dry_run_time_estimate(policy, small_dataset_characteristics):
    import torch

    policy.enable_dry_run_measurement = True
    profile = _simulated_profile_for_this_test(cuda=torch.cuda.is_available())
    evaluator = ResourceEvaluator(profile, policy)
    assessment = evaluator.evaluate_model("resnet18", small_dataset_characteristics)

    assert assessment.feasible is True
    assert assessment.memory_estimate.get("measured") is True
    assert assessment.memory_estimate["estimated_total_mb"] > 0
    assert assessment.time_estimate["estimation_method"] in ("dry_run_estimate", "measured_hardware_estimate")


def _simulated_profile_for_this_test(cuda: bool):
    from hospital_client.resource_training.resource_profile import ResourceProfile

    gpu = (
        {"cuda_available": True, "gpu_name": "test-gpu", "gpu_vendor": "NVIDIA", "total_vram_mb": 8000.0,
         "free_vram_mb": 8000.0, "cuda_capability": "8.6", "fallback_reason": None}
        if cuda else
        {"cuda_available": False, "gpu_name": None, "gpu_vendor": None, "total_vram_mb": None,
         "free_vram_mb": None, "cuda_capability": None, "fallback_reason": "no gpu"}
    )
    return ResourceProfile(
        timestamp="t", cpu={"processor": "t", "physical_cores": 4, "logical_cores": 8, "utilization_percent": 5.0},
        ram={"total_mb": 16000.0, "available_mb": 12000.0, "utilization_percent": 20.0}, gpu=gpu,
        storage={"path": ".", "total_mb": 100000.0, "free_mb": 50000.0, "utilization_percent": 50.0},
        network={"network_available": None, "latency_ms": None, "checked_host": None},
    )
