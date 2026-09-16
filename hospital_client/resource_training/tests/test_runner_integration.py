"""
Real M4 -> M5 -> M6 -> M8 -> M7 integration: a genuine synthetic dataset
built through Module 4/5, real Module 8 hardware detection and
recommendation generation, resolved into a real Module 7 TrainingConfig, and
executed through the real Module 7 Trainer. No fabricated results.
"""
import torch
import pytest

from hospital_client.resource_training.dataset_characteristics import inspect_dataset
from hospital_client.resource_training.estimator import HistoricalThroughputStore
from hospital_client.resource_training.policy import default_policy
from hospital_client.resource_training.recommender import FAST, generate_recommendations
from hospital_client.resource_training.resolved_config import to_training_config
from hospital_client.resource_training.resource_profile import build_resource_profile
from hospital_client.resource_training.runner import run_resource_aware_training
from hospital_client.training.result import TrainingStatus


def test_full_pipeline_cpu(real_small_dataset_dir, tmp_path):
    policy = default_policy()
    dataset = inspect_dataset(real_small_dataset_dir)
    assert dataset.num_train_samples > 0

    profile = build_resource_profile(storage_path=str(tmp_path), check_network=False, policy=policy)
    rec_set = generate_recommendations(profile, policy, dataset)
    assert rec_set.recommendations  # some feasible config always exists on CPU for tiny models

    fast = next((r for r in rec_set.recommendations if r.recommendation_type == FAST), rec_set.recommendations[0])
    training_config = to_training_config(
        fast, model_id="itest_cpu", dataset_dir=real_small_dataset_dir,
        output_dir=str(tmp_path / "out"), num_classes=dataset.num_classes,
    )
    # This test exercises the CPU path specifically (a separate skipif-gated
    # test below exercises real CUDA) - force CPU/fp32 regardless of what the
    # real detected hardware recommended, and keep it fast with epochs=1.
    training_config.device = "cpu"
    training_config.precision = "fp32"
    training_config.epochs = 1

    history_path = str(tmp_path / "history.json")
    result, stats, handoff = run_resource_aware_training(training_config, policy, history_path, profile)

    assert result.status in (TrainingStatus.TRAINING_COMPLETED, TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION)
    assert result.epochs_run == 1
    assert stats.actual_training_duration_seconds > 0
    assert stats.hardware["cpu"]["logical_cores"] >= 1
    assert stats.monitor_summary["sample_count"] >= 1

    # -> M9: a federation-ready run produces a real Module 7 FederationHandoff.
    if result.ready_for_federation:
        assert handoff is not None
        assert handoff.architecture == training_config.model_config.architecture
        assert handoff.parameter_count > 0

    # A second recommendation call after a real run should be able to see
    # the newly recorded (real, not fabricated) measured throughput.
    if stats.measured_samples_per_second:
        history = HistoricalThroughputStore(history_path)
        assert history.lookup(fast.architecture, "cpu", result.resolved_precision) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_full_pipeline_cuda(real_small_dataset_dir, tmp_path):
    policy = default_policy()
    dataset = inspect_dataset(real_small_dataset_dir)
    profile = build_resource_profile(storage_path=str(tmp_path), check_network=False, policy=policy)
    assert profile.gpu["cuda_available"] is True

    rec_set = generate_recommendations(profile, policy, dataset)
    fast = next((r for r in rec_set.recommendations if r.recommendation_type == FAST), rec_set.recommendations[0])
    assert fast.device == "cuda"

    training_config = to_training_config(
        fast, model_id="itest_cuda", dataset_dir=real_small_dataset_dir,
        output_dir=str(tmp_path / "out"), num_classes=dataset.num_classes,
    )
    training_config.epochs = 1

    result, stats, handoff = run_resource_aware_training(training_config, policy, str(tmp_path / "history.json"), profile)
    assert result.status in (TrainingStatus.TRAINING_COMPLETED, TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION)
    assert stats.monitor_summary["peak_vram_mb"] is not None
    if result.ready_for_federation:
        assert handoff is not None
        assert handoff.device == "cuda"
