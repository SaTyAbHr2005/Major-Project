"""
Exercises runner.py's CUDA-OOM adaptation/retry loop with a MONKEYPATCHED
Trainer.run() (simulating OOM results), the same technique Module 7's own
test suite uses for its CUDA-OOM handling test - never a fabricated claim
about real GPU behavior, just a controlled unit test of the retry/adaptation
bookkeeping around Trainer.run().
"""
from hospital_client.model_management.config import ModelConfig
from hospital_client.resource_training import runner as runner_module
from hospital_client.resource_training.policy import default_policy
from hospital_client.resource_training.resource_profile import ResourceProfile
from hospital_client.training.config import TrainingConfig
from hospital_client.training.result import TrainingResult, TrainingStatus


def _cuda_profile():
    return ResourceProfile(
        timestamp="t",
        cpu={"processor": "sim", "physical_cores": 4, "logical_cores": 8, "utilization_percent": 5.0},
        ram={"total_mb": 16000.0, "available_mb": 12000.0, "utilization_percent": 20.0},
        gpu={"cuda_available": True, "gpu_name": "Simulated GPU", "gpu_vendor": "NVIDIA",
             "total_vram_mb": 4000.0, "free_vram_mb": 4000.0, "cuda_capability": "8.6", "fallback_reason": None},
        storage={"path": ".", "total_mb": 100000.0, "free_mb": 50000.0, "utilization_percent": 50.0},
        network={"network_available": None, "latency_ms": None, "checked_host": None},
    )


def _config(batch_size=16):
    return TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2),
        dataset_dir="d", output_dir="o", device="cuda", batch_size=batch_size, epochs=1,
    )


def _oom_result():
    return TrainingResult(status=TrainingStatus.TRAINING_FAILED, model_id="m", architecture="resnet18",
                           errors=["CUDA out of memory: tried to allocate 200 MiB"])


def _ok_result():
    return TrainingResult(
        status=TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION, model_id="m", architecture="resnet18",
        epochs_run=1, training_duration_seconds=2.0, training_metrics={"sample_count": 20}, resolved_precision="fp16",
    )


def test_adapts_batch_size_and_retries_after_simulated_oom(monkeypatch):
    calls = []

    class _FakeTrainer:
        def __init__(self, config):
            calls.append(config.batch_size)
        def run(self):
            return _oom_result() if len(calls) == 1 else _ok_result()

    monkeypatch.setattr(runner_module, "Trainer", _FakeTrainer)
    # The handoff step (-> M9) is exercised for real in test_runner_integration.py;
    # here it is stubbed out since these fake TrainingResults have no real
    # Module 6 checkpoint on disk to load.
    monkeypatch.setattr(runner_module, "_build_federation_handoff", lambda result, output_dir: "stub-handoff")
    policy = default_policy()
    policy.max_adaptation_attempts = 2
    policy.monitor_interval_seconds = 0.05

    result, stats, handoff = runner_module.run_resource_aware_training(_config(16), policy, pre_training_profile=_cuda_profile())
    assert handoff == "stub-handoff"

    assert calls == [16, 8]  # halved exactly once, then succeeded
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert len(stats.adaptations) == 1
    assert stats.adaptations[0]["outcome"] == "retrying"
    assert stats.adaptations[0]["old_batch_size"] == 16
    assert stats.adaptations[0]["new_batch_size"] == 8
    assert stats.cuda_oom_event_count == 1


def test_stops_retrying_after_max_adaptation_attempts(monkeypatch):
    calls = []

    class _AlwaysOomTrainer:
        def __init__(self, config):
            calls.append(config.batch_size)
        def run(self):
            return _oom_result()

    monkeypatch.setattr(runner_module, "Trainer", _AlwaysOomTrainer)
    policy = default_policy()
    policy.max_adaptation_attempts = 2
    policy.monitor_interval_seconds = 0.05

    result, stats, handoff = runner_module.run_resource_aware_training(_config(16), policy, pre_training_profile=_cuda_profile())
    assert handoff is None  # TRAINING_FAILED never reaches federation

    assert calls == [16, 8, 4]  # initial attempt + 2 adaptations, then stops (attempts exhausted)
    assert result.status == TrainingStatus.TRAINING_FAILED
    retrying = [a for a in stats.adaptations if a["outcome"] == "retrying"]
    assert len(retrying) == 2


def test_gives_up_immediately_when_batch_size_already_at_minimum(monkeypatch):
    calls = []

    class _AlwaysOomTrainer:
        def __init__(self, config):
            calls.append(config.batch_size)
        def run(self):
            return _oom_result()

    monkeypatch.setattr(runner_module, "Trainer", _AlwaysOomTrainer)
    policy = default_policy()
    policy.monitor_interval_seconds = 0.05

    result, stats, handoff = runner_module.run_resource_aware_training(_config(1), policy, pre_training_profile=_cuda_profile())
    assert handoff is None

    assert calls == [1]  # never retried - batch_size=1 cannot be reduced further
    assert result.status == TrainingStatus.TRAINING_FAILED
    assert stats.adaptations[0]["outcome"] == "exhausted"
    assert stats.adaptations[0]["new_batch_size"] is None


def test_device_check_fails_clearly_when_cuda_config_but_no_cuda_hardware(monkeypatch):
    from hospital_client.resource_training.runner import ResourceCheckFailed

    no_cuda_profile = _cuda_profile()
    no_cuda_profile.gpu = {
        "cuda_available": False, "gpu_name": None, "gpu_vendor": None, "total_vram_mb": None,
        "free_vram_mb": None, "cuda_capability": None, "fallback_reason": "Simulated: CUDA vanished.",
    }
    policy = default_policy()

    try:
        runner_module.run_resource_aware_training(_config(16), policy, pre_training_profile=no_cuda_profile)
        assert False, "expected ResourceCheckFailed"
    except ResourceCheckFailed as e:
        assert "cuda" in str(e).lower()
