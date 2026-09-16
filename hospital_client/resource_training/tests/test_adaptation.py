from hospital_client.model_management.config import ModelConfig
from hospital_client.resource_training.adaptation import adapt_config_for_oom, detect_cuda_oom
from hospital_client.resource_training.policy import default_policy
from hospital_client.training.config import TrainingConfig
from hospital_client.training.result import TrainingResult, TrainingStatus


def _config(batch_size):
    return TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2),
        dataset_dir="d", output_dir="o", device="cuda", batch_size=batch_size,
    )


def test_detect_cuda_oom_true_when_failed_with_oom_message():
    result = TrainingResult(status=TrainingStatus.TRAINING_FAILED, model_id="m", architecture="resnet18", errors=["CUDA out of memory: tried to allocate ..."])
    assert detect_cuda_oom(result) is True


def test_detect_cuda_oom_false_when_completed():
    result = TrainingResult(status=TrainingStatus.TRAINING_COMPLETED, model_id="m", architecture="resnet18")
    assert detect_cuda_oom(result) is False


def test_detect_cuda_oom_false_when_failed_for_other_reason():
    result = TrainingResult(status=TrainingStatus.TRAINING_FAILED, model_id="m", architecture="resnet18", errors=["Some other RuntimeError"])
    assert detect_cuda_oom(result) is False


def test_adapt_config_halves_batch_size():
    policy = default_policy()
    adapted = adapt_config_for_oom(_config(16), policy)
    assert adapted.batch_size == 8
    assert adapted.model_config.architecture == "resnet18"  # architecture never changed
    assert adapted.device == "cuda"  # device never silently changed


def test_adapt_config_never_reduces_below_one():
    policy = default_policy()
    adapted = adapt_config_for_oom(_config(1), policy)
    assert adapted is None  # caller must stop retrying


def test_adapt_config_floors_at_one_not_zero():
    policy = default_policy()
    policy.batch_size_reduction_factor = 0.1
    adapted = adapt_config_for_oom(_config(2), policy)
    assert adapted.batch_size == 1


def test_original_config_never_mutated():
    policy = default_policy()
    original = _config(16)
    adapt_config_for_oom(original, policy)
    assert original.batch_size == 16
