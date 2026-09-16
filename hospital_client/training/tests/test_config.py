import pytest

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig, validate_training_config


def _base_config(**overrides):
    defaults = dict(
        model_id="m",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)),
        dataset_dir="/tmp/dataset",
        output_dir="/tmp/output",
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_valid_config_passes():
    validate_training_config(_base_config())


def test_invalid_model_config_is_rejected_via_module6():
    bad = ModelConfig(architecture="not_real", num_classes=2)
    with pytest.raises(ValueError, match="Unknown architecture"):
        validate_training_config(_base_config(model_config=bad))


def test_missing_dataset_dir_rejected():
    with pytest.raises(ValueError, match="dataset_dir"):
        validate_training_config(_base_config(dataset_dir=""))


def test_missing_output_dir_rejected():
    with pytest.raises(ValueError, match="output_dir"):
        validate_training_config(_base_config(output_dir=""))


def test_invalid_device_rejected():
    with pytest.raises(ValueError, match="device"):
        validate_training_config(_base_config(device="tpu"))


def test_invalid_epochs_rejected():
    with pytest.raises(ValueError, match="epochs"):
        validate_training_config(_base_config(epochs=0))


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        validate_training_config(_base_config(batch_size=0))


def test_invalid_learning_rate_rejected():
    with pytest.raises(ValueError, match="learning_rate"):
        validate_training_config(_base_config(learning_rate=0))


def test_invalid_optimizer_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        validate_training_config(_base_config(optimizer="rmsprop"))


def test_invalid_scheduler_rejected():
    with pytest.raises(ValueError, match="scheduler"):
        validate_training_config(_base_config(scheduler="cyclic"))


def test_invalid_class_weighting_rejected():
    with pytest.raises(ValueError, match="class_weighting"):
        validate_training_config(_base_config(class_weighting="magic"))


def test_early_stopping_requires_valid_metric():
    with pytest.raises(ValueError, match="early_stopping_metric"):
        validate_training_config(_base_config(early_stopping=True, early_stopping_metric="bogus"))


def test_early_stopping_requires_validation():
    with pytest.raises(ValueError, match="require_validation"):
        validate_training_config(_base_config(early_stopping=True, require_validation=False))


def test_negative_num_workers_rejected():
    with pytest.raises(ValueError, match="num_workers"):
        validate_training_config(_base_config(num_workers=-1))


def test_config_to_dict_roundtrips_model_config():
    config = _base_config()
    d = config.to_dict()
    assert d["model_config"]["architecture"] == "resnet18"
    assert d["model_id"] == "m"
