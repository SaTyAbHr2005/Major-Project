import pytest

from hospital_client.model_management.config import ModelConfig, validate_model_config


def test_valid_config_passes():
    validate_model_config(ModelConfig(architecture="resnet18", num_classes=3, input_size=(64, 64), color_mode="RGB"))


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        validate_model_config(ModelConfig(architecture="not_a_real_model", num_classes=2))


def test_invalid_num_classes_rejected():
    with pytest.raises(ValueError, match="num_classes"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=1))


def test_num_classes_bool_rejected():
    with pytest.raises(ValueError, match="num_classes"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=True))


def test_invalid_input_size_rejected():
    with pytest.raises(ValueError, match="input_size"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=2, input_size=(0, 64)))


def test_invalid_input_size_wrong_length_rejected():
    with pytest.raises(ValueError, match="input_size"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64, 64)))


def test_invalid_color_mode_rejected():
    with pytest.raises(ValueError, match="color_mode"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=2, color_mode="CMYK"))


def test_invalid_pretrained_type_rejected():
    with pytest.raises(ValueError, match="pretrained"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=2, pretrained="yes"))


def test_invalid_dropout_rejected():
    with pytest.raises(ValueError, match="dropout"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=2, dropout=1.5))


def test_negative_dropout_rejected():
    with pytest.raises(ValueError, match="dropout"):
        validate_model_config(ModelConfig(architecture="resnet18", num_classes=2, dropout=-0.1))


def test_config_to_dict_and_from_dict_roundtrip():
    config = ModelConfig(architecture="mobilenet_v2", num_classes=5, input_size=(96, 96), color_mode="L", pretrained=False, dropout=0.2)
    restored = ModelConfig.from_dict(config.to_dict())
    assert restored == config
