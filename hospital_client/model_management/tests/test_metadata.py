import pytest

from hospital_client.model_management.metadata import ModelMetadata


def test_metadata_defaults():
    m = ModelMetadata(model_id="a", version=1, architecture="resnet18")
    assert m.status == "draft"
    assert m.epochs is None
    assert m.warnings == []


def test_invalid_status_rejected():
    with pytest.raises(ValueError, match="status"):
        ModelMetadata(model_id="a", version=1, architecture="resnet18", status="approved_by_hospital")


def test_to_dict_from_dict_roundtrip():
    m = ModelMetadata(
        model_id="a", version=2, architecture="resnet18", status="trained",
        config={"architecture": "resnet18", "num_classes": 2},
        num_classes=2, class_mapping={"normal": 0, "disease": 1},
        parent_version=1, total_parameters=100, trainable_parameters=100,
    )
    restored = ModelMetadata.from_dict(m.to_dict())
    assert restored.to_dict() == m.to_dict()


def test_from_dict_ignores_unknown_fields():
    data = {"model_id": "a", "version": 1, "architecture": "resnet18", "unknown_backend_field": 123}
    m = ModelMetadata.from_dict(data)
    assert m.model_id == "a"


def test_training_fields_are_none_until_module7_sets_them():
    m = ModelMetadata(model_id="a", version=1, architecture="resnet18")
    assert m.epochs is None
    assert m.optimizer is None
    assert m.learning_rate is None
    assert m.random_seed is None
