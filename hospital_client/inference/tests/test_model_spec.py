"""E. Class mapping / required metadata: the adapter never guesses what Module 6/7 did not record."""
import pytest

from hospital_client.inference.errors import ModelCompatibilityError
from hospital_client.inference.model_spec import UNAVAILABLE, build_model_spec
from hospital_client.model_management.metadata import ModelMetadata


def _metadata(**overrides):
    base = dict(
        model_id="m", version=2, architecture="resnet18",
        config={"architecture": "resnet18", "num_classes": 2, "input_size": [32, 32], "color_mode": "RGB",
                "pretrained": False, "dropout": None},
        status="trained", artifact_sha256="ab" * 32,
        input_spec={"input_size": [32, 32], "color_mode": "RGB"}, num_classes=2,
        class_mapping={"benign": 0, "malignant": 1},
    )
    base.update(overrides)
    return ModelMetadata(**base)


def test_valid_metadata_builds_spec_with_inverted_mapping():
    spec = build_model_spec(_metadata())
    assert spec.index_to_label == {0: "benign", 1: "malignant"}
    assert spec.input_size == (32, 32) and spec.color_mode == "RGB" and spec.version == 2


def test_index_meaning_comes_from_the_stored_mapping_not_from_an_assumption():
    spec = build_model_spec(_metadata(class_mapping={"malignant": 0, "benign": 1}))
    assert spec.index_to_label[0] == "malignant" and spec.index_to_label[1] == "benign"


@pytest.mark.parametrize("mapping", [
    {}, None, {"a": 0}, {"a": 0, "b": 2}, {"a": 0, "b": 0}, {"a": 0, "b": 1, "c": 2},
    {"a": 0, "b": True}, {"a": 0, "": 1}, {"a": "0", "b": "1"}, ["benign", "malignant"],
])
def test_missing_or_invalid_class_mapping_rejected(mapping):
    with pytest.raises(ModelCompatibilityError):
        build_model_spec(_metadata(class_mapping=mapping))


@pytest.mark.parametrize("input_spec", [
    {}, {"color_mode": "RGB"}, {"input_size": [32, 32]}, {"input_size": [0, 32], "color_mode": "RGB"},
    {"input_size": [32], "color_mode": "RGB"}, {"input_size": [32, 32], "color_mode": "HSV"},
    {"input_size": [32.5, 32], "color_mode": "RGB"},
])
def test_missing_or_invalid_required_preprocessing_metadata_rejected(input_spec):
    with pytest.raises(ModelCompatibilityError):
        build_model_spec(_metadata(input_spec=input_spec))


def test_input_spec_disagreeing_with_config_rejected():
    with pytest.raises(ModelCompatibilityError, match="disagrees"):
        build_model_spec(_metadata(input_spec={"input_size": [64, 64], "color_mode": "RGB"}))
    with pytest.raises(ModelCompatibilityError, match="disagrees"):
        build_model_spec(_metadata(input_spec={"input_size": [32, 32], "color_mode": "L"}))


def test_missing_hash_rejected():
    with pytest.raises(ModelCompatibilityError, match="artifact hash"):
        build_model_spec(_metadata(artifact_sha256=""))


def test_num_classes_validated():
    with pytest.raises(ModelCompatibilityError):
        build_model_spec(_metadata(num_classes=1, config={"architecture": "resnet18", "num_classes": 1, "input_size": [32, 32], "color_mode": "RGB"}))


def test_task_is_explicitly_marked_inferred_and_checked_when_expected():
    spec = build_model_spec(_metadata())
    assert spec.task == "image_classification" and spec.task_source.startswith("inferred")
    assert build_model_spec(_metadata(), expected_task="image_classification").task == "image_classification"
    with pytest.raises(ModelCompatibilityError, match="expected"):
        build_model_spec(_metadata(), expected_task="segmentation")


def test_unrecorded_preprocessing_reference_is_explicitly_unavailable_not_invented():
    spec = build_model_spec(_metadata())
    assert spec.preprocessing_reference is None
    assert spec.to_dict()["preprocessing_reference"] == UNAVAILABLE
    assert any("preprocessing_reference" in w for w in spec.warnings)
    with pytest.raises(ModelCompatibilityError, match="preprocessing_reference"):
        build_model_spec(_metadata(), require_preprocessing_reference=True)


def test_recorded_preprocessing_reference_is_preserved():
    spec = build_model_spec(_metadata(preprocessing_reference="preprocessing_report_v3"), require_preprocessing_reference=True)
    assert spec.to_dict()["preprocessing_reference"] == "preprocessing_report_v3" and not any("preprocessing" in w for w in spec.warnings)


def test_unsupported_architecture_rejected():
    with pytest.raises(ModelCompatibilityError, match="Unsupported architecture"):
        build_model_spec(_metadata(architecture="made_up_net"))
