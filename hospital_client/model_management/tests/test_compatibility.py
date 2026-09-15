"""
Compatibility tests using Module 5's ACTUAL ImageConfig class (read-only
import; Module 5 is never modified).
"""
from hospital_client.preprocessing.config import ImageConfig
from hospital_client.model_management.compatibility import check_compatibility
from hospital_client.model_management.metadata import ModelMetadata


def _metadata(input_size=(224, 224), color_mode="RGB", num_classes=2, class_mapping=None):
    return ModelMetadata(
        model_id="m", version=1, architecture="resnet18",
        input_spec={"input_size": list(input_size), "color_mode": color_mode},
        num_classes=num_classes,
        class_mapping=class_mapping or {},
    )


def test_matching_preprocessing_config_is_compatible():
    metadata = _metadata()
    image_config = ImageConfig(target_size=(224, 224), color_mode="RGB")
    result = check_compatibility(metadata, image_config=image_config, expected_num_classes=2)
    assert result.compatible
    assert result.mismatches == []


def test_wrong_input_size_detected():
    metadata = _metadata(input_size=(224, 224))
    image_config = ImageConfig(target_size=(128, 128), color_mode="RGB")
    result = check_compatibility(metadata, image_config=image_config)
    assert not result.compatible
    assert any("input_size" in m for m in result.mismatches)


def test_wrong_color_mode_detected():
    metadata = _metadata(color_mode="RGB")
    image_config = ImageConfig(target_size=(224, 224), color_mode="L")
    result = check_compatibility(metadata, image_config=image_config)
    assert not result.compatible
    assert any("color_mode" in m for m in result.mismatches)


def test_wrong_class_count_detected():
    metadata = _metadata(num_classes=2)
    result = check_compatibility(metadata, expected_num_classes=5)
    assert not result.compatible
    assert any("num_classes" in m for m in result.mismatches)


def test_class_mapping_mismatch_detected():
    metadata = _metadata(class_mapping={"normal": 0, "disease": 1})
    result = check_compatibility(metadata, expected_class_mapping={"normal": 0, "disease": 1, "extra": 2})
    assert not result.compatible
    assert any("class_mapping" in m for m in result.mismatches)


def test_matching_class_mapping_is_compatible():
    mapping = {"normal": 0, "disease": 1}
    metadata = _metadata(class_mapping=mapping)
    result = check_compatibility(metadata, expected_class_mapping=mapping)
    assert result.compatible


def test_missing_preprocessing_fields_produce_warnings_not_hard_failures():
    metadata = _metadata()
    image_config = ImageConfig()  # target_size/color_mode both None by default
    result = check_compatibility(metadata, image_config=image_config)
    assert result.compatible
    assert result.warnings
