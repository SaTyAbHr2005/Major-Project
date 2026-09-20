"""
B. Preprocessing. The strongest checks compare Module 16's tensor against the
tensor the REAL Module 7 dataset (and, for materialized data, the REAL Module 5
BaseTransformer feeding it) produces - i.e. inference sees exactly what training saw.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import ModelCompatibilityError
from hospital_client.inference.image_input import load_validated_image
from hospital_client.inference.model_spec import build_model_spec
from hospital_client.inference.preprocessing import PIPELINE_ID, InferencePreprocessor
from hospital_client.inference.tests.conftest import build_artifact, make_image
from hospital_client.inference.tests.test_model_spec import _metadata
from hospital_client.preprocessing.config import ImageConfig
from hospital_client.preprocessing.image.transforms import BaseTransformer
from hospital_client.training.dataset import ManifestImageDataset


def _spec(input_size=(32, 32), color_mode="RGB"):
    return build_model_spec(_metadata(
        config={"architecture": "resnet18", "num_classes": 2, "input_size": list(input_size), "color_mode": color_mode},
        input_spec={"input_size": list(input_size), "color_mode": color_mode},
    ))


def _noise_png(path, size=(61, 47), seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(str(path))  # PNG = lossless
    return str(path)


def _module7_tensor(source_path, color_mode, input_size, processed_path=None):
    dataset = ManifestImageDataset(
        {"records": [{"source_path": source_path, "processed_path": processed_path, "label": "benign"}]},
        {"benign": 0}, color_mode, input_size,
    )
    return dataset[0][0]


def test_tensor_shape_dtype_and_range_rgb(tmp_path):
    tensor, meta = InferencePreprocessor(_spec()).prepare(load_validated_image(_noise_png(tmp_path / "a.png")))
    assert tuple(tensor.shape) == (1, 3, 32, 32) and tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    assert meta["pipeline"] == PIPELINE_ID and meta["tensor_shape"] == [1, 3, 32, 32]
    assert meta["normalization"].startswith("none") and meta["source_size"] == [61, 47]


def test_grayscale_model_gets_single_channel_and_conversion_is_recorded(tmp_path):
    tensor, meta = InferencePreprocessor(_spec((40, 24), "L")).prepare(load_validated_image(_noise_png(tmp_path / "a.png")))
    assert tuple(tensor.shape) == (1, 1, 40, 24)
    assert "RGB->L" in meta["conversions"]


def test_rgba_and_palette_and_gray_sources_convert_to_model_mode(tmp_path):
    pre = InferencePreprocessor(_spec())
    for mode in ["RGBA", "P", "L", "1"]:
        tensor, meta = pre.prepare(load_validated_image(make_image(tmp_path / f"{mode}.png", mode=mode)))
        assert tuple(tensor.shape) == (1, 3, 32, 32) and f"{mode}->RGB" in meta["conversions"]


@pytest.mark.parametrize("color_mode", ["RGB", "L"])
def test_identical_to_module7_training_transform_lazy_mode(tmp_path, color_mode):
    src = _noise_png(tmp_path / "src.png")
    ours, _ = InferencePreprocessor(_spec((32, 32), color_mode)).prepare(load_validated_image(src))
    assert torch.equal(ours[0], _module7_tensor(src, color_mode, (32, 32)))


def test_identical_to_module5_materialized_then_module7_pipeline(tmp_path):
    src = _noise_png(tmp_path / "src.png", size=(70, 50))
    config = ImageConfig(target_size=(32, 32), color_mode="RGB")
    processed = tmp_path / "processed.png"
    assert BaseTransformer(config).transform(src, str(processed))  # the REAL Module 5 transformer
    expected = _module7_tensor(src, "RGB", (32, 32), processed_path=str(processed))

    ours, meta = InferencePreprocessor(_spec(), config, emulate_module5_materialization=True).prepare(load_validated_image(src))
    assert torch.equal(ours[0], expected)
    assert meta["module5_materialization_emulated"] is True
    assert any("lanczos" in c for c in meta["conversions"])

    plain, _ = InferencePreprocessor(_spec(), config).prepare(load_validated_image(src))  # bilinear: measurably different
    assert not torch.equal(plain[0], expected)


def test_preprocessing_is_deterministic_and_unaugmented(tmp_path):
    pre = InferencePreprocessor(_spec())
    loaded = load_validated_image(_noise_png(tmp_path / "a.png"))
    first, _ = pre.prepare(loaded)
    for _ in range(5):
        assert torch.equal(pre.prepare(loaded)[0], first)


@pytest.mark.parametrize("normalization", ["imagenet", "standard", "minmax", "anything"])
def test_requested_normalization_rejected_not_ignored(normalization):
    with pytest.raises(ModelCompatibilityError, match="normalization"):
        InferencePreprocessor(_spec(), ImageConfig(normalization=normalization))


def test_emulation_requires_image_config():
    with pytest.raises(ModelCompatibilityError, match="requires the Module 5 ImageConfig"):
        InferencePreprocessor(_spec(), None, emulate_module5_materialization=True)


def test_preprocessing_reference_recorded_or_marked_unavailable(tmp_path):
    loaded = load_validated_image(_noise_png(tmp_path / "a.png"))
    assert InferencePreprocessor(_spec()).prepare(loaded)[1]["preprocessing_reference"] == "unavailable"
    spec = build_model_spec(_metadata(preprocessing_reference="report_7"))
    assert InferencePreprocessor(spec).prepare(loaded)[1]["preprocessing_reference"] == "report_7"


# --- engine-level Module 5 <-> Module 6 compatibility (uses Module 6's own check_compatibility) ---

def test_engine_rejects_image_config_input_size_mismatch(tmp_path):
    artifact = build_artifact(tmp_path / "models", input_size=(32, 32))
    with pytest.raises(ModelCompatibilityError, match="input_size mismatch"):
        LocalInferenceEngine(artifact, InferenceConfig(image_config=ImageConfig(target_size=(64, 64), color_mode="RGB")))


def test_engine_rejects_image_config_color_mode_mismatch(tmp_path):
    artifact = build_artifact(tmp_path / "models", input_size=(32, 32), color_mode="RGB")
    with pytest.raises(ModelCompatibilityError, match="color_mode mismatch"):
        LocalInferenceEngine(artifact, InferenceConfig(image_config=ImageConfig(target_size=(32, 32), color_mode="L")))


def test_engine_accepts_matching_image_config(tmp_path):
    artifact = build_artifact(tmp_path / "models", input_size=(32, 32))
    engine = LocalInferenceEngine(artifact, InferenceConfig(image_config=ImageConfig(target_size=(32, 32), color_mode="RGB")))
    result = engine.predict(make_image(tmp_path / "x.png"))
    assert result.succeeded
    engine.close()


def test_engine_rejects_normalization_in_image_config(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    with pytest.raises(ModelCompatibilityError, match="normalization"):
        LocalInferenceEngine(artifact, InferenceConfig(image_config=ImageConfig(target_size=(32, 32), color_mode="RGB", normalization="imagenet")))


def test_engine_uses_the_models_own_input_size_for_any_source_size(shared_engine, tmp_path):
    for size in [(8, 8), (300, 20), (20, 300), (32, 32)]:
        result = shared_engine.predict(make_image(tmp_path / f"s{size[0]}x{size[1]}.png", size=size))
        assert result.succeeded and result.preprocessing["tensor_shape"] == [1, 3, 32, 32]
        assert result.preprocessing["source_size"] == list(size)


def test_module5_color_mode_step_matches_real_transformer_for_a_grayscale_source(tmp_path):
    src = tmp_path / "gray.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (50, 70), dtype=np.uint8)).save(str(src))  # mode "L"
    config = ImageConfig(target_size=(32, 32), color_mode="RGB")
    processed = tmp_path / "processed.png"
    assert BaseTransformer(config).transform(str(src), str(processed))  # Module 5: L -> RGB -> LANCZOS
    expected = _module7_tensor(str(src), "RGB", (32, 32), processed_path=str(processed))

    ours, meta = InferencePreprocessor(_spec(), config, emulate_module5_materialization=True).prepare(load_validated_image(str(src)))
    assert torch.equal(ours[0], expected) and "module5_color_mode:RGB" in meta["conversions"]


def test_tensor_shape_and_finiteness_guards(tmp_path):
    from hospital_client.inference.errors import ImageValidationError
    loaded = load_validated_image(_noise_png(tmp_path / "a.png"))
    pre = InferencePreprocessor(_spec())
    pre._transform = lambda img: torch.zeros(3, 16, 16)
    with pytest.raises(ImageValidationError, match="does not match the model input"):
        pre.prepare(loaded)
    pre._transform = lambda img: torch.full((3, 32, 32), float("nan"))
    with pytest.raises(ImageValidationError, match="non-finite"):
        pre.prepare(loaded)
