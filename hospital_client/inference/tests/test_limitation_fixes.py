"""
Fixes for the limitations found after the first Module 16 pass: recorded task/preprocessing
spec (Modules 6/7 -> 16), lossy re-encode reproduction, single-read image loading, auto-device
VRAM fallback, reference/path sanitization, environment metadata.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine, sanitize_reference
from hospital_client.inference.errors import (
    ImageValidationError, InferenceConfigError, ModelCompatibilityError, ModelProvisioningError,
)
from hospital_client.inference.image_input import load_validated_image
from hospital_client.inference.provisioning import LocalModelArtifact
from hospital_client.inference.preprocessing import InferencePreprocessor
from hospital_client.inference.tests.conftest import build_artifact, make_image
from hospital_client.inference.tests.test_preprocessing import _module7_tensor, _spec
from hospital_client.model_management.config import ModelConfig
from hospital_client.preprocessing.config import ImageConfig
from hospital_client.preprocessing.image.transforms import BaseTransformer
from hospital_client.training.dataset import TASK_TYPE, build_preprocessing_spec


def _extra(**over):
    extra = {"task_type": TASK_TYPE, "preprocessing_spec": build_preprocessing_spec("RGB", (32, 32))}
    extra.update(over)
    return extra


# ---- 1/2. recorded task + preprocessing spec ------------------------------------------------
def test_module7_records_task_and_preprocessing_spec(tmp_path):
    from hospital_client.training.config import TrainingConfig
    from hospital_client.training.tests.conftest import make_preprocessed_dataset
    from hospital_client.training.trainer import Trainer

    ds = make_preprocessed_dataset(tmp_path, {"a": 10, "b": 10}, image_size=32)
    cfg = ModelConfig(architecture="resnet18", num_classes=2, input_size=(32, 32))
    res = Trainer(TrainingConfig(model_id="t", model_config=cfg, dataset_dir=ds, output_dir=str(tmp_path / "o"),
                                 epochs=1, batch_size=4)).run()
    engine = LocalInferenceEngine(LocalModelArtifact(str(tmp_path / "o" / "models"), "t", res.checkpoint_version))
    assert engine.spec.task == TASK_TYPE and engine.spec.task_source.startswith("recorded")
    recorded = dict(engine.spec.preprocessing_spec)
    assert recorded.pop("upstream")["output_mode"] == "lazy"  # recorded by Module 5, carried by Module 7
    assert recorded == build_preprocessing_spec("RGB", (32, 32))
    assert not any("preprocessing_spec" in w or "task_type" in w for w in engine.spec.warnings)
    result = engine.predict(make_image(tmp_path / "i.png"))
    assert result.succeeded and result.preprocessing["preprocessing_spec_verified"] is True
    engine.close()


def test_recorded_spec_engine_reports_environment(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "m", extra=_extra()))
    result = engine.predict(make_image(tmp_path / "i.png"))
    assert result.environment["torch"] == torch.__version__
    assert result.environment["device_type"] == "cpu" and result.environment["precision"] == "fp32"
    assert result.environment["preprocessing_spec_version"] == 1
    assert result.to_dict()["output"]["confidence_kind"] == "uncalibrated softmax probability"
    engine.close()


def test_legacy_artifact_without_spec_still_works_but_warns(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "m"))
    assert engine.spec.task_source.startswith("inferred")
    joined = " ".join(engine.spec.warnings)
    assert "preprocessing_spec" in joined and "task_type" in joined
    engine.close()


@pytest.mark.parametrize("override", [
    {"normalization": "imagenet"},
    {"interpolation": "bicubic"},
    {"input_size": [64, 64]},
    {"color_mode": "L"},
    {"version": 99},
    {"dtype": "float16"},
])
def test_mismatching_recorded_spec_is_rejected(tmp_path, override):
    spec = dict(build_preprocessing_spec("RGB", (32, 32)), **override)
    with pytest.raises(ModelCompatibilityError, match="preprocessing_spec"):
        LocalInferenceEngine(build_artifact(tmp_path / "m", extra=_extra(preprocessing_spec=spec)))


def test_unsupported_recorded_task_is_rejected(tmp_path):
    with pytest.raises(ModelCompatibilityError, match="not supported"):
        LocalInferenceEngine(build_artifact(tmp_path / "m", extra=_extra(task_type="image_segmentation")))


def test_require_flag_is_satisfied_by_a_recorded_spec(tmp_path):
    artifact = build_artifact(tmp_path / "m", extra=_extra())
    LocalInferenceEngine(artifact, InferenceConfig(require_preprocessing_reference=True)).close()


# ---- 3. lossy re-encode reproduction ---------------------------------------------------------
@pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("WEBP", "webp")])
def test_lossy_reencode_matches_real_module5_materialization(tmp_path, fmt, ext):
    rng = np.random.default_rng(1)
    src = str(tmp_path / f"src.{ext}")
    Image.fromarray(rng.integers(0, 256, (50, 70, 3), dtype=np.uint8)).save(src, format=fmt)
    config = ImageConfig(target_size=(32, 32), color_mode="RGB")
    processed = tmp_path / f"processed.{ext}"
    assert BaseTransformer(config).transform(src, str(processed))
    expected = _module7_tensor(src, "RGB", (32, 32), processed_path=str(processed))

    ours, meta = InferencePreprocessor(_spec(), config, emulate_module5_materialization=True).prepare(load_validated_image(src))
    assert torch.equal(ours[0], expected)
    assert any("reencode" in c for c in meta["conversions"])


# ---- 4. reference sanitization ----------------------------------------------------------------
@pytest.mark.parametrize("bad", ["../secret", "C:\\scans\\p1.png", "a b", "x" * 129, "", "line\nbreak", 5])
def test_unsafe_reference_rejected(shared_engine, image_path, bad):
    with pytest.raises(InferenceConfigError, match="reference"):
        shared_engine.predict(image_path, reference=bad)


def test_safe_reference_accepted():
    assert sanitize_reference("study-42_v1.0:a") == "study-42_v1.0:a" and sanitize_reference(None) is None


# ---- 5. store paths never leak through provisioning errors ------------------------------------
def test_provisioning_error_does_not_expose_store_path(tmp_path):
    root = tmp_path / "very_private_hospital_root"
    build_artifact(root)
    with pytest.raises(ModelProvisioningError) as caught:
        LocalInferenceEngine(LocalModelArtifact(str(root), "m", 9))
    assert "very_private_hospital_root" not in str(caught.value)


# ---- 6. single read / no check-then-use gap ---------------------------------------------------
def test_image_is_read_once_and_decoded_from_memory(tmp_path, monkeypatch):
    path = make_image(tmp_path / "p.png")
    opened = []
    real_open = Image.open

    def spy(fp, *a, **k):
        opened.append(fp)
        return real_open(fp, *a, **k)

    monkeypatch.setattr(Image, "open", spy)
    load_validated_image(path)
    assert opened and not any(isinstance(fp, str) for fp in opened)  # never re-opened by path


def test_dicom_is_rejected_with_a_specific_message(tmp_path):
    p = tmp_path / "scan.dcm"
    p.write_bytes(b"DICM" * 64)
    with pytest.raises(ImageValidationError, match="DICOM"):
        load_validated_image(str(p))


# ---- 7. auto device: no CUDA headroom -> explicit CPU fallback --------------------------------
def test_auto_device_falls_back_to_cpu_explicitly_when_vram_is_insufficient(tmp_path, monkeypatch):
    import hospital_client.inference.engine as engine_module

    monkeypatch.setattr(engine_module, "resolve_inference_device", lambda requested: (torch.device("cuda"), "auto-selected via Module 8: cuda"))
    monkeypatch.setattr(engine_module, "cuda_has_headroom", lambda *_: (False, "free VRAM 10 MB < estimated need 90 MB"))
    engine = LocalInferenceEngine(build_artifact(tmp_path / "m"), InferenceConfig(device="auto"))
    result = engine.predict(make_image(tmp_path / "i.png"))
    assert result.succeeded and result.device == "cpu"
    assert "explicitly fell back to CPU" in result.device_note and "free VRAM 10 MB" in result.device_note
    engine.close()


def test_explicit_cuda_never_falls_back_when_unavailable(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    from hospital_client.inference.errors import DeviceUnavailableError

    with pytest.raises(DeviceUnavailableError):
        LocalInferenceEngine(build_artifact(tmp_path / "m"), InferenceConfig(device="cuda"))


# ---- #2 Module 5 upstream mode recorded -> Module 16 reproduces it automatically ----------------
def test_materialized_training_data_is_reproduced_automatically_without_image_config(tmp_path):
    from hospital_client.preprocessing.config import OutputMode
    from hospital_client.training.config import TrainingConfig
    from hospital_client.training.dataset import load_manifest
    from hospital_client.training.tests.conftest import make_preprocessed_dataset
    from hospital_client.training.trainer import Trainer

    ds = make_preprocessed_dataset(tmp_path, {"a": 10, "b": 10}, output_mode=OutputMode.MATERIALIZED, image_size=40)
    cfg = ModelConfig(architecture="resnet18", num_classes=2, input_size=(32, 32))
    res = Trainer(TrainingConfig(model_id="mat", model_config=cfg, dataset_dir=ds, output_dir=str(tmp_path / "o"),
                                 epochs=1, batch_size=4)).run()
    engine = LocalInferenceEngine(LocalModelArtifact(str(tmp_path / "o" / "models"), "mat", res.checkpoint_version))
    assert engine.spec.preprocessing_spec["upstream"]["output_mode"] == "materialized"

    record = load_manifest(f"{ds}/train_manifest.json")["records"][0]
    assert record["processed_path"]  # really materialized (and JPEG re-encoded by Module 5)
    expected = _module7_tensor(record["source_path"], "RGB", (32, 32), processed_path=record["processed_path"])
    ours, meta = engine._preprocessor.prepare(load_validated_image(record["source_path"]))
    assert torch.equal(ours[0], expected)
    assert meta["module5_materialization_emulated"] is True and meta["module5_emulation_source"] == "recorded_upstream"
    engine.close()


def test_lazy_or_unrecorded_upstream_does_not_emulate(tmp_path):
    for upstream in (None, {"output_mode": "lazy", "color_mode": None, "target_size": None, "resize_method": None}):
        spec = build_preprocessing_spec("RGB", (32, 32), upstream)
        engine = LocalInferenceEngine(build_artifact(tmp_path / f"m{upstream is None}", extra=_extra(preprocessing_spec=spec)))
        _, meta = engine._preprocessor.prepare(load_validated_image(make_image(tmp_path / "i.png")))
        assert meta["module5_materialization_emulated"] is False
        engine.close()
    assert any("upstream" in w for w in LocalInferenceEngine(
        build_artifact(tmp_path / "mw", extra=_extra())).spec.warnings)


def test_malformed_upstream_is_rejected(tmp_path):
    spec = build_preprocessing_spec("RGB", (32, 32), {"output_mode": "materialized", "target_size": [0, 5], "color_mode": "RGB"})
    with pytest.raises(ModelCompatibilityError, match="upstream"):
        LocalInferenceEngine(build_artifact(tmp_path / "m", extra=_extra(preprocessing_spec=spec)))


# ---- #8 determinism scope, #11 engine thread-safety ---------------------------------------------
def test_deterministic_mode_restores_global_flags(tmp_path):
    before = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    engine = LocalInferenceEngine(build_artifact(tmp_path / "m"))
    result = engine.predict(make_image(tmp_path / "i.png"))
    assert result.environment["deterministic_algorithms_requested"] is True
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled()) == before
    engine.close()


def test_concurrent_predictions_on_one_engine_are_safe_and_identical(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    engine = LocalInferenceEngine(build_artifact(tmp_path / "m"))
    image = make_image(tmp_path / "i.png")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: engine.predict(image), range(12)))
    assert all(r.succeeded for r in results)
    assert len({tuple(r.class_probabilities.values()) for r in results}) == 1
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_repeat_predictions_are_bitwise_identical(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "m"), InferenceConfig(device="cuda"))
    image = make_image(tmp_path / "i.png")
    outs = {tuple(engine.predict(image).class_probabilities.values()) for _ in range(5)}
    assert len(outs) == 1
    engine.close()
