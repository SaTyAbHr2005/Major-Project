"""D/E/G. Inference behaviour: outputs, class-mapping, determinism, no gradients, weights untouched, version preserved."""
import hashlib
import json
import logging
import math

import pytest
import torch

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import InferenceError
from hospital_client.inference.result import ClassificationOutput, InferenceStatus
from hospital_client.inference.tests.conftest import build_artifact, make_image


def _weights_fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_cpu_inference_produces_a_complete_structured_result(shared_engine, image_path):
    r = shared_engine.predict(image_path, reference="study-42")
    assert r.status == InferenceStatus.COMPLETED and r.succeeded and r.errors == [] and r.error_category is None
    assert (r.model_id, r.model_version, r.architecture, r.device) == ("m", 3, "resnet18", "cpu")
    assert r.task == "image_classification" and r.task_source.startswith("inferred")
    assert r.reference == "study-42"
    assert r.inference_duration_seconds > 0 and r.total_duration_seconds >= r.inference_duration_seconds
    assert r.peak_memory_mb is None  # CPU: no safe peak-memory measurement is claimed
    assert r.timestamp and r.preprocessing["tensor_shape"] == [1, 3, 32, 32]
    assert r.model_info["integrity_verified"] is True


def test_output_is_a_valid_probability_distribution(shared_engine, image_path):
    out = shared_engine.predict(image_path).output
    assert isinstance(out, ClassificationOutput) and out.output_transform == "softmax"
    assert set(out.class_probabilities) == {"benign", "malignant"}
    assert math.isclose(sum(out.class_probabilities.values()), 1.0, abs_tol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in out.class_probabilities.values())
    assert out.confidence == max(out.class_probabilities.values())
    assert out.class_probabilities[out.predicted_label] == out.confidence
    assert out.predicted_index in (0, 1)


def test_result_convenience_properties(shared_engine, image_path):
    r = shared_engine.predict(image_path)
    assert (r.predicted_label, r.confidence, r.class_probabilities) == (
        r.output.predicted_label, r.output.confidence, r.output.class_probabilities)


@pytest.mark.parametrize("mapping", [{"benign": 0, "malignant": 1}, {"malignant": 0, "benign": 1}])
def test_predicted_label_comes_from_the_stored_class_mapping(tmp_path, mapping):
    artifact = build_artifact(tmp_path / "models", class_mapping=mapping)
    engine = LocalInferenceEngine(artifact)
    image = make_image(tmp_path / "x.png")
    result = engine.predict(image)

    from hospital_client.inference.image_input import load_validated_image
    tensor, _ = engine._preprocessor.prepare(load_validated_image(image))
    with torch.no_grad():
        logits = engine.model(tensor)
    expected_index = int(torch.argmax(logits, dim=1))
    assert result.output.predicted_index == expected_index
    assert result.output.predicted_label == {v: k for k, v in mapping.items()}[expected_index]
    for label, index in mapping.items():  # each label is paired with ITS index's probability
        assert result.output.class_probabilities[label] == pytest.approx(float(torch.softmax(logits, 1)[0, index]), abs=1e-6)
    engine.close()


def test_inference_is_deterministic_on_cpu(shared_engine, image_path):
    runs = [shared_engine.predict(image_path).output.class_probabilities for _ in range(4)]
    assert all(r == runs[0] for r in runs)


def test_model_version_is_preserved_for_each_explicit_version(tmp_path):
    artifact = build_artifact(tmp_path / "models", versions=3)
    for version in (1, 2, 3):
        engine = LocalInferenceEngine(type(artifact)(artifact.store_root, artifact.model_id, version))
        assert engine.predict(make_image(tmp_path / "x.png")).model_version == version
        engine.close()


def test_no_gradients_are_computed_during_inference(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    seen = {}

    def hook(module, inputs, output):
        seen["inference_mode"] = torch.is_inference_mode_enabled()
        seen["grad_enabled"] = torch.is_grad_enabled()
        seen["output_requires_grad"] = output.requires_grad

    handle = engine.model.register_forward_hook(hook)
    engine.predict(make_image(tmp_path / "x.png"))
    handle.remove()
    assert seen == {"inference_mode": True, "grad_enabled": False, "output_requires_grad": False}
    assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
    assert not engine.model.training
    engine.close()


def test_inference_never_modifies_model_weights_or_mode(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    before = _weights_fingerprint(engine.model)
    for i in range(3):
        engine.predict(make_image(tmp_path / f"x{i}.png", color=(i * 40, 50, 90)))
    assert _weights_fingerprint(engine.model) == before
    assert not engine.model.training  # BatchNorm running stats would drift if this were train mode
    engine.close()


def test_invalid_image_returns_a_failed_result_not_an_exception(shared_engine, tmp_path):
    r = shared_engine.predict(str(tmp_path / "ghost.png"), reference="r1")
    assert r.status == InferenceStatus.FAILED and not r.succeeded
    assert r.error_category == "invalid_image" and r.output is None and r.reference == "r1"
    assert r.predicted_label is None and r.confidence is None and r.class_probabilities is None
    assert r.model_version == 3 and r.errors


def test_corrupt_image_returns_failed_result(shared_engine, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"garbage" * 30)
    assert shared_engine.predict(str(bad)).error_category == "invalid_image"


def test_wrong_output_shape_is_reported_as_model_incompatibility(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    engine.model = lambda x: torch.zeros(1, 5)
    r = engine.predict(make_image(tmp_path / "x.png"))
    assert r.status == InferenceStatus.FAILED and r.error_category == "model_compatibility"


def test_non_finite_output_is_rejected(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    engine.model = lambda x: torch.tensor([[float("nan"), 1.0]])
    assert engine.predict(make_image(tmp_path / "x.png")).error_category == "model_compatibility"


def test_unexpected_runtime_errors_are_not_swallowed(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    def boom(x):
        raise RuntimeError("a genuine bug")
    engine.model = boom
    with pytest.raises(RuntimeError, match="genuine bug"):
        engine.predict(make_image(tmp_path / "x.png"))


def test_cpu_never_treats_an_oom_message_as_recoverable(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    def boom(x):
        raise RuntimeError("CUDA out of memory (pretend)")
    engine.model = boom
    with pytest.raises(RuntimeError):  # the OOM handler is CUDA-only, mirroring Module 7
        engine.predict(make_image(tmp_path / "x.png"))


def test_closed_engine_refuses_to_predict(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    engine.close()
    with pytest.raises(InferenceError, match="closed"):
        engine.predict(make_image(tmp_path / "x.png"))


def test_extension_metadata_is_passed_through_untouched(tmp_path):
    ext = {"privacy": {"mechanism": "future_module_10"}, "n": 3}
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"), InferenceConfig(extension_metadata=ext))
    assert engine.predict(make_image(tmp_path / "x.png")).extension_metadata == ext
    engine.close()


def test_logs_contain_only_safe_operational_fields(tmp_path, caplog):
    secret = "Jane_Roe_MRN99887"
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    with caplog.at_level(logging.DEBUG, logger="hospital_client.inference"):
        engine.predict(make_image(tmp_path / f"{secret}.png"))
        engine.predict(str(tmp_path / f"{secret}_missing.png"))
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "inference completed" in text and "category=invalid_image" in text and "model_id=m" in text
    assert secret not in text and str(tmp_path) not in text
    engine.close()


def test_result_to_dict_is_json_serializable_and_carries_the_disclaimer(shared_engine, image_path):
    data = shared_engine.predict(image_path).to_dict()
    text = json.dumps(data)
    assert json.loads(text)["status"] == "COMPLETED"
    assert "not a clinical diagnosis" in data["disclaimer"]
    assert data["output"]["type"] == "classification"
    assert data["model_info"]["approval"]["platform_approved"] is False


def test_invalid_probability_distribution_is_rejected(tmp_path, monkeypatch):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    image = make_image(tmp_path / "x.png")
    monkeypatch.setattr(torch, "softmax", lambda *a, **k: torch.tensor([[0.2, 0.2]]))
    result = engine.predict(image)
    assert result.status == InferenceStatus.FAILED and result.error_category == "model_compatibility"
