"""LocalInferenceService (Module 3 facade), the CLI, and result wording/serialization."""
import json
import sys
import threading

import pytest

from hospital_client.inference.cli import main
from hospital_client.inference.engine import InferenceConfig
from hospital_client.inference.errors import InferenceError, ModelProvisioningError
from hospital_client.inference.provisioning import LocalStoreModelProvider
from hospital_client.inference.result import CONFIDENCE_NOTE, DISCLAIMER, InferenceResult, InferenceStatus, format_summary
from hospital_client.inference.service import LocalInferenceService
from hospital_client.inference.tests.conftest import build_artifact, make_image

BANNED_CLAIMS = [
    "definitely", "confirmed diagnosis", "diagnosis is confirmed", "disease-free", "replaces a doctor",
    "clinically accurate", "guaranteed", "patient has", "clinical certainty is",
]


# --- service ---------------------------------------------------------------

def test_service_lifecycle(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    service = LocalInferenceService()
    assert not service.is_loaded
    with pytest.raises(InferenceError, match="No model is loaded"):
        service.predict(make_image(tmp_path / "x.png"))
    with pytest.raises(InferenceError):
        service.model_info()

    info = service.load_model(LocalStoreModelProvider(artifact.store_root, "m", 1))
    assert service.is_loaded and info["model_id"] == "m" and service.model_info() == info
    assert service.predict(make_image(tmp_path / "x.png"), reference="r").reference == "r"
    service.unload()
    assert not service.is_loaded
    service.unload()  # idempotent


def test_failed_reload_keeps_the_previously_loaded_model(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    service = LocalInferenceService()
    service.load_model(artifact)
    with pytest.raises(ModelProvisioningError):
        service.load_model(LocalStoreModelProvider(artifact.store_root, "m", 99))
    assert service.is_loaded and service.predict(make_image(tmp_path / "x.png")).succeeded
    service.unload()


def test_reload_replaces_the_model(tmp_path):
    a = build_artifact(tmp_path / "models", model_id="first")
    b = build_artifact(tmp_path / "models", model_id="second", architecture="mobilenet_v3_small", input_size=(64, 64))
    service = LocalInferenceService()
    service.load_model(a)
    service.load_model(b)
    assert service.predict(make_image(tmp_path / "x.png")).architecture == "mobilenet_v3_small"
    service.unload()


def test_concurrent_predictions_are_serialized_safely(tmp_path):
    service = LocalInferenceService()
    service.load_model(build_artifact(tmp_path / "models"))
    image = make_image(tmp_path / "x.png")
    results, errors = [], []

    def worker():
        try:
            results.append(service.predict(image).class_probabilities)
        except Exception as e:  # pragma: no cover - would indicate a threading bug
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(results) == 6 and all(r == results[0] for r in results)
    service.unload()


# --- CLI ---------------------------------------------------------------------

def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["__main__.py"] + argv)
    main()


@pytest.fixture
def cli_setup(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    return ["--models-root", artifact.store_root, "--model-id", "m", "--version", "1"], make_image(tmp_path / "scan.png")


def test_cli_predict_prints_summary_with_disclaimer(cli_setup, monkeypatch, capsys):
    model_args, image = cli_setup
    _run(["predict"] + model_args + ["--image", image], monkeypatch)
    out = capsys.readouterr().out
    assert "Predicted class (model output):" in out and "Model confidence" in out and DISCLAIMER in out
    assert image not in out and "scan.png" not in out


def test_cli_predict_json(cli_setup, monkeypatch, capsys):
    model_args, image = cli_setup
    _run(["predict"] + model_args + ["--image", image, "--json", "--reference", "REF-1"], monkeypatch)
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "COMPLETED" and data["reference"] == "REF-1" and data["output"]["type"] == "classification"


def test_cli_failed_image_exits_nonzero(cli_setup, tmp_path, monkeypatch, capsys):
    model_args, _ = cli_setup
    with pytest.raises(SystemExit) as info:
        _run(["predict"] + model_args + ["--image", str(tmp_path / "ghost.png")], monkeypatch)
    assert info.value.code == 1 and "Inference failed" in capsys.readouterr().out


def test_cli_missing_model_exits_nonzero(cli_setup, monkeypatch, capsys):
    model_args, image = cli_setup
    model_args[model_args.index("--version") + 1] = "9"
    with pytest.raises(SystemExit) as info:
        _run(["predict"] + model_args + ["--image", image], monkeypatch)
    assert info.value.code == 1 and "Error:" in capsys.readouterr().out


def test_cli_requires_an_explicit_version(cli_setup, monkeypatch):
    model_args, image = cli_setup
    without_version = [a for i, a in enumerate(model_args) if a != "--version" and (i == 0 or model_args[i - 1] != "--version")]
    with pytest.raises(SystemExit) as info:
        _run(["predict"] + without_version + ["--image", image], monkeypatch)
    assert info.value.code == 2  # argparse: --version is required


def test_cli_validate_model(cli_setup, monkeypatch, capsys):
    model_args, _ = cli_setup
    _run(["validate-model"] + model_args, monkeypatch)
    info = json.loads(capsys.readouterr().out)
    assert info["integrity_verified"] is True and info["approval"]["platform_approved"] is False


def test_cli_no_args_prints_help(monkeypatch, capsys):
    _run([], monkeypatch)
    assert "usage" in capsys.readouterr().out.lower()


# --- result wording / shape ---------------------------------------------------------

def test_failed_result_summary_and_shape():
    r = InferenceResult(status=InferenceStatus.FAILED, model_id="m", model_version=1, architecture="resnet18", task="image_classification",
                        task_source="inferred", device="cpu", errors=["bad image"], error_category="invalid_image")
    text = format_summary(r)
    assert "Inference failed" in text and "invalid_image" in text and DISCLAIMER in text
    data = r.to_dict()
    assert data["status"] == "FAILED" and data["output"] is None and data["disclaimer"] == DISCLAIMER


def test_no_unsupported_clinical_claims_in_any_user_facing_text(tmp_path):
    from hospital_client.inference.engine import LocalInferenceEngine
    mapping = {"normal": 0, "malignant": 1}
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models", class_mapping=mapping))
    ok = engine.predict(make_image(tmp_path / "x.png"))
    bad = engine.predict(str(tmp_path / "ghost.png"))
    engine.close()
    text = (format_summary(ok) + format_summary(bad) + json.dumps(ok.to_dict()) + CONFIDENCE_NOTE + DISCLAIMER).lower()
    for phrase in BANNED_CLAIMS:
        assert phrase not in text, phrase
    assert "not a clinical diagnosis" in text and "not a measure of clinical certainty" in text
    assert "model output" in text and "predicted class" in text


def test_task_output_base_class_requires_a_subclass():
    from hospital_client.inference.result import TaskOutput
    with pytest.raises(NotImplementedError):
        TaskOutput().to_dict()
