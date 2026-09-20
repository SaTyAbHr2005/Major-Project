"""H/D. Device handling: explicit devices never fall back silently; 'auto' reuses Module 8; CUDA runs for real when present."""
import pytest
import torch

from hospital_client.inference.device import resolve_inference_device
from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import DeviceUnavailableError, InferenceConfigError
from hospital_client.inference.tests.conftest import build_artifact, make_image


def test_cpu_is_resolved_without_a_note():
    device, note = resolve_inference_device("cpu")
    assert device.type == "cpu" and note is None


@pytest.mark.parametrize("bad", ["tpu", "cuda:0", "", None, "GPU"])
def test_invalid_device_rejected(bad):
    with pytest.raises(InferenceConfigError):
        resolve_inference_device(bad)


def test_cuda_requested_but_unavailable_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="does not silently fall back"):
        resolve_inference_device("cuda")


def test_engine_fails_before_loading_when_cuda_unavailable(tmp_path, monkeypatch):
    artifact = build_artifact(tmp_path / "models")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError):
        LocalInferenceEngine(artifact, InferenceConfig(device="cuda"))


def test_auto_delegates_to_module8_and_records_its_reason(monkeypatch):
    calls = {}
    import hospital_client.resource_training.evaluator as m8_evaluator
    import hospital_client.resource_training.resource_profile as m8_profile

    real_build = m8_profile.build_resource_profile
    def spy_build(*args, **kwargs):
        calls["kwargs"] = kwargs
        return real_build(*args, **kwargs)
    monkeypatch.setattr(m8_profile, "build_resource_profile", spy_build)
    monkeypatch.setattr(m8_evaluator, "select_device", lambda profile: ("cpu", "simulated: no usable GPU"))

    device, note = resolve_inference_device("auto")
    assert device.type == "cpu" and "simulated: no usable GPU" in note and "Module 8" in note
    assert calls["kwargs"] == {"check_network": False}  # the inference path never probes the network


def test_auto_matches_module8_selection_on_this_machine():
    from hospital_client.resource_training.evaluator import select_device
    from hospital_client.resource_training.resource_profile import build_resource_profile
    expected, _ = select_device(build_resource_profile(check_network=False))
    device, note = resolve_inference_device("auto")
    assert device.type == expected and note.startswith("auto-selected via Module 8")


def test_auto_engine_reports_the_device_note(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"), InferenceConfig(device="auto"))
    result = engine.predict(make_image(tmp_path / "x.png"))
    assert result.succeeded and result.device_note.startswith("auto-selected via Module 8")
    engine.close()


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")


@cuda_only
def test_real_cuda_inference_matches_cpu_and_reports_peak_memory(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    image = make_image(tmp_path / "x.png")
    cpu_engine, cuda_engine = LocalInferenceEngine(artifact, InferenceConfig(device="cpu")), LocalInferenceEngine(artifact, InferenceConfig(device="cuda"))
    cpu, cuda = cpu_engine.predict(image), cuda_engine.predict(image)
    assert cuda.succeeded and cuda.device == "cuda" and cuda.peak_memory_mb is not None and cuda.peak_memory_mb > 0
    assert cuda.output.predicted_label == cpu.output.predicted_label
    for label, p in cpu.class_probabilities.items():
        assert cuda.class_probabilities[label] == pytest.approx(p, abs=1e-2)
    assert next(cuda_engine.model.parameters()).device.type == "cuda"
    cpu_engine.close(); cuda_engine.close()


@cuda_only
def test_real_cuda_oom_is_a_failed_result_not_a_crash(tmp_path):
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"), InferenceConfig(device="cuda"))
    def oom(x):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
    engine.model = oom
    result = engine.predict(make_image(tmp_path / "x.png"))
    assert result.status.value == "FAILED" and result.error_category == "out_of_memory"
    engine.close()
