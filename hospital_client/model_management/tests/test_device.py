"""
Device portability tests.

CUDA tests only run assertions about real CUDA behavior when
torch.cuda.is_available() is genuinely True on the machine running the
suite - they are never faked. The "CUDA unavailable" error path is verified
with monkeypatch (simulating unavailability to test our error handling),
which is different from pretending CUDA IS available.
"""
import pytest
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.device import resolve_device, to_device
from hospital_client.model_management.registry import build_model


def test_cpu_is_always_supported():
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_invalid_device_string_rejected():
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")


def test_cuda_requested_but_unavailable_raises_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_cuda_resolves_when_genuinely_available():
    device = resolve_device("cuda")
    assert device.type == "cuda"


def test_model_construction_and_explicit_cpu_placement():
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = build_model(config)
    model = to_device(model, "cpu")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_explicit_cpu_selection_used_even_when_cuda_exists():
    # CUDA genuinely exists on this machine, but requesting CPU must still
    # place the model on CPU - Module 6 never auto-selects CUDA.
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = to_device(build_model(config), "cpu")
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_model_runs_on_real_cuda_when_available():
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = to_device(build_model(config), "cuda")
    x = torch.randn(1, 3, 64, 64, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert out.device.type == "cuda"
    assert out.shape == (1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_gpu_saved_weights_load_correctly_on_cpu(tmp_path):
    from hospital_client.model_management.store import ModelStore

    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = to_device(build_model(config), "cuda")
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(model, config, "cuda_trained")

    cpu_model, _ = store.load_checkpoint("cuda_trained", 1, device="cpu")
    assert next(cpu_model.parameters()).device.type == "cpu"
    out = cpu_model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_cpu_saved_weights_load_correctly_on_cuda(tmp_path):
    from hospital_client.model_management.store import ModelStore

    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = build_model(config)  # CPU by default
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(model, config, "cpu_trained")

    cuda_model, _ = store.load_checkpoint("cpu_trained", 1, device="cuda")
    assert next(cuda_model.parameters()).device.type == "cuda"
    out = cuda_model(torch.randn(1, 3, 64, 64, device="cuda"))
    assert out.shape == (1, 2)
