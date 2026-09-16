import torch
import pytest

from hospital_client.resource_training import hardware


def test_detect_cpu_returns_well_formed_dict():
    result = hardware.detect_cpu()
    assert result["logical_cores"] >= 1
    assert result["physical_cores"] is None or result["physical_cores"] >= 1
    assert 0.0 <= result["utilization_percent"] <= 100.0


def test_detect_ram_returns_well_formed_dict():
    result = hardware.detect_ram()
    assert result["total_mb"] > 0
    assert result["available_mb"] >= 0
    assert result["available_mb"] <= result["total_mb"] * 1.01  # allow tiny rounding


def test_detect_storage_returns_well_formed_dict(tmp_path):
    result = hardware.detect_storage(str(tmp_path))
    assert result["total_mb"] > 0
    assert result["free_mb"] >= 0


def test_detect_network_never_raises_and_is_fast():
    import time
    started = time.monotonic()
    result = hardware.detect_network(timeout_seconds=0.3)
    elapsed = time.monotonic() - started
    assert "network_available" in result
    assert elapsed < 3.0  # must not block local training


def test_detect_network_unreachable_host_reports_unavailable():
    result = hardware.detect_network(host="203.0.113.1", port=1, timeout_seconds=0.2)
    assert result["network_available"] is False
    assert result["latency_ms"] is None


def test_detect_gpu_cuda_unavailable_reports_fallback_reason(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = hardware.detect_gpu()
    assert result["cuda_available"] is False
    assert result["gpu_name"] is None
    assert result["fallback_reason"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_detect_gpu_cuda_available_reports_real_values():
    result = hardware.detect_gpu()
    assert result["cuda_available"] is True
    assert result["gpu_name"]
    assert result["total_vram_mb"] > 0
    assert result["free_vram_mb"] is not None
    assert result["cuda_capability"]


def test_detect_gpu_utilization_never_raises_and_is_fast():
    import time
    started = time.monotonic()
    result = hardware.detect_gpu_utilization(timeout_seconds=0.5)
    assert time.monotonic() - started < 3.0
    assert result is None or 0.0 <= result <= 100.0


def test_detect_gpu_utilization_none_when_nvidia_smi_missing(monkeypatch):
    import subprocess as subprocess_module

    def _raise(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess_module, "run", _raise)
    assert hardware.detect_gpu_utilization(timeout_seconds=0.5) is None
