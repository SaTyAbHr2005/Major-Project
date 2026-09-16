import time

import torch
import pytest

from hospital_client.resource_training import hardware, monitor as monitor_module
from hospital_client.resource_training.monitor import ResourceMonitor
from hospital_client.resource_training.policy import default_policy


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    # `monitor._probe_unavailable` caches across samples/instances within a
    # process - reset it around every test in this file so one test's
    # monkeypatched failure never poisons another test's real probe.
    monitor_module._probe_unavailable = False
    yield
    monitor_module._probe_unavailable = False


def test_monitor_captures_cpu_and_ram_samples():
    policy = default_policy()
    policy.monitor_interval_seconds = 0.05
    monitor = ResourceMonitor(policy, device_type="cpu")
    monitor.start()
    time.sleep(0.25)
    summary = monitor.stop()
    assert summary.sample_count >= 2
    assert 0.0 <= summary.peak_cpu_percent <= 100.0
    assert summary.peak_ram_mb > 0
    assert summary.peak_vram_mb is None  # cpu run - no VRAM samples


def test_monitor_works_for_very_short_runs():
    policy = default_policy()
    policy.monitor_interval_seconds = 5.0  # longer than the run itself
    monitor = ResourceMonitor(policy, device_type="cpu")
    monitor.start()
    summary = monitor.stop()
    assert summary.sample_count >= 1  # at least the forced start/stop samples


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_monitor_captures_vram_on_cuda():
    policy = default_policy()
    policy.monitor_interval_seconds = 0.05
    monitor = ResourceMonitor(policy, device_type="cuda")
    monitor.start()
    tensor = torch.zeros((1024, 1024), device="cuda")
    time.sleep(0.2)
    del tensor
    summary = monitor.stop()
    assert summary.peak_vram_mb is not None
    assert summary.peak_vram_mb >= 0


def test_gpu_utilization_none_when_device_is_cpu():
    policy = default_policy()
    policy.monitor_interval_seconds = 0.05
    monitor = ResourceMonitor(policy, device_type="cpu")
    monitor.start()
    time.sleep(0.1)
    summary = monitor.stop()
    assert summary.gpu_utilization_available is False
    assert summary.peak_gpu_utilization_percent is None


def test_gpu_utilization_degrades_gracefully_when_nvidia_smi_unavailable(monkeypatch):
    # Simulates a machine with no NVIDIA driver/CLI present - must never
    # fabricate a utilization value, regardless of whether THIS development
    # machine happens to have nvidia-smi.
    monkeypatch.setattr(hardware, "detect_gpu_utilization", lambda timeout_seconds=0.5: None)
    policy = default_policy()
    policy.monitor_interval_seconds = 0.05
    monitor = ResourceMonitor(policy, device_type="cuda")
    monitor.start()
    time.sleep(0.1)
    summary = monitor.stop()
    assert summary.gpu_utilization_available is False
    assert summary.peak_gpu_utilization_percent is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_gpu_utilization_is_real_when_nvidia_smi_available():
    # This is a REAL measurement via the nvidia-smi CLI (no pynvml
    # dependency) - not a fabricated or hard-coded value.
    if hardware.detect_gpu_utilization(timeout_seconds=1.0) is None:
        pytest.skip("nvidia-smi CLI not available on this machine")
    policy = default_policy()
    policy.monitor_interval_seconds = 0.05
    monitor = ResourceMonitor(policy, device_type="cuda")
    monitor.start()
    time.sleep(0.2)
    summary = monitor.stop()
    assert summary.gpu_utilization_available is True
    assert summary.peak_gpu_utilization_percent is not None
    assert 0.0 <= summary.peak_gpu_utilization_percent <= 100.0
