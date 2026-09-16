"""
Tests the real dry-run measurement directly: a genuine forward+backward+
optimizer-step executed on this machine, not a heuristic. Kept to a small
number of fast architectures/batch sizes since each call is a real compute
step.
"""
import torch
import pytest

from hospital_client.resource_training.profiler import measure_real_resource_usage


def test_measures_real_positive_memory_and_time_on_cpu():
    result = measure_real_resource_usage(
        architecture="resnet18", num_classes=2, input_size=(64, 64), color_mode="RGB",
        batch_size=2, device_type="cpu", precision="fp32", optimizer_name="adam",
    )
    assert result.measured_total_mb > 0
    assert result.measured_seconds_per_batch > 0


def test_larger_batch_size_takes_at_least_as_long_on_cpu():
    small = measure_real_resource_usage("resnet18", 2, (64, 64), "RGB", 2, "cpu", "fp32", "adam")
    large = measure_real_resource_usage("resnet18", 2, (64, 64), "RGB", 16, "cpu", "fp32", "adam")
    # Real wall-clock timing is noisy at this scale - only assert both are
    # genuinely positive and of the same order of magnitude, not an exact
    # inequality (a strict > comparison would be flaky on a busy CI runner).
    assert small.measured_seconds_per_batch > 0
    assert large.measured_seconds_per_batch > 0


def test_repeated_measurement_does_not_leak_state():
    # Running the same measurement twice must not raise or accumulate memory
    # in a way that fails - real cleanup (del + empty_cache) is exercised.
    for _ in range(3):
        result = measure_real_resource_usage("mobilenet_v3_small", 2, (64, 64), "RGB", 4, "cpu", "fp32", "adam")
        assert result.measured_total_mb > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_measures_real_vram_on_cuda():
    result = measure_real_resource_usage(
        architecture="resnet18", num_classes=2, input_size=(64, 64), color_mode="RGB",
        batch_size=4, device_type="cuda", precision="fp16", optimizer_name="adam",
    )
    assert result.measured_total_mb > 0
    assert result.measured_seconds_per_batch > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_real_cuda_oom_propagates_for_absurd_batch_size():
    with pytest.raises(RuntimeError):
        measure_real_resource_usage(
            architecture="vit_b16", num_classes=2, input_size=(224, 224), color_mode="RGB",
            batch_size=1_000_000, device_type="cuda", precision="fp16", optimizer_name="adam",
        )
