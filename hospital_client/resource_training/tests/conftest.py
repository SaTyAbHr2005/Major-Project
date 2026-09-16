"""
Shared fixtures for Module 8 tests.

Simulated ResourceProfile fixtures (very_low/low/medium/high) are clearly
labeled as SIMULATED for testing the decision logic - never presented as
physical measurements (spec §21). Real dataset generation reuses Module 7's
own conftest helper (`make_preprocessed_dataset`) rather than duplicating
synthetic-dataset-building logic.
"""
import pytest

from hospital_client.resource_training.dataset_characteristics import DatasetCharacteristics
from hospital_client.resource_training.policy import default_policy
from hospital_client.resource_training.resource_profile import ResourceProfile
from hospital_client.training.tests.conftest import make_preprocessed_dataset  # noqa: F401 (re-exported)


def _simulated_profile(cpu_cores, ram_mb, cuda, vram_mb=None, gpu_name="Simulated GPU"):
    gpu = (
        {
            "cuda_available": True, "gpu_name": gpu_name, "gpu_vendor": "NVIDIA",
            "total_vram_mb": vram_mb, "free_vram_mb": vram_mb, "cuda_capability": "8.6",
            "fallback_reason": None,
        }
        if cuda else
        {
            "cuda_available": False, "gpu_name": None, "gpu_vendor": None, "total_vram_mb": None,
            "free_vram_mb": None, "cuda_capability": None,
            "fallback_reason": "Simulated profile: no CUDA device.",
        }
    )
    return ResourceProfile(
        timestamp="2026-01-01T00:00:00+00:00",
        cpu={"processor": "Simulated CPU", "physical_cores": cpu_cores // 2, "logical_cores": cpu_cores, "utilization_percent": 15.0},
        ram={"total_mb": ram_mb, "available_mb": ram_mb * 0.8, "utilization_percent": 25.0},
        gpu=gpu,
        storage={"path": ".", "total_mb": 500_000.0, "free_mb": 300_000.0, "utilization_percent": 40.0},
        network={"network_available": False, "latency_ms": None, "checked_host": None},
        warnings=[], errors=[],
    )


@pytest.fixture
def very_low_profile():
    """Simulated: CPU-only, limited RAM (spec §21 VERY LOW)."""
    return _simulated_profile(cpu_cores=4, ram_mb=4000.0, cuda=False)


@pytest.fixture
def low_profile():
    """Simulated: 4GB VRAM GPU, 16GB RAM (spec §21 LOW)."""
    return _simulated_profile(cpu_cores=8, ram_mb=16_000.0, cuda=True, vram_mb=4000.0)


@pytest.fixture
def medium_profile():
    """Simulated: 8GB VRAM GPU, 32GB RAM (spec §21 MEDIUM)."""
    return _simulated_profile(cpu_cores=12, ram_mb=32_000.0, cuda=True, vram_mb=8000.0)


@pytest.fixture
def high_profile():
    """Simulated: 16+GB VRAM GPU, 64GB RAM (spec §21 HIGH)."""
    return _simulated_profile(cpu_cores=16, ram_mb=65_000.0, cuda=True, vram_mb=16_000.0)


@pytest.fixture
def all_simulated_profiles(very_low_profile, low_profile, medium_profile, high_profile):
    return {"very_low": very_low_profile, "low": low_profile, "medium": medium_profile, "high": high_profile}


@pytest.fixture
def policy():
    """
    Real dry-run measurement (policy.enable_dry_run_measurement, on by
    default in default_policy() for real usage) is disabled for this shared
    fixture so the many pure-decision-logic tests using it stay fast - they
    are testing selection/recommendation logic, not the measurement itself.
    The measurement is exercised for real in test_profiler.py, in the
    dedicated real-measurement tests in test_evaluator.py, and throughout
    test_runner_integration.py (which uses default_policy() directly).
    """
    p = default_policy()
    p.enable_dry_run_measurement = False
    return p


@pytest.fixture
def small_dataset_characteristics():
    return DatasetCharacteristics(
        dataset_dir="synthetic", num_train_samples=200, num_validation_samples=40,
        num_test_samples=0, num_classes=2,
    )


@pytest.fixture
def real_small_dataset_dir(tmp_path):
    """A REAL Module 4 -> Module 5 output directory (synthetic images only)."""
    return make_preprocessed_dataset(tmp_path, {"cat": 12, "dog": 12})
