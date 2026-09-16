from hospital_client.resource_training import hardware
from hospital_client.resource_training.resource_profile import ResourceProfile, build_resource_profile


def test_build_resource_profile_populates_all_sections(tmp_path):
    profile = build_resource_profile(storage_path=str(tmp_path), check_network=False)
    assert profile.cpu["logical_cores"] >= 1
    assert profile.ram["total_mb"] > 0
    assert "cuda_available" in profile.gpu
    assert profile.storage["total_mb"] > 0
    assert profile.network["network_available"] is None  # check_network=False


def test_to_dict_contains_only_hardware_fields(tmp_path):
    profile = build_resource_profile(storage_path=str(tmp_path), check_network=False)
    data = profile.to_dict()
    assert set(data.keys()) == {"timestamp", "cpu", "ram", "gpu", "storage", "network", "warnings", "errors"}


def test_degrades_gracefully_when_a_detector_raises(monkeypatch, tmp_path):
    def _boom():
        raise RuntimeError("simulated CPU detection failure")

    monkeypatch.setattr(hardware, "detect_cpu", _boom)
    profile = build_resource_profile(storage_path=str(tmp_path), check_network=False)
    assert profile.cpu["logical_cores"] is None
    assert any("CPU" in w for w in profile.warnings)
    # other sections still populated despite the CPU failure
    assert profile.ram["total_mb"] > 0


def test_network_check_disabled_by_default_flag_is_fast(tmp_path):
    import time
    started = time.monotonic()
    build_resource_profile(storage_path=str(tmp_path), check_network=True)
    assert time.monotonic() - started < 3.0
