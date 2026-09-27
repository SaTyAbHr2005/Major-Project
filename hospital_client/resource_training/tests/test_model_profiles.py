from hospital_client.model_management.registry import ARCHITECTURE_CATALOG
from hospital_client.resource_training.model_profiles import measure_all_model_profiles, measure_model_profile


def test_measures_all_8_architectures():
    profiles = measure_all_model_profiles(num_classes=2)
    assert set(profiles.keys()) == set(ARCHITECTURE_CATALOG.keys())
    for profile in profiles.values():
        assert profile.param_count > 0


def test_param_counts_are_real_measurements_not_guesses():
    # ResNet-50 has substantially more parameters than ResNet-18 - this is a
    # real architectural fact verifiable via the measured counts, not an
    # assumption baked into the code.
    resnet18 = measure_model_profile("resnet18", num_classes=2)
    resnet50 = measure_model_profile("resnet50", num_classes=2)
    assert resnet50.param_count > resnet18.param_count


def test_measurement_is_cached(monkeypatch):
    from hospital_client.resource_training import model_profiles as mp
    mp._CACHE.clear()
    calls = {"count": 0}
    original = mp.build_model
    def _counting_build_model(config):
        calls["count"] += 1
        return original(config)
    monkeypatch.setattr(mp, "build_model", _counting_build_model)

    mp.measure_model_profile("resnet18", num_classes=2)
    mp.measure_model_profile("resnet18", num_classes=2)
    assert calls["count"] == 1


def test_num_classes_affects_param_count_for_classifier_head():
    small = measure_model_profile("resnet18", num_classes=2)
    large = measure_model_profile("resnet18", num_classes=100)
    assert large.param_count > small.param_count
