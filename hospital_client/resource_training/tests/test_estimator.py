from hospital_client.resource_training.estimator import (
    HistoricalThroughputStore, estimate_memory_mb, estimate_training_time, format_duration,
)
from hospital_client.resource_training.policy import default_policy


def test_estimate_memory_mb_scales_with_batch_size():
    policy = default_policy()
    small = estimate_memory_mb(10_000_000, 4, (224, 224), "fp32", "adam", "medium_end", policy)
    large = estimate_memory_mb(10_000_000, 32, (224, 224), "fp32", "adam", "medium_end", policy)
    assert large.estimated_total_mb > small.estimated_total_mb


def test_estimate_memory_mb_fp16_uses_less_than_fp32():
    policy = default_policy()
    fp32 = estimate_memory_mb(10_000_000, 16, (224, 224), "fp32", "adam", "medium_end", policy)
    fp16 = estimate_memory_mb(10_000_000, 16, (224, 224), "fp16", "adam", "medium_end", policy)
    assert fp16.estimated_total_mb < fp32.estimated_total_mb


def test_format_duration_variants():
    assert "second" in format_duration(45)
    assert "minute" in format_duration(190)
    assert "hour" in format_duration(4000)


def test_estimate_training_time_is_positive_and_labeled_baseline():
    policy = default_policy()
    estimate = estimate_training_time(
        "resnet18", 11_700_000, (224, 224), num_train_samples=1000, num_validation_samples=200,
        batch_size=16, epochs=5, device_type="cpu", precision="fp32", policy=policy,
    )
    assert estimate.estimated_training_time_seconds > 0
    assert estimate.estimation_method == "baseline_estimate"
    assert estimate.estimation_confidence == "low"
    assert isinstance(estimate.estimated_training_time_display, str) and estimate.estimated_training_time_display


def test_more_epochs_increase_estimated_time():
    policy = default_policy()
    base_kwargs = dict(
        architecture="resnet18", param_count=11_700_000, input_size=(224, 224),
        num_train_samples=1000, num_validation_samples=0, batch_size=16,
        device_type="cpu", precision="fp32", policy=policy,
    )
    short = estimate_training_time(epochs=2, **base_kwargs)
    long = estimate_training_time(epochs=10, **base_kwargs)
    assert long.estimated_training_time_seconds > short.estimated_training_time_seconds


def test_history_store_never_fabricates_and_upgrades_estimation_method(tmp_path):
    path = str(tmp_path / "history.json")
    store = HistoricalThroughputStore(path)
    assert store.lookup("resnet18", "cpu", "fp32") is None

    policy = default_policy()
    baseline = estimate_training_time(
        "resnet18", 11_700_000, (224, 224), 1000, 0, 16, 5, "cpu", "fp32", policy, history=store,
    )
    assert baseline.estimation_method == "baseline_estimate"

    store.record("resnet18", "cpu", "fp32", samples_per_second=50.0)
    measured = estimate_training_time(
        "resnet18", 11_700_000, (224, 224), 1000, 0, 16, 5, "cpu", "fp32", policy, history=store,
    )
    assert measured.estimation_method == "measured_hardware_estimate"
    assert measured.estimation_confidence == "medium"


def test_history_store_reaches_high_confidence_after_enough_samples(tmp_path):
    path = str(tmp_path / "history.json")
    store = HistoricalThroughputStore(path)
    policy = default_policy()
    for _ in range(policy.high_confidence_history_samples):
        store.record("resnet18", "cpu", "fp32", samples_per_second=40.0)
    estimate = estimate_training_time("resnet18", 11_700_000, (224, 224), 1000, 0, 16, 5, "cpu", "fp32", policy, history=store)
    assert estimate.estimation_confidence == "high"


def test_history_store_persists_across_instances(tmp_path):
    path = str(tmp_path / "history.json")
    HistoricalThroughputStore(path).record("resnet18", "cpu", "fp32", 33.0)
    reloaded = HistoricalThroughputStore(path)
    assert reloaded.lookup("resnet18", "cpu", "fp32") == [33.0]


def test_history_store_ignores_non_positive_throughput(tmp_path):
    store = HistoricalThroughputStore(str(tmp_path / "history.json"))
    store.record("resnet18", "cpu", "fp32", 0.0)
    store.record("resnet18", "cpu", "fp32", -5.0)
    assert store.lookup("resnet18", "cpu", "fp32") is None
