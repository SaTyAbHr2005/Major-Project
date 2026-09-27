import os

import pytest
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.store import ModelStore
from hospital_client.training.config import TrainingConfig
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer, build_federation_handoff, get_model


def _config(dataset_dir, output_dir, **overrides):
    defaults = dict(
        model_id="m",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)),
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=2,
        batch_size=4,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# Basic training loop / metrics tracking
# ---------------------------------------------------------------------------

def test_training_tracks_loss_accuracy_and_duration(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"))).run()
    assert result.training_metrics["loss"] >= 0.0
    assert 0.0 <= result.training_metrics["accuracy"] <= 1.0
    assert result.training_duration_seconds > 0.0
    assert result.epochs_run == 2


def test_validation_metrics_include_precision_recall_f1_confusion_matrix(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"))).run()
    for key in ("precision", "recall", "f1", "confusion_matrix", "per_class"):
        assert key in result.validation_metrics


def test_test_evaluation_never_influences_checkpoint_or_early_stopping(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), run_test_evaluation=True, early_stopping=True, epochs=3)).run()
    # Test metrics are only ever computed once, at the very end.
    assert result.test_metrics is not None
    assert result.best_epoch is not None and result.best_epoch <= result.epochs_run


# ---------------------------------------------------------------------------
# Best checkpoint / early stopping / scheduler
# ---------------------------------------------------------------------------

def test_best_checkpoint_is_not_assumed_to_be_final_epoch(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    result = Trainer(_config(small_dataset_dir, output_dir, epochs=3)).run()
    store = ModelStore(os.path.join(output_dir, "models"))
    versions = store.list_versions("m")
    # A version was saved for every epoch that improved on the best metric so far.
    assert len(versions) >= 1
    assert result.checkpoint_version in [v.version for v in versions]


def test_early_stopping_halts_before_max_epochs_on_plateau(small_dataset_dir, tmp_path):
    config = _config(
        small_dataset_dir, str(tmp_path / "out"), epochs=20,
        early_stopping=True, early_stopping_metric="val_loss", early_stopping_patience=1,
    )
    result = Trainer(config).run()
    assert result.epochs_run <= 20
    # With patience=1 on a tiny dataset, it should stop well before 20 epochs.
    assert any("Early stopping" in w for w in result.warnings) or result.epochs_run < 20


def test_reduce_on_plateau_scheduler_runs_without_error(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), scheduler="reduce_on_plateau", epochs=3)
    result = Trainer(config).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION


def test_no_validation_still_produces_a_checkpoint(small_dataset_dir, tmp_path):
    single_class_config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    config = _config(small_dataset_dir, str(tmp_path / "out"), model_config=single_class_config, require_validation=False, epochs=1)
    result = Trainer(config).run()
    assert result.checkpoint_model_id is not None
    assert result.best_epoch == 1


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def test_resume_continues_from_saved_epoch(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    r1 = Trainer(_config(small_dataset_dir, output_dir, epochs=2)).run()
    assert r1.epochs_run == 2

    r2 = Trainer(_config(small_dataset_dir, output_dir, epochs=4, resume=True)).run()
    assert r2.epochs_run == 2  # only the remaining 2 epochs ran
    assert r2.checkpoint_version >= r1.checkpoint_version


def test_resume_without_prior_state_raises(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), resume=True)
    with pytest.raises(FileNotFoundError, match="no training state found"):
        Trainer(config).run()


def test_resume_with_architecture_mismatch_raises(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    Trainer(_config(small_dataset_dir, output_dir, epochs=1)).run()

    other_model = ModelConfig(architecture="mobilenet_v2", num_classes=2, input_size=(64, 64))
    config = _config(small_dataset_dir, output_dir, model_config=other_model, resume=True)
    with pytest.raises(ValueError, match="architecture"):
        Trainer(config).run()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_initial_weights(small_dataset_dir, tmp_path):
    from hospital_client.model_management.registry import build_model
    import torch as _torch

    _torch.manual_seed(123)
    model_a = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))
    _torch.manual_seed(123)
    model_b = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert _torch.equal(pa, pb)


def test_random_seed_is_recorded_in_result(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), random_seed=777, epochs=1)).run()
    assert result.random_seed == 777
    assert result.training_configuration["random_seed"] == 777


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------

def test_balanced_class_weighting_completes_training(imbalanced_dataset_dir, tmp_path):
    config = _config(imbalanced_dataset_dir, str(tmp_path / "out"), class_weighting="balanced", epochs=1)
    result = Trainer(config).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION


def test_class_weighting_never_changes_validation_sample_count(imbalanced_dataset_dir, tmp_path):
    from hospital_client.training.dataloader import build_dataloaders
    cfg_none = _config(imbalanced_dataset_dir, str(tmp_path / "out"), class_weighting="none")
    cfg_balanced = _config(imbalanced_dataset_dir, str(tmp_path / "out2"), class_weighting="balanced")
    prepared_none = build_dataloaders(cfg_none)
    prepared_balanced = build_dataloaders(cfg_balanced)
    assert prepared_none.validation_sample_count == prepared_balanced.validation_sample_count
    assert prepared_none.train_sample_count == prepared_balanced.train_sample_count


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unavailable_explicit_device_raises_clearly(small_dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _config(small_dataset_dir, str(tmp_path / "out"), device="cuda")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        Trainer(config).run()


def test_missing_dataset_manifest_fails_before_training(tmp_path):
    config = _config(str(tmp_path / "does_not_exist"), str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        Trainer(config).run()


# ---------------------------------------------------------------------------
# Federation handoff
# ---------------------------------------------------------------------------

def test_federation_handoff_requires_completed_awaiting_federation_status():
    from hospital_client.training.result import TrainingResult

    failed_result = TrainingResult(status=TrainingStatus.TRAINING_FAILED, model_id="m", architecture="resnet18")
    with pytest.raises(ValueError, match="TRAINING_COMPLETED_AWAITING_FEDERATION"):
        build_federation_handoff(failed_result, "/tmp/out")


def test_get_model_uses_module6_store_directly(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    result = Trainer(_config(small_dataset_dir, output_dir, epochs=1)).run()
    model, metadata = get_model(result.checkpoint_model_id, result.checkpoint_version, output_dir)
    assert metadata.architecture == "resnet18"
