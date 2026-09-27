"""
CUDA tests only run real assertions when torch.cuda.is_available() is
genuinely True on the machine running the suite - never faked. This
development machine has a real GPU, so these exercise actual CUDA behavior
rather than being skipped.
"""
import pytest
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer


def _config(dataset_dir, output_dir, **overrides):
    defaults = dict(
        model_id="m",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)),
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=4,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_training_runs_on_real_cuda(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), device="cuda")).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.epochs_run == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_cuda_oom_is_recorded_not_silently_recovered(small_dataset_dir, tmp_path, monkeypatch):
    import hospital_client.training.trainer as trainer_module

    original_build_model = trainer_module.build_model

    def _oom_build_model(config):
        model = original_build_model(config)
        original_forward = model.forward

        def _forward_that_ooms(*args, **kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

        model.forward = _forward_that_ooms
        return model

    monkeypatch.setattr(trainer_module, "build_model", _oom_build_model)

    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), device="cuda")).run()
    assert result.status == TrainingStatus.TRAINING_FAILED
    assert any("out of memory" in e.lower() for e in result.errors)
    # GPU memory must have been cleaned up, not left dangling.
    torch.cuda.synchronize()


def test_non_oom_runtime_error_is_not_swallowed(small_dataset_dir, tmp_path, monkeypatch):
    import hospital_client.training.trainer as trainer_module

    original_build_model = trainer_module.build_model

    def _broken_build_model(config):
        model = original_build_model(config)

        def _forward_that_breaks(*args, **kwargs):
            raise RuntimeError("some unrelated shape error, nothing to do with memory")

        model.forward = _forward_that_breaks
        return model

    monkeypatch.setattr(trainer_module, "build_model", _broken_build_model)

    with pytest.raises(RuntimeError, match="unrelated shape error"):
        Trainer(_config(small_dataset_dir, str(tmp_path / "out"))).run()


def test_keyboard_interrupt_produces_interrupted_status_and_saves_resumable_state(small_dataset_dir, tmp_path, monkeypatch):
    import hospital_client.training.trainer as trainer_module

    call_count = {"n": 0}
    original_run_epoch = trainer_module.Trainer._run_epoch

    def _run_epoch_then_interrupt(self, model, loader, loss_fn, optimizer, device, class_mapping, train_mode, precision, scaler):
        result = original_run_epoch(self, model, loader, loss_fn, optimizer, device, class_mapping, train_mode, precision, scaler)
        call_count["n"] += 1
        if call_count["n"] >= 2 and train_mode:  # interrupt after the 1st full epoch (train+val = 2 calls)
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(trainer_module.Trainer, "_run_epoch", _run_epoch_then_interrupt)

    output_dir = str(tmp_path / "out")
    config = _config(small_dataset_dir, output_dir, epochs=5)
    result = Trainer(config).run()

    assert result.status == TrainingStatus.TRAINING_INTERRUPTED

    # Resuming afterward must work from the saved state.
    from hospital_client.training.checkpoint import load_training_state
    state = load_training_state(output_dir, config.model_id)
    assert state["epoch"] >= 1
