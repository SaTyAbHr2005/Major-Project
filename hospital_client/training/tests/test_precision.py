"""
Mixed precision tests. GPU-specific assertions only run real checks when
torch.cuda.is_available() is genuinely True on this machine - never faked.
FP32 remains the safe baseline and must be unaffected on CPU-only systems.
"""
import pytest
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig, validate_training_config
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer, resolve_precision


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


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_default_precision_is_fp32():
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2), dataset_dir="d", output_dir="o"
    )
    assert config.precision == "fp32"


def test_invalid_precision_value_rejected():
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2),
        dataset_dir="d", output_dir="o", precision="bf16",
    )
    with pytest.raises(ValueError, match="precision"):
        validate_training_config(config)


def test_fp16_on_cpu_rejected_at_config_validation():
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2),
        dataset_dir="d", output_dir="o", device="cpu", precision="fp16",
    )
    with pytest.raises(ValueError, match="does not pretend FP16 is available"):
        validate_training_config(config)


def test_fp32_and_auto_allowed_on_cpu():
    for precision in ("fp32", "auto"):
        config = TrainingConfig(
            model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2),
            dataset_dir="d", output_dir="o", device="cpu", precision=precision,
        )
        validate_training_config(config)  # must not raise


# ---------------------------------------------------------------------------
# resolve_precision() - the runtime device/precision resolution
# ---------------------------------------------------------------------------

def test_resolve_precision_fp32_always_fp32():
    assert resolve_precision("fp32", torch.device("cpu")) == "fp32"
    assert resolve_precision("fp32", torch.device("cuda")) == "fp32"


def test_resolve_precision_auto_on_cpu_is_fp32():
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"


def test_resolve_precision_fp16_on_cpu_raises():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_precision("fp16", torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_resolve_precision_auto_on_cuda_is_fp16():
    assert resolve_precision("auto", torch.device("cuda")) == "fp16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_resolve_precision_fp16_on_cuda_is_fp16():
    assert resolve_precision("fp16", torch.device("cuda")) == "fp16"


# ---------------------------------------------------------------------------
# Real training verification
# ---------------------------------------------------------------------------

def test_fp32_cpu_training_unaffected(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), precision="fp32")).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.resolved_precision == "fp32"
    assert "accuracy" in result.training_metrics


def test_auto_precision_on_cpu_resolves_to_fp32_and_trains(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), precision="auto")).run()
    assert result.resolved_precision == "fp32"
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION


def test_fp16_on_cpu_device_raises_when_config_bypassed(small_dataset_dir, tmp_path):
    # validate_training_config already blocks this combination at
    # Trainer.__init__ time; this test exercises the runtime backstop in
    # run() directly (defense in depth), by constructing a Trainer without
    # going through __init__'s validation.
    config = _config(small_dataset_dir, str(tmp_path / "out"))
    config.precision = "fp16"  # mutate after construction to bypass __init__ validation
    trainer = object.__new__(Trainer)
    trainer.config = config
    with pytest.raises(ValueError, match="requires a CUDA device"):
        trainer.run()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_fp32_cuda_training_still_works(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), device="cuda", precision="fp32")).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.resolved_precision == "fp32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_fp16_cuda_training_works_and_generates_metrics(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), device="cuda", precision="fp16")).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.resolved_precision == "fp16"
    for key in ("loss", "accuracy", "precision", "recall", "f1"):
        assert key in result.validation_metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_auto_precision_on_cuda_resolves_to_fp16(small_dataset_dir, tmp_path):
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), device="cuda", precision="auto")).run()
    assert result.resolved_precision == "fp16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_fp16_checkpoint_is_loadable_like_any_other(small_dataset_dir, tmp_path):
    """
    Model parameters are always stored in FP32 regardless of the autocast
    precision used during training (the standard AMP pattern: only certain
    ops run in reduced precision, master weights stay FP32) - so a
    checkpoint trained under FP16 loads exactly like an FP32 one.
    """
    from hospital_client.training.trainer import get_model

    output_dir = str(tmp_path / "out")
    result = Trainer(_config(small_dataset_dir, output_dir, device="cuda", precision="fp16")).run()
    model, metadata = get_model(result.checkpoint_model_id, result.checkpoint_version, output_dir, device="cpu")
    assert next(model.parameters()).dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_fp16_resume_restores_scaler_state(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    Trainer(_config(small_dataset_dir, output_dir, device="cuda", precision="fp16", epochs=1)).run()
    r2 = Trainer(_config(small_dataset_dir, output_dir, device="cuda", precision="fp16", epochs=2, resume=True)).run()
    assert r2.epochs_run == 1
    assert r2.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA genuinely unavailable on this machine")
def test_cuda_oom_still_explicit_under_fp16(small_dataset_dir, tmp_path, monkeypatch):
    import hospital_client.training.trainer as trainer_module

    original_build_model = trainer_module.build_model

    def _oom_build_model(config):
        model = original_build_model(config)
        def _forward_that_ooms(*args, **kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        model.forward = _forward_that_ooms
        return model

    monkeypatch.setattr(trainer_module, "build_model", _oom_build_model)
    result = Trainer(_config(small_dataset_dir, str(tmp_path / "out"), device="cuda", precision="fp16")).run()
    assert result.status == TrainingStatus.TRAINING_FAILED
    assert any("out of memory" in e.lower() for e in result.errors)
