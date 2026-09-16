import pytest
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig, validate_training_config
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer


def _config(dataset_dir, output_dir, **overrides):
    defaults = dict(
        model_id="m",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)),
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=3,
        batch_size=4,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_none_and_string_none_both_mean_no_scheduler():
    for value in (None, "none"):
        config = _config("d", "o", scheduler=value)
        validate_training_config(config)  # must not raise


def test_existing_reduce_on_plateau_config_still_valid():
    config = _config("d", "o", scheduler="reduce_on_plateau", scheduler_patience=3, scheduler_factor=0.1)
    validate_training_config(config)


def test_step_scheduler_valid_config():
    config = _config("d", "o", scheduler="step", scheduler_step_size=5, scheduler_gamma=0.5)
    validate_training_config(config)


def test_step_scheduler_invalid_step_size_rejected():
    config = _config("d", "o", scheduler="step", scheduler_step_size=0)
    with pytest.raises(ValueError, match="scheduler_step_size"):
        validate_training_config(config)


def test_step_scheduler_invalid_gamma_rejected():
    config = _config("d", "o", scheduler="step", scheduler_gamma=0)
    with pytest.raises(ValueError, match="scheduler_gamma"):
        validate_training_config(config)


def test_cosine_scheduler_valid_config():
    config = _config("d", "o", scheduler="cosine", scheduler_t_max=10, scheduler_eta_min=0.0001)
    validate_training_config(config)


def test_cosine_scheduler_defaults_t_max_to_none():
    config = _config("d", "o", scheduler="cosine")
    validate_training_config(config)  # scheduler_t_max=None is valid - defaults to epochs at build time


def test_cosine_scheduler_invalid_t_max_rejected():
    config = _config("d", "o", scheduler="cosine", scheduler_t_max=0)
    with pytest.raises(ValueError, match="scheduler_t_max"):
        validate_training_config(config)


def test_cosine_scheduler_invalid_eta_min_rejected():
    config = _config("d", "o", scheduler="cosine", scheduler_eta_min=-1.0)
    with pytest.raises(ValueError, match="scheduler_eta_min"):
        validate_training_config(config)


def test_invalid_scheduler_name_rejected():
    config = _config("d", "o", scheduler="exponential")
    with pytest.raises(ValueError, match="scheduler"):
        validate_training_config(config)


# ---------------------------------------------------------------------------
# Real training runs / LR actually changes / resume
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheduler_name,kwargs", [
    ("reduce_on_plateau", {}),
    ("step", {"scheduler_step_size": 1, "scheduler_gamma": 0.5}),
    ("cosine", {"scheduler_t_max": 3}),
    (None, {}),
])
def test_each_scheduler_completes_a_real_training_run(scheduler_name, kwargs, small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), scheduler=scheduler_name, **kwargs)
    result = Trainer(config).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.epochs_run == 3


def test_step_scheduler_actually_changes_learning_rate(small_dataset_dir, tmp_path):
    from hospital_client.model_management.registry import build_model

    config = _config(
        small_dataset_dir, str(tmp_path / "out"), scheduler="step",
        scheduler_step_size=1, scheduler_gamma=0.5, learning_rate=0.1, epochs=3,
    )
    trainer = Trainer(config)
    model = build_model(config.model_config)
    optimizer = trainer._build_optimizer(model)
    scheduler = trainer._build_scheduler(optimizer)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs == pytest.approx([0.1, 0.05, 0.025, 0.0125])


def test_cosine_scheduler_actually_changes_learning_rate(small_dataset_dir, tmp_path):
    from hospital_client.model_management.registry import build_model

    config = _config(
        small_dataset_dir, str(tmp_path / "out"), scheduler="cosine",
        scheduler_t_max=4, learning_rate=0.1, epochs=4,
    )
    trainer = Trainer(config)
    model = build_model(config.model_config)
    optimizer = trainer._build_optimizer(model)
    scheduler = trainer._build_scheduler(optimizer)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Cosine annealing must actually vary the LR, not hold it constant.
    assert len(set(round(lr, 8) for lr in lrs)) > 1
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)  # reaches eta_min=0 at T_max


def test_cosine_scheduler_t_max_defaults_to_epochs(small_dataset_dir, tmp_path):
    from hospital_client.model_management.registry import build_model

    config = _config(small_dataset_dir, str(tmp_path / "out"), scheduler="cosine", epochs=5)  # scheduler_t_max unset
    trainer = Trainer(config)
    model = build_model(config.model_config)
    optimizer = trainer._build_optimizer(model)
    scheduler = trainer._build_scheduler(optimizer)
    assert scheduler.T_max == 5


def test_resume_restores_step_scheduler_state(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    config1 = _config(output_dir=output_dir, dataset_dir=small_dataset_dir, scheduler="step", scheduler_step_size=1, epochs=2)
    r1 = Trainer(config1).run()
    assert r1.epochs_run == 2

    config2 = _config(
        output_dir=output_dir, dataset_dir=small_dataset_dir, scheduler="step",
        scheduler_step_size=1, epochs=4, resume=True,
    )
    r2 = Trainer(config2).run()
    assert r2.epochs_run == 2  # only the remaining 2 epochs
    assert r2.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION


def test_resume_restores_cosine_scheduler_state(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    config1 = _config(output_dir=output_dir, dataset_dir=small_dataset_dir, scheduler="cosine", scheduler_t_max=4, epochs=2)
    Trainer(config1).run()

    config2 = _config(
        output_dir=output_dir, dataset_dir=small_dataset_dir, scheduler="cosine",
        scheduler_t_max=4, epochs=4, resume=True,
    )
    r2 = Trainer(config2).run()
    assert r2.epochs_run == 2
    assert r2.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION


def test_default_scheduler_is_still_none_unchanged():
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2), dataset_dir="d", output_dir="o"
    )
    assert config.scheduler is None


def test_scheduler_is_deterministic_under_same_config(small_dataset_dir, tmp_path):
    from hospital_client.model_management.registry import build_model

    def _lr_trace(seed):
        config = _config(
            small_dataset_dir, str(tmp_path / f"out_{seed}"), scheduler="step",
            scheduler_step_size=1, scheduler_gamma=0.3, random_seed=seed,
        )
        trainer = Trainer(config)
        torch.manual_seed(seed)
        model = build_model(config.model_config)
        optimizer = trainer._build_optimizer(model)
        scheduler = trainer._build_scheduler(optimizer)
        trace = []
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            trace.append(optimizer.param_groups[0]["lr"])
        return trace

    assert _lr_trace(1) == _lr_trace(1)
