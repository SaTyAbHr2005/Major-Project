import pytest
import torch
import torch.nn as nn

from hospital_client.training.checkpoint import (
    load_training_state,
    restore_rng_state,
    save_training_state,
    training_state_path,
)


def test_save_and_load_training_state_roundtrip(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    path = save_training_state(
        str(tmp_path), "model_a", epoch=3, optimizer=optimizer, scheduler=scheduler,
        best_metric=0.42, best_epoch=2, class_mapping={"cat": 0, "dog": 1},
        model_config_dict={"architecture": "resnet18"}, checkpoint_model_id="model_a",
        checkpoint_version=2, random_seed=99,
    )
    assert path == training_state_path(str(tmp_path), "model_a")

    state = load_training_state(str(tmp_path), "model_a")
    assert state["epoch"] == 3
    assert state["best_metric"] == 0.42
    assert state["best_epoch"] == 2
    assert state["class_mapping"] == {"cat": 0, "dog": 1}
    assert state["checkpoint_version"] == 2
    assert state["random_seed"] == 99
    assert state["optimizer_state"] is not None
    assert state["scheduler_state"] is not None


def test_load_missing_training_state_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no training state found"):
        load_training_state(str(tmp_path), "nonexistent_model")


def test_optimizer_and_scheduler_can_be_restored_from_state(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    save_training_state(
        str(tmp_path), "m", epoch=1, optimizer=optimizer, scheduler=scheduler,
        best_metric=1.0, best_epoch=1, class_mapping={"a": 0}, model_config_dict={},
        checkpoint_model_id=None, checkpoint_version=None, random_seed=1,
    )
    state = load_training_state(str(tmp_path), "m")

    new_model = nn.Linear(4, 2)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(new_optimizer, patience=2)
    new_optimizer.load_state_dict(state["optimizer_state"])
    new_scheduler.load_state_dict(state["scheduler_state"])  # should not raise


def test_restore_rng_state_changes_subsequent_random_draws(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    torch.manual_seed(42)
    before_snapshot_draw = torch.rand(1)  # advance RNG from the seeded state

    torch.manual_seed(42)
    save_training_state(
        str(tmp_path), "m2", epoch=1, optimizer=optimizer, scheduler=None,
        best_metric=None, best_epoch=None, class_mapping={}, model_config_dict={},
        checkpoint_model_id=None, checkpoint_version=None, random_seed=42,
    )
    state = load_training_state(str(tmp_path), "m2")

    torch.manual_seed(0)  # perturb the RNG
    torch.rand(5)

    restore_rng_state(state)
    restored_draw = torch.rand(1)
    assert torch.equal(before_snapshot_draw, restored_draw)


def test_no_optimizer_or_scheduler_saved_gracefully(tmp_path):
    save_training_state(
        str(tmp_path), "m3", epoch=1, optimizer=None, scheduler=None,
        best_metric=None, best_epoch=None, class_mapping={}, model_config_dict={},
        checkpoint_model_id=None, checkpoint_version=None, random_seed=1,
    )
    state = load_training_state(str(tmp_path), "m3")
    assert state["optimizer_state"] is None
    assert state["scheduler_state"] is None
