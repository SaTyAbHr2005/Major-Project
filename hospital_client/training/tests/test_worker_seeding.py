"""
DataLoader worker RNG reproducibility. See module7_readme.md for the exact
reproducibility guarantees and limitations (this does NOT claim bit-for-bit
GPU determinism).
"""
import random

import numpy as np
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig
from hospital_client.training.dataloader import _seed_worker, build_dataloaders, derive_worker_seed
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer


def _config(dataset_dir, output_dir, **overrides):
    defaults = dict(
        model_id="m",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(32, 32)),
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=4,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# Pure seed derivation - no hard-coded values, deterministic, reproducible
# ---------------------------------------------------------------------------

def test_same_seed_and_worker_id_reproducible():
    assert derive_worker_seed(42, 0) == derive_worker_seed(42, 0)
    assert derive_worker_seed(42, 3) == derive_worker_seed(42, 3)


def test_different_worker_ids_get_different_seeds():
    seeds = {derive_worker_seed(42, w) for w in range(8)}
    assert len(seeds) == 8  # all distinct for a small worker range


def test_different_base_seeds_produce_different_worker_seeds():
    assert derive_worker_seed(1, 0) != derive_worker_seed(2, 0)


def test_seed_worker_actually_seeds_python_numpy_and_torch_rng():
    _seed_worker(worker_id=0, base_seed=123)
    py_val = random.random()
    np_val = np.random.rand()
    torch_val = torch.rand(1).item()

    _seed_worker(worker_id=0, base_seed=123)  # re-seed identically
    assert random.random() == py_val
    assert np.random.rand() == np_val
    assert torch.rand(1).item() == torch_val


def test_seed_worker_is_not_a_hardcoded_constant():
    _seed_worker(worker_id=1, base_seed=100)
    val_a = random.random()
    _seed_worker(worker_id=1, base_seed=200)
    val_b = random.random()
    assert val_a != val_b


# ---------------------------------------------------------------------------
# num_workers=0 and num_workers>0 both work (Windows-safe: module-level
# worker_init_fn via functools.partial, never a lambda/closure)
# ---------------------------------------------------------------------------

def test_num_workers_zero_still_works(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), num_workers=0)
    prepared = build_dataloaders(config)
    assert prepared.train_loader.num_workers == 0
    images, labels = next(iter(prepared.train_loader))
    assert images.shape[0] > 0


def test_num_workers_greater_than_zero_works_on_windows(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), num_workers=2)
    prepared = build_dataloaders(config)
    assert prepared.train_loader.num_workers == 2
    batches = list(prepared.train_loader)  # forces worker processes to spawn and run
    assert sum(b[0].shape[0] for b in batches) == prepared.train_sample_count


def test_default_num_workers_is_zero_low_resource_friendly():
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2), dataset_dir="d", output_dir="o"
    )
    assert config.num_workers == 0  # unchanged default - friendly to low-resource systems


def test_full_training_run_with_workers_completes(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), num_workers=2)
    result = Trainer(config).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
