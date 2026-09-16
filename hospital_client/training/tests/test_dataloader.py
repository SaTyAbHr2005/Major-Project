import json
import os

import pytest

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig
from hospital_client.training.dataloader import build_dataloaders


def _config(dataset_dir, output_dir, **overrides):
    defaults = dict(
        model_id="m",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(32, 32)),
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_lazy_manifest_dataset_loads_correctly(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"))
    prepared = build_dataloaders(config)
    assert prepared.train_sample_count > 0
    assert prepared.validation_sample_count > 0
    assert set(prepared.class_mapping.keys()) == {"cat", "dog"}

    images, labels = next(iter(prepared.train_loader))
    assert images.shape[1:] == (3, 32, 32)  # RGB, resized to model input size
    assert labels.dtype.is_floating_point is False


def test_materialized_manifest_dataset_loads_correctly(small_dataset_dir_materialized, tmp_path):
    config = _config(small_dataset_dir_materialized, str(tmp_path / "out"))
    prepared = build_dataloaders(config)
    assert prepared.train_sample_count > 0
    images, labels = next(iter(prepared.train_loader))
    assert images.shape[1:] == (3, 32, 32)


def test_train_loader_shuffles_and_validation_does_not(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"))
    prepared = build_dataloaders(config)
    assert prepared.train_loader.sampler.__class__.__name__ != "SequentialSampler"
    assert prepared.validation_loader.sampler.__class__.__name__ == "SequentialSampler"


def test_test_loader_only_built_when_requested(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path / "out"), run_test_evaluation=False)
    prepared = build_dataloaders(config)
    assert prepared.test_loader is None

    config2 = _config(small_dataset_dir, str(tmp_path / "out2"), run_test_evaluation=True)
    prepared2 = build_dataloaders(config2)
    assert prepared2.test_loader is not None
    assert prepared2.test_sample_count > 0


def test_missing_validation_manifest_raises_when_required(tmp_path):
    dataset_dir = tmp_path / "empty_dataset"
    dataset_dir.mkdir()
    with open(dataset_dir / "train_manifest.json", "w") as f:
        json.dump({"split": "train", "metadata": {}, "total_records": 2, "records": [
            {"source_path": "x.jpg", "rel_path": "x.jpg", "label": "cat", "subtype": None,
             "pre_assigned_split": None, "size": 1, "assigned_split": "train", "processed_path": None},
            {"source_path": "y.jpg", "rel_path": "y.jpg", "label": "dog", "subtype": None,
             "pre_assigned_split": None, "size": 1, "assigned_split": "train", "processed_path": None},
        ]}, f)
    config = _config(str(dataset_dir), str(tmp_path / "out"))
    with pytest.raises(ValueError, match="Validation manifest not found"):
        build_dataloaders(config)


def test_missing_validation_manifest_allowed_when_not_required(tmp_path):
    dataset_dir = tmp_path / "empty_dataset"
    dataset_dir.mkdir()
    with open(dataset_dir / "train_manifest.json", "w") as f:
        json.dump({"split": "train", "metadata": {}, "total_records": 2, "records": [
            {"source_path": "x.jpg", "rel_path": "x.jpg", "label": "cat", "subtype": None,
             "pre_assigned_split": None, "size": 1, "assigned_split": "train", "processed_path": None},
            {"source_path": "y.jpg", "rel_path": "y.jpg", "label": "dog", "subtype": None,
             "pre_assigned_split": None, "size": 1, "assigned_split": "train", "processed_path": None},
        ]}, f)
    config = _config(str(dataset_dir), str(tmp_path / "out"), require_validation=False)
    prepared = build_dataloaders(config)
    assert prepared.validation_loader is None


def test_missing_train_manifest_raises(tmp_path):
    dataset_dir = tmp_path / "nonexistent"
    config = _config(str(dataset_dir), str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        build_dataloaders(config)


def test_empty_train_manifest_raises(tmp_path):
    dataset_dir = tmp_path / "empty"
    dataset_dir.mkdir()
    with open(dataset_dir / "train_manifest.json", "w") as f:
        json.dump({"split": "train", "metadata": {}, "total_records": 0, "records": []}, f)
    config = _config(str(dataset_dir), str(tmp_path / "out"))
    with pytest.raises(ValueError, match="zero records"):
        build_dataloaders(config)


def test_class_count_mismatch_rejected(small_dataset_dir, tmp_path):
    wrong_config = ModelConfig(architecture="resnet18", num_classes=5, input_size=(32, 32))
    config = _config(small_dataset_dir, str(tmp_path / "out"), model_config=wrong_config)
    with pytest.raises(ValueError, match="Class-count mismatch"):
        build_dataloaders(config)


def test_unknown_label_in_validation_split_rejected(small_dataset_dir, tmp_path):
    val_path = os.path.join(small_dataset_dir, "validation_manifest.json")
    with open(val_path) as f:
        manifest = json.load(f)
    if manifest["records"]:
        manifest["records"][0]["label"] = "totally_new_class"
    with open(val_path, "w") as f:
        json.dump(manifest, f)

    config = _config(small_dataset_dir, str(tmp_path / "out"))
    with pytest.raises(ValueError, match="never seen in the training manifest"):
        build_dataloaders(config)
