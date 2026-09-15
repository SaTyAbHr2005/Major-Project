"""
Regression tests for Module 5 respecting an existing Module 4-detected
train/validation/test split layout instead of always re-splitting by ratio.

Background (see the audit / previous fix report): SplitConfig.respect_existing_splits
existed but was never read anywhere, so Module 5 always re-split every valid
sample by ratio even when the hospital's dataset already had a valid
train/validation/test directory structure - silently discarding an existing
test set and reshuffling validation data that should never be touched.

These tests use REAL Module 4 output (hospital_client.dataset.ingestion.image_ingestor)
as the input to Module 5, not hand-built profile dictionaries.
"""
import json
import os

from PIL import Image

from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash
from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.preprocessing.config import OutputMode, PreprocessingConfig
from hospital_client.preprocessing.engine import PreprocessingEngine


_image_counter = 0


def _make_image(path, color=None):
    # Distinct content per call by default so these fixtures never trigger
    # cross-split duplicate-leakage detection (split_validation.py) by
    # accident; pass an explicit color to deliberately create duplicates.
    global _image_counter
    if color is None:
        n = _image_counter
        _image_counter += 1
        color = (n % 256, (n * 7) % 256, (n * 13) % 256)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def _norm(path):
    return os.path.normpath(str(path))


def _build_split_dataset(base_dir, layout):
    """
    layout: {split_name: {class_name: count}} builds a train/validation/test
    style directory tree. layout: {class_name: count} (values are ints, not
    dicts) builds a flat, no-split directory tree instead.

    Returns {normalized_absolute_path: (split_name_or_None, class_name)}.
    """
    created = {}
    is_split_layout = bool(layout) and isinstance(next(iter(layout.values())), dict)
    if is_split_layout:
        for split_name, classes in layout.items():
            for cls, count in classes.items():
                cls_dir = base_dir / split_name / cls
                cls_dir.mkdir(parents=True)
                for i in range(count):
                    path = cls_dir / f"{split_name}_{cls}_{i}.jpg"
                    _make_image(path)
                    created[_norm(path)] = (split_name, cls)
    else:
        for cls, count in layout.items():
            cls_dir = base_dir / cls
            cls_dir.mkdir(parents=True)
            for i in range(count):
                path = cls_dir / f"{cls}_{i}.jpg"
                _make_image(path)
                created[_norm(path)] = (None, cls)
    return created


def _run_engine(profile, tmp_path, output_mode=OutputMode.LAZY, respect_existing_splits=True):
    profile_path = tmp_path / "dataset_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile.to_dict(), f)

    output_dir = tmp_path / "out"
    config = PreprocessingConfig()
    config.dataset.output_mode = output_mode
    config.dataset.output_path = str(output_dir)
    config.split.respect_existing_splits = respect_existing_splits

    engine = PreprocessingEngine(config)
    engine.run(str(profile_path))
    return output_dir


def _load_manifests(output_dir):
    manifests = {}
    for split_name in ("train", "validation", "test"):
        with open(output_dir / f"{split_name}_manifest.json") as f:
            manifests[split_name] = json.load(f)
    return manifests


# ---------------------------------------------------------------------------
# (a) train/validation/test already present -> no re-splitting
# ---------------------------------------------------------------------------

def test_full_train_validation_test_layout_is_respected(tmp_path):
    data_dir = tmp_path / "data"
    layout = {
        "train": {"cat": 3, "dog": 3},
        "validation": {"cat": 2, "dog": 2},
        "test": {"cat": 2, "dog": 2},
    }
    created = _build_split_dataset(data_dir, layout)

    profile = ingest_image_dataset(str(data_dir))
    assert sorted(profile.splits["detected_splits"]) == ["test", "train", "validation"]

    output_dir = _run_engine(profile, tmp_path)
    manifests = _load_manifests(output_dir)

    assert manifests["train"]["total_records"] == 6
    assert manifests["validation"]["total_records"] == 4
    assert manifests["test"]["total_records"] == 4

    for split_name in ("train", "validation", "test"):
        for record in manifests[split_name]["records"]:
            expected_split, expected_cls = created[_norm(record["source_path"])]
            assert expected_split == split_name
            assert record["label"] == expected_cls
        assert manifests[split_name]["metadata"]["split_assignment"]["source"] == "existing"


# ---------------------------------------------------------------------------
# (b) train + test already present (no validation) -> respected, no
#     validation set is fabricated from train/test
# ---------------------------------------------------------------------------

def test_train_and_test_only_layout_validation_stays_empty(tmp_path):
    data_dir = tmp_path / "data"
    layout = {
        "train": {"cat": 4, "dog": 4},
        "test": {"cat": 2, "dog": 2},
    }
    created = _build_split_dataset(data_dir, layout)

    profile = ingest_image_dataset(str(data_dir))
    assert sorted(profile.splits["detected_splits"]) == ["test", "train"]

    output_dir = _run_engine(profile, tmp_path)
    manifests = _load_manifests(output_dir)

    assert manifests["train"]["total_records"] == 8
    assert manifests["test"]["total_records"] == 4
    assert manifests["validation"]["total_records"] == 0

    for split_name in ("train", "test"):
        for record in manifests[split_name]["records"]:
            expected_split, _ = created[_norm(record["source_path"])]
            assert expected_split == split_name
        assert manifests[split_name]["metadata"]["split_assignment"]["source"] == "existing"


def test_train_only_layout_is_respected(tmp_path):
    data_dir = tmp_path / "data"
    _build_split_dataset(data_dir, {"train": {"cat": 3, "dog": 3}})

    profile = ingest_image_dataset(str(data_dir))
    assert profile.splits["detected_splits"] == ["train"]

    output_dir = _run_engine(profile, tmp_path)
    manifests = _load_manifests(output_dir)

    assert manifests["train"]["total_records"] == 6
    assert manifests["validation"]["total_records"] == 0
    assert manifests["test"]["total_records"] == 0
    assert manifests["train"]["metadata"]["split_assignment"]["source"] == "existing"


# ---------------------------------------------------------------------------
# (c) no existing splits -> normal generated ratio-based splitting
# ---------------------------------------------------------------------------

def test_no_existing_splits_falls_back_to_generated_split(tmp_path):
    data_dir = tmp_path / "data"
    _build_split_dataset(data_dir, {"cat": 10, "dog": 10})

    profile = ingest_image_dataset(str(data_dir))
    # Flat class layout - Module 4 detects no split folders at all.
    assert profile.splits["detected_splits"] == []

    output_dir = _run_engine(profile, tmp_path)
    manifests = _load_manifests(output_dir)

    total = sum(m["total_records"] for m in manifests.values())
    assert total == 20  # complete accounting - nothing silently dropped
    for split_name in ("train", "validation", "test"):
        assert manifests[split_name]["metadata"]["split_assignment"]["source"] == "generated"


# ---------------------------------------------------------------------------
# (d) existing test data is preserved exactly, never dropped or moved
# ---------------------------------------------------------------------------

def test_existing_test_split_is_preserved_and_not_dropped(tmp_path):
    data_dir = tmp_path / "data"
    layout = {
        "train": {"cat": 3, "dog": 3},
        "test": {"cat": 1, "dog": 1},
    }
    created = _build_split_dataset(data_dir, layout)
    expected_test_paths = {p for p, (split, _) in created.items() if split == "test"}

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path)
    manifests = _load_manifests(output_dir)

    manifest_test_paths = {_norm(r["source_path"]) for r in manifests["test"]["records"]}
    assert manifest_test_paths == expected_test_paths
    assert manifests["train"]["total_records"] == 6


# ---------------------------------------------------------------------------
# (e) source dataset remains unchanged
# ---------------------------------------------------------------------------

def test_source_images_unchanged_when_respecting_existing_splits(tmp_path):
    data_dir = tmp_path / "data"
    created = _build_split_dataset(data_dir, {"train": {"cat": 2, "dog": 2}, "test": {"cat": 1, "dog": 1}})
    hashes_before = {p: compute_file_hash(p) for p in created}

    profile = ingest_image_dataset(str(data_dir))
    _run_engine(profile, tmp_path, output_mode=OutputMode.MATERIALIZED)

    hashes_after = {p: compute_file_hash(p) for p in created}
    assert hashes_before == hashes_after


# ---------------------------------------------------------------------------
# Materialized mode still works when respecting existing splits
# ---------------------------------------------------------------------------

def test_materialized_mode_respects_existing_splits(tmp_path):
    data_dir = tmp_path / "data"
    _build_split_dataset(data_dir, {"train": {"cat": 2, "dog": 2}, "test": {"cat": 1, "dog": 1}})

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, output_mode=OutputMode.MATERIALIZED)

    train_files = list((output_dir / "train").rglob("*.jpg"))
    test_files = list((output_dir / "test").rglob("*.jpg"))
    assert len(train_files) == 4
    assert len(test_files) == 2
    validation_dir = output_dir / "validation"
    assert not validation_dir.exists() or not list(validation_dir.rglob("*.jpg"))


# ---------------------------------------------------------------------------
# The config flag actually controls the behavior (not dead code anymore)
# ---------------------------------------------------------------------------

def test_respect_existing_splits_can_be_disabled(tmp_path):
    data_dir = tmp_path / "data"
    _build_split_dataset(data_dir, {"train": {"cat": 5, "dog": 5}, "test": {"cat": 5, "dog": 5}})

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, respect_existing_splits=False)
    manifests = _load_manifests(output_dir)

    for split_name in ("train", "validation", "test"):
        assert manifests[split_name]["metadata"]["split_assignment"]["source"] == "generated"
    total = sum(m["total_records"] for m in manifests.values())
    assert total == 20
