"""
Regression tests for the existing-split VALIDATION layer.

The previous behavior ("existing split detected -> trust it") is replaced by
"existing split detected -> inspect -> validate -> preserve only when
appropriate" (see hospital_client/preprocessing/splitting/split_validation.py
and its wiring in engine.py). These tests are split into two groups:

- Unit tests calling validate_image_split() directly with small hand-built
  record lists, for precise coverage of each validation rule (empty splits,
  class consistency, imbalance, duplicate/group leakage, accounting).
- Engine-level integration tests using REAL Module 4 output
  (hospital_client.dataset.ingestion.image_ingestor.ingest_image_dataset),
  for the end-to-end decision behavior (preserve / carve / regenerate / stop).
"""
import json
import os

import pytest
from PIL import Image

from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.preprocessing.config import OutputMode, PreprocessingConfig, SplitConfig
from hospital_client.preprocessing.engine import PreprocessingEngine
from hospital_client.preprocessing.splitting.split_validation import (
    SplitValidationStatus,
    validate_image_split,
)


# ===========================================================================
# Unit tests: validate_image_split() directly
# ===========================================================================

def _rec(source_path, rel_path, label, split):
    return {"source_path": str(source_path), "rel_path": rel_path, "label": label, "pre_assigned_split": split}


def _write(path, content=b"x"):
    with open(path, "wb") as f:
        f.write(content)


def test_empty_declared_split_is_invalid_not_valid_complete(tmp_path):
    p1 = tmp_path / "a.jpg"
    _write(p1, b"AAA")
    records = [_rec(p1, "train/cat/a.jpg", "cat", "train")]
    # "test" was detected by Module 4 (folder exists) but produced 0 valid records.
    result = validate_image_split(records, [], ["train", "test"], SplitConfig(), str(tmp_path))
    assert result.status == SplitValidationStatus.INVALID
    assert result.empty_splits == ["test"]


def test_inconsistent_classes_across_splits_is_reported(tmp_path):
    p1, p2, p3 = tmp_path / "t1.jpg", tmp_path / "t2.jpg", tmp_path / "t3.jpg"
    _write(p1, b"AAA"); _write(p2, b"BBB"); _write(p3, b"CCC")
    records = [
        _rec(p1, "train/cat/1.jpg", "cat", "train"),
        _rec(p2, "train/dog/1.jpg", "dog", "train"),
        _rec(p3, "test/cat/1.jpg", "cat", "test"),  # test is missing "dog"
    ]
    result = validate_image_split(records, [], ["train", "test"], SplitConfig(), str(tmp_path))
    assert result.missing_classes.get("test") == ["dog"]
    # A class mismatch is reported, not auto-invalidated (samples aren't
    # deleted/modified to force agreement).
    assert result.status == SplitValidationStatus.VALID_PARTIAL


def test_severe_class_imbalance_is_flagged_but_does_not_invalidate(tmp_path):
    records = []
    for i in range(20):
        p = tmp_path / f"cat_{i}.jpg"
        _write(p, f"CAT{i}".encode())
        records.append(_rec(p, f"train/cat/{i}.jpg", "cat", "train"))
    p_dog = tmp_path / "dog_0.jpg"
    _write(p_dog, b"DOG0")
    records.append(_rec(p_dog, "train/dog/0.jpg", "dog", "train"))

    config = SplitConfig(severe_class_imbalance_ratio=5.0)
    result = validate_image_split(records, [], ["train"], config, str(tmp_path))
    assert result.class_distribution_warnings  # flagged
    assert result.status == SplitValidationStatus.VALID_PARTIAL  # not invalidated


def test_mild_imbalance_under_threshold_is_not_flagged(tmp_path):
    records = []
    for i in range(3):
        p = tmp_path / f"cat_{i}.jpg"
        _write(p, f"CAT{i}".encode())
        records.append(_rec(p, f"train/cat/{i}.jpg", "cat", "train"))
    p_dog = tmp_path / "dog_0.jpg"
    _write(p_dog, b"DOG0")
    records.append(_rec(p_dog, "train/dog/0.jpg", "dog", "train"))  # ratio 3:1 < default 5.0

    result = validate_image_split(records, [], ["train"], SplitConfig(), str(tmp_path))
    assert result.class_distribution_warnings == []


@pytest.mark.parametrize("split_a,split_b", [("train", "test"), ("train", "validation"), ("validation", "test")])
def test_duplicate_across_two_splits_detected(tmp_path, split_a, split_b):
    shared = b"IDENTICAL_CONTENT"
    p_a = tmp_path / f"{split_a}_dup.jpg"
    p_b = tmp_path / f"{split_b}_dup.jpg"
    _write(p_a, shared)
    _write(p_b, shared)
    p_other = tmp_path / "unique.jpg"
    _write(p_other, b"UNIQUE")

    declared = sorted({"train", "validation", "test"})  # all three present, all non-empty
    records = [
        _rec(p_a, f"{split_a}/cat/dup.jpg", "cat", split_a),
        _rec(p_b, f"{split_b}/cat/dup.jpg", "cat", split_b),
    ]
    # give every declared split at least one (non-duplicate) sample too
    for s in declared:
        if s not in (split_a, split_b):
            records.append(_rec(p_other, f"{s}/cat/unique.jpg", "cat", s))

    result = validate_image_split(records, [], declared, SplitConfig(), str(tmp_path))
    assert result.status == SplitValidationStatus.INVALID
    assert any(set(d["splits"]) == {split_a, split_b} for d in result.duplicate_leakage)


def test_group_leakage_detected_with_real_identifier(tmp_path):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _write(p1, b"A"); _write(p2, b"B")
    records = [
        _rec(p1, "train/cat/a.jpg", "cat", "train"),
        _rec(p2, "test/cat/b.jpg", "cat", "test"),
    ]
    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps({"train/cat/a.jpg": "patient_102", "test/cat/b.jpg": "patient_102"}))

    config = SplitConfig(group_id_map_path=str(group_map_path))
    result = validate_image_split(records, [], ["train", "test"], config, str(tmp_path))
    assert result.status == SplitValidationStatus.INVALID
    assert result.group_leakage["checked"] is True
    assert "patient_102" in result.group_leakage["leaking_groups"]


def test_group_leakage_reported_as_unverifiable_without_identifier(tmp_path):
    p1 = tmp_path / "a.jpg"
    _write(p1, b"A")
    records = [_rec(p1, "train/cat/a.jpg", "cat", "train")]
    result = validate_image_split(records, [], ["train"], SplitConfig(), str(tmp_path))
    assert result.group_leakage["checked"] is False
    assert "could not be verified" in result.group_leakage["reason"]


def test_rejected_samples_are_counted_per_split():
    records = [_rec("a", "train/cat/a.jpg", "cat", "train"), _rec("b", "test/cat/b.jpg", "cat", "test")]
    rejected = [
        {"rel_path": "test/cat/corrupt.jpg", "pre_assigned_split": "test"},
        {"rel_path": "test/cat/corrupt2.jpg", "pre_assigned_split": "test"},
    ]
    result = validate_image_split(records, rejected, ["train", "test"], SplitConfig(), ".")
    assert result.invalid_sample_counts.get("test") == 2
    # Rejected samples don't empty the split (a valid record still exists there).
    assert result.status == SplitValidationStatus.VALID_PARTIAL


def test_mixed_layout_with_unassigned_file_is_ambiguous():
    records = [_rec("a", "train/cat/a.jpg", "cat", "train")]
    rejected = [{"rel_path": "disease/img.jpg", "pre_assigned_split": None}]
    result = validate_image_split(records, rejected, ["train"], SplitConfig(), ".")
    assert result.status == SplitValidationStatus.AMBIGUOUS
    assert result.unassigned_sample_count == 1


def test_no_train_split_present_is_invalid():
    records = [_rec("a", "test/cat/a.jpg", "cat", "test")]
    result = validate_image_split(records, [], ["test"], SplitConfig(), ".")
    assert result.status == SplitValidationStatus.INVALID


def test_no_detected_splits_is_no_existing_split():
    result = validate_image_split([], [], [], SplitConfig(), ".")
    assert result.status == SplitValidationStatus.NO_EXISTING_SPLIT


# ===========================================================================
# Engine-level integration tests (real Module 4 output)
# ===========================================================================

_image_counter = 0


def _make_image(path, color=None):
    global _image_counter
    if color is None:
        n = _image_counter
        _image_counter += 1
        color = (n % 256, (n * 7) % 256, (n * 13) % 256)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def _run_engine(profile, tmp_path, **split_overrides):
    profile_path = tmp_path / "dataset_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile.to_dict(), f)

    output_dir = tmp_path / "out"
    config = PreprocessingConfig()
    config.dataset.output_path = str(output_dir)
    for key, value in split_overrides.items():
        setattr(config.split, key, value)

    engine = PreprocessingEngine(config)
    engine.run(str(profile_path))
    return output_dir


def _report(output_dir):
    with open(output_dir / "preprocessing_report.json") as f:
        return json.load(f)


def test_ambiguous_mixed_layout_end_to_end_is_not_silently_resplit(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train" / "normal").mkdir(parents=True)
    _make_image(data_dir / "train" / "normal" / "1.jpg")
    (data_dir / "disease").mkdir(parents=True)
    _make_image(data_dir / "disease" / "img1.jpg")
    _make_image(data_dir / "disease" / "img2.jpg")

    profile = ingest_image_dataset(str(data_dir))

    with pytest.raises(ValueError, match="ambiguous"):
        _run_engine(profile, tmp_path)

    # Must not have silently produced a re-split output.
    assert not (tmp_path / "out" / "train_manifest.json").exists()


def test_ambiguous_layout_can_be_explicitly_regenerated(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train" / "normal").mkdir(parents=True)
    _make_image(data_dir / "train" / "normal" / "1.jpg")
    (data_dir / "disease").mkdir(parents=True)
    _make_image(data_dir / "disease" / "img1.jpg")
    _make_image(data_dir / "disease" / "img2.jpg")

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, invalid_split_policy="regenerate")

    manifests_total = 0
    for split_name in ("train", "validation", "test"):
        with open(output_dir / f"{split_name}_manifest.json") as f:
            manifest = json.load(f)
        manifests_total += manifest["total_records"]
        assert manifest["metadata"]["split_assignment"]["source"] == "regenerated_after_ambiguous"
    assert manifests_total == 3


def test_duplicate_leakage_end_to_end_errors_by_default(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train" / "cat").mkdir(parents=True)
    (data_dir / "test" / "cat").mkdir(parents=True)
    _make_image(data_dir / "train" / "cat" / "1.jpg", color=(9, 9, 9))
    _make_image(data_dir / "train" / "cat" / "2.jpg")
    # Byte-for-byte copy of the train image placed in test.
    import shutil
    shutil.copy2(data_dir / "train" / "cat" / "1.jpg", data_dir / "test" / "cat" / "1.jpg")

    profile = ingest_image_dataset(str(data_dir))
    with pytest.raises(ValueError, match="invalid"):
        _run_engine(profile, tmp_path)
    assert not (tmp_path / "out" / "train_manifest.json").exists()


def test_duplicate_leakage_can_be_explicitly_regenerated(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train" / "cat").mkdir(parents=True)
    (data_dir / "test" / "cat").mkdir(parents=True)
    _make_image(data_dir / "train" / "cat" / "1.jpg", color=(9, 9, 9))
    _make_image(data_dir / "train" / "cat" / "2.jpg")
    import shutil
    shutil.copy2(data_dir / "train" / "cat" / "1.jpg", data_dir / "test" / "cat" / "1.jpg")

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, invalid_split_policy="regenerate")
    with open(output_dir / "train_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["metadata"]["split_assignment"]["source"] == "regenerated_after_invalid"


def test_generate_missing_validation_from_train_when_explicitly_configured(tmp_path):
    data_dir = tmp_path / "data"
    for cls in ("cat", "dog"):
        (data_dir / "train" / cls).mkdir(parents=True)
        (data_dir / "test" / cls).mkdir(parents=True)
        for i in range(10):
            _make_image(data_dir / "train" / cls / f"{i}.jpg")
        for i in range(2):
            _make_image(data_dir / "test" / cls / f"{i}.jpg")
    original_test_files = {str(p) for p in (data_dir / "test").rglob("*.jpg")}

    profile = ingest_image_dataset(str(data_dir))

    # Default: validation NOT fabricated.
    default_output = _run_engine(profile, tmp_path)
    with open(default_output / "validation_manifest.json") as f:
        assert json.load(f)["total_records"] == 0

    # Explicitly configured: validation carved out of TRAIN only.
    configured_output = _run_engine(profile, tmp_path, generate_missing_splits_from_train=True)
    with open(configured_output / "validation_manifest.json") as f:
        val_manifest = json.load(f)
    with open(configured_output / "test_manifest.json") as f:
        test_manifest = json.load(f)
    with open(configured_output / "train_manifest.json") as f:
        train_manifest = json.load(f)

    assert val_manifest["total_records"] > 0
    assert val_manifest["metadata"]["split_assignment"]["source"] == "existing_partial_generated_from_train"
    # Test set is untouched - exact same files, none moved into validation.
    test_files = {r["source_path"] for r in test_manifest["records"]}
    assert test_files == original_test_files
    # Every carved validation record originally came from train, never from test.
    val_sources = {r["source_path"] for r in val_manifest["records"]}
    assert val_sources.isdisjoint(original_test_files)
    # Nothing lost: train + validation together still account for all 20 original train images.
    assert train_manifest["total_records"] + val_manifest["total_records"] == 20


def test_corrupted_image_inside_existing_split_is_reported_not_silent(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train" / "cat").mkdir(parents=True)
    (data_dir / "test" / "cat").mkdir(parents=True)
    for i in range(3):
        _make_image(data_dir / "train" / "cat" / f"{i}.jpg")
    _make_image(data_dir / "test" / "cat" / "0.jpg")
    # A corrupted file inside the existing train split.
    with open(data_dir / "train" / "cat" / "corrupt.jpg", "wb") as f:
        f.write(b"NOT_AN_IMAGE")

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path)

    report = _report(output_dir)
    validation = report["split_validation"]
    assert validation["status"] == "valid_complete" or validation["status"] == "valid_partial"
    assert validation["invalid_sample_counts"].get("train", 0) == 1

    # Complete accounting: discovered = accepted + rejected + review.
    q = report["quarantine"]
    assert q["total_discovered"] == q["accepted"] + q["rejected"] + q["review"]
    assert q["total_discovered"] == 3 + 1 + 1  # 3 good train + 1 corrupt train + 1 good test


def test_source_dataset_unchanged_after_ambiguous_and_invalid_failures(tmp_path):
    from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash

    data_dir = tmp_path / "data"
    (data_dir / "train" / "normal").mkdir(parents=True)
    _make_image(data_dir / "train" / "normal" / "1.jpg")
    (data_dir / "disease").mkdir(parents=True)
    _make_image(data_dir / "disease" / "img1.jpg")

    all_files = list(data_dir.rglob("*.jpg"))
    hashes_before = {str(p): compute_file_hash(str(p)) for p in all_files}

    profile = ingest_image_dataset(str(data_dir))
    with pytest.raises(ValueError):
        _run_engine(profile, tmp_path)

    hashes_after = {str(p): compute_file_hash(str(p)) for p in all_files}
    assert hashes_before == hashes_after
