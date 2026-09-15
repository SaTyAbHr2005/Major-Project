"""
Regression tests for the limitations closed after the split-validation task:

1. Duplicate/group leakage severity is now configurable via
   SplitConfig.leakage_policy ("invalid" default vs "warning").
2. generate_missing_splits_from_train is verified for every VALID_PARTIAL
   combination (only validation missing, only test missing, both missing),
   not just the train+test example from the original task.
3. The image "generated split" path no longer fabricates group identifiers
   (previously np.arange(...)) when GROUPED strategy + group_column were
   set; it now requires a real group_id_map_path and raises rather than
   silently faking patient/group information, for both the ratio-based
   generated split and the partial-split carve-from-train path.
"""
import json

import numpy as np
import pytest
from PIL import Image

from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.preprocessing.config import PreprocessingConfig, SplitConfig, SplitStrategy
from hospital_client.preprocessing.engine import PreprocessingEngine
from hospital_client.preprocessing.splitting.split_validation import (
    SplitValidationStatus,
    validate_image_split,
)

_image_counter = 0


def _make_image(path, color=None):
    global _image_counter
    if color is None:
        n = _image_counter
        _image_counter += 1
        color = (n % 256, (n * 7) % 256, (n * 13) % 256)
    Image.new("RGB", (32, 32), color=color).save(path)


def _rec(source_path, rel_path, label, split):
    return {"source_path": str(source_path), "rel_path": rel_path, "label": label, "pre_assigned_split": split}


def _write(path, content=b"x"):
    with open(path, "wb") as f:
        f.write(content)


def _run_engine(profile, tmp_path, **split_overrides):
    profile_path = tmp_path / "dataset_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile.to_dict(), f)
    output_dir = tmp_path / "out"
    config = PreprocessingConfig()
    config.dataset.output_path = str(output_dir)
    for key, value in split_overrides.items():
        setattr(config.split, key, value)
    PreprocessingEngine(config).run(str(profile_path))
    return output_dir


def _manifest(output_dir, split_name):
    with open(output_dir / f"{split_name}_manifest.json") as f:
        return json.load(f)


# ===========================================================================
# 1. Leakage severity is configurable (WARNING vs INVALID)
# ===========================================================================

def test_duplicate_leakage_warning_policy_preserves_the_split(tmp_path):
    shared = b"IDENTICAL"
    p_train, p_test = tmp_path / "train.jpg", tmp_path / "test.jpg"
    _write(p_train, shared)
    _write(p_test, shared)
    records = [_rec(p_train, "train/cat/1.jpg", "cat", "train"), _rec(p_test, "test/cat/1.jpg", "cat", "test")]

    config = SplitConfig(leakage_policy="warning")
    result = validate_image_split(records, [], ["train", "test"], config, str(tmp_path))
    assert result.status == SplitValidationStatus.VALID_PARTIAL  # not INVALID
    assert result.duplicate_leakage  # still reported, never hidden
    assert any("leakage_policy='warning'" in r for r in result.reasons)


def test_duplicate_leakage_default_policy_is_still_invalid(tmp_path):
    shared = b"IDENTICAL"
    p_train, p_test = tmp_path / "train.jpg", tmp_path / "test.jpg"
    _write(p_train, shared)
    _write(p_test, shared)
    records = [_rec(p_train, "train/cat/1.jpg", "cat", "train"), _rec(p_test, "test/cat/1.jpg", "cat", "test")]

    result = validate_image_split(records, [], ["train", "test"], SplitConfig(), str(tmp_path))
    assert result.status == SplitValidationStatus.INVALID


def test_group_leakage_warning_policy_preserves_the_split(tmp_path):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _write(p1, b"A"); _write(p2, b"B")
    records = [_rec(p1, "train/cat/a.jpg", "cat", "train"), _rec(p2, "test/cat/b.jpg", "cat", "test")]
    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps({"train/cat/a.jpg": "patient_1", "test/cat/b.jpg": "patient_1"}))

    config = SplitConfig(group_id_map_path=str(group_map_path), leakage_policy="warning")
    result = validate_image_split(records, [], ["train", "test"], config, str(tmp_path))
    assert result.status == SplitValidationStatus.VALID_PARTIAL
    assert result.group_leakage["leaking_groups"]  # still reported


def test_engine_end_to_end_with_leakage_warning_policy_completes(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "train" / "cat").mkdir(parents=True)
    (data_dir / "test" / "cat").mkdir(parents=True)
    _make_image(data_dir / "train" / "cat" / "1.jpg", color=(1, 1, 1))
    _make_image(data_dir / "train" / "cat" / "2.jpg")
    import shutil
    shutil.copy2(data_dir / "train" / "cat" / "1.jpg", data_dir / "test" / "cat" / "1.jpg")

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, leakage_policy="warning")

    train_manifest = _manifest(output_dir, "train")
    validation = train_manifest["metadata"]["split_assignment"]["validation"]
    assert validation["status"] == "valid_partial"
    assert validation["duplicate_leakage"]
    assert train_manifest["metadata"]["split_assignment"]["source"] == "existing"


# ===========================================================================
# 2. generate_missing_splits_from_train covers every partial combination
# ===========================================================================

def _build_layout(data_dir, layout):
    for split_name, classes in layout.items():
        for cls, count in classes.items():
            cls_dir = data_dir / split_name / cls
            cls_dir.mkdir(parents=True)
            for i in range(count):
                _make_image(cls_dir / f"{i}.jpg")


def test_carve_only_test_missing_leaves_validation_untouched(tmp_path):
    data_dir = tmp_path / "data"
    _build_layout(data_dir, {
        "train": {"cat": 10, "dog": 10},
        "validation": {"cat": 3, "dog": 3},
    })
    original_validation_files = {str(p) for p in (data_dir / "validation").rglob("*.jpg")}

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, generate_missing_splits_from_train=True)

    val_manifest = _manifest(output_dir, "validation")
    test_manifest = _manifest(output_dir, "test")
    train_manifest = _manifest(output_dir, "train")

    assert test_manifest["total_records"] > 0
    assert test_manifest["metadata"]["split_assignment"]["source"] == "existing_partial_generated_from_train"
    # Existing validation is exactly preserved - none of it was touched.
    val_files = {r["source_path"] for r in val_manifest["records"]}
    assert val_files == original_validation_files
    assert train_manifest["total_records"] + test_manifest["total_records"] == 20


def test_carve_both_validation_and_test_from_train_only_layout(tmp_path):
    data_dir = tmp_path / "data"
    _build_layout(data_dir, {"train": {"cat": 10, "dog": 10}})

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(profile, tmp_path, generate_missing_splits_from_train=True)

    train_manifest = _manifest(output_dir, "train")
    val_manifest = _manifest(output_dir, "validation")
    test_manifest = _manifest(output_dir, "test")

    assert val_manifest["total_records"] > 0
    assert test_manifest["total_records"] > 0
    assert val_manifest["metadata"]["split_assignment"]["source"] == "existing_partial_generated_from_train"
    assert train_manifest["total_records"] + val_manifest["total_records"] + test_manifest["total_records"] == 20


# ===========================================================================
# 3. No fabricated group identifiers in the generated / carve paths
# ===========================================================================

def test_grouped_strategy_without_group_map_raises_instead_of_fabricating(tmp_path):
    # A flat (no split-folder) layout takes the NO_EXISTING_SPLIT -> generated
    # ratio-split path, which is where the previous np.arange(...) fabrication lived.
    data_dir = tmp_path / "flat"
    for cls, count in {"cat": 5, "dog": 5}.items():
        (data_dir / cls).mkdir(parents=True)
        for i in range(count):
            _make_image(data_dir / cls / f"{i}.jpg")

    profile = ingest_image_dataset(str(data_dir))
    with pytest.raises(ValueError, match="does not fabricate group identifiers"):
        _run_engine(profile, tmp_path, strategy=SplitStrategy.GROUPED)


def test_grouped_strategy_with_incomplete_group_map_raises(tmp_path):
    data_dir = tmp_path / "flat"
    for cls, count in {"cat": 3, "dog": 3}.items():
        (data_dir / cls).mkdir(parents=True)
        for i in range(count):
            _make_image(data_dir / cls / f"{i}.jpg")

    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps({"cat/0.jpg": "p1"}))  # only 1 of 6 samples covered

    profile = ingest_image_dataset(str(data_dir))
    with pytest.raises(ValueError, match="missing from group_id_map_path"):
        _run_engine(profile, tmp_path, strategy=SplitStrategy.GROUPED, group_id_map_path=str(group_map_path))


def test_grouped_strategy_with_real_complete_map_keeps_groups_together(tmp_path):
    import os as _os
    data_dir = tmp_path / "flat"
    rel_to_group = {}
    for cls in ("cat", "dog"):
        (data_dir / cls).mkdir(parents=True)
        for i in range(6):
            path = data_dir / cls / f"{i}.jpg"
            _make_image(path)
            rel_path = _os.path.relpath(str(path), str(data_dir))
            # Two files per patient (patients don't respect class boundaries here,
            # only used to prove group integrity across the resulting splits).
            rel_to_group[rel_path] = f"patient_{i // 2}_{cls}"

    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps(rel_to_group))

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(
        profile, tmp_path, strategy=SplitStrategy.GROUPED, group_id_map_path=str(group_map_path)
    )

    split_to_groups = {}
    for split_name in ("train", "validation", "test"):
        manifest = _manifest(output_dir, split_name)
        groups_in_split = set()
        for r in manifest["records"]:
            rel = _os.path.relpath(r["source_path"], str(data_dir))
            groups_in_split.add(rel_to_group[rel])
        split_to_groups[split_name] = groups_in_split

    # No group/patient appears in more than one resulting split.
    all_groups_seen = []
    for groups_in_split in split_to_groups.values():
        all_groups_seen.extend(groups_in_split)
    assert len(all_groups_seen) == len(set(all_groups_seen)), (
        f"A patient/group id was split across multiple output splits: {split_to_groups}"
    )


def test_carve_from_train_with_real_groups_preserves_group_integrity(tmp_path):
    import os as _os
    import shutil
    data_dir = tmp_path / "data"
    rel_to_group = {}
    for cls in ("cat", "dog"):
        (data_dir / "train" / cls).mkdir(parents=True)
        (data_dir / "test" / cls).mkdir(parents=True)
        for i in range(10):
            path = data_dir / "train" / cls / f"{i}.jpg"
            _make_image(path)
            rel_to_group[_os.path.relpath(str(path), str(data_dir))] = f"patient_{i}_{cls}"
        for i in range(2):
            path = data_dir / "test" / cls / f"{i}.jpg"
            _make_image(path)
            rel_to_group[_os.path.relpath(str(path), str(data_dir))] = f"testpatient_{i}_{cls}"

    group_map_path = tmp_path / "groups.json"
    group_map_path.write_text(json.dumps(rel_to_group))

    profile = ingest_image_dataset(str(data_dir))
    output_dir = _run_engine(
        profile,
        tmp_path,
        generate_missing_splits_from_train=True,
        strategy=SplitStrategy.GROUPED,
        group_id_map_path=str(group_map_path),
    )

    train_manifest = _manifest(output_dir, "train")
    val_manifest = _manifest(output_dir, "validation")

    def groups_of(manifest):
        return {rel_to_group[_os.path.relpath(r["source_path"], str(data_dir))] for r in manifest["records"]}

    train_groups = groups_of(train_manifest)
    val_groups = groups_of(val_manifest)
    assert train_groups.isdisjoint(val_groups)
    assert val_manifest["total_records"] > 0
