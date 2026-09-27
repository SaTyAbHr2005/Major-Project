"""
Integration tests for the Module 4 -> Module 5 contract.

These tests deliberately do NOT hand-build fake DatasetProfile dictionaries.
They run the real Module 4 ingestion functions to produce an actual
DatasetProfile, serialize it exactly as the Module 4 CLI does, and feed that
literal output into the Module 5 PreprocessingEngine. This is the path the
audit found was broken:

1. dataset_type: Module 4 emits "Image"/"CSV"/"Excel"; Module 5 only
   recognized lowercase "image"/"tabular".
2. splits: Module 4 stores split names under splits["detected_splits"];
   Module 5 was treating the dict's own keys as split names.
3. tabular target: Module 4 only ever reports candidate_target_columns;
   Module 5 was silently defaulting to np.zeros(...) as the label array.
"""
import json
import os

import pandas as pd
import pytest
from PIL import Image

from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.dataset.ingestion.tabular_ingestor import ingest_tabular_dataset
from hospital_client.preprocessing.config import PreprocessingConfig, OutputMode
from hospital_client.preprocessing.engine import (
    PreprocessingEngine,
    _normalize_dataset_type,
    _extract_split_names,
)


_image_counter = 0


def _make_image(path, color=None):
    # Distinct content per call by default so these synthetic fixtures never
    # trigger cross-split duplicate-leakage detection (split_validation.py)
    # by accident; pass an explicit color to deliberately create duplicates.
    global _image_counter
    if color is None:
        n = _image_counter
        _image_counter += 1
        color = (n % 256, (n * 7) % 256, (n * 13) % 256)
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def _write_profile(profile, path):
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f)


# ---------------------------------------------------------------------------
# Bug 1: dataset_type contract (unit-level regression on the real M4 values)
# ---------------------------------------------------------------------------

def test_normalize_dataset_type_matches_real_module4_values():
    assert _normalize_dataset_type("Image") == "image"
    assert _normalize_dataset_type("CSV") == "tabular"
    assert _normalize_dataset_type("Excel") == "tabular"
    # legacy/hand-built profiles already used elsewhere in this suite
    assert _normalize_dataset_type("image") == "image"
    assert _normalize_dataset_type("tabular") == "tabular"
    assert _normalize_dataset_type("magic") == "magic"


# ---------------------------------------------------------------------------
# Bug 2: splits contract (unit-level regression on the real M4 shape)
# ---------------------------------------------------------------------------

def test_extract_split_names_uses_detected_splits_value_not_dict_keys():
    # Real Module 4 shape: splits["detected_splits"] holds the split names.
    # The dict's own key ("detected_splits") must NOT be treated as a split.
    real_module4_shape = {"splits": {"detected_splits": ["train", "test"]}}
    assert _extract_split_names(real_module4_shape) == ["train", "test"]

    # Legacy/hand-built shape where the dict's keys ARE the split names
    # (used by this suite's own engine tests) must still work.
    legacy_shape = {"splits": {"train": ["a.jpg"], "test": ["b.jpg"]}}
    assert sorted(_extract_split_names(legacy_shape)) == ["test", "train"]

    assert _extract_split_names({}) == []


# ---------------------------------------------------------------------------
# End-to-end: real Module 4 image profile -> Module 5 (bugs 1 + 2 together)
# ---------------------------------------------------------------------------

def test_real_module4_image_profile_preprocesses_with_correct_labels(tmp_path):
    data_dir = tmp_path / "data"
    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(3):
                _make_image(cls_dir / f"img_{i}.jpg")

    # Step 1: real Module 4 ingestion produces the actual DatasetProfile.
    profile = ingest_image_dataset(str(data_dir))
    assert profile.dataset_type == "Image"
    assert sorted(profile.splits["detected_splits"]) == ["test", "train"]
    assert sorted(profile.classes) == ["cat", "dog"]

    profile_path = tmp_path / "dataset_profile.json"
    _write_profile(profile, profile_path)

    # Step 2: feed the EXACT Module 4 output into Module 5, unmodified.
    output_dir = tmp_path / "out"
    config = PreprocessingConfig()
    config.dataset.output_mode = OutputMode.LAZY
    config.dataset.output_path = str(output_dir)

    engine = PreprocessingEngine(config)
    engine.run(str(profile_path))

    all_labels = set()
    for split_name in ("train", "validation", "test"):
        manifest_path = output_dir / f"{split_name}_manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        for record in manifest["records"]:
            all_labels.add(record["label"])

    # Regression guard for the splits-contract bug: labels must be the real
    # class names, never the split folder names ("train"/"test").
    assert all_labels
    assert all_labels <= {"cat", "dog"}
    assert "train" not in all_labels
    assert "test" not in all_labels

    # Source images must remain untouched (non-destructive processing).
    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            for i in range(3):
                assert (data_dir / split / cls / f"img_{i}.jpg").exists()


# ---------------------------------------------------------------------------
# End-to-end: real Module 4 tabular profile -> Module 5 (bug 3)
# ---------------------------------------------------------------------------

def _make_tabular_dataset(csv_path):
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "outcome": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })
    df.to_csv(csv_path, index=False)
    return df


def test_real_module4_tabular_profile_without_target_raises_clear_error(tmp_path):
    csv_path = tmp_path / "patients.csv"
    _make_tabular_dataset(csv_path)

    # Step 1: real Module 4 ingestion - it only ever reports candidates.
    profile = ingest_tabular_dataset(str(csv_path))
    assert profile.dataset_type == "CSV"
    candidate_cols = [c["column"] for c in profile.tabular_statistics["candidate_target_columns"]]
    assert "outcome" in candidate_cols
    assert "target_column" not in profile.tabular_statistics

    profile_path = tmp_path / "dataset_profile.json"
    _write_profile(profile, profile_path)

    config = PreprocessingConfig()
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)

    # Regression guard for the target-column bug: Module 5 must fail safely
    # and mention the real candidates instead of silently using np.zeros(...).
    with pytest.raises(ValueError, match="Tabular target column is ambiguous"):
        engine.run(str(profile_path))


def test_real_module4_tabular_profile_with_explicit_target_preprocesses(tmp_path):
    csv_path = tmp_path / "patients.csv"
    original_df = _make_tabular_dataset(csv_path)

    profile = ingest_tabular_dataset(str(csv_path))
    profile_path = tmp_path / "dataset_profile.json"
    _write_profile(profile, profile_path)

    output_dir = tmp_path / "out"
    config = PreprocessingConfig()
    config.dataset.output_mode = OutputMode.MATERIALIZED
    config.dataset.output_path = str(output_dir)
    config.tabular.target_column = "outcome"  # explicit configuration, as required

    engine = PreprocessingEngine(config)
    engine.run(str(profile_path))

    assert (output_dir / "train.csv").exists()
    with open(output_dir / "train_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["metadata"]["target_resolution"]["source"] == "config_override"
    assert manifest["metadata"]["target_resolution"]["target_column"] == "outcome"

    # Source CSV must remain untouched.
    reread_df = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(reread_df, original_df)
