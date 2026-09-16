"""
Shared fixtures for Module 7 tests. Builds a REAL Module 4 -> Module 5
pipeline output (not hand-built fake manifests) so tests exercise the
actual contract Module 7 must consume. Uses only synthetic, generated
images - never real patient/medical data.
"""
import json
import os

import pytest
from PIL import Image

from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.preprocessing.config import PreprocessingConfig, OutputMode
from hospital_client.preprocessing.engine import PreprocessingEngine

_counter = 0


def _make_image(path, base_color, index, size=48):
    global _counter
    _counter += 1
    color = tuple((c + index * 7 + _counter) % 256 for c in base_color)
    Image.new("RGB", (size, size), color=color).save(path)


def make_preprocessed_dataset(tmp_path, class_counts, output_mode=OutputMode.LAZY, image_size=48):
    """
    class_counts: e.g. {"cat": 12, "dog": 12}. Returns the Module 5 output
    directory (containing train/validation/test manifests).
    """
    data_dir = tmp_path / "data"
    palette = [(200, 50, 50), (50, 50, 200), (50, 200, 50), (200, 200, 50)]
    for i, (cls, count) in enumerate(class_counts.items()):
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        base_color = palette[i % len(palette)]
        for j in range(count):
            _make_image(str(cls_dir / f"{j}.jpg"), base_color, j, size=image_size)

    profile = ingest_image_dataset(str(data_dir))
    profile_path = tmp_path / "profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile.to_dict(), f)

    output_dir = tmp_path / "preprocessed"
    config = PreprocessingConfig()
    config.dataset.output_mode = output_mode
    config.dataset.output_path = str(output_dir)
    PreprocessingEngine(config).run(str(profile_path))
    return str(output_dir)


@pytest.fixture
def small_dataset_dir(tmp_path):
    return make_preprocessed_dataset(tmp_path, {"cat": 12, "dog": 12})


@pytest.fixture
def small_dataset_dir_materialized(tmp_path):
    return make_preprocessed_dataset(tmp_path, {"cat": 12, "dog": 12}, output_mode=OutputMode.MATERIALIZED)


@pytest.fixture
def imbalanced_dataset_dir(tmp_path):
    return make_preprocessed_dataset(tmp_path, {"normal": 20, "disease": 4})
