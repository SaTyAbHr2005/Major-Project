"""
Shared fixtures for Module 16 tests. Models are REAL Module 6 artifacts saved
through ModelStore (safetensors + metadata.json + SHA-256), and images are
synthetic - never real patient data. Read-only tests share one module-scoped
artifact/engine to keep the suite fast; anything that mutates a model or its
files builds its own.
"""
import os

import pytest
from PIL import Image

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.provisioning import LocalModelArtifact
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import build_model
from hospital_client.model_management.store import ModelStore

DEFAULT_MAPPING = {"benign": 0, "malignant": 1}


def build_artifact(
    root, model_id="m", architecture="resnet18", input_size=(32, 32), color_mode="RGB",
    class_mapping=DEFAULT_MAPPING, status="trained", num_classes=2, versions=1, extra=None,
):
    """Saves `versions` real Module 6 checkpoints and returns a LocalModelArtifact for the last."""
    config = ModelConfig(architecture=architecture, num_classes=num_classes, input_size=input_size, color_mode=color_mode)
    model = build_model(config)
    store = ModelStore(str(root))
    extra_metadata = dict(extra or {})
    if class_mapping is not None:
        extra_metadata["class_mapping"] = class_mapping
    metadata = None
    for _ in range(versions):
        metadata = store.save_checkpoint(model, config, model_id, status=status, extra_metadata=extra_metadata)
    return LocalModelArtifact(store_root=str(root), model_id=model_id, version=metadata.version)


def make_image(path, size=(50, 40), mode="RGB", color=None, fmt=None):
    color = color if color is not None else {"RGB": (10, 200, 30), "L": 120, "RGBA": (10, 200, 30, 255), "P": 5, "1": 1}.get(mode, 0)
    Image.new(mode, size, color=color).save(str(path), format=fmt)
    return str(path)


@pytest.fixture(scope="module")
def shared_artifact(tmp_path_factory):
    return build_artifact(tmp_path_factory.mktemp("models"), versions=3)


@pytest.fixture(scope="module")
def shared_engine(shared_artifact):
    engine = LocalInferenceEngine(shared_artifact, InferenceConfig(device="cpu"))
    yield engine
    engine.close()


@pytest.fixture
def image_path(tmp_path):
    return make_image(tmp_path / "patient_scan_0001.png")


def store_paths(artifact):
    version_dir = os.path.join(artifact.store_root, artifact.model_id, f"v{artifact.version}")
    return os.path.join(version_dir, "model.safetensors"), os.path.join(version_dir, "metadata.json")
