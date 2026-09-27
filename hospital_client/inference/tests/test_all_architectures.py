"""
One generic engine serves every Module 6 architecture - no per-architecture
inference code. Real Module 6 artifacts, real forward passes, all 8 models.
"""
import math

import pytest

from hospital_client.inference.engine import LocalInferenceEngine
from hospital_client.inference.tests.conftest import build_artifact, make_image
from hospital_client.model_management.registry import list_architectures

INPUT_SIZE = (64, 64)  # valid for all 8 (vit_b16 needs a multiple of 16; mobilevit_xxs needs spatial room)


@pytest.mark.parametrize("architecture", sorted(list_architectures()))
def test_generic_engine_runs_every_module6_architecture(architecture, tmp_path):
    artifact = build_artifact(
        tmp_path / "models", model_id=architecture, architecture=architecture, input_size=INPUT_SIZE,
        class_mapping={"a": 0, "b": 1, "c": 2}, num_classes=3,
    )
    engine = LocalInferenceEngine(artifact)
    result = engine.predict(make_image(tmp_path / "x.png", size=(90, 70)))
    engine.close()

    assert result.succeeded and result.architecture == architecture and result.model_id == architecture
    assert set(result.class_probabilities) == {"a", "b", "c"}
    assert math.isclose(sum(result.class_probabilities.values()), 1.0, abs_tol=1e-5)
    assert result.preprocessing["tensor_shape"] == [1, 3, 64, 64]
