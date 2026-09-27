"""
Tests for the versioned Module 7 -> Module 9 FederationHandoff contract
(federation.py). Covers construction, validation, serialization round-trip,
and rejection of malformed/tampered handoffs.
"""
import numpy as np
import pytest

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.federation import (
    PROTOCOL_VERSION,
    FederationHandoff,
    build_federation_handoff,
    validate_federation_handoff,
)
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer
from hospital_client.training.trainer import build_federation_handoff as trainer_build_federation_handoff


def _params():
    return [np.random.rand(4, 3).astype(np.float32), np.random.rand(2).astype(np.float32)]


def _valid_handoff(**overrides):
    defaults = dict(
        parameters=_params(),
        model_id="chest_xray_resnet18",
        model_version=1,
        architecture="resnet18",
        num_train_samples=100,
        class_mapping={"normal": 0, "disease": 1},
        training_metrics={"loss": 0.1, "accuracy": 0.95},
        validation_metrics={"loss": 0.2, "accuracy": 0.9},
        training_configuration={"epochs": 5, "device": "cpu"},
        device="cpu",
        precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    defaults.update(overrides)
    return build_federation_handoff(**defaults)


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

def test_valid_handoff_constructs_successfully():
    handoff = _valid_handoff()
    assert handoff.protocol_version == PROTOCOL_VERSION
    assert handoff.parameter_count == 2
    assert handoff.model_id == "chest_xray_resnet18"


def test_round_id_and_client_id_default_to_none_never_fabricated():
    handoff = _valid_handoff()
    assert handoff.round_id is None
    assert handoff.client_id is None


def test_round_id_and_client_id_used_when_explicitly_supplied():
    handoff = _valid_handoff(round_id="round_007", client_id="hospital_A")
    assert handoff.round_id == "round_007"
    assert handoff.client_id == "hospital_A"


def test_parameter_shapes_and_dtypes_recorded_correctly():
    params = [np.zeros((5, 5), dtype=np.float32), np.zeros((3,), dtype=np.float64)]
    handoff = _valid_handoff(parameters=params)
    assert handoff.parameter_shapes == [[5, 5], [3]]
    assert handoff.parameter_dtypes == ["float32", "float64"]


def test_parameter_ordering_is_preserved():
    params = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32), np.array([3.0], dtype=np.float32)]
    handoff = _valid_handoff(parameters=params)
    assert [p.item() for p in handoff.parameters] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Serialization / deserialization round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip_preserves_parameters(tmp_path):
    handoff = _valid_handoff()
    directory = str(tmp_path / "handoff")
    handoff.save(directory)

    loaded = FederationHandoff.load(directory)
    assert loaded.model_id == handoff.model_id
    assert loaded.model_version == handoff.model_version
    assert loaded.parameter_count == handoff.parameter_count
    for original, restored in zip(handoff.parameters, loaded.parameters):
        assert np.array_equal(original, restored)


def test_round_trip_preserves_all_metadata_fields(tmp_path):
    handoff = _valid_handoff(round_id="r1", client_id="hospitalX")
    directory = str(tmp_path / "handoff")
    handoff.save(directory)
    loaded = FederationHandoff.load(directory)
    assert loaded.to_dict() == handoff.to_dict()


def test_round_trip_preserves_parameter_order(tmp_path):
    params = [np.array([i], dtype=np.float32) for i in range(10)]
    handoff = _valid_handoff(parameters=params)
    directory = str(tmp_path / "handoff")
    handoff.save(directory)
    loaded = FederationHandoff.load(directory)
    assert [p.item() for p in loaded.parameters] == list(range(10))


def test_save_does_not_embed_raw_arrays_in_json(tmp_path):
    handoff = _valid_handoff(parameters=[np.random.rand(1000, 1000).astype(np.float32)])
    directory = str(tmp_path / "handoff")
    handoff.save(directory)
    import os
    metadata_size = os.path.getsize(os.path.join(directory, "handoff_metadata.json"))
    # A 1000x1000 float32 array is ~4MB; the JSON metadata must stay tiny.
    assert metadata_size < 10_000


# ---------------------------------------------------------------------------
# Validation: shapes, counts, model identity
# ---------------------------------------------------------------------------

def test_parameter_count_mismatch_rejected():
    handoff = _valid_handoff()
    handoff.parameter_count = 99
    with pytest.raises(ValueError, match="parameter_count"):
        validate_federation_handoff(handoff)


def test_parameter_shape_mismatch_rejected():
    handoff = _valid_handoff()
    handoff.parameter_shapes[0] = [999, 999]
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_federation_handoff(handoff)


def test_parameter_dtype_mismatch_rejected():
    handoff = _valid_handoff()
    handoff.parameter_dtypes[0] = "float64"
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_federation_handoff(handoff)


def test_empty_model_id_rejected():
    with pytest.raises(ValueError, match="model_id"):
        _valid_handoff(model_id="")


def test_invalid_model_version_rejected():
    with pytest.raises(ValueError, match="model_version"):
        _valid_handoff(model_version=0)


def test_num_classes_class_mapping_mismatch_rejected():
    handoff = _valid_handoff()
    handoff.num_classes = 5  # class_mapping only has 2 entries
    with pytest.raises(ValueError, match="num_classes"):
        validate_federation_handoff(handoff)


def test_negative_sample_count_rejected():
    with pytest.raises(ValueError, match="num_train_samples"):
        _valid_handoff(num_train_samples=-1)


# ---------------------------------------------------------------------------
# Protocol version handling
# ---------------------------------------------------------------------------

def test_unsupported_protocol_version_rejected():
    handoff = _valid_handoff()
    handoff.protocol_version = 999
    with pytest.raises(ValueError, match="Unsupported FederationHandoff protocol_version"):
        validate_federation_handoff(handoff)


def test_current_protocol_version_is_one():
    assert PROTOCOL_VERSION == 1


# ---------------------------------------------------------------------------
# Integrity checksum / malformed handoff rejection
# ---------------------------------------------------------------------------

def test_tampered_parameters_fail_checksum_validation():
    handoff = _valid_handoff()
    handoff.parameters[0] = handoff.parameters[0] + 1.0  # mutate after construction
    with pytest.raises(ValueError, match="integrity check failed"):
        validate_federation_handoff(handoff)


def test_load_missing_metadata_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata"):
        FederationHandoff.load(str(tmp_path / "nonexistent"))


def test_load_missing_parameters_file_raises(tmp_path):
    import json, os
    directory = str(tmp_path / "partial_handoff")
    os.makedirs(directory)
    handoff = _valid_handoff()
    with open(os.path.join(directory, "handoff_metadata.json"), "w") as f:
        json.dump(handoff.to_dict(), f)
    with pytest.raises(FileNotFoundError, match="parameters"):
        FederationHandoff.load(directory)


def test_load_rejects_truncated_parameter_array_count(tmp_path):
    import numpy as _np
    import os
    directory = str(tmp_path / "truncated_handoff")
    handoff = _valid_handoff()
    handoff.save(directory)
    # Simulate truncation: overwrite the .npz with fewer arrays than parameter_count claims.
    os.remove(os.path.join(directory, "handoff_parameters.npz"))
    _np.savez(os.path.join(directory, "handoff_parameters"), arr_0=handoff.parameters[0])  # only 1 of 2 arrays
    with pytest.raises(ValueError, match="missing array"):
        FederationHandoff.load(directory)


# ---------------------------------------------------------------------------
# No mutation of original model parameters
# ---------------------------------------------------------------------------

def test_building_handoff_does_not_mutate_input_parameters():
    original = _params()
    snapshot = [p.copy() for p in original]
    _valid_handoff(parameters=original)
    for before, after in zip(snapshot, original):
        assert np.array_equal(before, after)


def test_to_dict_never_contains_raw_parameter_arrays():
    handoff = _valid_handoff()
    d = handoff.to_dict()
    assert "parameters" not in d


# ---------------------------------------------------------------------------
# End-to-end integration through Trainer
# ---------------------------------------------------------------------------

def _train_config(dataset_dir, output_dir):
    from hospital_client.training.config import TrainingConfig
    return TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)),
        dataset_dir=dataset_dir, output_dir=output_dir, epochs=1, batch_size=4,
    )


def test_trainer_produces_a_valid_versioned_handoff(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    result = Trainer(_train_config(small_dataset_dir, output_dir)).run()
    handoff = trainer_build_federation_handoff(result, output_dir)

    validate_federation_handoff(handoff)  # must not raise
    assert handoff.protocol_version == PROTOCOL_VERSION
    assert handoff.model_id == result.checkpoint_model_id
    assert handoff.model_version == result.checkpoint_version
    assert handoff.round_id is None
    assert handoff.client_id is None
    assert handoff.status == "TRAINING_COMPLETED_AWAITING_FEDERATION"
    assert handoff.device == "cpu"
    assert handoff.precision == "fp32"


def test_trainer_handoff_round_trips_through_save_load(small_dataset_dir, tmp_path):
    output_dir = str(tmp_path / "out")
    result = Trainer(_train_config(small_dataset_dir, output_dir)).run()
    handoff = trainer_build_federation_handoff(result, output_dir)

    handoff_dir = str(tmp_path / "handoff_dir")
    handoff.save(handoff_dir)
    loaded = FederationHandoff.load(handoff_dir)
    assert loaded.parameter_count == handoff.parameter_count
    for original, restored in zip(handoff.parameters, loaded.parameters):
        assert np.array_equal(original, restored)
