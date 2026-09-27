"""
ModelStore tests: save/load round-trip, tampering, corruption, architecture/
weight mismatches, and versioning. Every load goes through validate_artifact
internally - there is no blind-load path.
"""
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import build_model
from hospital_client.model_management.store import ModelStore


@pytest.fixture(params=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
def representative_config(request):
    # One architecture per a spread of resource tiers, kept small for speed.
    return ModelConfig(architecture=request.param, num_classes=3, input_size=(64, 64))


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip_state_dict_identical(tmp_path, representative_config):
    model = build_model(representative_config)
    store = ModelStore(str(tmp_path))
    metadata = store.save_checkpoint(model, representative_config, "m1")
    assert metadata.version == 1
    assert metadata.artifact_sha256

    loaded_model, loaded_metadata = store.load_checkpoint("m1", 1)
    original_sd, loaded_sd = model.state_dict(), loaded_model.state_dict()
    assert set(original_sd.keys()) == set(loaded_sd.keys())
    for key in original_sd:
        assert torch.equal(original_sd[key], loaded_sd[key]), f"tensor mismatch for {key}"
    assert loaded_metadata.model_id == "m1"


def test_metadata_round_trips_through_json(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=4, input_size=(64, 64), dropout=0.3)
    model = build_model(config)
    store = ModelStore(str(tmp_path))
    saved = store.save_checkpoint(model, config, "m2", extra_metadata={"notes": "synthetic test model"})
    fetched = store.get_version("m2", 1)
    assert fetched.to_dict() == saved.to_dict()
    assert fetched.notes == "synthetic test model"


def test_artifact_hash_is_generated_and_verified(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    metadata = store.save_checkpoint(build_model(config), config, "m3")
    assert len(metadata.artifact_sha256) == 64  # sha256 hex digest length
    result = store.validate("m3", 1)
    assert result.valid
    assert result.integrity_checked


# ---------------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------------

def test_tampered_artifact_bytes_fail_integrity_check(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m4")

    artifact_path = tmp_path / "m4" / "v1" / "model.safetensors"
    data = bytearray(artifact_path.read_bytes())
    data[-1] ^= 0xFF  # flip the last byte
    artifact_path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="hash mismatch"):
        store.load_checkpoint("m4", 1)
    result = store.validate("m4", 1)
    assert not result.valid
    assert not result.integrity_checked


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

def test_missing_artifact_file_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m5")
    os.remove(tmp_path / "m5" / "v1" / "model.safetensors")

    result = store.validate("m5", 1)
    assert not result.valid
    assert any("Missing artifact file" in r for r in result.reasons)


def test_empty_artifact_file_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m6")
    (tmp_path / "m6" / "v1" / "model.safetensors").write_bytes(b"")

    result = store.validate("m6", 1)
    assert not result.valid


def test_missing_metadata_file_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m7")
    os.remove(tmp_path / "m7" / "v1" / "metadata.json")

    result = store.validate("m7", 1)
    assert not result.valid
    assert any("Missing metadata" in r for r in result.reasons)


def test_malformed_metadata_json_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m8")
    (tmp_path / "m8" / "v1" / "metadata.json").write_text("{not valid json")

    result = store.validate("m8", 1)
    assert not result.valid
    assert any("Malformed metadata" in r for r in result.reasons)


def test_metadata_missing_required_fields_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m9")
    (tmp_path / "m9" / "v1" / "metadata.json").write_text(json.dumps({"only": "junk"}))

    result = store.validate("m9", 1)
    assert not result.valid


def test_missing_artifact_hash_in_metadata_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m10")
    meta_path = tmp_path / "m10" / "v1" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["artifact_sha256"] = ""
    meta_path.write_text(json.dumps(data))

    result = store.validate("m10", 1)
    assert not result.valid
    assert any("no recorded artifact hash" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Architecture / weight mismatch
# ---------------------------------------------------------------------------

def test_unknown_architecture_in_metadata_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m11")
    meta_path = tmp_path / "m11" / "v1" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["config"]["architecture"] = "not_a_real_architecture"
    meta_path.write_text(json.dumps(data))

    result = store.validate("m11", 1)
    assert not result.valid
    assert any("Invalid model configuration" in r or "Unknown architecture" in r for r in result.reasons)


def test_architecture_config_mismatch_with_weights_rejected(tmp_path):
    """
    Simulates the artifact's declared architecture not matching the actual
    saved weights (e.g. copy/paste error or manual tampering): weights for
    resnet18 stored, but metadata now claims resnet50 - strict key/shape
    validation must catch this rather than partially loading.
    """
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m12")
    meta_path = tmp_path / "m12" / "v1" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["config"]["architecture"] = "resnet50"  # different architecture, same weight file
    meta_path.write_text(json.dumps(data))

    result = store.validate("m12", 1)
    assert not result.valid
    # Hash check is against the (unmodified) artifact file, so it still
    # passes; the mismatch must be caught at the key/shape validation stage.
    assert any("weight key" in r or "shape mismatch" in r for r in result.reasons)


def test_missing_and_unexpected_weight_keys_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = build_model(config)
    store = ModelStore(str(tmp_path))
    metadata = store.save_checkpoint(model, config, "m13")

    # Overwrite the artifact with a state_dict missing/renaming keys, then
    # fix up the recorded hash so only the key-mismatch check is exercised.
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    del sd["fc.weight"]
    sd["totally_unexpected_key"] = torch.zeros(3)
    artifact_path = tmp_path / "m13" / "v1" / "model.safetensors"
    save_file(sd, str(artifact_path))
    from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash
    meta_path = tmp_path / "m13" / "v1" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["artifact_sha256"] = compute_file_hash(str(artifact_path))
    meta_path.write_text(json.dumps(data))

    result = store.validate("m13", 1)
    assert not result.valid
    joined = " ".join(result.reasons)
    assert "missing expected weight" in joined
    assert "unexpected weight" in joined


def test_incorrect_tensor_shape_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = build_model(config)
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(model, config, "m14")

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sd["fc.weight"] = torch.zeros(999, sd["fc.weight"].shape[1])  # wrong shape, same key set
    artifact_path = tmp_path / "m14" / "v1" / "model.safetensors"
    save_file(sd, str(artifact_path))
    from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash
    meta_path = tmp_path / "m14" / "v1" / "metadata.json"
    data = json.loads(meta_path.read_text())
    data["artifact_sha256"] = compute_file_hash(str(artifact_path))
    meta_path.write_text(json.dumps(data))

    result = store.validate("m14", 1)
    assert not result.valid
    assert any("shape mismatch" in r for r in result.reasons)


def test_load_never_partially_succeeds_on_invalid_artifact(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m15")
    os.remove(tmp_path / "m15" / "v1" / "model.safetensors")

    with pytest.raises(ValueError):
        store.load_checkpoint("m15", 1)


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

def test_versioning_v1_v2_monotonic_latest_and_parent(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    v1 = store.save_checkpoint(build_model(config), config, "m16")
    v2 = store.save_checkpoint(build_model(config), config, "m16", parent_version=v1.version)

    assert v1.version == 1
    assert v2.version == 2
    assert v2.parent_version == 1
    versions = [m.version for m in store.list_versions("m16")]
    assert versions == [1, 2]
    assert store.get_latest("m16").version == 2


def test_duplicate_version_is_rejected(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m17")
    with pytest.raises(FileExistsError):
        store.save_checkpoint(build_model(config), config, "m17", version=1)


def test_explicit_deletion_removes_a_version(tmp_path):
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m18")
    store.save_checkpoint(build_model(config), config, "m18")
    assert [m.version for m in store.list_versions("m18")] == [1, 2]

    store.delete_version("m18", 1)
    assert [m.version for m in store.list_versions("m18")] == [2]


def test_deleting_nonexistent_version_raises(tmp_path):
    store = ModelStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.delete_version("does_not_exist", 1)


def test_get_latest_with_no_versions_raises(tmp_path):
    store = ModelStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.get_latest("nothing_here")


def test_deletion_is_never_automatic(tmp_path):
    """Saving/loading/listing must never remove a version as a side effect."""
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    store = ModelStore(str(tmp_path))
    store.save_checkpoint(build_model(config), config, "m19")
    store.load_checkpoint("m19", 1)
    store.list_versions("m19")
    store.get_latest("m19")
    assert [m.version for m in store.list_versions("m19")] == [1]
