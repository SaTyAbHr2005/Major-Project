"""A. Model loading / provisioning: valid model loads; invalid, tampered and mismatched models are rejected."""
import json
import os
from types import SimpleNamespace

import pytest

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import ModelCompatibilityError, ModelProvisioningError
from hospital_client.inference.provisioning import (
    ApprovedModelProvider, ApprovedModelSource, LocalModelArtifact, LocalStoreModelProvider,
)
from hospital_client.inference.tests.conftest import build_artifact, store_paths


def test_valid_model_loads_and_reports_identity(shared_artifact):
    engine = LocalInferenceEngine(LocalStoreModelProvider(shared_artifact.store_root, "m", 3))
    info = engine.model_info()
    assert (info["model_id"], info["model_version"], info["architecture"]) == ("m", 3, "resnet18")
    assert info["artifact_format"] == "safetensors" and info["integrity_verified"] is True
    assert len(info["artifact_sha256"]) == 64
    assert info["input_size"] == [32, 32] and info["color_mode"] == "RGB" and info["num_classes"] == 2
    assert not engine.model.training
    assert all(not p.requires_grad for p in engine.model.parameters())
    engine.close()


def test_locally_provisioned_model_is_never_reported_as_platform_approved(shared_artifact):
    engine = LocalInferenceEngine(shared_artifact)
    approval = engine.model_info()["approval"]
    assert approval["source"] == "local_standalone" and approval["platform_approved"] is False
    engine.close()


def test_provider_interface_alias_and_custom_provider(shared_artifact):
    assert ApprovedModelSource is ApprovedModelProvider

    class FutureProvider(ApprovedModelProvider):  # stands in for a Module 15/14/11-backed provider
        def provision(self):
            return LocalModelArtifact(shared_artifact.store_root, "m", 3, source="future_module_15", platform_approved=True,
                                      provenance={"approved_by": "test"})

    engine = LocalInferenceEngine(FutureProvider())
    approval = engine.model_info()["approval"]
    assert approval["source"] == "future_module_15" and approval["provenance"] == {"approved_by": "test"}
    engine.close()


def test_source_must_be_provider_or_artifact():
    with pytest.raises(Exception, match="source must be"):
        LocalInferenceEngine("not-a-source")


def test_missing_store_directory_is_rejected_and_never_created(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ModelProvisioningError, match="does not exist"):
        LocalInferenceEngine(LocalStoreModelProvider(str(missing), "m", 1))
    assert not missing.exists()


@pytest.mark.parametrize("model_id, version", [("nope", 1), ("m", 99)])
def test_unknown_model_or_version_rejected(shared_artifact, model_id, version):
    with pytest.raises(ModelProvisioningError):
        LocalInferenceEngine(LocalStoreModelProvider(shared_artifact.store_root, model_id, version))


@pytest.mark.parametrize("version", [None, 0, -1, True, "1", 1.0])
def test_version_must_be_explicit_positive_int(shared_artifact, version):
    with pytest.raises(ModelProvisioningError, match="explicit positive integer"):
        LocalInferenceEngine(LocalStoreModelProvider(shared_artifact.store_root, "m", version))


@pytest.mark.parametrize("model_id", ["../escape", "a/b", "", "..", "m\\..\\x"])
def test_malicious_model_ids_rejected(shared_artifact, model_id):
    with pytest.raises(ModelProvisioningError):
        LocalInferenceEngine(LocalStoreModelProvider(shared_artifact.store_root, model_id, 1))


def test_integrity_mismatch_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    weights, _ = store_paths(artifact)
    with open(weights, "r+b") as f:
        f.seek(-8, os.SEEK_END)
        byte = f.read(1)
        f.seek(-8, os.SEEK_END)
        f.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ModelProvisioningError, match="hash mismatch"):
        LocalInferenceEngine(artifact)


def test_missing_weights_file_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    os.remove(store_paths(artifact)[0])
    with pytest.raises(ModelProvisioningError, match="Missing artifact"):
        LocalInferenceEngine(artifact)


def test_missing_and_malformed_metadata_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    _, metadata_path = store_paths(artifact)
    with open(metadata_path, "w") as f:
        f.write("{ not json")
    with pytest.raises(ModelProvisioningError, match="Malformed metadata"):
        LocalInferenceEngine(artifact)
    os.remove(metadata_path)
    with pytest.raises(ModelProvisioningError, match="Missing metadata"):
        LocalInferenceEngine(artifact)


def _edit_metadata(artifact, edit):
    _, metadata_path = store_paths(artifact)
    with open(metadata_path) as f:
        data = json.load(f)
    edit(data)
    with open(metadata_path, "w") as f:
        json.dump(data, f)


def test_architecture_mismatch_between_config_and_weights_rejected(tmp_path):
    # metadata claims efficientnet_b0 everywhere, but the weights are resnet18's
    artifact = build_artifact(tmp_path / "models")
    def edit(d):
        d["architecture"] = "efficientnet_b0"
        d["config"]["architecture"] = "efficientnet_b0"
    _edit_metadata(artifact, edit)
    with pytest.raises(ModelProvisioningError, match="weight key"):
        LocalInferenceEngine(artifact)


def test_metadata_architecture_disagreeing_with_its_config_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    _edit_metadata(artifact, lambda d: d.__setitem__("architecture", "resnet50"))
    with pytest.raises(ModelCompatibilityError, match="does not match its stored model configuration"):
        LocalInferenceEngine(artifact)


def test_unsupported_architecture_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    def edit(d):
        d["architecture"] = "made_up_net"
        d["config"]["architecture"] = "made_up_net"
    _edit_metadata(artifact, edit)
    with pytest.raises(ModelProvisioningError):
        LocalInferenceEngine(artifact)


def test_checkpoint_config_mismatch_num_classes_rejected(tmp_path):
    # config says 5 classes but the checkpoint's head has 2 outputs -> Module 6 shape validation
    artifact = build_artifact(tmp_path / "models")
    def edit(d):
        d["config"]["num_classes"] = 5
        d["num_classes"] = 5
    _edit_metadata(artifact, edit)
    with pytest.raises(ModelProvisioningError, match="shape mismatch"):
        LocalInferenceEngine(artifact)


def test_metadata_num_classes_disagreeing_with_config_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    _edit_metadata(artifact, lambda d: d.__setitem__("num_classes", 3))
    with pytest.raises(ModelCompatibilityError, match="num_classes"):
        LocalInferenceEngine(artifact)


def test_invalid_model_configuration_in_metadata_rejected(tmp_path):
    artifact = build_artifact(tmp_path / "models")
    _edit_metadata(artifact, lambda d: d["config"].__setitem__("input_size", [0, 32]))
    with pytest.raises(ModelProvisioningError, match="Invalid model configuration"):
        LocalInferenceEngine(artifact)


@pytest.mark.parametrize("status", ["draft", "deprecated"])
def test_unusable_local_status_rejected(tmp_path, status):
    artifact = build_artifact(tmp_path / "models", status=status)
    with pytest.raises(ModelCompatibilityError, match="not usable for inference"):
        LocalInferenceEngine(artifact)


def test_allowed_statuses_are_configurable(tmp_path):
    artifact = build_artifact(tmp_path / "models", status="draft")
    engine = LocalInferenceEngine(artifact, InferenceConfig(allowed_model_statuses=("draft",)))
    engine.close()


def test_from_training_result_requires_a_checkpoint(tmp_path):
    with pytest.raises(ModelProvisioningError, match="no saved checkpoint"):
        LocalStoreModelProvider.from_training_result(SimpleNamespace(checkpoint_model_id=None, checkpoint_version=None), str(tmp_path))
    provider = LocalStoreModelProvider.from_training_result(SimpleNamespace(checkpoint_model_id="m", checkpoint_version=2), str(tmp_path))
    assert provider.store_root == os.path.join(str(tmp_path), "models") and provider.version == 2


def test_invalid_engine_configuration_rejected(shared_artifact):
    from hospital_client.inference.errors import InferenceConfigError
    with pytest.raises(InferenceConfigError):
        LocalInferenceEngine(shared_artifact, InferenceConfig(device="tpu"))
    with pytest.raises(InferenceConfigError):
        LocalInferenceEngine(shared_artifact, InferenceConfig(max_image_bytes=0))
