"""
Artifact validation for Module 6.

Every artifact is treated as untrusted input (it may have been corrupted,
tampered with, or produced by an incompatible/older version of the code).
Validation never partially loads, never drops mismatched keys, never
silently falls back to a different architecture, and never zero-fills a
missing parameter. On any failure the specific reason is recorded and
loading is refused - the only recovery paths are selecting a different
explicit version or retraining through Module 7.

This module deliberately reuses hospital_client.dataset.ingestion utilities
(compute_file_hash, check_file_readability) instead of reimplementing
hashing/file-integrity checks that Module 4 already provides and tests.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from safetensors.torch import load_file

from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash
from hospital_client.dataset.ingestion.validation import check_file_readability
from hospital_client.model_management.config import ModelConfig, validate_model_config
from hospital_client.model_management.metadata import ModelMetadata
from hospital_client.model_management.registry import build_model

ARTIFACT_FILENAME = "model.safetensors"
METADATA_FILENAME = "metadata.json"


@dataclass
class ModelValidationResult:
    valid: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    integrity_checked: bool = False
    architecture_checked: bool = False
    weights_checked: bool = False
    metadata_checked: bool = False
    metadata: Optional[ModelMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "integrity_checked": self.integrity_checked,
            "architecture_checked": self.architecture_checked,
            "weights_checked": self.weights_checked,
            "metadata_checked": self.metadata_checked,
        }


def validate_artifact(version_dir: str) -> ModelValidationResult:
    """
    version_dir is an already path-safety-validated directory (see
    store.ModelStore._version_dir) expected to contain model.safetensors and
    metadata.json.
    """
    result = ModelValidationResult(valid=False)
    artifact_path = os.path.join(version_dir, ARTIFACT_FILENAME)
    metadata_path = os.path.join(version_dir, METADATA_FILENAME)

    if not os.path.exists(metadata_path):
        result.reasons.append(f"Missing metadata file: {metadata_path}")
        return result
    is_readable, reason = check_file_readability(metadata_path)
    if not is_readable:
        result.reasons.append(f"Metadata file unreadable: {reason}")
        return result

    try:
        with open(metadata_path, "r") as f:
            raw_metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        result.reasons.append(f"Malformed metadata JSON: {e}")
        return result

    try:
        metadata = ModelMetadata.from_dict(raw_metadata)
    except (KeyError, TypeError, ValueError) as e:
        result.reasons.append(f"Metadata schema invalid: {e}")
        return result
    result.metadata_checked = True

    if not os.path.exists(artifact_path):
        result.reasons.append(f"Missing artifact file: {artifact_path}")
        result.metadata = metadata
        return result
    is_readable, reason = check_file_readability(artifact_path)
    if not is_readable:
        result.reasons.append(f"Artifact file unreadable or empty: {reason}")
        result.metadata = metadata
        return result

    if not metadata.artifact_sha256:
        result.reasons.append("Metadata has no recorded artifact hash to verify integrity against.")
        result.metadata = metadata
        return result
    actual_hash = compute_file_hash(artifact_path)
    if actual_hash != metadata.artifact_sha256:
        result.reasons.append(
            f"Artifact hash mismatch: expected {metadata.artifact_sha256}, got {actual_hash} - "
            "the artifact may be corrupted or tampered with."
        )
        result.metadata = metadata
        return result
    result.integrity_checked = True

    try:
        config = ModelConfig.from_dict(metadata.config)
        validate_model_config(config)
    except (ValueError, KeyError) as e:
        result.reasons.append(f"Invalid model configuration in metadata: {e}")
        result.metadata = metadata
        return result
    result.architecture_checked = True

    try:
        model = build_model(config)
    except Exception as e:
        result.reasons.append(f"Failed to construct architecture '{config.architecture}': {e}")
        result.metadata = metadata
        return result

    try:
        state_dict = load_file(artifact_path, device="cpu")
    except Exception as e:
        result.reasons.append(f"Failed to read safetensors artifact: {e}")
        result.metadata = metadata
        return result

    expected = model.state_dict()
    expected_keys, actual_keys = set(expected.keys()), set(state_dict.keys())
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        result.reasons.append(f"Artifact is missing expected weight key(s): {missing[:10]}")
    if unexpected:
        result.reasons.append(f"Artifact has unexpected weight key(s): {unexpected[:10]}")
    if missing or unexpected:
        result.metadata = metadata
        return result

    shape_mismatches = [
        f"{key}: expected {tuple(expected[key].shape)}, got {tuple(state_dict[key].shape)}"
        for key in expected
        if tuple(state_dict[key].shape) != tuple(expected[key].shape)
    ]
    if shape_mismatches:
        result.reasons.append(f"Tensor shape mismatch: {shape_mismatches[:10]}")
        result.metadata = metadata
        return result

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        result.reasons.append(f"strict state_dict load failed: {e}")
        result.metadata = metadata
        return result
    result.weights_checked = True

    result.valid = True
    result.metadata = metadata
    return result
