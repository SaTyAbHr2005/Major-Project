"""
The versioned, explicit Module 7 -> Module 9 FederationHandoff contract.

Module 7 does NOT implement Flower, FedAvg/FedProx, federated rounds,
network transport, secure communication, differential privacy, or
Byzantine defense. This module only defines and validates the DATA
CONTRACT a future Module 9 will consume - it invents no transport and
couples to no network/library.

Design:
- Parameters stay as Module 6's existing `get_parameters()` output (an
  ordered List[np.ndarray]) - that interface is reused, not replaced.
- The handoff is versioned via PROTOCOL_VERSION so Module 9 can reject or
  branch on an incompatible contract version explicitly, rather than
  guessing at an undocumented shape.
- `round_id`/`client_id` are Optional[str] and are NEVER fabricated here:
  Module 7 has no federation round or hospital-identity context, so these
  stay None unless a caller (a future integration layer) explicitly
  supplies real values it obtained elsewhere.
- Serialization keeps large parameter arrays OUT of JSON: `save()` writes a
  compact `.npz` (NumPy's own binary array format) alongside a JSON
  metadata sidecar, mirroring the same artifact+metadata split already
  used by Module 6 (`model.safetensors` + `metadata.json`).
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

PROTOCOL_VERSION = 1
SUPPORTED_PROTOCOL_VERSIONS = {1}

PARAMETERS_FILENAME = "handoff_parameters.npz"
METADATA_FILENAME = "handoff_metadata.json"


def _compute_parameters_checksum(parameters: List[np.ndarray]) -> str:
    """SHA-256 over the ordered, raw bytes of every parameter array."""
    hasher = hashlib.sha256()
    for array in parameters:
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


@dataclass
class FederationHandoff:
    protocol_version: int
    model_id: str
    model_version: int
    architecture: str

    # Parameters (Module 6's existing get_parameters() contract - unchanged).
    parameters: List[np.ndarray]
    parameter_count: int
    parameter_shapes: List[List[int]]
    parameter_dtypes: List[str]
    parameters_checksum: str

    # Data provenance.
    num_train_samples: int
    num_classes: int
    class_mapping: Dict[str, int]

    # Metrics / configuration context.
    training_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    training_configuration: Dict[str, Any]
    device: str
    precision: str

    status: str

    # Assigned only by the federation layer (Module 9) / a future identity
    # integration - Module 7 has no such context and never invents these.
    round_id: Optional[str] = None
    client_id: Optional[str] = None

    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Metadata only - never embeds the raw parameter arrays (see save())."""
        return {
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "parameter_shapes": self.parameter_shapes,
            "parameter_dtypes": self.parameter_dtypes,
            "parameters_checksum": self.parameters_checksum,
            "num_train_samples": self.num_train_samples,
            "num_classes": self.num_classes,
            "class_mapping": self.class_mapping,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "training_configuration": self.training_configuration,
            "device": self.device,
            "precision": self.precision,
            "status": self.status,
            "round_id": self.round_id,
            "client_id": self.client_id,
            "generated_at": self.generated_at,
        }

    def save(self, directory: str) -> None:
        """Writes handoff_parameters.npz + handoff_metadata.json into `directory`."""
        os.makedirs(directory, exist_ok=True)
        np.savez(
            os.path.join(directory, PARAMETERS_FILENAME.replace(".npz", "")),
            **{f"arr_{i}": arr for i, arr in enumerate(self.parameters)},
        )
        with open(os.path.join(directory, METADATA_FILENAME), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "FederationHandoff":
        metadata_path = os.path.join(directory, METADATA_FILENAME)
        parameters_path = os.path.join(directory, PARAMETERS_FILENAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Missing handoff metadata file: {metadata_path}")
        if not os.path.exists(parameters_path):
            raise FileNotFoundError(f"Missing handoff parameters file: {parameters_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        with np.load(parameters_path) as npz:
            keys = [f"arr_{i}" for i in range(metadata["parameter_count"])]
            missing = [k for k in keys if k not in npz.files]
            if missing:
                raise ValueError(
                    f"Malformed handoff: parameters file is missing array(s) {missing} "
                    f"expected from parameter_count={metadata['parameter_count']}. "
                    "Module 7 does not reconstruct a handoff with truncated/reordered parameters."
                )
            parameters = [npz[k] for k in keys]

        handoff = cls(
            protocol_version=metadata["protocol_version"],
            model_id=metadata["model_id"],
            model_version=metadata["model_version"],
            architecture=metadata["architecture"],
            parameters=parameters,
            parameter_count=metadata["parameter_count"],
            parameter_shapes=metadata["parameter_shapes"],
            parameter_dtypes=metadata["parameter_dtypes"],
            parameters_checksum=metadata["parameters_checksum"],
            num_train_samples=metadata["num_train_samples"],
            num_classes=metadata["num_classes"],
            class_mapping=metadata["class_mapping"],
            training_metrics=metadata["training_metrics"],
            validation_metrics=metadata["validation_metrics"],
            training_configuration=metadata["training_configuration"],
            device=metadata["device"],
            precision=metadata["precision"],
            status=metadata["status"],
            round_id=metadata.get("round_id"),
            client_id=metadata.get("client_id"),
            generated_at=metadata["generated_at"],
        )
        validate_federation_handoff(handoff)
        return handoff


def build_federation_handoff(
    parameters: List[np.ndarray],
    model_id: str,
    model_version: int,
    architecture: str,
    num_train_samples: int,
    class_mapping: Dict[str, int],
    training_metrics: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    training_configuration: Dict[str, Any],
    device: str,
    precision: str,
    status: str,
    round_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> FederationHandoff:
    handoff = FederationHandoff(
        protocol_version=PROTOCOL_VERSION,
        model_id=model_id,
        model_version=model_version,
        architecture=architecture,
        parameters=parameters,
        parameter_count=len(parameters),
        parameter_shapes=[list(p.shape) for p in parameters],
        parameter_dtypes=[str(p.dtype) for p in parameters],
        parameters_checksum=_compute_parameters_checksum(parameters),
        num_train_samples=num_train_samples,
        num_classes=len(class_mapping),
        class_mapping=class_mapping,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        training_configuration=training_configuration,
        device=device,
        precision=precision,
        status=status,
        round_id=round_id,
        client_id=client_id,
    )
    validate_federation_handoff(handoff)
    return handoff


def validate_federation_handoff(handoff: FederationHandoff) -> None:
    """
    Raises ValueError describing the first problem found. Never silently
    truncates, reorders, zero-fills, or "corrects" a malformed handoff.
    """
    if handoff.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(
            f"Unsupported FederationHandoff protocol_version {handoff.protocol_version!r}; "
            f"this Module 7 build supports {sorted(SUPPORTED_PROTOCOL_VERSIONS)}."
        )
    if not handoff.model_id or not isinstance(handoff.model_id, str):
        raise ValueError(f"model_id must be a non-empty string, got {handoff.model_id!r}")
    if not isinstance(handoff.model_version, int) or isinstance(handoff.model_version, bool) or handoff.model_version < 1:
        raise ValueError(f"model_version must be a positive integer, got {handoff.model_version!r}")
    if not handoff.architecture:
        raise ValueError("architecture must be a non-empty string.")

    n = handoff.parameter_count
    if n != len(handoff.parameters):
        raise ValueError(f"parameter_count ({n}) does not match len(parameters) ({len(handoff.parameters)}).")
    if n != len(handoff.parameter_shapes):
        raise ValueError(f"parameter_count ({n}) does not match len(parameter_shapes) ({len(handoff.parameter_shapes)}).")
    if n != len(handoff.parameter_dtypes):
        raise ValueError(f"parameter_count ({n}) does not match len(parameter_dtypes) ({len(handoff.parameter_dtypes)}).")

    for i, (array, expected_shape, expected_dtype) in enumerate(
        zip(handoff.parameters, handoff.parameter_shapes, handoff.parameter_dtypes)
    ):
        if list(array.shape) != list(expected_shape):
            raise ValueError(f"Parameter {i} shape mismatch: array has {list(array.shape)}, metadata says {expected_shape}.")
        if str(array.dtype) != expected_dtype:
            raise ValueError(f"Parameter {i} dtype mismatch: array has {array.dtype}, metadata says {expected_dtype}.")

    recomputed_checksum = _compute_parameters_checksum(handoff.parameters)
    if recomputed_checksum != handoff.parameters_checksum:
        raise ValueError(
            f"Parameter integrity check failed: recomputed checksum {recomputed_checksum} does not "
            f"match recorded checksum {handoff.parameters_checksum}."
        )

    if not isinstance(handoff.num_train_samples, int) or handoff.num_train_samples < 0:
        raise ValueError(f"num_train_samples must be an integer >= 0, got {handoff.num_train_samples!r}")
    if handoff.num_classes != len(handoff.class_mapping):
        raise ValueError(
            f"num_classes ({handoff.num_classes}) does not match len(class_mapping) ({len(handoff.class_mapping)})."
        )
    if not handoff.device:
        raise ValueError("device must be a non-empty string.")
    if not handoff.precision:
        raise ValueError("precision must be a non-empty string.")
    if not handoff.status:
        raise ValueError("status must be a non-empty string.")
