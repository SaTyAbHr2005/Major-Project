"""
Local, file-based model artifact store for Module 6.

Layout:
    <artifact_store_root>/<model_id>/v<N>/model.safetensors
    <artifact_store_root>/<model_id>/v<N>/metadata.json

Versions are immutable and integer-monotonic per model_id. Every load goes
through validate_artifact() (integrity + architecture + strict weight
checks) - there is no "blind load" path. Deletion is only ever explicit;
nothing here runs automatic garbage collection, TTL expiry, or silent
overwrites.
"""
import json
import os
import re
import shutil
from typing import List, Optional, Tuple

import torch
import torchvision
import safetensors
from safetensors.torch import save_file

from hospital_client.model_management.architectures import count_parameters
from hospital_client.model_management.config import ModelConfig, validate_model_config
from hospital_client.model_management.device import resolve_device
from hospital_client.model_management.metadata import ModelMetadata
from hospital_client.model_management.registry import build_model
from hospital_client.model_management.validation import ARTIFACT_FILENAME, METADATA_FILENAME, validate_artifact

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,200}$")
_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def _validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
        raise ValueError(
            f"Invalid model_id {model_id!r}: must be 1-200 characters matching "
            f"{_MODEL_ID_RE.pattern} (letters, digits, '_' and '-' only)."
        )


def _validate_version(version) -> None:
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"Invalid version {version!r}: must be a positive integer.")


class ModelStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    # -- path safety ---------------------------------------------------

    def _model_dir(self, model_id: str) -> str:
        _validate_model_id(model_id)
        path = os.path.join(self.root, model_id)
        self._assert_within_root(path)
        return path

    def _version_dir(self, model_id: str, version: int) -> str:
        _validate_version(version)
        model_dir = self._model_dir(model_id)
        path = os.path.join(model_dir, f"v{version}")
        self._assert_within_root(path)
        return path

    def _assert_within_root(self, path: str) -> None:
        root_abs = os.path.abspath(self.root)
        path_abs = os.path.abspath(path)
        if os.path.commonpath([root_abs, path_abs]) != root_abs:
            raise ValueError(f"Resolved artifact path escapes the artifact store root: {path!r}")

    # -- versioning ------------------------------------------------------

    def _existing_versions(self, model_id: str) -> List[int]:
        model_dir = self._model_dir(model_id)
        if not os.path.isdir(model_dir):
            return []
        versions = []
        for entry in os.listdir(model_dir):
            m = _VERSION_DIR_RE.match(entry)
            if m and os.path.isdir(os.path.join(model_dir, entry)):
                versions.append(int(m.group(1)))
        return sorted(versions)

    def _next_version(self, model_id: str) -> int:
        existing = self._existing_versions(model_id)
        return existing[-1] + 1 if existing else 1

    # -- save/load ---------------------------------------------------------

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        model_id: str,
        status: str = "draft",
        version: Optional[int] = None,
        parent_version: Optional[int] = None,
        extra_metadata: Optional[dict] = None,
    ) -> ModelMetadata:
        validate_model_config(config)
        _validate_model_id(model_id)

        resolved_version = version if version is not None else self._next_version(model_id)
        _validate_version(resolved_version)
        version_dir = self._version_dir(model_id, resolved_version)
        if os.path.exists(version_dir):
            raise FileExistsError(
                f"Version {resolved_version} of model '{model_id}' already exists. "
                "Model versions are immutable - saving never overwrites an existing version."
            )
        os.makedirs(version_dir)

        artifact_path = os.path.join(version_dir, ARTIFACT_FILENAME)
        state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, artifact_path)

        # Reuse Module 4's SHA-256 utility rather than reimplementing hashing.
        from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash
        artifact_hash = compute_file_hash(artifact_path)
        params = count_parameters(model)

        metadata = ModelMetadata(
            model_id=model_id,
            version=resolved_version,
            architecture=config.architecture,
            config=config.to_dict(),
            status=status,
            artifact_sha256=artifact_hash,
            artifact_size_bytes=os.path.getsize(artifact_path),
            framework_versions={
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "safetensors": safetensors.__version__,
            },
            input_spec={"input_size": list(config.input_size), "color_mode": config.color_mode},
            num_classes=config.num_classes,
            parent_version=parent_version,
            total_parameters=params["total"],
            trainable_parameters=params["trainable"],
        )

        if extra_metadata:
            known_fields = set(ModelMetadata.__dataclass_fields__.keys())
            for key, value in extra_metadata.items():
                if key in known_fields:
                    setattr(metadata, key, value)
                else:
                    metadata.warnings.append(f"Ignored unknown metadata field '{key}'.")

        with open(os.path.join(version_dir, METADATA_FILENAME), "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return metadata

    def load_checkpoint(self, model_id: str, version: int, device: str = "cpu") -> Tuple[torch.nn.Module, ModelMetadata]:
        version_dir = self._version_dir(model_id, version)
        result = validate_artifact(version_dir)
        if not result.valid:
            raise ValueError(
                f"Artifact validation failed for model '{model_id}' version {version}: {result.reasons}"
            )

        config = ModelConfig.from_dict(result.metadata.config)
        model = build_model(config)
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(version_dir, ARTIFACT_FILENAME), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(resolve_device(device))
        return model, result.metadata

    def validate(self, model_id: str, version: int):
        return validate_artifact(self._version_dir(model_id, version))

    # -- listing / lifecycle ------------------------------------------------

    def list_versions(self, model_id: str) -> List[ModelMetadata]:
        results = []
        for version in self._existing_versions(model_id):
            metadata_path = os.path.join(self._version_dir(model_id, version), METADATA_FILENAME)
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    results.append(ModelMetadata.from_dict(json.load(f)))
        return results

    def get_version(self, model_id: str, version: int) -> ModelMetadata:
        metadata_path = os.path.join(self._version_dir(model_id, version), METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found for model '{model_id}' version {version}.")
        with open(metadata_path, "r") as f:
            return ModelMetadata.from_dict(json.load(f))

    def get_latest(self, model_id: str) -> ModelMetadata:
        versions = self._existing_versions(model_id)
        if not versions:
            raise FileNotFoundError(f"No versions found for model_id '{model_id}'.")
        return self.get_version(model_id, versions[-1])

    def list_model_ids(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(
            entry for entry in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, entry)) and _MODEL_ID_RE.match(entry)
        )

    def delete_version(self, model_id: str, version: int) -> None:
        """Explicit, deliberate deletion only. Never called automatically."""
        version_dir = self._version_dir(model_id, version)
        if not os.path.isdir(version_dir):
            raise FileNotFoundError(f"No such version to delete: model '{model_id}' version {version}.")
        shutil.rmtree(version_dir)
