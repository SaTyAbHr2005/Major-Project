"""
Canonical base model of a federation round, stored as a Module 6 checkpoint
that every participating hospital downloads before local training.

Round 1 of a job starts from a freshly built model (SEED_<job_id>); later rounds
start from the job's latest global model. The checksum is computed exactly the
way the hospital Trainer verifies it (SHA-256 over the loaded parameters), so a
downloaded copy that differs in any way is refused before training starts.
"""
import os
from typing import Optional, Tuple

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters, set_parameters
from hospital_client.model_management.registry import build_model, get_architecture_info
from hospital_client.model_management.store import ModelStore
from hospital_client.model_management.validation import ARTIFACT_FILENAME, METADATA_FILENAME
from hospital_client.training.federation import FederationHandoff, _compute_parameters_checksum

BASE_MODEL_VERSION = 1
# The two files of a Module 6 checkpoint version, the only files the download route serves.
BASE_MODEL_FILES = (ARTIFACT_FILENAME, METADATA_FILENAME)


def base_model_store(storage) -> ModelStore:
    return ModelStore(os.path.join(storage.artifacts_dir, "base_models"))


def prepare_base_model(storage, job, previous_global_model: Optional[dict] = None) -> Tuple[str, int, str]:
    """Creates (or reuses) the round's base checkpoint; returns (model_id, version, checksum)."""
    store = base_model_store(storage)
    model_id = previous_global_model["global_model_id"] if previous_global_model else f"SEED_{job.federation_job_id}"

    if not os.path.isdir(os.path.join(store.root, model_id, f"v{BASE_MODEL_VERSION}")):
        reqs = job.training_requirements or {}
        dp_enabled = reqs.get("dp_noise_multiplier") is not None and float(reqs.get("dp_noise_multiplier", 0)) > 0
        norm_type = "group" if dp_enabled else "batch"
        config = ModelConfig(
            architecture=job.architecture,
            num_classes=job.num_classes,
            input_size=tuple(get_architecture_info(job.architecture).default_input_size),
            norm_type=norm_type,
        )
        model = build_model(config)
        if previous_global_model:
            set_parameters(model, FederationHandoff.load(previous_global_model["artifact_location"]).parameters)
        # An untrained seed is a draft; a global model came out of federated training.
        status = "trained" if previous_global_model else "draft"
        store.save_checkpoint(model, config, model_id, status=status, version=BASE_MODEL_VERSION)

    loaded, _ = store.load_checkpoint(model_id, BASE_MODEL_VERSION, device="cpu")
    return model_id, BASE_MODEL_VERSION, _compute_parameters_checksum(get_parameters(loaded))


def base_model_dir(storage, model_id: str, version: int) -> str:
    """Absolute folder of a stored base checkpoint (the id is validated by ModelStore)."""
    path = os.path.abspath(base_model_store(storage)._version_dir(model_id, version))
    if not os.path.isdir(path):
        raise ValueError(f"Base model {model_id} v{version} is not stored on the federation server.")
    return path
