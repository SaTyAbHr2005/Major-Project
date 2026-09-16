"""
Two distinct kinds of "checkpoint", deliberately kept separate:

1. Model artifact (Module 6): weights + ModelMetadata, safetensors-only,
   strictly validated on every load, meant to be portable and eventually
   shared with other modules (Module 9/15/16). Saved via
   hospital_client.model_management.store.ModelStore - Module 7 never
   reimplements artifact I/O.

2. Training state (Module 7, private to this module): optimizer/scheduler
   state, epoch, best-metric bookkeeping, RNG state - needed only to RESUME
   an interrupted *local* training run. This is Module 7's own scratch
   state, never exchanged with another module or hospital, so plain
   torch.save()/torch.load() is appropriate here - unlike Module 6's
   artifact format, which must defend against untrusted/exchanged files,
   this file only ever round-trips within one hospital's own training
   process.
"""
import os
from typing import Any, Dict, Optional

import torch


def training_state_path(output_dir: str, model_id: str) -> str:
    state_dir = os.path.join(output_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{model_id}_state.pt")


def save_training_state(
    output_dir: str,
    model_id: str,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    best_metric: Optional[float],
    best_epoch: Optional[int],
    class_mapping: Dict[str, int],
    model_config_dict: Dict[str, Any],
    checkpoint_model_id: Optional[str],
    checkpoint_version: Optional[int],
    random_seed: int,
    scaler: Optional[Any] = None,
) -> str:
    state = {
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "class_mapping": class_mapping,
        "model_config": model_config_dict,
        "checkpoint_model_id": checkpoint_model_id,
        "checkpoint_version": checkpoint_version,
        "random_seed": random_seed,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path = training_state_path(output_dir, model_id)
    torch.save(state, path)
    return path


def load_training_state(output_dir: str, model_id: str) -> Dict[str, Any]:
    path = training_state_path(output_dir, model_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"resume=True but no training state found at {path}. Cannot resume a run that "
            "never saved state; start a fresh run instead."
        )
    return torch.load(path, weights_only=True)


def restore_rng_state(state: Dict[str, Any]) -> None:
    rng = state.get("rng_state") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
