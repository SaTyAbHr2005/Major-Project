"""Glue between what arrives over Flower and the Module 9 engine (storage, intake, global models)."""
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from hospital_client.training.federation import FederationHandoff, _compute_parameters_checksum, validate_federation_handoff

BN_STATISTICS = ("running_mean", "running_var")
_PATH = re.compile(r"^([A-Za-z]:[\\/]|/|\\\\|~[\\/])")   # C:\..., /..., \\server\..., ~/...


def strip_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Training configuration without filesystem paths (folder names and OS user names stay at the hospital)."""
    out = {}
    for key, value in (config or {}).items():
        if key in ("dataset_dir", "output_dir"):
            continue
        if isinstance(value, dict):
            out[key] = strip_paths(value)
        elif isinstance(value, str) and _PATH.match(value):
            continue
        else:
            out[key] = value
    return out


def handoff_from_parameters(parameters: List[np.ndarray], metadata: Dict[str, Any], directory: str) -> str:
    """Writes a FederationHandoff from weights received over Flower plus the hospital's handoff metadata.

    The metadata's parameters_checksum was computed by the hospital before sending, so a weight that
    changed in transit fails validation here (ValueError). Returns `directory`.
    """
    fields = {k: metadata[k] for k in FederationHandoff.__dataclass_fields__ if k in metadata and k != "parameters"}
    fields["training_configuration"] = strip_paths(fields.get("training_configuration") or {})
    handoff = FederationHandoff(parameters=[np.asarray(p) for p in parameters], **fields)
    validate_federation_handoff(handoff)
    handoff.save(directory)
    return directory


def average_bn_statistics(global_params: Sequence[np.ndarray], keys: Sequence[str],
                          stats: Sequence[Tuple[Sequence[np.ndarray], int]]) -> List[np.ndarray]:
    """Global parameters with BatchNorm running statistics replaced by the sample-weighted mean of the
    hospitals' calibrated statistics. Every other array (weights, counters) is the global one, unchanged."""
    total = float(sum(n for _, n in stats))
    out = []
    for i, key in enumerate(keys):
        g = np.asarray(global_params[i])
        if key.endswith(BN_STATISTICS) and total > 0:
            mean = sum(np.asarray(p[i], dtype=np.float64) * n for p, n in stats) / total
            out.append(mean.astype(g.dtype))
        else:
            out.append(g)
    return out


def replace_global_model_parameters(storage, global_model_id: str, parameters: List[np.ndarray]) -> None:
    """Re-saves a global model's artifact with new parameters and updates its checksum and size."""
    gm = storage.get_global_model(global_model_id)
    if gm is None:
        raise ValueError(f"Global model {global_model_id} not found.")
    handoff = FederationHandoff.load(gm.artifact_location)
    handoff.parameters = [np.asarray(p) for p in parameters]
    handoff.parameters_checksum = _compute_parameters_checksum(handoff.parameters)
    validate_federation_handoff(handoff)
    handoff.save(gm.artifact_location)
    gm.artifact_checksum = handoff.parameters_checksum
    gm.artifact_size = os.path.getsize(os.path.join(gm.artifact_location, "handoff_parameters.npz"))
    storage.save_global_model(gm)
