"""
Framework-light parameter interface: Module 6 -> future Module 9.

CLAUDE.md already defines the ML <-> FL contract as get_model()/
get_parameters()/set_parameters()/train_local()/evaluate(); Module 6 owns
the first three, Module 7 owns the last two. This module implements
get_parameters()/set_parameters() as plain ordered lists of NumPy arrays so
a future Module 9 can plug in (e.g. as a Flower NumPyClient) without Module 6
importing Flower or knowing anything about FedAvg/aggregation.

Ordering is deterministic: torch.nn.Module.state_dict() preserves parameter
registration order, and this module always converts in that exact order.
"""
from typing import List

import numpy as np
import torch


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(parameters) != len(keys):
        raise ValueError(f"Parameter count mismatch: model has {len(keys)} tensors, got {len(parameters)} arrays.")

    new_state = {}
    for key, array, existing in zip(keys, parameters, model.state_dict().values()):
        tensor = torch.as_tensor(np.asarray(array), dtype=existing.dtype)
        if tuple(tensor.shape) != tuple(existing.shape):
            raise ValueError(f"Shape mismatch for '{key}': expected {tuple(existing.shape)}, got {tuple(tensor.shape)}")
        new_state[key] = tensor

    model.load_state_dict(new_state, strict=True)
