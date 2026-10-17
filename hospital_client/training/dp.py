"""
Differential Privacy engine and accounting for Module 10.

Implements Opacus-based DP-SGD with:
- Poisson sampling (make_private with poisson_sampling=True)
- Rényi Differential Privacy (RDP) accountant over the complete training history
- Pre-training privacy budget check (refuses before training if projected epsilon exceeds cap)
- Physical batch management via BatchMemoryManager (Module 8 can adjust physical batch
  size for memory without altering the logical batch size or privacy budget)
- Local DP state persistence (dp_state.json)
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader


class PrivacyBudgetExceededError(ValueError):
    """Raised when an operation would exceed or has exceeded the differential privacy budget."""
    pass


def compute_rdp_epsilon(history: List[Tuple[float, float, int]], delta: float) -> float:
    """
    Computes cumulative epsilon over a full history of (noise_multiplier, sample_rate, steps).
    Uses Opacus's RDPAccountant across the entire sequence of steps — never adds per-round epsilons.
    """
    if not history or delta <= 0 or delta >= 1:
        return 0.0
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    accountant.history = [
        (float(entry[0]), float(entry[1]), int(entry[2]))
        for entry in history
        if int(entry[2]) > 0
    ]
    if not accountant.history:
        return 0.0
    eps = accountant.get_epsilon(delta=delta)
    return float(eps)


def load_dp_state(state_path: str, job_id: str) -> Dict[str, Any]:
    """Loads the local DP state file for a job, or returns a blank state."""
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("job_id") == job_id and isinstance(data.get("history"), list):
                    return data
        except Exception:
            pass
    return {
        "job_id": job_id,
        "history": [],
    }


def save_dp_state(state_path: str, state: Dict[str, Any]) -> None:
    """Persists the local DP state atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def project_and_check_budget(
    existing_history: List[Tuple[float, float, int]],
    noise_multiplier: float,
    sample_rate: float,
    planned_steps: int,
    target_delta: float,
    target_epsilon: Optional[float],
) -> float:
    """
    Projects cumulative epsilon for the planned round BEFORE training executes.
    Raises PrivacyBudgetExceededError if projected epsilon would exceed target_epsilon.
    """
    if planned_steps <= 0:
        return compute_rdp_epsilon(existing_history, target_delta)

    projected_history = list(existing_history) + [(noise_multiplier, sample_rate, planned_steps)]
    projected_eps = compute_rdp_epsilon(projected_history, target_delta)

    if target_epsilon is not None and target_epsilon > 0:
        if projected_eps > target_epsilon:
            raise PrivacyBudgetExceededError(
                f"Privacy budget exceeded: projected cumulative epsilon {projected_eps:.4f} "
                f"would exceed the target budget cap of {target_epsilon:.4f} (delta={target_delta}). "
                f"Training refused before starting."
            )

    return projected_eps


def prepare_dp_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    noise_multiplier: float,
    max_grad_norm: float,
    poisson_sampling: bool = True,
) -> Tuple[Any, torch.nn.Module, torch.optim.Optimizer, DataLoader]:
    """
    Wraps model, optimizer, and data_loader with Opacus PrivacyEngine using Poisson sampling.
    """
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
    )
    return privacy_engine, model, optimizer, train_loader
