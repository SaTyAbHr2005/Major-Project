"""
Device resolution for Module 6.

Module 6 never chooses a device automatically - that is Module 8's
responsibility (hardware detection, resource-aware selection). Module 6 only
resolves an EXPLICITLY requested device and fails clearly when it cannot be
honored, rather than silently substituting a different one.
"""
import torch

SUPPORTED_DEVICES = {"cpu", "cuda"}


def resolve_device(requested: str) -> torch.device:
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of {sorted(SUPPORTED_DEVICES)}, got {requested!r}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was explicitly requested but CUDA is not available on this "
            "machine. Module 6 does not silently fall back to CPU - automatic device "
            "selection/fallback policy belongs to Module 8. Request device='cpu' explicitly instead."
        )
    return torch.device(requested)


def to_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
    return model.to(resolve_device(device))
