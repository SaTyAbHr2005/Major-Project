"""
Inference device resolution. Explicit devices are validated with Module 6's
`resolve_device` and NEVER silently fall back (CUDA unavailable -> error).

"auto" is an opt-in hook for applications that want a device chosen for them:
it delegates to Module 8's existing `select_device` (no second resource
evaluator here), with the network probe disabled, and returns Module 8's
fallback reason so a CPU choice is never hidden. Module 8 is imported lazily so
the explicit-device path stays free of Module 8's dependencies.
"""
from typing import Optional, Tuple

import torch

from hospital_client.inference.errors import DeviceUnavailableError, InferenceConfigError
from hospital_client.model_management.device import resolve_device

VALID_DEVICE_REQUESTS = ("cpu", "cuda", "auto")
# Weights + activations + CUDA context need more than the raw weight file; this is a conservative
# heuristic (not a measurement), used only by device="auto" to decide whether CUDA is safe.
AUTO_VRAM_HEADROOM_FACTOR = 3.0


def cuda_has_headroom(weights_bytes: int, device: torch.device) -> Tuple[bool, str]:
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return False, "free VRAM could not be queried"
    needed = int(weights_bytes * AUTO_VRAM_HEADROOM_FACTOR)
    if free < needed:
        return False, f"free VRAM {free / 1024 ** 2:.0f} MB < estimated need {needed / 1024 ** 2:.0f} MB"
    return True, f"free VRAM {free / 1024 ** 2:.0f} MB >= estimated need {needed / 1024 ** 2:.0f} MB"


def resolve_inference_device(requested: str) -> Tuple[torch.device, Optional[str]]:
    if requested not in VALID_DEVICE_REQUESTS:
        raise InferenceConfigError(f"device must be one of {list(VALID_DEVICE_REQUESTS)}, got {requested!r}")

    if requested == "auto":
        from hospital_client.resource_training.evaluator import select_device
        from hospital_client.resource_training.resource_profile import build_resource_profile

        device_name, fallback_reason = select_device(build_resource_profile(check_network=False))
        note = f"auto-selected via Module 8: {device_name}" + (f" ({fallback_reason})" if fallback_reason else "")
        return torch.device(device_name), note

    try:
        return resolve_device(requested), None
    except RuntimeError as e:
        raise DeviceUnavailableError(str(e)) from None
