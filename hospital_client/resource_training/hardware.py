"""
Raw hardware detection for Module 8. Each function inspects the ACTUAL local
machine (via psutil/torch.cuda/platform/socket) and never hard-codes any
particular machine's numbers - this development machine's RTX 3050 is only
one possible measurement, not a target. Every function degrades gracefully
(returns an "unavailable"/None-filled dict) rather than raising, so a
detection failure on one axis never blocks the others - see profile.py.
"""
import platform
import socket
import subprocess
import time
from typing import Any, Dict, Optional

import psutil
import torch


def detect_cpu() -> Dict[str, Any]:
    return {
        "processor": platform.processor() or platform.machine() or "unknown",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "utilization_percent": psutil.cpu_percent(interval=0.1),
    }


def detect_ram() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "total_mb": vm.total / (1024 ** 2),
        "available_mb": vm.available / (1024 ** 2),
        "utilization_percent": vm.percent,
    }


def detect_gpu() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        return {
            "cuda_available": False,
            "gpu_name": None,
            "gpu_vendor": None,
            "total_vram_mb": None,
            "free_vram_mb": None,
            "cuda_capability": None,
            "fallback_reason": "torch.cuda.is_available() returned False on this machine.",
        }

    device_index = 0
    name = torch.cuda.get_device_name(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    return {
        "cuda_available": True,
        "gpu_name": name,
        # torch's CUDA backend targets NVIDIA hardware; this is a label, not
        # a claim of support for other vendors' CUDA-alike stacks.
        "gpu_vendor": "NVIDIA",
        "total_vram_mb": total_bytes / (1024 ** 2),
        "free_vram_mb": free_bytes / (1024 ** 2),
        "cuda_capability": f"{capability[0]}.{capability[1]}",
        "fallback_reason": None,
    }


def detect_storage(path: str = ".") -> Dict[str, Any]:
    usage = psutil.disk_usage(path)
    return {
        "path": path,
        "total_mb": usage.total / (1024 ** 2),
        "free_mb": usage.free / (1024 ** 2),
        "utilization_percent": usage.percent,
    }


def detect_gpu_utilization(timeout_seconds: float = 0.5) -> Optional[float]:
    """
    REAL GPU compute utilization via the `nvidia-smi` CLI (bundled with every
    NVIDIA driver install - no new Python dependency such as pynvml is
    required). Returns None (never a fabricated number) when no NVIDIA
    driver/GPU is present, the CLI is missing, or the call fails/times out.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().splitlines()[0]
        return float(first_line.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def detect_network(host: str = "8.8.8.8", port: int = 53, timeout_seconds: float = 0.5) -> Dict[str, Any]:
    """
    Best-effort, non-blocking connectivity probe. Never raises, never takes
    longer than timeout_seconds. Recorded only for future federated/resource
    telemetry (Module 8 §22) - local training never depends on this.
    """
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            latency_ms = (time.monotonic() - started) * 1000.0
            return {"network_available": True, "latency_ms": latency_ms, "checked_host": host}
    except OSError:
        return {"network_available": False, "latency_ms": None, "checked_host": host}
