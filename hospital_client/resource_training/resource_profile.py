"""
ResourceProfile: the standardized snapshot of the local machine's hardware
Module 8 evaluates against. Hardware/operational information only - never
medical images, patient records, PHI, raw dataset contents, or model
weights (see module8_readme.md privacy section).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from hospital_client.resource_training import hardware


@dataclass
class ResourceProfile:
    timestamp: str
    cpu: Dict[str, Any]
    ram: Dict[str, Any]
    gpu: Dict[str, Any]
    storage: Dict[str, Any]
    network: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu": self.cpu,
            "ram": self.ram,
            "gpu": self.gpu,
            "storage": self.storage,
            "network": self.network,
            "warnings": self.warnings,
            "errors": self.errors,
        }


_UNAVAILABLE_CPU = {"processor": None, "physical_cores": None, "logical_cores": None, "utilization_percent": None}
_UNAVAILABLE_RAM = {"total_mb": None, "available_mb": None, "utilization_percent": None}
_UNAVAILABLE_GPU = {
    "cuda_available": False, "gpu_name": None, "gpu_vendor": None, "total_vram_mb": None,
    "free_vram_mb": None, "cuda_capability": None, "fallback_reason": "Detection failed - see warnings.",
}
_UNAVAILABLE_STORAGE = {"path": None, "total_mb": None, "free_mb": None, "utilization_percent": None}
_UNAVAILABLE_NETWORK = {"network_available": False, "latency_ms": None, "checked_host": None}


def build_resource_profile(storage_path: str = ".", check_network: bool = True, policy=None) -> ResourceProfile:
    """
    Detects the actual local machine. Each detector is isolated - a failure
    on one axis (e.g. storage path doesn't exist) is recorded as a warning
    and degrades that section to an "unavailable" placeholder rather than
    aborting the whole profile.
    """
    warnings: List[str] = []
    errors: List[str] = []

    def _safe(label: str, fn, fallback):
        try:
            return fn()
        except Exception as e:
            warnings.append(f"{label} detection failed: {e}")
            return fallback

    cpu = _safe("CPU", hardware.detect_cpu, dict(_UNAVAILABLE_CPU))
    ram = _safe("RAM", hardware.detect_ram, dict(_UNAVAILABLE_RAM))
    gpu = _safe("GPU", hardware.detect_gpu, dict(_UNAVAILABLE_GPU))
    storage = _safe("Storage", lambda: hardware.detect_storage(storage_path), dict(_UNAVAILABLE_STORAGE))

    if check_network:
        if policy is not None:
            network = _safe(
                "Network",
                lambda: hardware.detect_network(policy.network_check_host, policy.network_check_port, policy.network_check_timeout_seconds),
                dict(_UNAVAILABLE_NETWORK),
            )
        else:
            network = _safe("Network", hardware.detect_network, dict(_UNAVAILABLE_NETWORK))
    else:
        network = {"network_available": None, "latency_ms": None, "checked_host": None}

    return ResourceProfile(
        timestamp=datetime.now(timezone.utc).isoformat(),
        cpu=cpu, ram=ram, gpu=gpu, storage=storage, network=network,
        warnings=warnings, errors=errors,
    )
