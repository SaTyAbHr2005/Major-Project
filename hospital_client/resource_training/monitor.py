"""
ResourceMonitor: samples CPU/RAM/VRAM on a background thread while Module 7's
Trainer.run() executes normally in the foreground. Module 7 is never
modified or wrapped internally - this only observes the process/GPU from
the outside at a configurable interval, so overhead stays low and Module 7's
training loop is untouched (see module8_readme.md limitations: this is
run-level observation, not an in-loop callback).
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil
import torch

from hospital_client.resource_training import hardware
from hospital_client.resource_training.policy import ResourcePolicy

_probe_unavailable = False  # cached across samples so we don't retry a failing probe every tick


def _sample_gpu_utilization(timeout_seconds: float) -> Optional[float]:
    global _probe_unavailable
    if _probe_unavailable or not torch.cuda.is_available():
        return None
    value = hardware.detect_gpu_utilization(timeout_seconds)
    if value is None:
        _probe_unavailable = True  # no nvidia-smi on this machine - stop retrying every tick
        return None
    return value


@dataclass
class MonitorSummary:
    sample_count: int
    peak_cpu_percent: Optional[float]
    average_cpu_percent: Optional[float]
    peak_ram_mb: Optional[float]
    peak_vram_mb: Optional[float]
    peak_gpu_utilization_percent: Optional[float]
    gpu_utilization_available: bool
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "peak_cpu_percent": self.peak_cpu_percent,
            "average_cpu_percent": self.average_cpu_percent,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "peak_gpu_utilization_percent": self.peak_gpu_utilization_percent,
            "gpu_utilization_available": self.gpu_utilization_available,
            "duration_seconds": self.duration_seconds,
        }


class ResourceMonitor:
    def __init__(self, policy: ResourcePolicy, device_type: str = "cpu"):
        self.interval = policy.monitor_interval_seconds
        self.device_type = device_type
        self.gpu_utilization_probe_timeout_seconds = policy.gpu_utilization_probe_timeout_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cpu_samples: List[float] = []
        self._ram_samples_mb: List[float] = []
        self._vram_samples_mb: List[float] = []
        self._gpu_util_samples: List[float] = []
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None

    def _sample_once(self) -> None:
        self._cpu_samples.append(psutil.cpu_percent(interval=None))
        self._ram_samples_mb.append(psutil.virtual_memory().used / (1024 ** 2))
        if self.device_type == "cuda" and torch.cuda.is_available():
            self._vram_samples_mb.append(torch.cuda.memory_reserved(0) / (1024 ** 2))
            util = _sample_gpu_utilization(self.gpu_utilization_probe_timeout_seconds)
            if util is not None:
                self._gpu_util_samples.append(util)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._sample_once()

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._sample_once()  # at least one sample even for very short runs
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> MonitorSummary:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1.0)
        self._stopped_at = time.monotonic()
        self._sample_once()  # final sample

        duration = (self._stopped_at - self._started_at) if self._started_at else 0.0
        return MonitorSummary(
            sample_count=len(self._cpu_samples),
            peak_cpu_percent=max(self._cpu_samples) if self._cpu_samples else None,
            average_cpu_percent=(sum(self._cpu_samples) / len(self._cpu_samples)) if self._cpu_samples else None,
            peak_ram_mb=max(self._ram_samples_mb) if self._ram_samples_mb else None,
            peak_vram_mb=max(self._vram_samples_mb) if self._vram_samples_mb else None,
            peak_gpu_utilization_percent=max(self._gpu_util_samples) if self._gpu_util_samples else None,
            gpu_utilization_available=bool(self._gpu_util_samples),
            duration_seconds=duration,
        )
