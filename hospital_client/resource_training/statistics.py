"""
ResourceStatistics: structured, PHI-free resource/training statistics for a
completed resource-aware run, intended as raw input for a future Module 20
(Research Experiments and Evaluation) - Module 8 records facts, it never
performs research analysis, model comparison, or benchmarking itself.

Contains hardware/config/resource numbers only - never medical images,
patient records, or raw dataset contents (Module 7's TrainingResult already
excludes those; this only adds resource-observation data on top).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hospital_client.resource_training.adaptation import AdaptationEvent
from hospital_client.resource_training.monitor import MonitorSummary
from hospital_client.resource_training.resource_profile import ResourceProfile


@dataclass
class ResourceStatistics:
    hardware: Dict[str, Any]
    training_configuration: Dict[str, Any]
    monitor_summary: Dict[str, Any]
    adaptations: List[Dict[str, Any]]
    actual_training_duration_seconds: float
    measured_samples_per_second: Optional[float]
    estimated_training_time_seconds: Optional[float]
    estimation_error_seconds: Optional[float]
    estimation_method: Optional[str]
    cuda_oom_event_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware": self.hardware,
            "training_configuration": self.training_configuration,
            "monitor_summary": self.monitor_summary,
            "adaptations": self.adaptations,
            "actual_training_duration_seconds": self.actual_training_duration_seconds,
            "measured_samples_per_second": self.measured_samples_per_second,
            "estimated_training_time_seconds": self.estimated_training_time_seconds,
            "estimation_error_seconds": self.estimation_error_seconds,
            "estimation_method": self.estimation_method,
            "cuda_oom_event_count": self.cuda_oom_event_count,
        }


def build_resource_statistics(
    profile: ResourceProfile, training_configuration: Dict[str, Any], monitor_summary: MonitorSummary,
    adaptations: List[AdaptationEvent], actual_duration_seconds: float,
    measured_samples_per_second: Optional[float], estimated_training_time_seconds: Optional[float] = None,
    estimation_method: Optional[str] = None,
) -> ResourceStatistics:
    estimation_error = None
    if estimated_training_time_seconds is not None:
        estimation_error = actual_duration_seconds - estimated_training_time_seconds

    oom_count = sum(1 for a in adaptations if "out of memory" in a.reason.lower())

    return ResourceStatistics(
        hardware=profile.to_dict(),
        training_configuration=training_configuration,
        monitor_summary=monitor_summary.to_dict(),
        adaptations=[a.to_dict() for a in adaptations],
        actual_training_duration_seconds=actual_duration_seconds,
        measured_samples_per_second=measured_samples_per_second,
        estimated_training_time_seconds=estimated_training_time_seconds,
        estimation_error_seconds=estimation_error,
        estimation_method=estimation_method,
        cuda_oom_event_count=oom_count,
    )
