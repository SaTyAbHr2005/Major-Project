"""
The structured result Module 7 produces at the end of a run. See
federation.py for the versioned Module 7 -> Module 9 FederationHandoff
contract Module 7 prepares (but never transmits) from a completed result.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class TrainingStatus(Enum):
    TRAINING_FAILED = "TRAINING_FAILED"
    TRAINING_INTERRUPTED = "TRAINING_INTERRUPTED"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    TRAINING_COMPLETED_AWAITING_FEDERATION = "TRAINING_COMPLETED_AWAITING_FEDERATION"


@dataclass
class TrainingResult:
    status: TrainingStatus
    model_id: str
    architecture: str
    checkpoint_model_id: Optional[str] = None
    checkpoint_version: Optional[int] = None
    best_epoch: Optional[int] = None
    epochs_run: int = 0
    training_duration_seconds: float = 0.0
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Optional[Dict[str, Any]] = None
    class_mapping: Dict[str, int] = field(default_factory=dict)
    training_configuration: Dict[str, Any] = field(default_factory=dict)
    preprocessing_reference: Optional[str] = None
    dataset_reference: Optional[str] = None
    random_seed: Optional[int] = None
    resolved_precision: str = "fp32"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def ready_for_federation(self) -> bool:
        return self.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model_id": self.model_id,
            "architecture": self.architecture,
            "checkpoint_model_id": self.checkpoint_model_id,
            "checkpoint_version": self.checkpoint_version,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "training_duration_seconds": self.training_duration_seconds,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "class_mapping": self.class_mapping,
            "training_configuration": self.training_configuration,
            "preprocessing_reference": self.preprocessing_reference,
            "dataset_reference": self.dataset_reference,
            "random_seed": self.random_seed,
            "resolved_precision": self.resolved_precision,
            "warnings": self.warnings,
            "errors": self.errors,
            "ready_for_federation": self.ready_for_federation,
            "generated_at": self.generated_at,
        }


def format_summary(result: TrainingResult, display_name: str) -> str:
    """Human-readable local training summary (see module7_readme.md)."""
    lines = []
    if result.status in (TrainingStatus.TRAINING_COMPLETED, TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION):
        lines.append("Local training completed")
    elif result.status == TrainingStatus.TRAINING_INTERRUPTED:
        lines.append("Local training interrupted")
    else:
        lines.append("Local training failed")

    lines.append("")
    lines.append(f"Model: {display_name}")
    lines.append(f"Epochs run: {result.epochs_run}")
    if result.best_epoch is not None:
        lines.append(f"Best epoch: {result.best_epoch}")

    val = result.validation_metrics
    if val:
        lines.append(f"Validation accuracy: {val.get('accuracy', 0.0) * 100:.2f}%")
        lines.append(f"Validation F1: {val.get('f1', 0.0) * 100:.2f}%")
        lines.append(f"Validation precision: {val.get('precision', 0.0) * 100:.2f}%")
        lines.append(f"Validation recall: {val.get('recall', 0.0) * 100:.2f}%")

    lines.append(f"Training duration: {result.training_duration_seconds / 60.0:.2f} minutes")
    if result.checkpoint_model_id is not None:
        lines.append(f"Best checkpoint: {result.checkpoint_model_id} v{result.checkpoint_version}")

    lines.append(f"Status: {result.status.value}")
    if result.warnings:
        lines.append(f"Warnings: {len(result.warnings)}")
    if result.errors:
        lines.append(f"Errors: {result.errors}")

    lines.append("")
    lines.append(
        "This is an experimental research result from local training on the provided "
        "dataset. It is not a clinical validation and does not represent guaranteed "
        "diagnostic performance."
    )
    return "\n".join(lines)
