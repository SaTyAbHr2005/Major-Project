"""
The structured result Module 16 returns. Persistence is deliberately NOT
handled here (Module 14 does not exist yet) - `to_dict()` gives any future
consumer a plain, JSON-serializable view.

Privacy: a result never contains the image path, filename, pixels, or
base64 data. The only image-derived fields are format/mode/dimensions/byte
size. `reference` is an opaque, caller-supplied identifier echoed back
unchanged (the caller decides what, if anything, it means).

Wording: outputs are "model output"/"predicted class"/"model confidence" -
never a diagnosis.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

DISCLAIMER = (
    "This result is a model prediction for the configured medical imaging task "
    "and is not a clinical diagnosis."
)
CONFIDENCE_NOTE = (
    "Confidence is the model's uncalibrated softmax output probability for the predicted class. "
    "It is a model output, not a measure of clinical certainty, and is not reliable for inputs unlike the training data."
)


class InferenceStatus(Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskOutput:
    """Base for task-specific outputs. A future task type adds its own subclass
    (plus a handler in engine.py) without changing InferenceResult."""

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class ClassificationOutput(TaskOutput):
    predicted_index: int
    predicted_label: str
    confidence: float
    class_probabilities: Dict[str, float]
    output_transform: str = "softmax"
    confidence_kind: str = "uncalibrated softmax probability"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "classification",
            "predicted_index": self.predicted_index,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "output_transform": self.output_transform,
            "confidence_kind": self.confidence_kind,
        }


@dataclass
class InferenceResult:
    status: InferenceStatus
    model_id: str
    model_version: int
    architecture: str
    task: str
    task_source: str
    device: str
    output: Optional[TaskOutput] = None
    device_note: Optional[str] = None
    inference_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    peak_memory_mb: Optional[float] = None
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    error_category: Optional[str] = None
    # Extension point for future modules (e.g. Module 10 privacy metadata,
    # Module 15 approval details) - Module 16 never interprets it.
    extension_metadata: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)  # software/hardware context for reproducibility
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def succeeded(self) -> bool:
        return self.status == InferenceStatus.COMPLETED

    @property
    def predicted_label(self) -> Optional[str]:
        return getattr(self.output, "predicted_label", None)

    @property
    def confidence(self) -> Optional[float]:
        return getattr(self.output, "confidence", None)

    @property
    def class_probabilities(self) -> Optional[Dict[str, float]]:
        return getattr(self.output, "class_probabilities", None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "architecture": self.architecture,
            "task": self.task,
            "task_source": self.task_source,
            "output": self.output.to_dict() if self.output is not None else None,
            "device": self.device,
            "device_note": self.device_note,
            "inference_duration_seconds": self.inference_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "preprocessing": self.preprocessing,
            "model_info": self.model_info,
            "reference": self.reference,
            "warnings": self.warnings,
            "errors": self.errors,
            "error_category": self.error_category,
            "extension_metadata": self.extension_metadata,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "disclaimer": DISCLAIMER,
        }


def format_summary(result: InferenceResult) -> str:
    lines = [
        f"Inference {'completed' if result.succeeded else 'failed'}",
        "",
        f"Model: {result.model_id} v{result.model_version} ({result.architecture})",
        f"Task: {result.task} ({result.task_source})",
        f"Device: {result.device}",
    ]
    if result.succeeded and isinstance(result.output, ClassificationOutput):
        out = result.output
        lines.append(f"Predicted class (model output): {out.predicted_label}")
        lines.append(f"Model confidence (uncalibrated softmax probability): {out.confidence * 100:.2f}%")
        lines.append("Class probabilities (model output):")
        for label, prob in sorted(out.class_probabilities.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {label}: {prob * 100:.2f}%")
        lines.append(f"Inference duration: {result.inference_duration_seconds * 1000:.1f} ms")
        lines.append("")
        lines.append(CONFIDENCE_NOTE)
    if result.errors:
        lines.append(f"Errors ({result.error_category}): {result.errors}")
    if result.warnings:
        lines.append(f"Warnings: {result.warnings}")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)
