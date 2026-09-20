"""
Model metadata schema for Module 6.

Follows the dataclass + to_dict()/from_dict() convention already used by
DatasetProfile (Module 4) and PreprocessingConfig (Module 5).

Local `status` values (draft/trained/available/deprecated) are Module 6's
OWN local lifecycle bookkeeping. They are NOT the authoritative platform
approval decision - that belongs to Module 15 (platform-wide, database-
backed model versioning/governance), which does not exist in this
repository. A hospital's local "available" status must not be presented as
platform-level approval.

Training-specific fields (epochs/optimizer/learning_rate/random_seed) are
part of the contract so Module 7 has somewhere to record them after an
actual training run, but Module 6 itself never populates them - doing so
would misrepresent that training occurred when it did not.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOCAL_STATUSES = {"draft", "trained", "available", "deprecated"}


@dataclass
class ModelMetadata:
    model_id: str
    version: int
    architecture: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "draft"

    artifact_sha256: str = ""
    artifact_size_bytes: int = 0
    framework_versions: Dict[str, str] = field(default_factory=dict)

    input_spec: Dict[str, Any] = field(default_factory=dict)
    num_classes: int = 0
    class_mapping: Dict[str, int] = field(default_factory=dict)

    preprocessing_reference: Optional[str] = None
    # Recorded by Module 7 at training time so inference can reproduce the exact input pipeline.
    task_type: Optional[str] = None
    preprocessing_spec: Optional[Dict[str, Any]] = None
    dataset_profile_reference: Optional[str] = None
    training_reference: Optional[str] = None
    parent_version: Optional[int] = None

    total_parameters: Optional[int] = None
    trainable_parameters: Optional[int] = None

    # Populated only by Module 7, only after real training. Never set by Module 6.
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = None
    random_seed: Optional[int] = None

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.status not in LOCAL_STATUSES:
            raise ValueError(f"status must be one of {sorted(LOCAL_STATUSES)}, got {self.status!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "architecture": self.architecture,
            "config": self.config,
            "status": self.status,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "framework_versions": self.framework_versions,
            "input_spec": self.input_spec,
            "num_classes": self.num_classes,
            "class_mapping": self.class_mapping,
            "preprocessing_reference": self.preprocessing_reference,
            "task_type": self.task_type,
            "preprocessing_spec": self.preprocessing_spec,
            "dataset_profile_reference": self.dataset_profile_reference,
            "training_reference": self.training_reference,
            "parent_version": self.parent_version,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "random_seed": self.random_seed,
            "created_at": self.created_at,
            "notes": self.notes,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        known_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in known_fields})
