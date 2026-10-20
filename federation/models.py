import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

class JobStatus(str, enum.Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ARCHIVED = "ARCHIVED"

class RoundStatus(str, enum.Enum):
    CREATED = "CREATED"
    OPEN = "OPEN"
    RECEIVING = "RECEIVING"
    READY_FOR_AGGREGATION = "READY_FOR_AGGREGATION"
    AGGREGATING = "AGGREGATING"
    GLOBAL_MODEL_CREATED = "GLOBAL_MODEL_CREATED"
    EVALUATION_PENDING = "EVALUATION_PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    INSUFFICIENT_PARTICIPANTS = "INSUFFICIENT_PARTICIPANTS"
    CANCELLED = "CANCELLED"

class UpdateStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    QUARANTINED = "QUARANTINED"
    EXPIRED = "EXPIRED"
    USED_IN_AGGREGATION = "USED_IN_AGGREGATION"
    NOT_USED = "NOT_USED"
    FAILED = "FAILED"

class GlobalModelStatus(str, enum.Enum):
    CREATED = "CREATED"
    EVALUATION_PENDING = "EVALUATION_PENDING"
    EVALUATED = "EVALUATED"
    MAIN = "MAIN"
    HISTORICAL = "HISTORICAL"
    EVALUATED_NOT_PROMOTED = "EVALUATED_NOT_PROMOTED"
    FAILED = "FAILED"

class PromotionDecisionValue(str, enum.Enum):
    PROMOTED = "PROMOTED"
    NOT_PROMOTED = "NOT_PROMOTED"
    REJECTED = "REJECTED"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class FederationJob:
    federation_job_id: str
    task: str # e.g. Cancer classification
    task_type: str
    architecture: str
    num_classes: int
    class_mapping: Dict[str, int]
    expected_participants: List[str]
    minimum_participants: int
    deadline: str
    training_requirements: Dict[str, Any]
    aggregation_configuration: Dict[str, Any]
    status: JobStatus = JobStatus.CREATED
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

@dataclass
class FederationParticipant:
    participant_id: str
    federation_job_id: str
    hospital_identifier: str
    authorization_status: str
    expected_status: str
    last_submission: Optional[str] = None
    participation_status: str = "EXPECTED"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    # Join request (federation/participation.py): the hospital's own eligibility summary and the admin decision.
    eligibility: Dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    decided_by: str = ""

@dataclass
class FederationRound:
    round_id: str
    federation_job_id: str
    round_number: int
    expected_participants: List[str]
    minimum_participants: int
    deadline: str
    base_model_id: str = ""
    base_model_version: int = 1
    base_model_checksum: str = ""
    status: RoundStatus = RoundStatus.CREATED
    received_participants: List[str] = field(default_factory=list)
    accepted_participants: List[str] = field(default_factory=list)
    rejected_participants: List[str] = field(default_factory=list)
    quarantined_participants: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso)
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    aggregation_started_at: Optional[str] = None
    aggregation_completed_at: Optional[str] = None

@dataclass
class FederationUpdate:
    update_id: str
    federation_job_id: str
    round_id: str
    participant_id: str
    model_id: str
    architecture: str
    artifact_location: str
    artifact_checksum: str
    parameter_count: int
    parameter_metadata: Dict[str, Any]
    num_train_samples: int
    training_configuration: Dict[str, Any]
    validation_result: str = ""
    rejection_reason: str = ""
    security_status: str = "ACCEPT"
    security_reason: str = ""
    security_metrics: Dict[str, Any] = field(default_factory=dict)
    status: UpdateStatus = UpdateStatus.RECEIVED
    received_at: str = field(default_factory=now_iso)
    validated_at: Optional[str] = None
    used_in_aggregation_at: Optional[str] = None

@dataclass
class GlobalModelVersion:
    global_model_id: str
    task: str
    task_type: str
    architecture: str
    version: int
    artifact_location: str
    artifact_checksum: str
    artifact_size: int
    parameter_count: int
    parameter_metadata: Dict[str, Any]
    federation_job_id: str
    round_id: str
    source_update_ids: List[str]
    aggregation_method: str
    aggregation_metadata: Dict[str, Any]
    class_mapping: Dict[str, int] = field(default_factory=dict)
    schema_hash: str = "LEGACY"
    status: GlobalModelStatus = GlobalModelStatus.CREATED
    created_at: str = field(default_factory=now_iso)

@dataclass
class EvaluationRecord:
    evaluation_id: str
    global_model_id: str
    model_version: int
    task: str
    architecture: str
    evaluation_dataset_id: str
    evaluation_dataset_version: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class_metrics: Dict[str, Any]
    confusion_matrix: List[List[int]]
    predictions_location: str
    metrics_location: str
    inference_configuration: Dict[str, Any]
    status: str
    started_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None

@dataclass
class PromotionDecision:
    promotion_decision_id: str
    candidate_global_model_id: str
    previous_main_global_model_id: Optional[str]
    decision: PromotionDecisionValue
    promotion_policy_version: str
    evaluation_record_id: str
    reason: str
    metric_summary: Dict[str, Any]
    created_at: str = field(default_factory=now_iso)
