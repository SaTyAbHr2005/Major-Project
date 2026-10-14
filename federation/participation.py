"""
Who may take part in a federation job (Module 9).

The server never sees a hospital's data, so it cannot judge eligibility itself. Instead:
  1. the hospital runs the eligibility check locally (hospital_client/federation/eligibility.py),
  2. it requests to join the job, sending only an aggregate summary of that check
     (eligible, task match, class coverage, image counts, hardware tier; no images, no paths),
  3. the admin approves or rejects the request,
  4. only approved hospitals can join the job's rounds (and only their updates pass validation,
     which checks job.expected_participants).

Hospitals that were already in a job's expected_participants before requests existed count as
approved ("legacy"), so jobs created earlier keep working.

The summary is reported by the hospital and cannot be verified by the server; it is decision
support for the admin, not proof. (The update's own sample count is trusted the same way today.)
"""
from typing import Any, Dict, List, Optional

from .models import FederationJob, FederationParticipant, JobStatus, now_iso
from .storage import FederationStorage

PENDING, APPROVED, REJECTED = "PENDING", "APPROVED", "REJECTED"
DECISIONS = (APPROVED, REJECTED)
JOINABLE_JOB_STATUSES = (JobStatus.CREATED, JobStatus.ACTIVE)

_BOOL_FIELDS = ("eligible", "dataset_eligible", "architecture_eligible", "task_match")
_INT_FIELDS = ("train_sample_count", "validation_sample_count", "test_sample_count")
_STR_FIELDS = ("dataset_eligibility_type", "architecture_tier")
_LIST_FIELDS = ("required_classes", "available_classes", "missing_classes", "extra_classes",
                "dataset_reasons", "architecture_reasons")
_MAX_ITEMS, _MAX_TEXT = 50, 300


def participant_id(job_id: str, hospital_id: str) -> str:
    return f"{job_id}:{hospital_id}"


def summarize_eligibility(raw: Any) -> Dict[str, Any]:
    """Keep only the aggregate fields of an eligibility result, with checked types and bounded sizes."""
    if not isinstance(raw, dict):
        raise ValueError("An eligibility summary from the hospital's local check is required.")
    out: Dict[str, Any] = {}
    for k in _BOOL_FIELDS:
        if k in raw:
            if not isinstance(raw[k], bool):
                raise ValueError(f"Eligibility field {k} must be true or false.")
            out[k] = raw[k]
    for k in _INT_FIELDS:
        if k in raw:
            v = raw[k]
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"Eligibility field {k} must be a whole number >= 0.")
            out[k] = v
    for k in _STR_FIELDS:
        if k in raw and raw[k] is not None:
            out[k] = str(raw[k])[:_MAX_TEXT]
    for k in _LIST_FIELDS:
        if k in raw:
            if not isinstance(raw[k], list):
                raise ValueError(f"Eligibility field {k} must be a list.")
            out[k] = [str(v)[:_MAX_TEXT] for v in raw[k][:_MAX_ITEMS]]
    if "eligible" not in out:
        raise ValueError("The eligibility summary must say whether the hospital is eligible.")
    return out


def get_participant(storage: FederationStorage, job: FederationJob, hospital_id: str) -> Optional[FederationParticipant]:
    p = storage.get_participant(participant_id(job.federation_job_id, hospital_id))
    if p is None and hospital_id in job.expected_participants:
        # Joined before requests existed: treat as approved.
        p = FederationParticipant(
            participant_id=participant_id(job.federation_job_id, hospital_id),
            federation_job_id=job.federation_job_id, hospital_identifier=hospital_id,
            authorization_status=APPROVED, expected_status="LEGACY", decision_reason="Joined before participation requests existed.")
    return p


def list_participants(storage: FederationStorage, job: FederationJob) -> List[FederationParticipant]:
    stored = {p.hospital_identifier: p for p in storage.list_participants(job.federation_job_id)}
    for hospital_id in job.expected_participants:
        if hospital_id not in stored:
            stored[hospital_id] = get_participant(storage, job, hospital_id)
    return sorted(stored.values(), key=lambda p: p.created_at)


def request_participation(storage: FederationStorage, job: FederationJob, hospital_id: str, raw_eligibility: Any) -> FederationParticipant:
    if job.status not in JOINABLE_JOB_STATUSES:
        raise ValueError(f"Job {job.federation_job_id} is {job.status.value} and does not accept hospitals.")
    summary = summarize_eligibility(raw_eligibility)
    if not summary["eligible"]:
        reasons = summary.get("dataset_reasons", []) + summary.get("architecture_reasons", [])
        raise ValueError("Your local eligibility check says this dataset is not eligible for the job"
                         + (f": {'; '.join(reasons)}" if reasons else "."))
    p = get_participant(storage, job, hospital_id)
    if p is None or p.expected_status == "LEGACY":
        p = FederationParticipant(
            participant_id=participant_id(job.federation_job_id, hospital_id),
            federation_job_id=job.federation_job_id, hospital_identifier=hospital_id,
            authorization_status=APPROVED if p else PENDING, expected_status="REQUESTED")
    elif p.authorization_status == REJECTED:
        p.authorization_status, p.decision_reason, p.decided_by = PENDING, "", ""
    p.eligibility = summary          # an approved hospital may refresh its summary without losing approval
    p.updated_at = now_iso()
    storage.save_participant(p)
    return p


def decide_participation(storage: FederationStorage, job: FederationJob, hospital_id: str,
                         decision: str, reason: str = "", admin_id: str = "") -> FederationParticipant:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {list(DECISIONS)}, got {decision!r}")
    p = get_participant(storage, job, hospital_id)
    if p is None:
        raise ValueError(f"Hospital {hospital_id} has not requested to join job {job.federation_job_id}.")
    p.authorization_status, p.decided_by, p.updated_at = decision, admin_id, now_iso()
    p.decision_reason = (reason or "").strip()[:_MAX_TEXT]
    if p.expected_status == "LEGACY":
        p.expected_status = "REQUESTED"
    storage.save_participant(p)

    if decision == APPROVED and hospital_id not in job.expected_participants:
        job.expected_participants.append(hospital_id)
    if decision == REJECTED:
        if hospital_id in job.expected_participants:
            job.expected_participants.remove(hospital_id)
        # Leave rounds it already sent an update to; drop it from rounds it only joined.
        for rnd in storage.list_rounds_for_job(job.federation_job_id):
            if hospital_id in rnd.expected_participants and hospital_id not in rnd.received_participants:
                rnd.expected_participants.remove(hospital_id)
                storage.save_round(rnd)
    job.updated_at = now_iso()
    storage.save_job(job)
    return p


def can_join(storage: FederationStorage, job: FederationJob, hospital_id: str) -> bool:
    p = get_participant(storage, job, hospital_id)
    return p is not None and p.authorization_status == APPROVED
