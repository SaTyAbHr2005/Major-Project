"""
Who may join a federation job (request + eligibility summary + admin decision), and the
evaluation refusing to report metrics when some evaluation images cannot be processed.
"""
import json
import os
import subprocess
import sys

import pytest
from PIL import Image

from federation import participation as part
from federation.evaluation import EvaluationManager
from federation.events import EventEmitter
from federation.jobs import JobManager
from federation.models import GlobalModelVersion, JobStatus
from federation.rounds import RoundManager
from federation.storage import FederationStorage
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters
from hospital_client.model_management.registry import build_model
from hospital_client.training.federation import build_federation_handoff

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ELIGIBLE = {"eligible": True, "dataset_eligible": True, "task_match": True, "required_classes": ["a", "b"],
            "available_classes": ["a", "b"], "missing_classes": [], "train_sample_count": 400,
            "validation_sample_count": 50, "test_sample_count": 50, "architecture_tier": "RECOMMENDED"}


@pytest.fixture
def storage(tmp_path):
    s = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    JobManager(s).create_job("JOB", "Malaria", "image_classification", "resnet18", 2, {"a": 0, "b": 1},
                             [], 1, "", {}, {})
    return s


def job(storage):
    return storage.get_job("JOB")


def open_round(storage):
    RoundManager(storage).create_round("JOB", "JOB_R1", 1, [], 1, "2099-01-01T00:00:00+00:00", "B", 1, "c")


# ── eligibility summary ───────────────────────────────────────────────────────────────

def test_summary_keeps_only_aggregate_fields():
    raw = {**ELIGIBLE, "dataset_path": "C:/hospital/patients", "images": ["x.png"], "patient_ids": [1, 2]}
    s = part.summarize_eligibility(raw)
    assert set(s) <= set(ELIGIBLE) and "dataset_path" not in s and "images" not in s
    assert s["train_sample_count"] == 400


@pytest.mark.parametrize("raw", [None, "yes", {}, {"eligible": "true"}, {"eligible": True, "train_sample_count": -1},
                                 {"eligible": True, "train_sample_count": 1.5}, {"eligible": True, "missing_classes": "a"}])
def test_bad_summaries_are_refused(raw):
    with pytest.raises(ValueError):
        part.summarize_eligibility(raw)


def test_long_lists_and_texts_are_bounded():
    s = part.summarize_eligibility({"eligible": True, "dataset_reasons": ["x" * 1000] * 200})
    assert len(s["dataset_reasons"]) == 50 and len(s["dataset_reasons"][0]) == 300


# ── request / decision / join ─────────────────────────────────────────────────────────

def test_request_then_approve_lets_the_hospital_join(storage):
    assert not part.can_join(storage, job(storage), "H1")
    p = part.request_participation(storage, job(storage), "H1", ELIGIBLE)
    assert p.authorization_status == "PENDING" and p.eligibility["train_sample_count"] == 400
    assert not part.can_join(storage, job(storage), "H1")          # pending is not enough
    assert "H1" not in job(storage).expected_participants

    part.decide_participation(storage, job(storage), "H1", "APPROVED", admin_id="ADM")
    assert part.can_join(storage, job(storage), "H1")
    assert "H1" in job(storage).expected_participants              # update validation checks this list
    assert storage.get_participant("JOB:H1").decided_by == "ADM"


def test_ineligible_hospital_cannot_request(storage):
    with pytest.raises(ValueError, match="not eligible.*missing class b"):
        part.request_participation(storage, job(storage), "H1", {"eligible": False, "dataset_reasons": ["missing class b"]})
    assert storage.get_participant("JOB:H1") is None


def test_rejection_removes_access_and_a_new_request_goes_back_to_pending(storage):
    open_round(storage)
    part.request_participation(storage, job(storage), "H1", ELIGIBLE)
    part.decide_participation(storage, job(storage), "H1", "APPROVED")
    rnd = storage.get_round("JOB_R1"); rnd.expected_participants.append("H1"); storage.save_round(rnd)

    part.decide_participation(storage, job(storage), "H1", "REJECTED", reason="Dataset too small")
    assert not part.can_join(storage, job(storage), "H1")
    assert "H1" not in job(storage).expected_participants
    assert "H1" not in storage.get_round("JOB_R1").expected_participants   # joined but sent nothing yet
    assert storage.get_participant("JOB:H1").decision_reason == "Dataset too small"

    p = part.request_participation(storage, job(storage), "H1", ELIGIBLE)
    assert p.authorization_status == "PENDING" and p.decision_reason == ""


def test_approved_hospital_can_refresh_its_summary_without_losing_approval(storage):
    part.request_participation(storage, job(storage), "H1", ELIGIBLE)
    part.decide_participation(storage, job(storage), "H1", "APPROVED")
    p = part.request_participation(storage, job(storage), "H1", {**ELIGIBLE, "train_sample_count": 900})
    assert p.authorization_status == "APPROVED" and p.eligibility["train_sample_count"] == 900


def test_hospitals_listed_before_requests_existed_count_as_approved(storage):
    j = job(storage); j.expected_participants.append("OLD"); storage.save_job(j)
    assert part.can_join(storage, job(storage), "OLD")
    listed = part.list_participants(storage, job(storage))
    assert [(p.hospital_identifier, p.authorization_status) for p in listed] == [("OLD", "APPROVED")]


def test_finished_jobs_take_no_requests_and_bad_decisions_are_refused(storage):
    with pytest.raises(ValueError, match="has not requested"):
        part.decide_participation(storage, job(storage), "H9", "APPROVED")
    part.request_participation(storage, job(storage), "H1", ELIGIBLE)
    with pytest.raises(ValueError, match="decision must be"):
        part.decide_participation(storage, job(storage), "H1", "MAYBE")
    j = job(storage); j.status = JobStatus.COMPLETED; storage.save_job(j)
    with pytest.raises(ValueError, match="does not accept hospitals"):
        part.request_participation(storage, job(storage), "H2", ELIGIBLE)


# ── adapter (what the backend calls) ──────────────────────────────────────────────────

def adapter(cwd, **req):
    out = subprocess.run([sys.executable, os.path.join(REPO, "federation", "api_adapter.py")], input=json.dumps(req),
                         capture_output=True, text=True, cwd=cwd, env={**os.environ, "PYTHONPATH": REPO}, check=True)
    return json.loads(out.stdout)


def test_adapter_refuses_round_join_until_approved(tmp_path):
    assert adapter(tmp_path, action="create_job", job_id="J", task="Malaria", architecture="mobilenet_v3_small",
                   num_classes=2, class_mapping={"a": 0, "b": 1}, minimum_participants=1)["success"]
    s = FederationStorage(str(tmp_path / "federation_metadata.db"), str(tmp_path / "federation_artifacts"))
    RoundManager(s).create_round("J", "J_R1", 1, [], 1, "2099-01-01T00:00:00+00:00", "B", 1, "c")

    res = adapter(tmp_path, action="register_participant", job_id="J", round_id="J_R1", participant_id="H1")
    assert not res["success"] and "has not requested" in res["error"] and "cannot join" in res["error"]

    assert adapter(tmp_path, action="request_participation", job_id="J", participant_id="H1", eligibility=ELIGIBLE)["success"]
    res = adapter(tmp_path, action="register_participant", job_id="J", round_id="J_R1", participant_id="H1")
    assert not res["success"] and "waiting for admin approval" in res["error"]

    jobs = adapter(tmp_path, action="list_jobs")["jobs"]
    assert jobs[0]["participants"][0]["authorization_status"] == "PENDING"
    assert jobs[0]["participants"][0]["eligibility"]["train_sample_count"] == 400

    assert adapter(tmp_path, action="decide_participation", job_id="J", participant_id="H1", decision="APPROVED")["success"]
    assert adapter(tmp_path, action="register_participant", job_id="J", round_id="J_R1", participant_id="H1")["success"]
    assert s.get_round("J_R1").expected_participants == ["H1"]


# ── evaluation with images that cannot be processed ───────────────────────────────────

def test_evaluation_stops_instead_of_scoring_only_the_images_that_worked(tmp_path):
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    cfg = ModelConfig(architecture="mobilenet_v3_small", num_classes=2, input_size=(224, 224), color_mode="RGB")
    params = get_parameters(build_model(cfg))
    gm_dir = str(tmp_path / "gm")
    build_federation_handoff(params, "GM-X-V1", 1, "mobilenet_v3_small", "B", 1, "c", 10, {"a": 0, "b": 1},
                             {}, {}, {}, "cpu", "fp32", "GLOBAL_MODEL_CREATED").save(gm_dir)
    storage.save_global_model(GlobalModelVersion(
        global_model_id="GM-X-V1", task="Malaria", task_type="image_classification", architecture="mobilenet_v3_small",
        version=1, artifact_location=gm_dir, artifact_checksum="", artifact_size=0, parameter_count=len(params),
        parameter_metadata={"num_classes": 2}, federation_job_id="JOB", round_id="R1", source_update_ids=[],
        aggregation_method="weighted_fedavg", aggregation_metadata={}, class_mapping={"a": 0, "b": 1}))
    from federation.models import GlobalModelStatus
    gm = storage.get_global_model("GM-X-V1"); gm.status = GlobalModelStatus.EVALUATION_PENDING; storage.save_global_model(gm)

    root = tmp_path / "eval" / "Malaria" / "v1"
    for cls in ("a", "b"):
        (root / cls).mkdir(parents=True)
        Image.new("RGB", (64, 64), "red").save(root / cls / "ok.png")
    (root / "b" / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"1 of 3 evaluation images could not be processed.*broken\.png"):
        EvaluationManager(storage, EventEmitter(), str(tmp_path / "eval")).evaluate("GM-X-V1")
    assert storage.get_latest_evaluation("GM-X-V1") is None          # no partial metrics were recorded
    with open(EvaluationManager(storage, EventEmitter()).progress_path("GM-X-V1")) as f:
        progress = json.load(f)                                       # the admin screen shows the failure
    assert progress["status"] == "failed" and "broken.png" in progress["error"] and progress["done"] == 3


def test_progress_file_that_cannot_be_replaced_does_not_stop_the_evaluation(tmp_path, monkeypatch):
    # Windows refuses os.replace while the backend is reading the progress file.
    def locked(*a, **k):
        raise PermissionError(5, "Access is denied")
    monkeypatch.setattr("federation.evaluation.os.replace", locked)
    monkeypatch.setattr("federation.evaluation.time.sleep", lambda s: None)
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    with pytest.raises(ValueError, match="not found"):     # the real error, not the progress write
        EvaluationManager(storage, EventEmitter(), str(tmp_path / "eval")).evaluate("GM-MISSING")
