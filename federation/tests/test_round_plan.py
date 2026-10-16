"""
Round plan of a federation job: strategy and FedProx settings, total rounds, early stopping,
and the next round opening automatically after aggregation (through the API adapter).
"""
import datetime
import json
import os
import subprocess
import sys

import pytest

from federation.base_models import base_model_store
from federation.round_plan import early_stopped, plan_status, validate_plan
from federation.storage import FederationStorage
from hospital_client.model_management.parameters import get_parameters
from hospital_client.training.federation import build_federation_handoff

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCH = "mobilenet_v3_small"
MAPPING = {"Parasitized": 0, "Uninfected": 1}


# ── validate_plan ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("agg,reqs", [
    ({}, {}),
    ({"strategy": "fedavg", "total_rounds": 3, "round_duration_hours": 24}, {}),
    ({"strategy": "fedprox"}, {"proximal_mu": 0.01}),
    ({"total_rounds": 5, "auto_continue": False}, {}),
    ({"total_rounds": 5, "round_duration_hours": 2, "early_stopping": {"metric": "f1", "min_delta": 0.01, "patience": 2}}, {}),
    ({"protocol": "manual"}, {}),
    ({"protocol": "flower", "total_rounds": 3, "round_duration_hours": 2}, {}),
])
def test_valid_plans(agg, reqs):
    validate_plan(agg, reqs)


@pytest.mark.parametrize("agg,reqs,msg", [
    ({"strategy": "fedsgd"}, {}, "strategy"),
    ({"strategy": "fedprox"}, {}, "proximal_mu"),
    ({"strategy": "fedprox"}, {"proximal_mu": 0}, "proximal_mu"),
    ({"strategy": "fedprox"}, {"proximal_mu": -1}, "proximal_mu"),
    ({}, {"proximal_mu": 0.01}, "only used"),
    ({"total_rounds": 0}, {}, "total_rounds"),
    ({"total_rounds": 2.5}, {}, "total_rounds"),
    ({"total_rounds": True}, {}, "total_rounds"),
    ({"total_rounds": 3}, {}, "round_duration_hours"),           # auto_continue defaults on with total_rounds
    ({"auto_continue": True}, {}, "auto_continue"),
    ({"total_rounds": 3, "auto_continue": False, "early_stopping": {"patience": 0}}, {}, "patience"),
    ({"total_rounds": 3, "auto_continue": False, "early_stopping": {"metric": "loss", "patience": 1}}, {}, "metric"),
    ({"total_rounds": 3, "auto_continue": False, "early_stopping": {"min_delta": -1, "patience": 1}}, {}, "min_delta"),
    ({"protocol": "p2p"}, {}, "protocol must be"),
    ({"protocol": "flower"}, {}, "Flower jobs need total_rounds"),
    ({"protocol": "flower", "total_rounds": 3}, {}, "round_duration_hours"),
    ({"protocol": "flower", "total_rounds": 3, "round_duration_hours": 2, "auto_continue": False}, {}, "Flower jobs run their rounds automatically"),
])
def test_invalid_plans_are_rejected(agg, reqs, msg):
    with pytest.raises(ValueError, match=msg):
        validate_plan(agg, reqs)


def test_protocol_defaults_to_manual_and_is_reported():
    assert plan_status({}, [], [])["protocol"] == "manual"
    assert plan_status({"protocol": "flower", "total_rounds": 2, "round_duration_hours": 1}, [], [])["protocol"] == "flower"


# ── stopping rules ────────────────────────────────────────────────────────────────────────

def test_early_stopping_counts_rounds_without_enough_improvement():
    assert not early_stopped([], 0.01, 1)
    assert not early_stopped([0.70, 0.75, 0.80], 0.01, 1)
    assert early_stopped([0.70, 0.80, 0.805], 0.01, 1)           # +0.005 < min_delta
    assert not early_stopped([0.70, 0.80, 0.805], 0.01, 2)
    assert early_stopped([0.70, 0.80, 0.79, 0.805], 0.01, 2)
    assert not early_stopped([0.70, 0.80, 0.79, 0.82], 0.01, 2)  # improvement resets the count


def rounds(*statuses):
    return [{"round_number": i + 1, "status": s} for i, s in enumerate(statuses)]


def test_plan_finishes_after_total_rounds():
    agg = {"total_rounds": 2, "round_duration_hours": 1}
    p = plan_status(agg, rounds("COMPLETED"), [])
    assert (p["rounds_completed"], p["next_round_number"], p["finished"]) == (1, 2, False)
    p = plan_status(agg, rounds("COMPLETED", "OPEN"), [])
    assert (p["next_round_number"], p["finished"]) == (None, False)   # round 2 exists; no round 3 planned
    p = plan_status(agg, rounds("COMPLETED", "EVALUATION_PENDING"), [])
    assert p["finished"] and "2 planned" in p["reason"]


def test_open_ended_plan_never_finishes_by_count():
    p = plan_status({}, rounds("COMPLETED", "COMPLETED", "COMPLETED"), [])
    assert (p["total_rounds"], p["next_round_number"], p["finished"], p["auto_continue"]) == (None, 4, False, False)


def test_plan_finishes_on_early_stopping():
    agg = {"total_rounds": 10, "round_duration_hours": 1, "early_stopping": {"metric": "accuracy", "min_delta": 0.01, "patience": 1}}
    p = plan_status(agg, rounds("COMPLETED", "COMPLETED"), [(2, 0.801), (1, 0.80)])
    assert p["finished"] and "Early stopping" in p["reason"] and p["next_round_number"] is None
    assert [s["round_number"] for s in p["scores"]] == [1, 2]


# ── adapter: two planned rounds, FedProx, automatic next round ────────────────────────────

def adapter(cwd, **req):
    out = subprocess.run([sys.executable, os.path.join(REPO, "federation", "api_adapter.py")], input=json.dumps(req),
                         capture_output=True, text=True, cwd=cwd, env={**os.environ, "PYTHONPATH": REPO}, check=True)
    return json.loads(out.stdout)


def future(hours=1):
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours)).isoformat()


ELIGIBLE = {"eligible": True, "task_match": True, "train_sample_count": 50}


def approve(cwd, hospital):
    """Hospitals join a job by request + admin approval (federation/participation.py)."""
    assert adapter(cwd, action="request_participation", job_id="JOB1", participant_id=hospital, eligibility=ELIGIBLE)["success"]
    assert adapter(cwd, action="decide_participation", job_id="JOB1", participant_id=hospital, decision="APPROVED")["success"]


def submit(cwd, round_id, hospital, mu):
    """A hospital update built from the round's base model, reporting the given proximal_mu."""
    storage = FederationStorage(db_path=str(cwd / "federation_metadata.db"), artifacts_dir=str(cwd / "federation_artifacts"))
    rnd = storage.get_round(round_id)
    model, _ = base_model_store(storage).load_checkpoint(rnd.base_model_id, rnd.base_model_version)
    d = str(cwd / f"upd_{round_id}_{hospital}")
    build_federation_handoff(get_parameters(model), f"LOCAL_{hospital}", 1, ARCH, rnd.base_model_id, rnd.base_model_version,
                             rnd.base_model_checksum, 50, MAPPING, {}, {}, {"proximal_mu": mu}, "cpu", "fp32",
                             "TRAINING_COMPLETED_AWAITING_FEDERATION").save(d)
    approve(cwd, hospital)
    assert adapter(cwd, action="register_participant", job_id="JOB1", round_id=round_id, participant_id=hospital)["success"]
    res = adapter(cwd, action="submit_update", job_id="JOB1", round_id=round_id, participant_id=hospital, handoff_dir=d)
    assert res["success"], res
    return res["update"]


def test_job_runs_its_planned_rounds_then_completes(tmp_path):
    bad = adapter(tmp_path, action="create_job", job_id="BAD", task="Malaria", architecture=ARCH, num_classes=2,
                  class_mapping=MAPPING, aggregation_configuration={"strategy": "fedprox"})
    assert not bad["success"] and "proximal_mu" in bad["error"]

    res = adapter(tmp_path, action="create_job", job_id="JOB1", task="Malaria", architecture=ARCH, num_classes=2,
                  class_mapping=MAPPING, minimum_participants=2, training_requirements={"proximal_mu": 0.01},
                  aggregation_configuration={"strategy": "fedprox", "total_rounds": 2, "round_duration_hours": 6})
    assert res["success"], res
    assert adapter(tmp_path, action="create_round", job_id="JOB1", round_id="JOB1_R1", round_number=1, deadline=future())["success"]

    # An update trained without the job's proximal term is rejected.
    assert submit(tmp_path, "JOB1_R1", "HOSP_X", 0.0)["status"] == "REJECTED"
    for h in ("HOSP_A", "HOSP_B"):
        assert submit(tmp_path, "JOB1_R1", h, 0.01)["status"] == "ACCEPTED"

    agg = adapter(tmp_path, action="aggregate", round_id="JOB1_R1", force_close=True)
    assert agg["success"], agg
    assert agg["global_model"]["aggregation_metadata"]["strategy"] == "fedprox"
    assert agg["plan"]["rounds_completed"] == 1 and not agg["plan"]["finished"]
    nxt = agg["next_round"]                                     # opened automatically from global model v1
    assert nxt["round_id"] == "JOB1_R2" and nxt["status"] == "OPEN"
    assert nxt["base_model_id"] == agg["global_model"]["global_model_id"]
    hours = (datetime.datetime.fromisoformat(nxt["deadline"]) - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 3600
    assert 5.9 < hours <= 6.0

    for h in ("HOSP_A", "HOSP_B"):
        assert submit(tmp_path, "JOB1_R2", h, 0.01)["status"] == "ACCEPTED"
    agg = adapter(tmp_path, action="aggregate", round_id="JOB1_R2", force_close=True)
    assert agg["success"], agg
    assert agg["plan"]["finished"] and agg["next_round"] is None

    job = adapter(tmp_path, action="list_jobs")["jobs"][0]
    assert job["status"] == "COMPLETED" and job["plan"]["rounds_completed"] == 2
    extra = adapter(tmp_path, action="create_round", job_id="JOB1", round_id="JOB1_R3", round_number=3, deadline=future())
    assert not extra["success"] and "finished" in extra["error"]


def test_an_archived_job_is_read_only(tmp_path):
    # Delete on a job with history archives it (the record stays); an archived job opens no more rounds.
    assert adapter(tmp_path, action="create_job", job_id="JOB1", task="Malaria", architecture=ARCH, num_classes=2,
                   class_mapping=MAPPING, minimum_participants=2)["success"]
    assert adapter(tmp_path, action="create_round", job_id="JOB1", round_id="JOB1_R1", round_number=1, deadline=future())["success"]
    for h in ("HOSP_A", "HOSP_B"):
        assert submit(tmp_path, "JOB1_R1", h, 0.0)["status"] == "ACCEPTED"
    assert adapter(tmp_path, action="aggregate", round_id="JOB1_R1", force_close=True)["success"]

    assert adapter(tmp_path, action="delete_job", job_id="JOB1")["action_taken"] == "archived"
    job = adapter(tmp_path, action="list_jobs")["jobs"][0]
    assert job["status"] == "ARCHIVED"
    assert job["plan"]["finished"] and "archived" in job["plan"]["reason"] and job["plan"]["next_round_number"] is None
    extra = adapter(tmp_path, action="create_round", job_id="JOB1", round_id="JOB1_R2", round_number=2, deadline=future())
    assert not extra["success"] and "archived" in extra["error"]
