"""
Hospital side of a Flower job: the connection answers the server with the operator's current
identity, runs the normal training job when asked, reports each step, gives up when the server
closes the round, and returns BatchNorm statistics in the calibration step.
"""
import threading
import time

import numpy as np
import pytest

from hospital_client.federation.flower_client import FlowerConnection

TRAIN = {"phase": "train", "job_id": "JOB", "round_id": "JOB_R2", "fl_round": 2, "total_rounds": 3,
         "base_model_id": "GM-1", "base_model_version": 1, "base_model_checksum": "abc"}
CAL = {"phase": "calibrate", "job_id": "JOB", "round_id": "JOB_R2", "fl_round": 2, "global_model_id": "GM-2"}
PARAMS = [np.ones(3, np.float32)]


def conn_with(run_training=None, calibrate=None, report=None, token="tok-1"):
    states = []
    c = FlowerConnection("HOSP_1", "JOB", "127.0.0.1:1", token,
                         run_training=run_training or (lambda p, cfg, cancel, on_job: (p, 40, {"model_id": "M"})),
                         calibrate=calibrate or (lambda p, cfg: (p, 40)),
                         report=report, poll_seconds=0.05)
    c._on_change = lambda st: states.append(st["state"])
    return c, states


def test_identity_is_the_operators_current_token_and_never_shown_in_status():
    c, _ = conn_with(token="old")
    assert c.client().get_properties({}) == {"hospital_id": "HOSP_1", "token": "old"}
    c.update_token("new")
    assert c.client().get_properties({})["token"] == "new"
    assert "new" not in str(c.status()) and "old" not in str(c.status())


def test_train_round_runs_the_training_job_and_reports_each_step():
    def run(p, cfg, cancel, on_job):
        on_job("job123")
        assert c.status()["state"] == "training" and c.status()["training_job_id"] == "job123"
        return [p[0] * 2], 40, {"model_id": "FED_JOB"}

    c, states = conn_with(run_training=run)
    params, n, metrics = c.client().fit(PARAMS, TRAIN)
    assert n == 40 and np.allclose(params[0], 2.0)
    assert metrics["hospital_id"] == "HOSP_1" and metrics["token"] == "tok-1" and '"model_id": "FED_JOB"' in metrics["handoff_metadata"]
    steps = [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]   # one entry per step
    assert steps[:3] == ["received_model", "training", "sending"]
    st = c.status()
    assert (st["fl_round"], st["total_rounds"], st["round_id"]) == (2, 3, "JOB_R2")


def test_training_stops_when_the_server_closes_the_round():
    reports = []

    def report(st):
        reports.append(st["state"])
        return {"round_id": "JOB_R2", "closing": len(reports) > 2}

    def run(p, cfg, cancel, on_job):
        assert cancel.wait(5), "the connection never asked the training to stop"
        raise RuntimeError("cancelled")

    c, states = conn_with(run_training=run, report=report)
    with pytest.raises(RuntimeError, match="closed"):
        c.client().fit(PARAMS, TRAIN)
    assert c.status()["state"] == "missed" and "closed" in c.status()["message"]


def test_the_backend_hears_the_update_was_sent_after_training_ends():
    # The admin's "updates sent" count and Aggregate now depend on this report.
    reports = []
    c, _ = conn_with(report=lambda st: reports.append(st["state"]) or {"round_id": "JOB_R2", "closing": False})
    c.client().fit(PARAMS, TRAIN)
    deadline = time.time() + 5
    while "waiting_for_others" not in reports and time.time() < deadline:
        time.sleep(0.05)
    assert "waiting_for_others" in reports, reports


def test_a_failed_training_is_reported_with_its_reason():
    def run(p, cfg, cancel, on_job):
        raise RuntimeError("A training run is already in progress on this machine.")

    c, _ = conn_with(run_training=run)
    with pytest.raises(RuntimeError):
        c.client().fit(PARAMS, TRAIN)
    assert c.status()["state"] == "failed" and "already in progress" in c.status()["message"]


def test_calibration_step_returns_statistics_and_marks_the_round_done():
    c, states = conn_with(calibrate=lambda p, cfg: ([p[0] + 1], 40))
    params, n, metrics = c.client().fit(PARAMS, CAL)
    assert np.allclose(params[0], 2.0) and n == 40 and metrics == {"hospital_id": "HOSP_1"}
    assert states == ["calibrating", "done"]


def test_a_stopped_connection_takes_no_more_work():
    c, _ = conn_with()
    c.stop()
    assert c.status()["state"] == "disconnected"
    with pytest.raises(RuntimeError, match="disconnected"):
        c.client().fit(PARAMS, TRAIN)
