import json
import os
import threading
import time
import flwr as fl
import numpy as np
import pytest
import torch
import torch.nn as nn

from federation import participation
from federation.events import EventEmitter
from federation.flower_app.server import serve
from federation.intake import IntakeManager
from federation.jobs import JobManager
from federation.models import UpdateStatus
from federation.security_hooks import SecurityDecision
from federation.storage import FederationStorage
from federation.validation import UpdateValidator
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters
from hospital_client.model_management.registry import build_model
from hospital_client.training.dp import compute_rdp_epsilon, load_dp_state, save_dp_state
from hospital_client.training.federation import build_federation_handoff

ARCH = "mobilenet_v3_small"
MAPPING = {"Parasitized": 0, "Uninfected": 1}


def _free_address():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{s.getsockname()[1]}"


def _sample_parameters():
    model = build_model(ModelConfig(architecture=ARCH, num_classes=2, input_size=(64, 64), norm_type="group"))
    return get_parameters(model)


class DummySecurityHook:
    def evaluate_update(self, update, handoff, job=None, rnd=None):
        return SecurityDecision(verdict="ACCEPT")


def test_dp_multi_round_accounting_equals_combined_history(tmp_path):
    """
    Privacy accounting across rounds: store one ledger row per accepted round
    and compute total epsilon by running the RDP accountant over the whole history.
    Never add per-round epsilons. Hospital dp_state.json keeps the same history.
    """
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    jm = JobManager(storage)
    job_id = "JOB_DP_MULTI"
    delta = 1e-5
    sigma = 1.2
    q = 0.05
    steps_r1 = 50
    steps_r2 = 60

    reqs = {
        "dp_enabled": True,
        "dp_noise_multiplier": sigma,
        "dp_max_grad_norm": 1.0,
        "dp_target_delta": delta,
        "dp_target_epsilon": 20.0,
        "dp_logical_batch_size": 2,  # q=0.05 * N=40
    }
    jm.create_job(
        federation_job_id=job_id,
        task="Malaria",
        task_type="image_classification",
        architecture=ARCH,
        num_classes=2,
        class_mapping=MAPPING,
        expected_participants=["H1"],
        minimum_participants=1,
        deadline="",
        training_requirements=reqs,
        aggregation_configuration={"protocol": "manual", "total_rounds": 2, "round_duration_hours": 1.0},
    )

    from federation.base_models import prepare_base_model
    job = storage.get_job(job_id)
    base_id, base_ver, base_chk = prepare_base_model(storage, job)

    from federation.rounds import RoundManager
    rm = RoundManager(storage)
    r1 = rm.create_round(
        federation_job_id=job_id,
        round_id=f"{job_id}_R1",
        round_number=1,
        expected_participants=["H1"],
        minimum_participants=1,
        deadline="2099-01-01T00:00:00Z",
        base_model_id=base_id,
        base_model_version=base_ver,
        base_model_checksum=base_chk,
    )
    rm.open_round(r1.round_id)

    validator = UpdateValidator(storage)
    intake = IntakeManager(storage, validator, DummySecurityHook(), EventEmitter())

    # Round 1 update
    eps_r1 = compute_rdp_epsilon([(sigma, q, steps_r1)], delta)
    params_r1 = _sample_parameters()
    h1 = build_federation_handoff(
        parameters=params_r1,
        model_id="H1_R1",
        model_version=1,
        architecture=ARCH,
        base_model_id=r1.base_model_id,
        base_model_version=r1.base_model_version,
        base_model_checksum=r1.base_model_checksum,
        num_train_samples=40,
        class_mapping=MAPPING,
        training_metrics={"loss": 0.5},
        validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps_r1,
            "dp_epsilon": eps_r1,
        },
        device="cpu",
        precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )

    h1_dir = str(tmp_path / "h1_handoff")
    h1.save(h1_dir)
    upd1 = intake.receive_handoff(job_id, r1.round_id, "H1", h1_dir)
    assert upd1.status == UpdateStatus.ACCEPTED

    # Verify server ledger after Round 1
    ledger1 = storage.get_dp_ledger(job_id, "H1")
    assert len(ledger1) == 1
    assert ledger1[0]["steps"] == steps_r1
    assert abs(ledger1[0]["round_epsilon"] - eps_r1) < 1e-4
    assert abs(storage.compute_cumulative_dp_epsilon(job_id, "H1", delta) - eps_r1) < 1e-4

    # Round 2
    r2 = rm.create_round(
        federation_job_id=job_id,
        round_id=f"{job_id}_R2",
        round_number=2,
        expected_participants=["H1"],
        minimum_participants=1,
        deadline="2099-01-01T00:00:00Z",
        base_model_id=base_id,
        base_model_version=base_ver,
        base_model_checksum=base_chk,
    )
    rm.open_round(r2.round_id)

    eps_r2 = compute_rdp_epsilon([(sigma, q, steps_r2)], delta)
    params_r2 = [p + np.asarray(0.001, dtype=p.dtype) if p.dtype.kind == "f" else p for p in params_r1]
    h2 = build_federation_handoff(
        parameters=params_r2,
        model_id="H1_R2",
        model_version=2,
        architecture=ARCH,
        base_model_id=r2.base_model_id,
        base_model_version=r2.base_model_version,
        base_model_checksum=r2.base_model_checksum,
        num_train_samples=40,
        class_mapping=MAPPING,
        training_metrics={"loss": 0.4},
        validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps_r2,
            "dp_epsilon": eps_r2,
        },
        device="cpu",
        precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )

    h2_dir = str(tmp_path / "h2_handoff")
    h2.save(h2_dir)
    upd2 = intake.receive_handoff(job_id, r2.round_id, "H1", h2_dir)
    assert upd2.status == UpdateStatus.ACCEPTED

    # Verify server ledger after Round 2
    ledger2 = storage.get_dp_ledger(job_id, "H1")
    assert len(ledger2) == 2
    assert ledger2[1]["steps"] == steps_r2

    # CRITICAL CHECK: Cumulative epsilon equals RDP accountant over combined history
    combined_history = [(sigma, q, steps_r1), (sigma, q, steps_r2)]
    expected_cumulative_eps = compute_rdp_epsilon(combined_history, delta)
    server_cumulative_eps = storage.compute_cumulative_dp_epsilon(job_id, "H1", delta)

    assert abs(server_cumulative_eps - expected_cumulative_eps) < 1e-4
    # Must NOT equal naive addition of per-round epsilons
    naive_sum = eps_r1 + eps_r2
    assert abs(server_cumulative_eps - naive_sum) > 0.01
    assert server_cumulative_eps < naive_sum  # RDP composition is strictly tighter than simple addition

    # Verify hospital dp_state.json mirrors the same history and computes the same epsilon
    hospital_dp_state_path = str(tmp_path / "hospital_dp_state.json")
    hospital_state = {
        "job_id": job_id,
        "history": [
            {"round_id": r1.round_id, "noise_multiplier": sigma, "sample_rate": q, "steps": steps_r1, "delta": delta},
            {"round_id": r2.round_id, "noise_multiplier": sigma, "sample_rate": q, "steps": steps_r2, "delta": delta},
        ],
    }
    save_dp_state(hospital_dp_state_path, hospital_state)
    loaded_state = load_dp_state(hospital_dp_state_path, job_id)
    hospital_history = [(e["noise_multiplier"], e["sample_rate"], e["steps"]) for e in loaded_state["history"]]
    hospital_cumulative_eps = compute_rdp_epsilon(hospital_history, delta)
    assert abs(hospital_cumulative_eps - server_cumulative_eps) < 1e-6


class FakeFlowerDPHospital(fl.client.NumPyClient):
    """Flower client that reports DP training handoff metadata."""

    def __init__(self, hid, sigma=1.0, q=0.05, steps=50, delta=1e-5):
        self.hid = hid
        self.sigma, self.q, self.steps, self.delta = sigma, q, steps, delta
        self.seen = []

    def get_properties(self, config):
        return {"hospital_id": self.hid, "token": f"tok-{self.hid}"}

    def fit(self, parameters, config):
        self.seen.append(dict(config))
        # DP jobs should NEVER request calibration
        if config.get("phase") == "calibrate":
            raise RuntimeError("Received calibrate call on a DP job!")

        params = [p + np.asarray(0.001, dtype=p.dtype) if p.dtype.kind == "f" else p for p in parameters]
        round_eps = compute_rdp_epsilon([(self.sigma, self.q, self.steps)], self.delta)
        h = build_federation_handoff(
            parameters=params,
            model_id=f"FED_{self.hid}_{config['fl_round']}",
            model_version=1,
            architecture=ARCH,
            base_model_id=config["base_model_id"],
            base_model_version=int(config["base_model_version"]),
            base_model_checksum=config["base_model_checksum"],
            num_train_samples=40,
            class_mapping=MAPPING,
            training_metrics={"loss": 0.4},
            validation_metrics={},
            training_configuration={
                "dp_enabled": True,
                "dp_noise_multiplier": self.sigma,
                "dp_max_grad_norm": 1.0,
                "dp_target_delta": self.delta,
                "dp_sample_rate": self.q,
                "dp_steps": self.steps,
                "dp_epsilon": round_eps,
            },
            device="cpu",
            precision="fp32",
            status="TRAINING_COMPLETED_AWAITING_FEDERATION",
        )
        return params, 40, {"hospital_id": self.hid, "token": f"tok-{self.hid}", "handoff_metadata": json.dumps(h.to_dict())}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}


def test_flower_dp_update_charged_to_ledger_exactly_once(tmp_path):
    """
    A Flower DP update is charged to the ledger exactly once.
    (Tests removal of duplicate storage.update_dp_budget() from flower_app/server.py).
    Also confirms client never receives a calibrate call for DP jobs.
    """
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    jm = JobManager(storage)
    job_id = "JOB_FLOWER_DP"
    delta = 1e-5
    sigma = 1.0

    reqs = {
        "dp_enabled": True,
        "dp_noise_multiplier": sigma,
        "dp_max_grad_norm": 1.0,
        "dp_target_delta": delta,
        "dp_target_epsilon": 25.0,
        "dp_logical_batch_size": 2,  # q=0.05 * N=40
    }
    jm.create_job(
        federation_job_id=job_id,
        task="Malaria",
        task_type="image_classification",
        architecture=ARCH,
        num_classes=2,
        class_mapping=MAPPING,
        expected_participants=["H1"],
        minimum_participants=1,
        deadline="",
        training_requirements=reqs,
        aggregation_configuration={"protocol": "flower", "total_rounds": 1, "round_duration_hours": 1.0},
    )

    participation.request_participation(storage, storage.get_job(job_id), "H1", {"eligible": True})
    participation.decide_participation(storage, storage.get_job(job_id), "H1", "APPROVED")

    address = _free_address()
    status_path = str(tmp_path / "flower" / "status.json")
    client = FakeFlowerDPHospital("H1", sigma=sigma, q=0.05, steps=50, delta=delta)

    thread = threading.Thread(
        target=lambda: (time.sleep(1), fl.client.start_client(server_address=address, client=client.to_client(), insecure=True)),
        daemon=True,
    )
    thread.start()

    final = serve(
        storage, job_id, address, status_path,
        verify_identity=lambda hid, tok: tok == f"tok-{hid}",
        close_path=str(tmp_path / "flower" / "close_round"),
        poll_seconds=0.2,
    )
    thread.join(timeout=30)

    # 1. Check client NEVER received calibrate call
    assert len(client.seen) == 1
    assert client.seen[0].get("phase") != "calibrate"
    assert client.seen[0].get("fl_round") == 1

    # 2. Check DP ledger has EXACTLY ONE entry for H1
    ledger = storage.get_dp_ledger(job_id, "H1")
    assert len(ledger) == 1, f"Expected exactly 1 ledger row, got {len(ledger)}"
    assert ledger[0]["steps"] == 50
    assert abs(ledger[0]["noise_multiplier"] - sigma) < 1e-4

    # 3. Check cumulative epsilon equals the round epsilon
    expected_eps = compute_rdp_epsilon([(sigma, 0.05, 50)], delta)
    assert abs(ledger[0]["cumulative_epsilon"] - expected_eps) < 1e-4


# ── 3a: Server rejects updates with missing / mismatched DP fields ────────────

def _make_valid_dp_handoff(job, rnd, sigma, q, steps, delta, params):
    """Build a valid DP FederationHandoff for the given round."""
    round_eps = compute_rdp_epsilon([(sigma, q, steps)], delta)
    return build_federation_handoff(
        parameters=params,
        model_id=f"MODEL_{rnd.round_number}",
        model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id,
        base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40,
        class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4},
        validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps,
            "dp_epsilon": round_eps,
        },
        device="cpu",
        precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )


@pytest.fixture()
def dp_job_setup(tmp_path):
    """Fixture: a DP job + open round + storage + intake for rejection tests."""
    from federation.base_models import prepare_base_model
    from federation.jobs import JobManager
    from federation.rounds import RoundManager

    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    jm = JobManager(storage)
    job_id = "JOB_REJECT"
    sigma, delta = 1.1, 1e-5
    reqs = {
        "dp_enabled": True,
        "dp_noise_multiplier": sigma,
        "dp_max_grad_norm": 1.0,
        "dp_target_delta": delta,
        "dp_target_epsilon": 20.0,
        "dp_logical_batch_size": 2,  # q=0.05 * N=40
    }
    jm.create_job(
        federation_job_id=job_id, task="T", task_type="image_classification",
        architecture=ARCH, num_classes=2, class_mapping=MAPPING,
        expected_participants=["H1"], minimum_participants=1, deadline="",
        training_requirements=reqs,
        aggregation_configuration={"protocol": "manual", "total_rounds": 1, "round_duration_hours": 1.0},
    )
    job = storage.get_job(job_id)
    base_id, base_ver, base_chk = prepare_base_model(storage, job)
    rm = RoundManager(storage)
    rnd = rm.create_round(
        federation_job_id=job_id, round_id=f"{job_id}_R1", round_number=1,
        expected_participants=["H1"], minimum_participants=1,
        deadline="2099-01-01T00:00:00Z",
        base_model_id=base_id, base_model_version=base_ver, base_model_checksum=base_chk,
    )
    rm.open_round(rnd.round_id)
    validator = UpdateValidator(storage)
    intake = IntakeManager(storage, validator, DummySecurityHook(), EventEmitter())
    params = _sample_parameters()
    return storage, intake, job, storage.get_round(rnd.round_id), sigma, delta, params, tmp_path


def test_server_rejects_update_with_dp_disabled(dp_job_setup):
    """Server rejects an update that does not report dp_enabled for a DP job."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    q, steps = 0.05, 50
    handoff = build_federation_handoff(
        parameters=params, model_id="NO_DP", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            # dp_enabled absent — simulates hospital training without DP
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
        },
        device="cpu", precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_no_dp")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "differential privacy" in upd.rejection_reason.lower() or "dp" in upd.rejection_reason.lower()


def test_server_rejects_update_with_mismatched_sigma(dp_job_setup):
    """Server rejects an update with a different noise_multiplier than the job requires."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    q, steps = 0.05, 50
    wrong_sigma = sigma + 0.5
    round_eps = compute_rdp_epsilon([(wrong_sigma, q, steps)], delta)
    handoff = build_federation_handoff(
        parameters=params, model_id="WRONG_SIGMA", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": wrong_sigma,   # WRONG
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps,
            "dp_epsilon": round_eps,
        },
        device="cpu", precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_bad_sigma")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "noise_multiplier" in upd.rejection_reason


def test_server_rejects_update_with_mismatched_epsilon(dp_job_setup):
    """Server rejects an update whose reported epsilon does not match server recomputation."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    q, steps = 0.05, 50
    correct_eps = compute_rdp_epsilon([(sigma, q, steps)], delta)
    handoff = build_federation_handoff(
        parameters=params, model_id="WRONG_EPS", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps,
            "dp_epsilon": correct_eps + 2.0,   # WRONG — inflated by 2
        },
        device="cpu", precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_bad_eps")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "epsilon" in upd.rejection_reason.lower()


def test_server_rejects_update_with_mismatched_delta(dp_job_setup):
    """Server rejects an update that reports a different delta than the job requires."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    q, steps = 0.05, 50
    wrong_delta = 1e-3   # job uses 1e-5
    round_eps = compute_rdp_epsilon([(sigma, q, steps)], wrong_delta)
    handoff = build_federation_handoff(
        parameters=params, model_id="WRONG_DELTA", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": wrong_delta,    # WRONG
            "dp_sample_rate": q,
            "dp_steps": steps,
            "dp_epsilon": round_eps,
        },
        device="cpu", precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_bad_delta")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "delta" in upd.rejection_reason.lower()


def test_server_rejects_update_with_mismatched_max_grad_norm(dp_job_setup):
    """Server rejects an update that reports a different dp_max_grad_norm than required."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    q, steps = 0.05, 50
    round_eps = compute_rdp_epsilon([(sigma, q, steps)], delta)
    handoff = build_federation_handoff(
        parameters=params, model_id="WRONG_C", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 0.5,    # Job requires 1.0
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps,
            "dp_epsilon": round_eps,
        },
        device="cpu", precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_bad_c")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "dp_max_grad_norm" in upd.rejection_reason.lower() or "max grad norm" in upd.rejection_reason.lower()


def test_server_rejects_update_with_non_fp32_precision(dp_job_setup):
    """Server rejects an update for a DP job that was trained in FP16."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    q, steps = 0.05, 50
    round_eps = compute_rdp_epsilon([(sigma, q, steps)], delta)
    handoff = build_federation_handoff(
        parameters=params, model_id="FP16_DP", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": q,
            "dp_steps": steps,
            "dp_epsilon": round_eps,
            "precision": "fp16",
        },
        device="cuda", precision="fp16",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_fp16")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "fp32" in upd.rejection_reason.lower()


def test_server_rejects_update_with_mismatched_sample_rate(dp_job_setup):
    """Server rejects an update when reported q does not match logical_batch_size / sample_count."""
    storage, intake, job, rnd, sigma, delta, params, tmp_path = dp_job_setup
    job.training_requirements["dp_logical_batch_size"] = 10
    storage.save_job(job)
    # Expected q = 10 / 40 = 0.25; client reports 0.05
    round_eps = compute_rdp_epsilon([(sigma, 0.05, 50)], delta)
    handoff = build_federation_handoff(
        parameters=params, model_id="WRONG_Q", model_version=1,
        architecture=job.architecture,
        base_model_id=rnd.base_model_id, base_model_version=rnd.base_model_version,
        base_model_checksum=rnd.base_model_checksum,
        num_train_samples=40, class_mapping=job.class_mapping,
        training_metrics={"loss": 0.4}, validation_metrics={},
        training_configuration={
            "dp_enabled": True,
            "dp_noise_multiplier": sigma,
            "dp_max_grad_norm": 1.0,
            "dp_target_delta": delta,
            "dp_sample_rate": 0.05,  # Mismatched! Expected 0.25
            "dp_steps": 50,
            "dp_epsilon": round_eps,
        },
        device="cpu", precision="fp32",
        status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    )
    hdir = str(tmp_path / "h_bad_q")
    handoff.save(hdir)
    upd = intake.receive_handoff(job.federation_job_id, rnd.round_id, "H1", hdir)
    assert upd.status == UpdateStatus.REJECTED
    assert "sample rate" in upd.rejection_reason.lower()


# ── 3b: round_plan DP range checks and FedProx+DP rejection ──────────────────

def test_round_plan_rejects_fedprox_with_dp():
    """round_plan.validate_plan must refuse strategy='fedprox' combined with dp_enabled."""
    from federation.round_plan import validate_plan
    with pytest.raises(ValueError, match="fedprox"):
        validate_plan(
            {"strategy": "fedprox", "protocol": "manual"},
            {"proximal_mu": 0.1, "dp_enabled": True, "dp_noise_multiplier": 1.1,
             "dp_max_grad_norm": 1.0, "dp_target_delta": 1e-5},
        )


def test_round_plan_rejects_zero_sigma():
    """round_plan.validate_plan must refuse dp_noise_multiplier <= 0."""
    from federation.round_plan import validate_plan
    with pytest.raises(ValueError, match="dp_noise_multiplier"):
        validate_plan(
            {"strategy": "fedavg", "protocol": "manual"},
            {"dp_enabled": True, "dp_noise_multiplier": 0.0,
             "dp_max_grad_norm": 1.0, "dp_target_delta": 1e-5},
        )


def test_round_plan_rejects_invalid_delta():
    """round_plan.validate_plan must refuse dp_target_delta outside (0, 1)."""
    from federation.round_plan import validate_plan
    with pytest.raises(ValueError, match="dp_target_delta"):
        validate_plan(
            {"strategy": "fedavg", "protocol": "manual"},
            {"dp_enabled": True, "dp_noise_multiplier": 1.1,
             "dp_max_grad_norm": 1.0, "dp_target_delta": 1.5},   # > 1
        )


def test_round_plan_rejects_negative_target_epsilon():
    """round_plan.validate_plan must refuse dp_target_epsilon <= 0."""
    from federation.round_plan import validate_plan
    with pytest.raises(ValueError, match="dp_target_epsilon"):
        validate_plan(
            {"strategy": "fedavg", "protocol": "manual"},
            {"dp_enabled": True, "dp_noise_multiplier": 1.1,
             "dp_max_grad_norm": 1.0, "dp_target_delta": 1e-5,
             "dp_logical_batch_size": 64,
             "dp_target_epsilon": -1.0},
        )


def test_round_plan_rejects_missing_logical_batch_size():
    """round_plan.validate_plan must refuse a DP plan without dp_logical_batch_size.
    This guarantees the server's sample-rate check (q = B/N) always has a concrete
    B and never depends on a per-hospital Module 8 batch size choice.
    """
    from federation.round_plan import validate_plan
    with pytest.raises(ValueError, match="dp_logical_batch_size"):
        validate_plan(
            {"strategy": "fedavg", "protocol": "manual"},
            {"dp_enabled": True, "dp_noise_multiplier": 1.1,
             "dp_max_grad_norm": 1.0, "dp_target_delta": 1e-5,
             "dp_target_epsilon": 10.0},
            # dp_logical_batch_size intentionally omitted
        )


def test_round_plan_accepts_valid_dp_plan():
    """round_plan.validate_plan accepts a fully valid DP plan (smoke test)."""
    from federation.round_plan import validate_plan
    validate_plan(
        {"strategy": "fedavg", "protocol": "flower", "total_rounds": 2, "round_duration_hours": 1.0},
        {"dp_enabled": True, "dp_noise_multiplier": 1.1,
         "dp_max_grad_norm": 1.0, "dp_target_delta": 1e-5,
         "dp_logical_batch_size": 64, "dp_target_epsilon": 10.0},
    )


# ── 3c: Flower DP job never sends a calibrate fit config ─────────────────────
# Covered by test_flower_dp_update_charged_to_ledger_exactly_once above.
# FakeFlowerDPHospital.fit raises RuntimeError on phase=="calibrate" and the
# test asserts client.seen[0]["phase"] != "calibrate".  Explicitly re-stated:

def test_flower_dp_configure_fit_never_produces_calibrate_phase(tmp_path):
    """
    AppStrategy.configure_fit for a DP job must always call _configure_train,
    never _configure_calibrate.  Verified by inspecting every FitIns received
    by the client across a 1-round Flower DP job.
    """
    storage = FederationStorage(str(tmp_path / "fed2.db"), str(tmp_path / "art2"))
    jm = JobManager(storage)
    job_id = "JOB_CAL_CHECK"
    delta = 1e-5
    sigma = 1.0
    reqs = {
        "dp_enabled": True,
        "dp_noise_multiplier": sigma,
        "dp_max_grad_norm": 1.0,
        "dp_target_delta": delta,
        "dp_target_epsilon": 25.0,
        "dp_logical_batch_size": 2,  # q=0.05 * N=40
    }
    jm.create_job(
        federation_job_id=job_id, task="T", task_type="image_classification",
        architecture=ARCH, num_classes=2, class_mapping=MAPPING,
        expected_participants=["H2"], minimum_participants=1, deadline="",
        training_requirements=reqs,
        aggregation_configuration={"protocol": "flower", "total_rounds": 1, "round_duration_hours": 1.0},
    )
    participation.request_participation(storage, storage.get_job(job_id), "H2", {"eligible": True})
    participation.decide_participation(storage, storage.get_job(job_id), "H2", "APPROVED")

    address = _free_address()
    status_path = str(tmp_path / "flower2" / "status.json")
    client = FakeFlowerDPHospital("H2", sigma=sigma, q=0.05, steps=50, delta=delta)

    thread = threading.Thread(
        target=lambda: (time.sleep(1), fl.client.start_client(
            server_address=address, client=client.to_client(), insecure=True)),
        daemon=True,
    )
    thread.start()
    serve(
        storage, job_id, address, status_path,
        verify_identity=lambda hid, tok: tok == f"tok-{hid}",
        close_path=str(tmp_path / "flower2" / "close_round"),
        poll_seconds=0.2,
    )
    thread.join(timeout=30)

    # Every FitIns the client received must have phase="train", never "calibrate"
    assert len(client.seen) >= 1
    for fit_config in client.seen:
        assert fit_config.get("phase") != "calibrate", (
            f"DP job sent a calibrate FitIns: {fit_config}"
        )


# ── 3d: Job creation refuses architecture that fails ModuleValidator ──────────

class _StubUnfixableLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1))


class _StubUnfixableModule(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unfixable = _StubUnfixableLayer()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        out = self.pool(self.conv(x)).flatten(1)
        return self.fc(out)


def _build_stub_bad(config):
    return _StubUnfixableModule(config.num_classes)


def test_jobs_refuses_architecture_failing_module_validator(tmp_path):
    """
    JobManager.create_job must refuse a DP job whose architecture cannot be made
    Opacus-compatible (i.e., ModuleValidator still reports errors after fix()).
    Uses a stub architecture registered temporarily that always fails validation.
    """
    from opacus.validators import ModuleValidator
    from opacus.validators.errors import UnsupportedModuleError
    from federation.jobs import JobManager
    from hospital_client.model_management import registry as reg

    stub_arch = "_test_bad_arch_dp"
    reg.ARCHITECTURE_REGISTRY[stub_arch] = _build_stub_bad
    reg.ARCHITECTURE_CATALOG[stub_arch] = reg.ArchitectureInfo(
        stub_arch, "Bad Arch", "test", reg.ResourceTier.LOW_END,
        supports_pretrained=False, default_input_size=(64, 64),
    )
    # Register validator with no corresponding fixer so ModuleValidator.fix() leaves it invalid
    ModuleValidator.VALIDATORS[_StubUnfixableLayer] = lambda m: [
        UnsupportedModuleError("Stub unfixable layer is incompatible with DP")
    ]

    storage = FederationStorage(str(tmp_path / "fed3.db"), str(tmp_path / "art3"))
    jm = JobManager(storage)
    try:
        with pytest.raises(ValueError, match="(?i)opacus|ModuleValidator|compatible|dp"):
            jm.create_job(
                federation_job_id="JOB_BAD_ARCH",
                task="T", task_type="image_classification",
                architecture=stub_arch, num_classes=2, class_mapping=MAPPING,
                expected_participants=["H1"], minimum_participants=1, deadline="",
                training_requirements={
                    "dp_enabled": True,
                    "dp_noise_multiplier": 1.1,
                    "dp_max_grad_norm": 1.0,
                    "dp_target_delta": 1e-5,
                    "dp_logical_batch_size": 64,
                    "norm_type": "group",
                },
                aggregation_configuration={"protocol": "manual", "total_rounds": 1, "round_duration_hours": 1.0},
            )
    finally:
        ModuleValidator.VALIDATORS.pop(_StubUnfixableLayer, None)
        reg.ARCHITECTURE_REGISTRY.pop(stub_arch, None)
        reg.ARCHITECTURE_CATALOG.pop(stub_arch, None)

