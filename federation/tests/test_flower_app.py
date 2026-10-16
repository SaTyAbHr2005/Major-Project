"""
Flower server for an application job (federation/flower_app): real Flower gRPC server and
in-process hospital clients. Every Flower round is a Module 9 round (base model, intake
validation, aggregation, BatchNorm calibration, round plan) and progress is written to a
status file the backend serves to the UIs.
"""
import json
import os
import threading
import time

import flwr as fl
import numpy as np
import pytest

from federation import participation
from federation.api_adapter import get_all_global_models
from federation.flower_app.server import serve
from federation.jobs import JobManager
from federation.storage import FederationStorage
from hospital_client.training.federation import FederationHandoff, build_federation_handoff

ARCH = "mobilenet_v3_small"
MAPPING = {"Parasitized": 0, "Uninfected": 1}


def free_address():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{s.getsockname()[1]}"


class FakeHospital(fl.client.NumPyClient):
    """Stands in for the hospital service: 'trains' by nudging the weights and reports a valid handoff."""

    def __init__(self, hid, keys, behaviour=None):
        self.hid, self.keys, self.behaviour = hid, keys, behaviour or {}
        self.seen = []

    def get_properties(self, config):
        return {"hospital_id": self.hid, "token": f"tok-{self.hid}"}

    def fit(self, parameters, config):
        self.seen.append(dict(config))
        if config["phase"] == "calibrate":
            k = float(self.behaviour.get("bn", 1.0))
            out = [np.full_like(p, k) if key.endswith(("running_mean", "running_var")) else p
                   for p, key in zip(parameters, self.keys)]
            return out, 40, {"hospital_id": self.hid}
        if "train" in self.behaviour:
            self.behaviour["train"](config)
        params = [p + np.asarray(0.001, dtype=p.dtype) if p.dtype.kind == "f" else p for p in parameters]
        h = build_federation_handoff(
            params, f"FED_{self.hid}_{config['fl_round']}", 1, ARCH, config["base_model_id"], int(config["base_model_version"]),
            config["base_model_checksum"], 40, MAPPING, {"loss": 0.4}, {"accuracy": 0.9},
            {"epochs": 1, "output_dir": "C:\\Users\\op\\trained"}, "cpu", "fp32", "TRAINING_COMPLETED_AWAITING_FEDERATION")
        return params, 40, {"hospital_id": self.hid, "token": f"tok-{self.hid}", "handoff_metadata": json.dumps(h.to_dict())}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}


def state_keys():
    from hospital_client.model_management.config import ModelConfig
    from hospital_client.model_management.registry import build_model
    return list(build_model(ModelConfig(architecture=ARCH, num_classes=2, input_size=(224, 224), color_mode="RGB")).state_dict())


def make_job(tmp_path, approved, minimum=2, total_rounds=2, hours=1.0):
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    JobManager(storage).create_job("JOB", "Malaria", "image_classification", ARCH, 2, MAPPING, [], minimum, "", {},
                                   {"protocol": "flower", "total_rounds": total_rounds, "round_duration_hours": hours})
    for h in approved:
        participation.request_participation(storage, storage.get_job("JOB"), h, {"eligible": True})
        participation.decide_participation(storage, storage.get_job("JOB"), h, "APPROVED")
    return storage


def run(tmp_path, storage, hospitals, behaviours=None):
    address = free_address()
    status_path = str(tmp_path / "flower" / "status.json")
    keys = state_keys()
    clients = {h: FakeHospital(h, keys, (behaviours or {}).get(h)) for h in hospitals}
    threads = [threading.Thread(target=lambda c=c: (time.sleep(1), fl.client.start_client(
        server_address=address, client=c.to_client(), insecure=True)), daemon=True) for c in clients.values()]
    for t in threads:
        t.start()
    final = serve(storage, "JOB", address, status_path, verify_identity=lambda hid, tok: tok == f"tok-{hid}",
                  close_path=str(tmp_path / "flower" / "close_round"), poll_seconds=0.2)
    for t in threads:
        t.join(timeout=30)
    return final, clients, status_path


def test_two_rounds_over_flower_produce_two_m9_global_models(tmp_path):
    storage = make_job(tmp_path, ["H1", "H2"])
    final, clients, status_path = run(tmp_path, storage, ["H1", "H2", "H3"],
                                      {"H1": {"bn": 2.0}, "H2": {"bn": 4.0}})
    rounds = storage.list_rounds_for_job("JOB")
    assert [r.round_number for r in rounds] == [1, 2]
    gms = sorted((m for m in get_all_global_models(storage) if m["federation_job_id"] == "JOB"), key=lambda m: m["version"])
    assert [m["version"] for m in gms] == [1, 2]
    used = storage.list_updates_for_round("JOB_R2")
    assert sorted(u.participant_id for u in used) == ["H1", "H2"]
    assert all(u.status.value == "USED_IN_AGGREGATION" for u in used)
    assert storage.get_job("JOB").status.value == "COMPLETED"
    assert clients["H3"].seen == []                                      # not approved: never asked to train

    st = json.load(open(status_path))
    assert st["state"] == "finished" and st["fl_round"] == 2 and st["total_rounds"] == 2
    assert st["hospitals"]["H1"]["state"] == "done" and st["hospitals"]["H2"]["state"] == "done"
    assert "token" not in json.dumps(st) and "tok-" not in json.dumps(st)
    types = [e["type"] for e in st["events"]]
    assert types.count("round_opened") == 2 and types.count("global_model_created") == 2

    # BatchNorm statistics of v1 are the clients' calibrated ones (2.0 and 4.0, equal sample counts) …
    keys = state_keys()
    v1 = FederationHandoff.load(gms[0]["artifact_location"])
    i = next(i for i, k in enumerate(keys) if k.endswith("running_mean"))
    assert np.allclose(v1.parameters[i], 3.0)
    # … and round 2 started from that calibrated model.
    assert storage.get_round("JOB_R2").base_model_checksum == v1.parameters_checksum
    # Hospital folder paths never reach the server.
    upd = FederationHandoff.load(used[0].artifact_location)
    assert "output_dir" not in upd.training_configuration


def test_round_fails_when_too_few_approved_hospitals_connect_before_the_deadline(tmp_path):
    storage = make_job(tmp_path, ["H1"], minimum=2, total_rounds=1, hours=4 / 3600)
    final, clients, status_path = run(tmp_path, storage, ["H1"])
    assert storage.get_round("JOB_R1").status.value == "INSUFFICIENT_PARTICIPANTS"
    assert [m for m in get_all_global_models(storage) if m["federation_job_id"] == "JOB"] == []
    st = json.load(open(status_path))
    assert st["state"] == "failed" and "approved hospital" in st["message"]


def test_admin_aggregate_now_closes_round_with_the_updates_sent(tmp_path):
    storage = make_job(tmp_path, ["H1", "H2"], minimum=1, total_rounds=1)
    close_path = tmp_path / "flower" / "close_round"

    def h1_done(config):          # H1 finishes, then the admin clicks Aggregate now
        threading.Timer(1.0, lambda: close_path.write_text(config["round_id"])).start()

    def h2_slow(config):          # H2 trains until the server says the round is closing, then gives up
        status = tmp_path / "flower" / "status.json"
        for _ in range(200):
            if status.exists() and json.loads(status.read_text()).get("closing"):
                raise RuntimeError("Round closed early by the admin")
            time.sleep(0.1)

    final, clients, status_path = run(tmp_path, storage, ["H1", "H2"], {"H1": {"train": h1_done}, "H2": {"train": h2_slow}})
    assert sorted(u.participant_id for u in storage.list_updates_for_round("JOB_R1")) == ["H1"]
    st = json.load(open(status_path))
    assert st["state"] == "finished" and st["hospitals"]["H2"]["state"] == "missed"
    assert "round_closing" in [e["type"] for e in st["events"]]
    assert not close_path.exists()


def test_identity_is_checked_with_the_backend():
    import http.server
    from federation.flower_app.__main__ import backend_identity_check

    users = {"good": {"role": "hospital_operator", "hospital_id": "H1"}, "admin": {"role": "admin", "hospital_id": "H1"}}

    class Me(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            user = users.get(self.headers.get("Authorization", "").removeprefix("Bearer "))
            self.send_response(200 if user else 401)
            self.end_headers()
            if user:
                self.wfile.write(json.dumps({"data": {"user": user}}).encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Me)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    check = backend_identity_check(f"http://127.0.0.1:{srv.server_port}/api/v1")
    try:
        assert check("H1", "good")
        assert not check("H2", "good")          # token belongs to another hospital
        assert not check("H1", "admin")         # not a hospital operator
        assert not check("H1", "expired")       # 401
        assert not check("H1", "")
    finally:
        srv.shutdown()
