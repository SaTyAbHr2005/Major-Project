"""
End to end on one machine: a real Flower server for a "flower" job and two hospitals' local ML
services (HOSP_1, HOSP_2) that connect through the service's own route, receive the round's model,
run the normal Module 8 training job on their own synthetic images, send the update and calibrate
BatchNorm. Both hospitals share this machine, as in a single-PC demo, so they take turns training.
"""
import socket
import time

from federation import participation
from federation.api_adapter import get_all_global_models
from federation.flower_app.server import serve
from federation.jobs import JobManager
from federation.storage import FederationStorage
from hospital_client.local_api.tests.test_server import api, call, make_images  # noqa: F401  (api is a fixture)

ARCH = "mobilenet_v3_small"
MAPPING = {"disease": 0, "normal": 1}


def wait(api_url, job, token):
    deadline = time.time() + 300
    while job["status"] == "running":
        assert time.time() < deadline, job["log"][-5:]
        time.sleep(0.5)
        job = call(f"{api_url}/jobs/{job['job_id']}", token=token)[2]
    return job


def prepared_dataset(api_url, token, folder):
    src = make_images(folder)
    _, _, job = call(f"{api_url}/datasets/inspect", "POST", {"path": str(src)}, token=token)
    job = wait(api_url, job, token)
    assert job["status"] == "succeeded", job["error"]
    dataset_id = job["result"]["dataset_id"]
    _, _, job = call(f"{api_url}/datasets/{dataset_id}/preprocess", "POST", {"mode": "lazy"}, token=token)
    assert wait(api_url, job, token)["status"] == "succeeded"
    return dataset_id


def test_one_flower_round_with_two_hospital_services_on_one_machine(api, tmp_path):
    datasets = {tok: prepared_dataset(api, tok, tmp_path / f"src_{tok}") for tok in ("good", "other")}

    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    JobManager(storage).create_job("JOB", "Cells", "image_classification", ARCH, 2, MAPPING, [], 2, "", {"epochs": 1},
                                   {"protocol": "flower", "total_rounds": 1, "round_duration_hours": 1})
    for h in ("HOSP_1", "HOSP_2"):
        participation.request_participation(storage, storage.get_job("JOB"), h, {"eligible": True})
        participation.decide_participation(storage, storage.get_job("JOB"), h, "APPROVED")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{s.getsockname()[1]}"
    job = {"federation_job_id": "JOB", "architecture": ARCH, "num_classes": 2, "class_mapping": MAPPING,
           "training_requirements": {"epochs": 1}}
    for tok, dataset_id in datasets.items():
        status, _, st = call(f"{api}/federation/flower/connect", "POST",
                             {"job": job, "dataset_id": dataset_id, "address": address}, token=tok)
        assert status == 202, st

    owners = {"good": "HOSP_1", "other": "HOSP_2"}
    final = serve(storage, "JOB", address, str(tmp_path / "flower" / "status.json"),
                  verify_identity=lambda hid, tok: owners.get(tok) == hid, poll_seconds=0.5)

    assert final["state"] == "finished", final
    updates = storage.list_updates_for_round("JOB_R1")
    assert sorted(u.participant_id for u in updates) == ["HOSP_1", "HOSP_2"]
    assert all(u.status.value == "USED_IN_AGGREGATION" for u in updates)
    assert [m["version"] for m in get_all_global_models(storage) if m["federation_job_id"] == "JOB"] == [1]
    assert storage.get_job("JOB").status.value == "COMPLETED"
    for tok in ("good", "other"):
        conn = call(f"{api}/federation/flower/status", token=tok)[2]["connections"][0]
        assert conn["state"] in ("done", "disconnected") and conn["training_job_id"]
        runs = call(f"{api}/training-runs", token=tok)[2]["runs"]            # the same training runs as a manual round
        assert any(r["model_id"].startswith("FED_JOB_") for r in runs), runs
