import json
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image

from hospital_client.local_api import server

ORIGIN = "http://localhost:5173"
HOSPITAL_USER = {"user_id": "USR_1", "role": "hospital_operator", "hospital_id": "HOSP_1"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    work = tmp_path / "work"
    monkeypatch.setattr(server, "WORK_DIR", str(work))
    monkeypatch.setattr(server, "ORIGINS", {ORIGIN})
    tokens = {"good": HOSPITAL_USER, "researcher": {"user_id": "USR_2", "role": "researcher"},
              "other": {"user_id": "USR_3", "role": "hospital_operator", "hospital_id": "HOSP_2"},
              "no_hospital": {"user_id": "USR_4", "role": "hospital_operator", "hospital_id": None},
              "traversal": {"user_id": "USR_5", "role": "hospital_operator", "hospital_id": "../HOSP_1"}}

    def fake_verify(token):
        if token not in tokens:
            raise server.ApiError(401, "Your session is not valid. Please sign in again.")
        return tokens[token]

    monkeypatch.setattr(server, "verify_token", fake_verify)
    server._flower.clear()
    httpd = server.make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/api/ml"
    httpd.shutdown()
    httpd.server_close()
    server._flower.clear()


def call(url, method="GET", body=None, token="good", origin=ORIGIN, ctype="application/json", raw=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if origin:
        headers["Origin"] = origin
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
    if data is not None or method == "POST":
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data if data is not None else (b"" if method == "POST" else None),
                                 method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = r.read()
            return r.status, dict(r.headers), (json.loads(payload) if r.headers["Content-Type"] == "application/json" else payload)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def wait_job(api, job, timeout=240):
    deadline = time.time() + timeout
    while job["status"] == "running":
        assert time.time() < deadline, f"job did not finish: {job['log'][-5:]}"
        time.sleep(0.5)
        _, _, job = call(f"{api}/jobs/{job['job_id']}")
    return job


def make_images(root, per_class=12):
    for cls, color in (("normal", (20, 20, 20)), ("disease", (220, 220, 220))):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (40 + i, 40), tuple(c + i for c in color)).save(d / f"{cls}_{i}.png")
    return root


def test_health_is_public_and_everything_else_needs_a_token(api):
    assert call(f"{api}/health", token=None)[0] == 200
    status, _, body = call(f"{api}/datasets", token=None)
    assert status == 401 and "token" in body["error"]
    assert call(f"{api}/datasets", token="bad")[0] == 401


def test_only_hospital_operators_are_allowed(api):
    status, _, body = call(f"{api}/datasets", token="researcher")
    assert status == 403 and "hospital operators" in body["error"]


def test_foreign_origin_is_rejected_and_preflight_is_scoped(api):
    assert call(f"{api}/health", origin="http://evil.example")[0] == 403
    req = urllib.request.Request(f"{api}/datasets/inspect", method="OPTIONS", headers={"Origin": ORIGIN})
    with urllib.request.urlopen(req) as r:
        assert r.status == 204
        assert r.headers["Access-Control-Allow-Origin"] == ORIGIN
        assert "Authorization" in r.headers["Access-Control-Allow-Headers"]


def test_post_requires_json_content_type(api):
    status, _, _ = call(f"{api}/datasets/inspect", method="POST", raw=b"path=x", ctype="text/plain")
    assert status == 415


def test_inspect_rejects_missing_or_unsupported_paths(api, tmp_path):
    assert call(f"{api}/datasets/inspect", "POST", {"path": str(tmp_path / "nope")})[0] == 400
    (tmp_path / "notes.txt").write_text("x")
    assert call(f"{api}/datasets/inspect", "POST", {"path": str(tmp_path / "notes.txt")})[0] == 400
    assert call(f"{api}/datasets/inspect", "POST", {})[0] == 400


def test_unknown_dataset_and_preprocess_before_inspect(api):
    assert call(f"{api}/datasets/0123456789ab")[0] == 404
    assert call(f"{api}/datasets/0123456789ab/recommend", "POST", {})[0] == 404


def test_inspect_then_preprocess_runs_the_real_cli_and_leaves_source_untouched(api, tmp_path):
    src = make_images(tmp_path / "src")
    before = sorted((p.name, p.stat().st_size) for p in src.rglob("*") if p.is_file())

    status, _, job = call(f"{api}/datasets/inspect", "POST", {"path": str(src)})
    assert status == 202 and job["kind"] == "inspect"
    job = wait_job(api, job)
    assert job["status"] == "succeeded", job["error"]
    profile = job["result"]["profile"]
    assert profile["total_samples"] == 24
    assert profile["class_distribution"] == {"normal": 12, "disease": 12}
    dataset_id = job["result"]["dataset_id"]

    assert call(f"{api}/datasets/{dataset_id}/recommend", "POST", {})[0] == 409  # not preprocessed yet

    status, _, job = call(f"{api}/datasets/{dataset_id}/preprocess", "POST", {"mode": "lazy"})
    assert status == 202
    job = wait_job(api, job)
    assert job["status"] == "succeeded", job["error"]
    report = job["result"]["preprocessing"]
    assert sum(report["splits"].values()) == report["quarantine"]["accepted"] == 24

    _, _, listing = call(f"{api}/datasets")
    assert [d["dataset_id"] for d in listing["datasets"]] == [dataset_id]
    assert listing["datasets"][0]["preprocessed"] is True
    assert sorted((p.name, p.stat().st_size) for p in src.rglob("*") if p.is_file()) == before


def test_preprocess_validates_options(api, tmp_path):
    src = make_images(tmp_path / "src", per_class=3)
    job = wait_job(api, call(f"{api}/datasets/inspect", "POST", {"path": str(src)})[2])
    did = job["result"]["dataset_id"]
    assert call(f"{api}/datasets/{did}/preprocess", "POST", {"mode": "zip"})[0] == 400
    assert call(f"{api}/datasets/{did}/preprocess", "POST", {"group_id_map_path": str(tmp_path / "x.json")})[0] == 400


def test_job_parses_epoch_lines_and_reports_failures():
    line = "[epoch 2/5] train_loss=0.5000 train_acc=0.7000 val_loss=0.6000 val_acc=0.6500 elapsed=12.5s"
    job = server.Job("t", [sys.executable, "-c", f"print({line!r}); print('done')"], lambda j: "ok")
    server._run_job(job)
    assert job.status == "succeeded" and job.result == "ok"
    assert job.epochs == [{"epoch": 2, "total_epochs": 5, "elapsed_seconds": 12.5, "train_loss": 0.5,
                           "train_acc": 0.7, "val_loss": 0.6, "val_acc": 0.65}]

    bad = server.Job("t", [sys.executable, "-c", "print('Error: dataset has no classes'); raise SystemExit(1)"], None)
    server._run_job(bad)
    assert bad.status == "failed" and bad.error == "Error: dataset has no classes"


def test_only_one_training_job_at_a_time(monkeypatch):
    monkeypatch.setattr(server, "_jobs", {})
    first = server.start_job("train", [sys.executable, "-c", "import time; time.sleep(3)"], lambda j: None)
    with pytest.raises(server.ApiError) as e:
        server.start_job("train", [sys.executable, "-c", "pass"], lambda j: None)
    assert e.value.status == 409
    server.cancel_job(first)
    deadline = time.time() + 10
    while first.status == "running" and time.time() < deadline:
        time.sleep(0.1)
    assert first.status == "cancelled"


def test_parse_json_output_ignores_surrounding_noise():
    assert server.parse_json_output('UserWarning: x\n{\n "a": {"b": 1}\n}\nmore') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        server.parse_json_output("no json here")


def test_predict_validates_input_and_removes_the_temp_copy(api, tmp_path):
    base = f"{api}/predict?model_id=m1&version=1&ext=.png"
    assert call(f"{api}/predict?model_id=m1&ext=.png", "POST", raw=b"x", ctype="application/octet-stream")[0] == 400
    assert call(f"{api}/predict?model_id=m1&version=1&ext=.exe", "POST", raw=b"x", ctype="application/octet-stream")[0] == 400
    assert call(base, "POST", raw=b"x", ctype="application/json")[0] == 415
    assert call(base, "POST", raw=b"x", ctype="application/octet-stream")[0] == 404  # not this hospital's model
    (tmp_path / "work" / "HOSP_1" / "trained" / "resnet18" / "models" / "m1").mkdir(parents=True)  # no version 1
    status, _, body = call(base, "POST", raw=b"not really a png", ctype="application/octet-stream")
    assert status in (200, 400)  # the CLI reports the failure
    assert list((tmp_path / "work" / "HOSP_1" / "tmp").iterdir()) == []


def fake_run(work, hospital_id, architecture, model_id):
    """Result files exactly where Module 8 --architecture-subdir puts them."""
    d = work / hospital_id / "trained" / architecture
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{model_id}_training_result.json").write_text(json.dumps(
        {"architecture": architecture, "status": "TRAINING_COMPLETED", "generated_at": "2026-10-08T12:00:00"}))
    (d / f"{model_id}_run.json").write_text(json.dumps({"dataset_name": "cells", "choice": "recommended"}))
    return d


def test_each_hospital_sees_only_its_own_training_runs(api, tmp_path):
    work = tmp_path / "work"
    fake_run(work, "HOSP_1", "resnet18", "cells_resnet18_1")
    fake_run(work, "HOSP_1", "mobilenet_v3_small", "cells_mnv3_2")
    fake_run(work, "HOSP_2", "resnet18", "other_resnet18_3")

    mine = {r["model_id"]: r for r in call(f"{api}/training-runs")[2]["runs"]}
    assert set(mine) == {"cells_resnet18_1", "cells_mnv3_2"}          # both architecture folders
    assert mine["cells_mnv3_2"]["dataset_name"] == "cells"              # run.json next to its results
    assert [r["model_id"] for r in call(f"{api}/training-runs", token="other")[2]["runs"]] == ["other_resnet18_3"]

    assert call(f"{api}/training-runs/cells_resnet18_1")[0] == 200
    assert call(f"{api}/training-runs/cells_resnet18_1", token="other")[0] == 404
    assert call(f"{api}/training-runs/other_resnet18_3")[0] == 404
    assert call(f"{api}/datasets", token="other")[2] == {"datasets": []}


def test_workspace_needs_a_valid_hospital_id(api, tmp_path):
    for token in ("no_hospital", "traversal"):
        status, _, body = call(f"{api}/training-runs", token=token)
        assert status == 403 and "hospital id" in body["error"]
    assert not (tmp_path / "HOSP_1").exists()  # nothing created outside WORK_DIR


def test_workspace_folder_cannot_be_shared_by_ids_differing_only_in_case(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORK_DIR", str(tmp_path))
    server.Workspace("HOSP_1")
    (tmp_path / "hosp_1").mkdir(exist_ok=True)          # what a case-insensitive filesystem would give
    (tmp_path / "hosp_1" / server.OWNER_FILE).write_text("HOSP_1")
    with pytest.raises(server.ApiError) as e:
        server.Workspace("hosp_1")
    assert e.value.status == 403


def test_jobs_are_visible_only_to_the_hospital_that_started_them(api, tmp_path):
    src = make_images(tmp_path / "src", per_class=3)
    status, _, job = call(f"{api}/datasets/inspect", "POST", {"path": str(src)})
    assert status == 202
    assert call(f"{api}/jobs/{job['job_id']}", token="other")[0] == 404
    assert call(f"{api}/jobs/{job['job_id']}/cancel", "POST", token="other")[0] == 404
    job = wait_job(api, job)
    assert job["status"] == "succeeded"
    assert (tmp_path / "work" / "HOSP_1" / "datasets" / job["result"]["dataset_id"] / "m4").is_dir()


def test_run_metadata_is_written_next_to_the_results(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "WORK_DIR", str(tmp_path))
    ws = server.Workspace("HOSP_1")
    write = server._run_meta_writer(ws, "m_1", {"dataset_name": "cells"})
    write(None)  # Module 8 wrote nothing (e.g. it failed before training): nothing to annotate
    assert ws.arch_dirs() == []
    d = fake_run(tmp_path, "HOSP_1", "resnet18", "m_1")
    write(None)
    assert json.loads((d / "m_1_run.json").read_text()) == {"dataset_name": "cells"}


def test_training_run_endpoints_reject_unknown_ids(api):
    assert call(f"{api}/training-runs")[2] == {"runs": []}
    assert call(f"{api}/training-runs/nope")[0] == 404
    assert call(f"{api}/training-runs/nope/plots/secret.png")[0] == 404
    assert call(f"{api}/models")[2] == {"models": []}


def test_train_validates_the_selection_before_starting_anything(api, tmp_path):
    src = make_images(tmp_path / "src", per_class=3)
    did = wait_job(api, call(f"{api}/datasets/inspect", "POST", {"path": str(src)})[2])["result"]["dataset_id"]
    wait_job(api, call(f"{api}/datasets/{did}/preprocess", "POST", {"mode": "lazy"})[2])
    train = f"{api}/datasets/{did}/train"
    assert call(train, "POST", {})[0] == 400
    assert call(train, "POST", {"choice": "turbo"})[0] == 400
    assert call(train, "POST", {"architecture": "../evil"})[0] == 400
    assert call(train, "POST", {"architecture": "resnet18", "batch_size": 0})[0] == 400
    assert call(train, "POST", {"architecture": "resnet18", "epochs": "10"})[0] == 400
    assert call(train, "POST", {"architecture": "resnet18", "batch_size": True})[0] == 400
    assert call(f"{api}/training-runs")[2] == {"runs": []}  # nothing was launched


def test_runs_are_named_after_the_recommended_model(tmp_path):
    recs = {"recommendations": [{"recommendation_type": "recommended", "model": "resnet50"},
                                {"recommendation_type": "fast", "model": "../evil"}]}
    (tmp_path / "recommendations.json").write_text(json.dumps(recs))
    assert server._choice_architecture(str(tmp_path), "recommended") == "resnet50"
    assert server._choice_architecture(str(tmp_path), "fast") is None          # unsafe value ignored
    assert server._choice_architecture(str(tmp_path), "high_capacity") is None  # caller falls back to the choice
    assert server._choice_architecture(str(tmp_path / "missing"), "recommended") is None


def test_retry_submits_the_handoff_to_the_round_saved_with_the_run(api, tmp_path, monkeypatch):
    work = tmp_path / "work"
    d = fake_run(work, "HOSP_1", "resnet18", "FED_JOB1_1")
    (d / "FED_JOB1_1_federation_handoff").mkdir()
    (d / "FED_JOB1_1_run.json").write_text(json.dumps({"job_id": "JOB1", "round_id": "ROUND1"}))
    fake_run(work, "HOSP_1", "resnet18", "FED_OLD_2")                 # started before round ids were saved

    sent = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"success": true, "update": {"status": "ACCEPTED"}}'

    real_urlopen = urllib.request.urlopen

    def fake_urlopen(req, timeout=None):
        if not req.full_url.startswith(server.BACKEND_URL):            # the test's own calls to the local API
            return real_urlopen(req, timeout=timeout)
        sent.append((req.full_url, json.loads(req.data), req.headers["Authorization"]))
        return Resp()

    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
    status, _, body = call(f"{api}/federation/submit/FED_JOB1_1", method="POST", body={})
    assert status == 200 and body["update"]["status"] == "ACCEPTED"
    url, payload, auth = sent[0]
    assert url.endswith("/federation/jobs/JOB1/rounds/ROUND1/submit")
    assert payload["handoff_dir"] == str(d / "FED_JOB1_1_federation_handoff")
    assert auth == "Bearer good"                                       # the caller's own token

    assert call(f"{api}/federation/submit/FED_OLD_2", method="POST", body={})[0] == 409
    assert call(f"{api}/federation/submit/FED_JOB1_1", method="POST", body={}, token="other")[0] == 409
    assert len(sent) == 1


def test_round_base_model_is_downloaded_once_into_the_hospital_model_store(tmp_path, monkeypatch):
    from types import SimpleNamespace
    ws = SimpleNamespace(trained=str(tmp_path / "trained"), tmp=str(tmp_path / "tmp"))
    rnd = {"round_id": "JOB1_R1", "base_model_id": "SEED_JOB1", "base_model_version": 1}
    fetched = []

    class Resp:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1):
            data, self.data = self.data, b""
            return data

    def fake_urlopen(req, timeout=None):
        fetched.append((req.full_url, req.headers["Authorization"]))
        return Resp(req.full_url.rsplit("/", 1)[1].encode())

    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
    server._ensure_base_model(ws, "resnet18", rnd, "tok")
    dest = tmp_path / "trained" / "resnet18" / "models" / "SEED_JOB1" / "v1"
    assert (dest / "model.safetensors").read_bytes() == b"model.safetensors"
    assert (dest / "metadata.json").read_bytes() == b"metadata.json"
    assert fetched == [(f"{server.BACKEND_URL}/federation/rounds/JOB1_R1/base-model/{f}", "Bearer tok")
                       for f in ("model.safetensors", "metadata.json")]

    server._ensure_base_model(ws, "resnet18", rnd, "tok")             # already present: no second download
    assert len(fetched) == 2

    with pytest.raises(server.ApiError) as e:
        server._ensure_base_model(ws, "resnet18", {**rnd, "base_model_id": "../escape"}, "tok")
    assert e.value.status == 400

    def refused(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(server.urllib.request, "urlopen", refused)
    with pytest.raises(server.ApiError) as e:
        server._ensure_base_model(ws, "resnet18", {**rnd, "base_model_id": "OTHER"}, "tok")
    assert e.value.status == 502
    assert not (tmp_path / "trained" / "resnet18" / "models" / "OTHER").exists()   # nothing half-written


@pytest.mark.parametrize("test_count, evaluated", [(5, True), (0, False)])
def test_federated_run_evaluates_the_test_split_when_there_is_one(api, tmp_path, monkeypatch, test_count, evaluated):
    import hospital_client.federation.manifest_filter as mf
    started = []
    monkeypatch.setattr(server, "_require_preprocessed", lambda ws, dataset_id: str(tmp_path))
    monkeypatch.setattr(mf, "create_job_manifests", lambda **kw: (str(tmp_path), {"train": 10, "val": 5, "test": test_count}))
    monkeypatch.setattr(server, "dataset_dir", lambda ws, dataset_id: str(tmp_path))
    (tmp_path / "source.json").write_text(json.dumps({"name": "cells"}))

    class Handle:
        def to_dict(self): return {"job_id": "abc", "status": "running"}

    monkeypatch.setattr(server, "start_job", lambda kind, cmd, *a, **kw: started.append(cmd) or Handle())
    job = {"federation_job_id": "JOB1", "architecture": "resnet18", "class_mapping": {"a": 0, "b": 1}, "rounds": []}
    status, _, _ = call(f"{api}/federation/participate", method="POST", body={"job": job, "dataset_id": "0123456789ab"})
    assert status == 202
    assert ("--test" in started[0] and "--plots" in started[0]) == evaluated


@pytest.mark.parametrize("body, generated", [({}, True), ({"generate_missing_splits": False}, False)])
def test_preprocess_creates_missing_splits_by_default(api, tmp_path, monkeypatch, body, generated):
    d = tmp_path / "work" / "HOSP_1" / "datasets" / "0123456789ab"
    (d / "m4").mkdir(parents=True)
    (d / "m4" / "dataset_profile.json").write_text("{}")
    (d / "source.json").write_text(json.dumps({"path": str(tmp_path), "name": "ds"}))
    started = []

    class Handle:
        def to_dict(self): return {"job_id": "abc", "status": "running"}

    monkeypatch.setattr(server, "start_job", lambda kind, cmd, *a, **kw: started.append(cmd) or Handle())
    status, _, _ = call(f"{api}/datasets/0123456789ab/preprocess", method="POST", body=body)
    assert status == 202
    assert ("--generate-missing-splits" in started[0]) == generated


def test_metadata_csv_columns_and_label_values(api, tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("isic_id,diagnosis,lesion_id\nA,MEL,L1\nB,NV,L1\nC,NV,L2\n")
    status, _, body = call(f"{api}/metadata-csv", method="POST", body={"path": str(csv)})
    assert status == 200 and body["columns"] == ["isic_id", "diagnosis", "lesion_id"] and "values" not in body
    status, _, body = call(f"{api}/metadata-csv", method="POST", body={"path": str(csv), "label_column": "diagnosis"})
    assert body["values"] == [{"value": "NV", "count": 2}, {"value": "MEL", "count": 1}]
    assert call(f"{api}/metadata-csv", method="POST", body={"path": str(csv), "label_column": "nope"})[0] == 400
    assert call(f"{api}/metadata-csv", method="POST", body={"path": str(tmp_path / "x.txt")})[0] == 400


def test_inspect_passes_the_metadata_label_source_to_module_4(api, tmp_path, monkeypatch):
    (tmp_path / "images").mkdir()
    csv = tmp_path / "meta.csv"
    csv.write_text("isic_id,diagnosis\nA,MEL\n")
    started = []

    class Handle:
        def to_dict(self): return {"job_id": "abc", "status": "running"}

    monkeypatch.setattr(server, "start_job", lambda kind, cmd, *a, **kw: started.append(cmd) or Handle())
    spec = {"csv_path": str(csv), "image_column": "isic_id", "label_column": "diagnosis",
            "group_column": "", "label_map": {"MEL": "malignant", "NV": None}}
    status, _, _ = call(f"{api}/datasets/inspect", method="POST", body={"path": str(tmp_path / "images"), "label_source": spec})
    assert status == 202
    cmd = started[0]
    assert cmd[cmd.index("--labels-csv") + 1] == str(csv) and "--group-column" not in cmd
    assert json.loads(cmd[cmd.index("--label-map") + 1]) == {"MEL": "malignant", "NV": None}

    bad = {**spec, "label_column": ""}
    assert call(f"{api}/datasets/inspect", method="POST", body={"path": str(tmp_path / "images"), "label_source": bad})[0] == 400
    assert call(f"{api}/datasets/inspect", method="POST", body={"path": str(csv), "label_source": spec})[0] == 400
    assert len(started) == 1


# ── Flower jobs ──────────────────────────────────────────────────────────────────────────

FLOWER_JOB = {"federation_job_id": "JOB1", "architecture": "mobilenet_v3_small", "num_classes": 2,
              "class_mapping": {"a": 0, "b": 1}, "training_requirements": {"epochs": 2}, "rounds": []}
ROUND = {"round_id": "JOB1_R1", "base_model_id": "SEED_JOB1", "base_model_version": 1, "base_model_checksum": "abc"}


def _fake_dataset(tmp_path, monkeypatch):
    import hospital_client.federation.manifest_filter as mf
    monkeypatch.setattr(server, "_require_preprocessed", lambda ws, dataset_id: str(tmp_path))
    monkeypatch.setattr(mf, "create_job_manifests", lambda **kw: (str(tmp_path), {"train": 10, "val": 5, "test": 5}))
    monkeypatch.setattr(server, "dataset_dir", lambda ws, dataset_id: str(tmp_path))
    (tmp_path / "source.json").write_text(json.dumps({"name": "cells"}))


def test_manual_and_flower_rounds_build_the_same_training_command(api, tmp_path, monkeypatch):
    _fake_dataset(tmp_path, monkeypatch)
    started = []

    class Handle:
        def to_dict(self): return {"job_id": "abc", "status": "running"}

    monkeypatch.setattr(server, "start_job", lambda kind, cmd, *a, **kw: started.append(cmd) or Handle())
    monkeypatch.setattr(server, "_ensure_base_model", lambda *a: None)
    job = {**FLOWER_JOB, "rounds": [ROUND]}
    assert call(f"{api}/federation/participate", method="POST", body={"job": job, "dataset_id": "0123456789ab", "round_id": "JOB1_R1"})[0] == 202
    cmd, meta, model_id, run_dir = server.build_federated_training(server.Workspace("HOSP_1"), job, "0123456789ab", ROUND)

    def norm(c):
        return [("MODEL" if a.startswith("FED_JOB1_") else a) for a in c]
    assert norm(cmd) == norm(started[0])
    assert "--canonical-base-model-checksum" in cmd and "--epochs" in cmd and meta["round_id"] == "JOB1_R1"


def test_model_received_over_flower_is_stored_where_the_trainer_reads_it(tmp_path):
    from types import SimpleNamespace
    from hospital_client.model_management.config import ModelConfig
    from hospital_client.model_management.parameters import get_parameters
    from hospital_client.model_management.registry import build_model
    from hospital_client.model_management.store import ModelStore
    from hospital_client.training.federation import _compute_parameters_checksum

    ws = SimpleNamespace(trained=str(tmp_path / "trained"))
    params = get_parameters(build_model(ModelConfig(architecture="mobilenet_v3_small", num_classes=2, input_size=(224, 224))))
    good = {**ROUND, "base_model_checksum": _compute_parameters_checksum(params)}
    server._save_received_base_model(ws, FLOWER_JOB, params, good)
    loaded, _ = ModelStore(str(tmp_path / "trained" / "mobilenet_v3_small" / "models")).load_checkpoint("SEED_JOB1", 1)
    assert _compute_parameters_checksum(get_parameters(loaded)) == good["base_model_checksum"]

    with pytest.raises(server.ApiError, match="checksum"):
        server._save_received_base_model(ws, FLOWER_JOB, params, {**ROUND, "base_model_id": "OTHER", "base_model_checksum": "wrong"})
    assert not (tmp_path / "trained" / "mobilenet_v3_small" / "models" / "OTHER").exists()


def test_flower_connect_status_and_disconnect(api, monkeypatch):
    from hospital_client.federation.flower_client import FlowerConnection
    monkeypatch.setattr(FlowerConnection, "start", lambda self: self._set(state="waiting"))
    body = {"job": FLOWER_JOB, "dataset_id": "0123456789ab", "address": "127.0.0.1:8090"}

    assert call(f"{api}/federation/flower/connect", method="POST", body={**body, "address": "evil host"})[0] == 400
    status, _, st = call(f"{api}/federation/flower/connect", method="POST", body=body)
    assert status == 202 and st["job_id"] == "JOB1" and st["state"] == "waiting"
    assert call(f"{api}/federation/flower/connect", method="POST", body=body)[0] == 409       # already connected

    status, _, res = call(f"{api}/federation/flower/status")
    assert status == 200 and [c["job_id"] for c in res["connections"]] == ["JOB1"] and "good" not in json.dumps(res)
    assert server._flower[("HOSP_1", "JOB1")].client().get_properties({})["token"] == "good"   # refreshed from the poll
    assert call(f"{api}/federation/flower/status", token="other")[2]["connections"] == []      # another hospital

    assert call(f"{api}/federation/flower/disconnect", method="POST", body={"job_id": "JOB1"})[0] == 200
    assert call(f"{api}/federation/flower/status")[2]["connections"][0]["state"] == "disconnected"
    assert call(f"{api}/federation/flower/connect", method="POST", body=body)[0] == 202       # can reconnect
