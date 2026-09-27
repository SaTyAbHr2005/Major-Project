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
    monkeypatch.setattr(server, "DATASETS_DIR", str(work / "datasets"))
    monkeypatch.setattr(server, "TRAINED_DIR", str(work / "trained"))
    monkeypatch.setattr(server, "TMP_DIR", str(work / "tmp"))
    monkeypatch.setattr(server, "ORIGINS", {ORIGIN})
    tokens = {"good": HOSPITAL_USER, "researcher": {"user_id": "USR_2", "role": "researcher"}}

    def fake_verify(token):
        if token not in tokens:
            raise server.ApiError(401, "Your session is not valid. Please sign in again.")
        return tokens[token]

    monkeypatch.setattr(server, "verify_token", fake_verify)
    httpd = server.make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/api/ml"
    httpd.shutdown()
    httpd.server_close()


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
    status, _, body = call(base, "POST", raw=b"not really a png", ctype="application/octet-stream")
    assert status in (200, 400)  # no such model: the CLI reports a failure
    assert list((tmp_path / "work" / "tmp").iterdir()) == []


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
