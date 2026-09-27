"""
Local ML service for the Hospital Desktop (see __init__.py).

Every ML action runs the existing module CLI exactly as a terminal user would:

    inspect     -> python -m hospital_client.dataset inspect <path> --output ... (Module 4)
    preprocess  -> python -m hospital_client.preprocessing preprocess ...      (Module 5)
    resources   -> python -m hospital_client.resource_training detect          (Module 8)
    recommend   -> python -m hospital_client.resource_training recommend ...   (Module 8)
    train       -> python -m hospital_client.resource_training train ... --test --plots (Modules 8 -> 7 -> 6)
    predict     -> python -m hospital_client.inference predict ... --json      (Module 16)

Long-running steps are background jobs the UI polls. Generated outputs live
under ML_WORK_DIR (default ~/medfl_ml_work); source datasets are only read.

Security: binds to 127.0.0.1; only the configured UI origins may call it;
every route except /health needs a backend-issued JWT for a
hospital_operator, verified against the backend's GET /auth/me.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOST = "127.0.0.1"
PORT = int(os.environ.get("ML_API_PORT", "8765"))
ORIGINS = {o.strip() for o in os.environ.get(
    "ML_API_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()}
BACKEND_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:5000/api/v1").rstrip("/")
WORK_DIR = os.path.abspath(os.environ.get("ML_WORK_DIR", os.path.join(os.path.expanduser("~"), "medfl_ml_work")))
DATASETS_DIR = os.path.join(WORK_DIR, "datasets")
TRAINED_DIR = os.path.join(WORK_DIR, "trained")
TMP_DIR = os.path.join(WORK_DIR, "tmp")

MAX_BODY = 50 * 1024 * 1024
MAX_JOB_LINES = 5000
LOG_TAIL = 400
MAX_JOBS = 100
AUTH_TTL_SECONDS = 60

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABULAR_EXTS = {".csv", ".xls", ".xlsx"}
CHOICES = {"recommended", "high_capacity", "fast"}
PLOT_FILES = ("confusion_matrix.png", "roc_curves.png", "pr_curves.png")
DATASET_ID_RE = re.compile(r"^[0-9a-f]{12}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")
ARCH_RE = re.compile(r"^[a-z0-9_]{1,50}$")
EPOCH_RE = re.compile(r"^\[epoch (\d+)/(\d+)\] (.*?) elapsed=([\d.]+)s$")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- auth

_auth_cache = {}
_auth_lock = threading.Lock()


def verify_token(token: str) -> dict:
    """Asks the backend who owns this token (no JWT secret on the Python side)."""
    now = time.time()
    with _auth_lock:
        hit = _auth_cache.get(token)
        if hit and hit[0] > now:
            return hit[1]
    req = urllib.request.Request(f"{BACKEND_URL}/auth/me",
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            user = json.load(r)["data"]["user"]
    except urllib.error.HTTPError:
        raise ApiError(401, "Your session is not valid. Please sign in again.")
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        raise ApiError(503, "Cannot verify your session: the backend API is unreachable.")
    with _auth_lock:
        _auth_cache[token] = (now + AUTH_TTL_SECONDS, user)
    return user


def authenticate(headers) -> dict:
    auth = headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or len(auth) <= 7:
        raise ApiError(401, "Missing bearer token. Please sign in.")
    user = verify_token(auth[7:])
    if user.get("role") != "hospital_operator":
        raise ApiError(403, "Only hospital operators can use the local ML service.")
    return user


# ---------------------------------------------------------------- jobs

_jobs = {}
_jobs_lock = threading.Lock()


def _module_cmd(module: str, *args) -> list:
    return [sys.executable, "-m", module, *[str(a) for a in args]]


def _child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class Job:
    def __init__(self, kind: str, cmd: list, finalize, meta=None):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.cmd, self.finalize = kind, cmd, finalize
        self.meta = meta or {}
        self.status = "running"
        self.lines, self.epochs = [], []
        self.result = self.error = None
        self.started_at, self.finished_at = time.time(), None
        self.proc = None
        self.cancelled = False

    def to_dict(self) -> dict:
        end = self.finished_at or time.time()
        return {
            "job_id": self.id, "kind": self.kind, "status": self.status, "meta": self.meta,
            "log": self.lines[-LOG_TAIL:], "epochs": self.epochs, "result": self.result, "error": self.error,
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "elapsed_seconds": round(end - self.started_at, 1),
        }


def _run_job(job: Job) -> None:
    try:
        job.proc = subprocess.Popen(job.cmd, cwd=REPO_ROOT, env=_child_env(), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        if job.cancelled:
            _kill(job.proc)
        for raw in job.proc.stdout:
            line = raw.rstrip("\r\n")
            job.lines.append(line)
            if len(job.lines) > MAX_JOB_LINES:
                del job.lines[: len(job.lines) - MAX_JOB_LINES]
            m = EPOCH_RE.match(line)
            if m:
                point = {"epoch": int(m.group(1)), "total_epochs": int(m.group(2)), "elapsed_seconds": float(m.group(4))}
                for pair in m.group(3).split():
                    key, _, value = pair.partition("=")
                    point[key] = float(value)
                job.epochs.append(point)
        code = job.proc.wait()
        if job.cancelled:
            job.status = "cancelled"
            return
        if code != 0:
            raise RuntimeError(_failure_message(job.lines, code))
        job.result = job.finalize(job)
        job.status = "succeeded"
    except Exception as e:
        job.status = "cancelled" if job.cancelled else "failed"
        job.error = str(e)
    finally:
        job.finished_at = time.time()


def _failure_message(lines: list, code: int) -> str:
    for line in reversed(lines):
        s = line.strip()
        if "error" in s.lower():
            return s
    return f"Process exited with code {code}. See the log for details."


def start_job(kind: str, cmd: list, finalize, meta=None) -> Job:
    job = Job(kind, cmd, finalize, meta)
    with _jobs_lock:
        if kind == "train" and any(j.kind == "train" and j.status == "running" for j in _jobs.values()):
            raise ApiError(409, "A training run is already in progress on this machine.")
        _jobs[job.id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.started_at)[:-MAX_JOBS]:
            if old.status != "running":
                _jobs.pop(old.id, None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def get_job(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise ApiError(404, "Unknown job.")
    return job


def _kill(proc) -> None:
    if os.name == "nt":  # kill the whole tree (DataLoader workers are child processes)
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()


def cancel_job(job: Job) -> None:
    if job.status != "running":
        return
    job.cancelled = True
    if job.proc is not None:  # otherwise _run_job kills it right after spawning
        _kill(job.proc)


def parse_json_output(text: str):
    """The CLIs print one JSON document; library warnings may surround it."""
    start = text.find("{")
    if start < 0:
        raise ValueError("The command produced no JSON output.")
    return json.JSONDecoder().raw_decode(text, start)[0]


# ---------------------------------------------------------------- files

def _read_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def dataset_dir(dataset_id: str) -> str:
    if not DATASET_ID_RE.match(dataset_id or ""):
        raise ApiError(400, "Invalid dataset id.")
    d = os.path.join(DATASETS_DIR, dataset_id)
    if not os.path.isfile(os.path.join(d, "source.json")):
        raise ApiError(404, "Unknown dataset. Inspect it first.")
    return d


def dataset_summary(dataset_id: str) -> dict:
    d = dataset_dir(dataset_id)
    source = _read_json(os.path.join(d, "source.json"))
    return {
        "dataset_id": dataset_id,
        "name": source["name"],
        "source_path": source["path"],
        "profile": _read_json(os.path.join(d, "m4", "dataset_profile.json")),
        "profile_report_md": _read_text(os.path.join(d, "m4", "dataset_report.md")),
        "preprocessing": _read_json(os.path.join(d, "m5", "preprocessing_report.json")),
        "preprocessing_report_md": _read_text(os.path.join(d, "m5", "preprocessing_report.md")),
        "recommendations": _read_json(os.path.join(d, "recommendations.json")),
    }


def training_run(model_id: str) -> dict:
    if not MODEL_ID_RE.match(model_id or ""):
        raise ApiError(400, "Invalid model id.")
    result = _read_json(os.path.join(TRAINED_DIR, f"{model_id}_training_result.json"))
    if result is None:
        raise ApiError(404, "No training result for this model id.")
    plots_dir = os.path.join(TRAINED_DIR, f"{model_id}_plots")
    return {
        "model_id": model_id,
        "run": _read_json(os.path.join(TRAINED_DIR, f"{model_id}_run.json")),
        "result": result,
        "resource_statistics": _read_json(os.path.join(TRAINED_DIR, f"{model_id}_resource_statistics.json")),
        "plots": [p for p in PLOT_FILES if os.path.isfile(os.path.join(plots_dir, p))],
        "curve_metrics": _read_json(os.path.join(plots_dir, "curve_metrics.json")),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text or "").strip("_")[:50] or "model"


# ---------------------------------------------------------------- routes

def r_health(h, m, q):
    return {"status": "ok", "service": "medfl-local-ml"}


def r_resources(h, m, q):
    os.makedirs(TRAINED_DIR, exist_ok=True)
    proc = subprocess.run(_module_cmd("hospital_client.resource_training", "detect", "--output", TRAINED_DIR),
                          cwd=REPO_ROOT, env=_child_env(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    if proc.returncode != 0:
        raise ApiError(500, _failure_message(proc.stdout.splitlines(), proc.returncode))
    return parse_json_output(proc.stdout)


_BROWSE_SCRIPT = (
    "import tkinter as tk\nfrom tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "print(filedialog.askdirectory(title='Locate the local medical imaging dataset') or '')\n"
)


def r_browse(h, m, q):
    """Native folder picker on THIS machine (a browser cannot reveal absolute paths).
    Runs in its own process so Tk owns its main thread."""
    proc = subprocess.run([sys.executable, "-c", _BROWSE_SCRIPT], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    path = proc.stdout.strip()
    return {"path": os.path.abspath(path) if path else None}


def r_list_datasets(h, m, q):
    out = []
    if os.path.isdir(DATASETS_DIR):
        for did in sorted(os.listdir(DATASETS_DIR)):
            if DATASET_ID_RE.match(did) and os.path.isfile(os.path.join(DATASETS_DIR, did, "source.json")):
                s = dataset_summary(did)
                out.append({k: s[k] for k in ("dataset_id", "name", "source_path")} | {
                    "total_samples": (s["profile"] or {}).get("total_samples"),
                    "classes": (s["profile"] or {}).get("classes"),
                    "preprocessed": s["preprocessing"] is not None,
                })
    return {"datasets": out}


def r_get_dataset(h, m, q):
    return dataset_summary(m.group(1))


def r_inspect(h, m, q):
    body = h.json_body()
    path = body.get("path")
    if not isinstance(path, str) or not path.strip() or len(path) > 1000:
        raise ApiError(400, "A dataset path is required.")
    path = os.path.abspath(os.path.expanduser(path.strip()))
    is_table = os.path.isfile(path) and os.path.splitext(path)[1].lower() in TABULAR_EXTS
    if not (os.path.isdir(path) or is_table):
        raise ApiError(400, "Path not found, or not a supported dataset (image folder, CSV, XLS or XLSX file).")
    if os.path.normcase(path).startswith(os.path.normcase(WORK_DIR)):
        raise ApiError(400, "That path is inside the ML work directory, not a source dataset.")

    dataset_id = hashlib.sha256(os.path.normcase(path).encode("utf-8")).hexdigest()[:12]
    d = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"path": path, "name": os.path.basename(path.rstrip("\\/")) or path}, f)
    # Generated outputs only - re-inspection invalidates everything derived from the old profile.
    for sub in ("m4", "m5"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    if os.path.exists(os.path.join(d, "recommendations.json")):
        os.remove(os.path.join(d, "recommendations.json"))

    cmd = _module_cmd("hospital_client.dataset", "inspect", path, "--output", os.path.join(d, "m4"))
    job = start_job("inspect", cmd, lambda j: _finish_inspect(dataset_id), {"dataset_id": dataset_id})
    return 202, job.to_dict()


def _finish_inspect(dataset_id: str) -> dict:
    s = dataset_summary(dataset_id)
    if s["profile"] is None:
        raise RuntimeError("Inspection finished but no dataset_profile.json was written.")
    return s


def r_preprocess(h, m, q):
    dataset_id = m.group(1)
    d = dataset_dir(dataset_id)
    profile = os.path.join(d, "m4", "dataset_profile.json")
    if not os.path.isfile(profile):
        raise ApiError(409, "Inspect the dataset (Module 4) before preprocessing.")
    body = h.json_body()
    mode = body.get("mode", "lazy")
    policy = body.get("invalid_split_policy", "error")
    if mode not in ("lazy", "materialized") or policy not in ("error", "regenerate"):
        raise ApiError(400, "Invalid preprocessing mode or split policy.")
    source = _read_json(os.path.join(d, "source.json"))["path"]
    out = os.path.join(d, "m5")
    shutil.rmtree(out, ignore_errors=True)
    args = ["preprocess", source, "--profile", profile, "--output", out, "--mode", mode,
            "--invalid-split-policy", policy]
    group_map = body.get("group_id_map_path")
    if group_map:
        if not isinstance(group_map, str) or not group_map.lower().endswith(".json") or not os.path.isfile(group_map):
            raise ApiError(400, "The group/patient id map must be an existing .json file.")
        args += ["--group-id-map", os.path.abspath(group_map)]
    target = body.get("target_column")
    if target:
        if not isinstance(target, str) or len(target) > 200:
            raise ApiError(400, "Invalid target column.")
        args += ["--target-column", target]
    if os.path.exists(os.path.join(d, "recommendations.json")):
        os.remove(os.path.join(d, "recommendations.json"))

    job = start_job("preprocess", _module_cmd("hospital_client.preprocessing", *args),
                    lambda j: _finish_preprocess(dataset_id), {"dataset_id": dataset_id})
    return 202, job.to_dict()


def _finish_preprocess(dataset_id: str) -> dict:
    s = dataset_summary(dataset_id)
    if s["preprocessing"] is None:
        raise RuntimeError("Preprocessing finished but no preprocessing_report.json was written.")
    return s


def _require_preprocessed(dataset_id: str) -> str:
    d = dataset_dir(dataset_id)
    m5 = os.path.join(d, "m5")
    if not os.path.isfile(os.path.join(m5, "preprocessing_report.json")):
        raise ApiError(409, "Preprocess the dataset (Module 5) first.")
    return m5


def r_recommend(h, m, q):
    dataset_id = m.group(1)
    m5 = _require_preprocessed(dataset_id)
    os.makedirs(TRAINED_DIR, exist_ok=True)
    cmd = _module_cmd("hospital_client.resource_training", "recommend", "--dataset", m5, "--output", TRAINED_DIR)

    def finish(job):
        recs = parse_json_output("\n".join(job.lines))
        with open(os.path.join(dataset_dir(dataset_id), "recommendations.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2)
        return recs

    job = start_job("recommend", cmd, finish, {"dataset_id": dataset_id})
    return 202, job.to_dict()


def r_train(h, m, q):
    dataset_id = m.group(1)
    m5 = _require_preprocessed(dataset_id)
    body = h.json_body()
    choice, architecture = body.get("choice"), body.get("architecture")
    selection_args = []
    if architecture is not None:
        # A specific model (a same-tier alternative, or a manual configuration). Module 8 itself
        # re-validates it: infeasible models and unsafe batch sizes are refused by the CLI.
        if not isinstance(architecture, str) or not ARCH_RE.match(architecture):
            raise ApiError(400, "Invalid architecture.")
        selection_args = ["--architecture", architecture]
        for key, flag in (("batch_size", "--batch-size"), ("epochs", "--epochs")):
            value = body.get(key)
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10000:
                    raise ApiError(400, f"{key} must be a positive integer.")
                selection_args += [flag, value]
        label = "manual" if body.get("batch_size") is not None or body.get("epochs") is not None else "alt"
        choice = f"{label}:{architecture}"
        suffix = f"{label}_{architecture}"
    elif choice in CHOICES:
        selection_args = ["--choice", choice]
        suffix = choice
    else:
        raise ApiError(400, f"Send choice (one of {sorted(CHOICES)}) or an architecture.")
    name = _read_json(os.path.join(dataset_dir(dataset_id), "source.json"))["name"]
    model_id = f"{_slug(body.get('name') or name)}_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(TRAINED_DIR, exist_ok=True)
    meta = {"dataset_id": dataset_id, "dataset_name": name, "choice": choice, "model_id": model_id,
            "request_id": body.get("request_id") if isinstance(body.get("request_id"), str) else None,
            "started_at": datetime.now().isoformat()}
    cmd = _module_cmd("hospital_client.resource_training", "train", "--dataset", m5, "--output", TRAINED_DIR,
                      "--model-id", model_id, *selection_args, "--test", "--plots")
    job = start_job("train", cmd, lambda j: training_run(model_id), meta)
    with open(os.path.join(TRAINED_DIR, f"{model_id}_run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 202, job.to_dict()


def r_get_job(h, m, q):
    return get_job(m.group(1)).to_dict()


def r_cancel_job(h, m, q):
    job = get_job(m.group(1))
    cancel_job(job)
    return job.to_dict()


def r_list_runs(h, m, q):
    runs = []
    if os.path.isdir(TRAINED_DIR):
        for fname in os.listdir(TRAINED_DIR):
            if not fname.endswith("_training_result.json"):
                continue
            model_id = fname[: -len("_training_result.json")]
            if not MODEL_ID_RE.match(model_id):
                continue
            res = _read_json(os.path.join(TRAINED_DIR, fname))
            run = _read_json(os.path.join(TRAINED_DIR, f"{model_id}_run.json")) or {}
            runs.append({
                "model_id": model_id, "architecture": res.get("architecture"), "status": res.get("status"),
                "epochs_run": res.get("epochs_run"), "duration_seconds": res.get("training_duration_seconds"),
                "validation_accuracy": (res.get("validation_metrics") or {}).get("accuracy"),
                "test_accuracy": (res.get("test_metrics") or {}).get("accuracy"),
                "checkpoint_version": res.get("checkpoint_version"),
                "ready_for_federation": res.get("ready_for_federation"),
                "dataset_name": run.get("dataset_name"), "choice": run.get("choice"),
                "generated_at": res.get("generated_at"),
            })
    runs.sort(key=lambda r: r.get("generated_at") or "", reverse=True)
    return {"runs": runs}


def r_get_run(h, m, q):
    return training_run(m.group(1))


def r_get_plot(h, m, q):
    model_id, name = m.group(1), m.group(2)
    if not MODEL_ID_RE.match(model_id) or name not in PLOT_FILES:
        raise ApiError(404, "Unknown plot.")
    path = os.path.join(TRAINED_DIR, f"{model_id}_plots", name)
    if not os.path.isfile(path):
        raise ApiError(404, "Plot not found.")
    with open(path, "rb") as f:
        return ("image/png", f.read())


def r_models(h, m, q):
    from hospital_client.model_management.store import ModelStore
    models_root = os.path.join(TRAINED_DIR, "models")
    out = []
    if os.path.isdir(models_root):
        store = ModelStore(models_root)
        for model_id in store.list_model_ids():
            run = _read_json(os.path.join(TRAINED_DIR, f"{model_id}_run.json")) or {}
            for meta in store.list_versions(model_id):
                out.append(meta.to_dict() | {"dataset_name": run.get("dataset_name")})
    out.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"models": out}


def r_predict(h, m, q):
    model_id = (q.get("model_id") or [""])[0]
    version = (q.get("version") or [""])[0]
    device = (q.get("device") or ["auto"])[0]
    ext = (q.get("ext") or [""])[0].lower()
    if not MODEL_ID_RE.match(model_id) or not version.isdigit() or int(version) < 1:
        raise ApiError(400, "An explicit model id and version are required.")
    if device not in ("cpu", "cuda", "auto"):
        raise ApiError(400, "device must be cpu, cuda or auto.")
    if ext not in IMAGE_EXTS:
        raise ApiError(400, f"Unsupported image type. Allowed: {sorted(IMAGE_EXTS)}")
    if h.headers.get("Content-Type", "").split(";")[0].strip() != "application/octet-stream":
        raise ApiError(415, "Send the image as application/octet-stream.")
    data = h.raw_body()
    if not data:
        raise ApiError(400, "Empty image.")

    os.makedirs(TMP_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=ext, dir=TMP_DIR)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        proc = subprocess.run(
            _module_cmd("hospital_client.inference", "predict", "--models-root", os.path.join(TRAINED_DIR, "models"),
                        "--model-id", model_id, "--version", version, "--device", device, "--image", tmp, "--json"),
            cwd=REPO_ROOT, env=_child_env(), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
    finally:
        os.remove(tmp)  # the uploaded copy never outlives the request
    try:
        return parse_json_output(proc.stdout)
    except ValueError:
        raise ApiError(400, _failure_message(proc.stdout.splitlines(), proc.returncode))


ROUTES = [
    ("GET", re.compile(r"/api/ml/health"), r_health, True),
    ("GET", re.compile(r"/api/ml/resources"), r_resources, False),
    ("POST", re.compile(r"/api/ml/browse-folder"), r_browse, False),
    ("GET", re.compile(r"/api/ml/datasets"), r_list_datasets, False),
    ("POST", re.compile(r"/api/ml/datasets/inspect"), r_inspect, False),
    ("GET", re.compile(r"/api/ml/datasets/([0-9a-f]{12})"), r_get_dataset, False),
    ("POST", re.compile(r"/api/ml/datasets/([0-9a-f]{12})/preprocess"), r_preprocess, False),
    ("POST", re.compile(r"/api/ml/datasets/([0-9a-f]{12})/recommend"), r_recommend, False),
    ("POST", re.compile(r"/api/ml/datasets/([0-9a-f]{12})/train"), r_train, False),
    ("GET", re.compile(r"/api/ml/jobs/([0-9a-f]{12})"), r_get_job, False),
    ("POST", re.compile(r"/api/ml/jobs/([0-9a-f]{12})/cancel"), r_cancel_job, False),
    ("GET", re.compile(r"/api/ml/training-runs"), r_list_runs, False),
    ("GET", re.compile(r"/api/ml/training-runs/([A-Za-z0-9_\-]+)"), r_get_run, False),
    ("GET", re.compile(r"/api/ml/training-runs/([A-Za-z0-9_\-]+)/plots/([a-z_]+\.png)"), r_get_plot, False),
    ("GET", re.compile(r"/api/ml/models"), r_models, False),
    ("POST", re.compile(r"/api/ml/predict"), r_predict, False),
]


class Handler(BaseHTTPRequestHandler):
    server_version = "MedFL-LocalML/1.0"

    def log_message(self, fmt, *args):  # path only - never query strings or bodies
        sys.stderr.write(f"[local_api] {self.command} {urlparse(self.path).path} {args[1] if len(args) > 1 else ''}\n")

    def _cors_headers(self):
        origin = self.headers.get("Origin")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _send(self, status: int, data: bytes, ctype: str):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def raw_body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError(400, "Invalid Content-Length.")
        if n < 0 or n > MAX_BODY:
            raise ApiError(413, "Request body too large.")
        return self.rfile.read(n)

    def json_body(self) -> dict:
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ApiError(415, "Expected application/json.")
        try:
            body = json.loads(self.raw_body() or b"{}")
        except ValueError:
            raise ApiError(400, "Malformed JSON body.")
        if not isinstance(body, dict):
            raise ApiError(400, "JSON body must be an object.")
        return body

    def do_OPTIONS(self):
        if self.headers.get("Origin") not in ORIGINS:
            return self._send(403, b'{"error": "Origin not allowed."}', "application/json")
        self.send_response(204)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        try:
            origin = self.headers.get("Origin")
            if origin is not None and origin not in ORIGINS:
                raise ApiError(403, "Origin not allowed.")
            url = urlparse(self.path)
            for route_method, pattern, handler, public in ROUTES:
                match = pattern.fullmatch(url.path)
                if match and route_method == method:
                    if not public:
                        authenticate(self.headers)
                    out = handler(self, match, parse_qs(url.query))
                    status = 200
                    if isinstance(out, tuple) and isinstance(out[0], int):
                        status, out = out
                    if isinstance(out, tuple):  # (content_type, bytes)
                        return self._send(status, out[1], out[0])
                    return self._send(status, json.dumps(out).encode("utf-8"), "application/json")
            raise ApiError(404, "Not found.")
        except ApiError as e:
            self._send(e.status, json.dumps({"error": str(e)}).encode("utf-8"), "application/json")
        except subprocess.TimeoutExpired:
            self._send(504, b'{"error": "The ML command timed out."}', "application/json")
        except Exception:
            traceback.print_exc()
            self._send(500, b'{"error": "Internal error in the local ML service."}', "application/json")


def make_server(port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, port), Handler)


def main():
    for d in (DATASETS_DIR, TRAINED_DIR, TMP_DIR):
        os.makedirs(d, exist_ok=True)
    server = make_server()
    print(f"MedFL local ML service on http://{HOST}:{PORT}/api/ml  (work dir: {WORK_DIR})")
    print(f"Allowed UI origins: {sorted(ORIGINS)} | auth via {BACKEND_URL}/auth/me")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in list(_jobs.values()):
            cancel_job(job)
        server.server_close()


if __name__ == "__main__":
    main()
