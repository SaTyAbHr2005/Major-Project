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
under ML_WORK_DIR (default ~/medfl_ml_work), one workspace per hospital
(see Workspace); source datasets are only read:

    <ML_WORK_DIR>/<hospital_id>/
        datasets/<dataset_id>/           Module 4/5 outputs + Module 8 recommendations
        trained/<architecture>/          one folder per model architecture:
            <model_id>_run.json, _training_result.json, _resource_statistics.json,
            _plots/, _federation_handoff/, models/ (Module 6 store), state/
        trained/resource_training_history.json   Module 8 throughput history
        federation_runs/<job_id>/run_*/  job-filtered manifests (reference local files)
        tmp/                             inference uploads, deleted per request

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
from typing import Optional
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOST = "127.0.0.1"
PORT = int(os.environ.get("ML_API_PORT", "8765"))
ORIGINS = {
    o.strip()
    for o in os.environ.get(
        "ML_API_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174",
    ).split(",")
    if o.strip()
}
BACKEND_URL = os.environ.get("ML_BACKEND_URL", "http://localhost:5000/api/v1").rstrip("/")
WORK_DIR = os.path.abspath(os.environ.get("ML_WORK_DIR", os.path.join(os.path.expanduser("~"), "medfl_ml_work")))

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
HOSPITAL_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
OWNER_FILE = ".hospital_id"
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
    def __init__(self, kind: str, cmd: list, finalize, meta=None, owner=None, on_exit=None):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.cmd, self.finalize = kind, cmd, finalize
        self.meta = meta or {}
        self.owner = owner      # hospital_id that started it; only it may poll/cancel
        self.on_exit = on_exit  # runs once the process exits, success or failure, before finalize
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
        if job.on_exit is not None:
            try:
                job.on_exit(job)
            except Exception:
                traceback.print_exc()  # bookkeeping must never hide the real outcome
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


def start_job(kind: str, cmd: list, finalize, meta=None, owner=None, on_exit=None) -> Job:
    job = Job(kind, cmd, finalize, meta, owner, on_exit)
    with _jobs_lock:
        # Machine-wide on purpose: hospitals sharing one machine also share its GPU.
        if kind == "train" and any(j.kind == "train" and j.status == "running" for j in _jobs.values()):
            raise ApiError(409, "A training run is already in progress on this machine.")
        _jobs[job.id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.started_at)[:-MAX_JOBS]:
            if old.status != "running":
                _jobs.pop(old.id, None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def get_job(job_id: str, owner=None) -> Job:
    job = _jobs.get(job_id)
    if job is None or job.owner != owner:  # another hospital's job looks exactly like a missing one
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


class Workspace:
    """One hospital's outputs under WORK_DIR/<hospital_id>/ (layout: module docstring).

    Every authenticated route resolves its paths through the signed-in operator's
    workspace, so on a shared machine a hospital never sees another hospital's
    datasets, training runs, models or federation manifests."""

    def __init__(self, hospital_id):
        if not isinstance(hospital_id, str) or not HOSPITAL_ID_RE.match(hospital_id):
            raise ApiError(403, "Your account has no valid hospital id, so no local ML workspace can be opened for it.")
        self.hospital_id = hospital_id
        self.root = os.path.join(WORK_DIR, hospital_id)
        self.datasets = os.path.join(self.root, "datasets")
        self.trained = os.path.join(self.root, "trained")
        self.federation_runs = os.path.join(self.root, "federation_runs")
        self.tmp = os.path.join(self.root, "tmp")
        self._claim()

    def _claim(self):
        # Windows/macOS paths ignore case, so HOSP_1 and hosp_1 would share one folder:
        # the folder remembers its exact owner and refuses any other id.
        os.makedirs(self.root, exist_ok=True)
        marker = os.path.join(self.root, OWNER_FILE)
        try:
            with open(marker, "x", encoding="utf-8") as f:
                f.write(self.hospital_id)
        except FileExistsError:
            if _read_text(marker) != self.hospital_id:
                raise ApiError(403, "This machine's ML workspace folder for your hospital id belongs to a different hospital id.")

    def arch_dirs(self) -> list:
        if not os.path.isdir(self.trained):
            return []
        return [os.path.join(self.trained, a) for a in sorted(os.listdir(self.trained))
                if ARCH_RE.match(a) and os.path.isdir(os.path.join(self.trained, a))]

    def run_dir(self, model_id: str):
        """The trained/<architecture> folder holding this run's results, or None."""
        return next((d for d in self.arch_dirs()
                     if os.path.isfile(os.path.join(d, f"{model_id}_training_result.json"))), None)

    def models_root(self, model_id: str):
        """The Module 6 store (trained/<architecture>/models) that has this model, or None."""
        return next((os.path.join(d, "models") for d in self.arch_dirs()
                     if os.path.isdir(os.path.join(d, "models", model_id))), None)


def dataset_dir(ws: Workspace, dataset_id: str) -> str:
    if not DATASET_ID_RE.match(dataset_id or ""):
        raise ApiError(400, "Invalid dataset id.")
    d = os.path.join(ws.datasets, dataset_id)
    if not os.path.isfile(os.path.join(d, "source.json")):
        raise ApiError(404, "Unknown dataset. Inspect it first.")
    return d


def dataset_summary(ws: Workspace, dataset_id: str) -> dict:
    d = dataset_dir(ws, dataset_id)
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


def training_run(ws: Workspace, model_id: str) -> dict:
    if not MODEL_ID_RE.match(model_id or ""):
        raise ApiError(400, "Invalid model id.")
    d = ws.run_dir(model_id)
    if d is None:
        raise ApiError(404, "No training result for this model id.")
    plots_dir = os.path.join(d, f"{model_id}_plots")
    return {
        "model_id": model_id,
        "run": _read_json(os.path.join(d, f"{model_id}_run.json")),
        "result": _read_json(os.path.join(d, f"{model_id}_training_result.json")),
        "resource_statistics": _read_json(os.path.join(d, f"{model_id}_resource_statistics.json")),
        "plots": [p for p in PLOT_FILES if os.path.isfile(os.path.join(plots_dir, p))],
        "curve_metrics": _read_json(os.path.join(plots_dir, "curve_metrics.json")),
    }


def _run_meta_writer(ws: Workspace, model_id: str, meta: dict):
    """Job on_exit hook: Module 8 only fixes the architecture folder once it has picked the
    model, so the run's metadata is written next to its results after the process exits."""
    def write(job):
        d = ws.run_dir(model_id)
        if d is not None:
            with open(os.path.join(d, f"{model_id}_run.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
    return write


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text or "").strip("_")[:50] or "model"


# ---------------------------------------------------------------- routes

def r_health(h, m, q):
    return {"status": "ok", "service": "medfl-local-ml"}


def r_resources(h, m, q):
    os.makedirs(h.ws.trained, exist_ok=True)
    proc = subprocess.run(_module_cmd("hospital_client.resource_training", "detect", "--output", h.ws.trained),
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
    ws = h.ws
    if os.path.isdir(ws.datasets):
        for did in sorted(os.listdir(ws.datasets)):
            if DATASET_ID_RE.match(did) and os.path.isfile(os.path.join(ws.datasets, did, "source.json")):
                s = dataset_summary(ws, did)
                src = _read_json(os.path.join(ws.datasets, did, "source.json")) or {}
                out.append({k: s[k] for k in ("dataset_id", "name", "source_path")} | {
                    "total_samples": (s["profile"] or {}).get("total_samples"),
                    "classes": (s["profile"] or {}).get("classes"),
                    "preprocessed": s["preprocessing"] is not None,
                    "declared_task": src.get("declared_task"),
                })
    return {"datasets": out}


def r_get_dataset(h, m, q):
    return dataset_summary(h.ws, m.group(1))


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
    label_args = _label_source_args(body.get("label_source"), path)  # validated before anything is written

    ws = h.ws
    dataset_id = hashlib.sha256(os.path.normcase(path).encode("utf-8")).hexdigest()[:12]
    d = os.path.join(ws.datasets, dataset_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "source.json"), "w", encoding="utf-8") as f:
        json.dump({"path": path, "name": os.path.basename(path.rstrip("\\/")) or path}, f)
    # Generated outputs only - re-inspection invalidates everything derived from the old profile.
    for sub in ("m4", "m5"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    if os.path.exists(os.path.join(d, "recommendations.json")):
        os.remove(os.path.join(d, "recommendations.json"))

    cmd = _module_cmd("hospital_client.dataset", "inspect", path, "--output", os.path.join(d, "m4"), *label_args)
    job = start_job("inspect", cmd, lambda j: _finish_inspect(ws, dataset_id), {"dataset_id": dataset_id},
                    owner=ws.hospital_id)
    return 202, job.to_dict()


def _metadata_csv_path(path) -> str:
    if not isinstance(path, str) or not path.strip() or len(path) > 1000:
        raise ApiError(400, "A metadata CSV path is required.")
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not (os.path.isfile(path) and path.lower().endswith(".csv")):
        raise ApiError(400, "The metadata file must be an existing .csv file.")
    return path


def _label_source_args(spec, dataset_path: str) -> list:
    """Module 4 CLI arguments for image labels from a metadata CSV (none for folder labels)."""
    if not spec:
        return []
    if not os.path.isdir(dataset_path) or not isinstance(spec, dict):
        raise ApiError(400, "Labels from a metadata CSV apply to image folders only.")
    args = ["--labels-csv", _metadata_csv_path(spec.get("csv_path"))]
    for key, flag, required in (("image_column", "--image-column", True), ("label_column", "--label-column", True),
                                ("group_column", "--group-column", False)):
        value = spec.get(key)
        if not value and not required:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ApiError(400, f"Choose the {key.replace('_', ' ')} of the metadata CSV.")
        args += [flag, value]
    label_map = spec.get("label_map")
    if label_map is not None:
        if not (isinstance(label_map, dict) and 0 < len(label_map) <= 500 and all(
                isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in label_map.items())):
            raise ApiError(400, "The label map must map each CSV value to a class name or null.")
        args += ["--label-map", json.dumps(label_map)]
    return args


def r_metadata_csv(h, m, q):
    """Columns of a metadata CSV and, for a chosen label column, its distinct values with counts."""
    from hospital_client.dataset.ingestion.metadata_labels import label_value_counts, read_csv_columns
    body = h.json_body()
    path = _metadata_csv_path(body.get("path"))
    try:
        out = {"columns": read_csv_columns(path)}
        if body.get("label_column"):
            counts = label_value_counts(path, str(body["label_column"]))
            out["values"] = [{"value": v, "count": c} for v, c in sorted(counts.items(), key=lambda x: -x[1])]
    except ValueError as e:
        raise ApiError(400, str(e))
    except Exception:
        raise ApiError(400, "Could not read that file as a CSV.")
    return out


_BROWSE_CSV_SCRIPT = (
    "import tkinter as tk\nfrom tkinter import filedialog\n"
    "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
    "print(filedialog.askopenfilename(title='Locate the metadata CSV with image labels', "
    "filetypes=[('CSV files', '*.csv')]) or '')\n"
)


def r_browse_csv(h, m, q):
    """Native file picker for a metadata CSV on THIS machine."""
    proc = subprocess.run([sys.executable, "-c", _BROWSE_CSV_SCRIPT], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    path = proc.stdout.strip()
    return {"path": os.path.abspath(path) if path else None}


def _finish_inspect(ws: Workspace, dataset_id: str) -> dict:
    s = dataset_summary(ws, dataset_id)
    if s["profile"] is None:
        raise RuntimeError("Inspection finished but no dataset_profile.json was written.")
    return s


def r_preprocess(h, m, q):
    dataset_id, ws = m.group(1), h.ws
    d = dataset_dir(ws, dataset_id)
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
    # A dataset with only train/ and test/ gets a validation split carved from train
    # (test untouched); Module 7 needs a validation split to train.
    if body.get("generate_missing_splits", True):
        args.append("--generate-missing-splits")
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
                    lambda j: _finish_preprocess(ws, dataset_id), {"dataset_id": dataset_id}, owner=ws.hospital_id)
    return 202, job.to_dict()


def _finish_preprocess(ws: Workspace, dataset_id: str) -> dict:
    s = dataset_summary(ws, dataset_id)
    if s["preprocessing"] is None:
        raise RuntimeError("Preprocessing finished but no preprocessing_report.json was written.")
    return s


def _require_preprocessed(ws: Workspace, dataset_id: str) -> str:
    d = dataset_dir(ws, dataset_id)
    m5 = os.path.join(d, "m5")
    if not os.path.isfile(os.path.join(m5, "preprocessing_report.json")):
        raise ApiError(409, "Preprocess the dataset (Module 5) first.")
    return m5


def r_recommend(h, m, q):
    dataset_id, ws = m.group(1), h.ws
    m5 = _require_preprocessed(ws, dataset_id)
    os.makedirs(ws.trained, exist_ok=True)
    cmd = _module_cmd("hospital_client.resource_training", "recommend", "--dataset", m5, "--output", ws.trained)

    def finish(job):
        recs = parse_json_output("\n".join(job.lines))
        with open(os.path.join(dataset_dir(ws, dataset_id), "recommendations.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2)
        return recs

    job = start_job("recommend", cmd, finish, {"dataset_id": dataset_id}, owner=ws.hospital_id)
    return 202, job.to_dict()


def _choice_architecture(d: str, choice: str):
    """The model in this dataset's saved Module 8 recommendations for `choice`, or None."""
    recs = _read_json(os.path.join(d, "recommendations.json")) or {}
    arch = next((r.get("model") for r in recs.get("recommendations", []) if r.get("recommendation_type") == choice), None)
    return arch if isinstance(arch, str) and ARCH_RE.match(arch) else None


def r_train(h, m, q):
    dataset_id, ws = m.group(1), h.ws
    m5 = _require_preprocessed(ws, dataset_id)
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
        model_name = architecture
    elif choice in CHOICES:
        selection_args = ["--choice", choice]
        # Name the run after the model Module 8 recommended for this choice (e.g. resnet50);
        # the choice itself is kept in <model_id>_run.json.
        model_name = _choice_architecture(dataset_dir(ws, dataset_id), choice) or choice
    else:
        raise ApiError(400, f"Send choice (one of {sorted(CHOICES)}) or an architecture.")
    name = _read_json(os.path.join(dataset_dir(ws, dataset_id), "source.json"))["name"]
    model_id = f"{_slug(body.get('name') or name)}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(ws.trained, exist_ok=True)
    meta = {"dataset_id": dataset_id, "dataset_name": name, "choice": choice, "model_id": model_id,
            "request_id": body.get("request_id") if isinstance(body.get("request_id"), str) else None,
            "started_at": datetime.now().isoformat()}
    # Module 8 picks the final architecture, so it also places the run in trained/<architecture>/.
    cmd = _module_cmd("hospital_client.resource_training", "train", "--dataset", m5, "--output", ws.trained,
                      "--model-id", model_id, *selection_args, "--test", "--plots", "--architecture-subdir")
    job = start_job("train", cmd, lambda j: training_run(ws, model_id), meta,
                    owner=ws.hospital_id, on_exit=_run_meta_writer(ws, model_id, meta))
    return 202, job.to_dict()


def r_get_job(h, m, q):
    return get_job(m.group(1), h.ws.hospital_id).to_dict()


def r_cancel_job(h, m, q):
    job = get_job(m.group(1), h.ws.hospital_id)
    cancel_job(job)
    return job.to_dict()


def r_list_runs(h, m, q):
    runs = []
    for d in h.ws.arch_dirs():
        for fname in os.listdir(d):
            if not fname.endswith("_training_result.json"):
                continue
            model_id = fname[: -len("_training_result.json")]
            if not MODEL_ID_RE.match(model_id):
                continue
            res = _read_json(os.path.join(d, fname))
            run = _read_json(os.path.join(d, f"{model_id}_run.json")) or {}
            runs.append({
                "model_id": model_id, "architecture": res.get("architecture"), "status": res.get("status"),
                "epochs_run": res.get("epochs_run"), "duration_seconds": res.get("training_duration_seconds"),
                "validation_accuracy": (res.get("validation_metrics") or {}).get("accuracy"),
                "test_accuracy": (res.get("test_metrics") or {}).get("accuracy"),
                "checkpoint_version": res.get("checkpoint_version"),
                "ready_for_federation": res.get("ready_for_federation"),
                "dataset_name": run.get("dataset_name"), "choice": run.get("choice"),
                "job_id": run.get("job_id"), "round_id": run.get("round_id"),
                "generated_at": res.get("generated_at"),
            })
    runs.sort(key=lambda r: r.get("generated_at") or "", reverse=True)
    return {"runs": runs}


def r_get_run(h, m, q):
    return training_run(h.ws, m.group(1))


def r_get_plot(h, m, q):
    model_id, name = m.group(1), m.group(2)
    if not MODEL_ID_RE.match(model_id) or name not in PLOT_FILES:
        raise ApiError(404, "Unknown plot.")
    d = h.ws.run_dir(model_id)
    path = os.path.join(d, f"{model_id}_plots", name) if d else None
    if path is None or not os.path.isfile(path):
        raise ApiError(404, "Plot not found.")
    with open(path, "rb") as f:
        return ("image/png", f.read())


def r_models(h, m, q):
    from hospital_client.model_management.store import ModelStore
    out = []
    for d in h.ws.arch_dirs():
        models_root = os.path.join(d, "models")
        if not os.path.isdir(models_root):
            continue
        store = ModelStore(models_root)
        for model_id in store.list_model_ids():
            run = _read_json(os.path.join(d, f"{model_id}_run.json")) or {}
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
    models_root = h.ws.models_root(model_id)
    if models_root is None:
        raise ApiError(404, "Unknown model for this hospital.")

    os.makedirs(h.ws.tmp, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=ext, dir=h.ws.tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        proc = subprocess.run(
            _module_cmd("hospital_client.inference", "predict", "--models-root", models_root,
                        "--model-id", model_id, "--version", version, "--device", device, "--image", tmp, "--json"),
            cwd=REPO_ROOT, env=_child_env(), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
    finally:
        os.remove(tmp)  # the uploaded copy never outlives the request
    try:
        return parse_json_output(proc.stdout)
    except ValueError:
        raise ApiError(400, _failure_message(proc.stdout.splitlines(), proc.returncode))


def r_evaluate_eligibility(h, m, q):
    from hospital_client.federation.eligibility import evaluate_eligibility
    body = h.json_body()
    job = body.get("job")
    dataset_id = body.get("dataset_id")
    if not job or not dataset_id:
        raise ApiError(400, "job and dataset_id are required")

    m5 = _require_preprocessed(h.ws, dataset_id)
    d = dataset_dir(h.ws, dataset_id)
    src = _read_json(os.path.join(d, "source.json")) or {}
    
    # Use explicit declared_task — NEVER use dataset name as task
    hospital_task = src.get("declared_task")  # None if not yet associated
    
    # Hardware compatibility - check recommendations
    recs = _read_json(os.path.join(d, "recommendations.json")) or {}
    
    # Dataset profile from M5 (class list comes from M4 profile for eligibility)
    m4_profile = _read_json(os.path.join(d, "m4", "dataset_profile.json")) or {}
    m5_profile = _read_json(os.path.join(m5, "preprocessing_report.json")) or {}
    # Use M4 classes as the authoritative class list; M5 may also have it
    profile = {"classes": m4_profile.get("classes", m5_profile.get("classes", []))}
    
    res = evaluate_eligibility(
        hospital_task=hospital_task,
        hospital_dataset_profile=profile,
        manifest_dir=m5,
        job=job,
        m8_recommendation_set=recs
    )
    return res.to_dict()


def r_set_dataset_task(h, m, q):
    """Associate a local dataset with an explicit declared_task."""
    dataset_id = m.group(1)
    d = dataset_dir(h.ws, dataset_id)
    body = h.json_body()
    declared_task = body.get("declared_task")
    if declared_task is not None and (not isinstance(declared_task, str) or not declared_task.strip()):
        raise ApiError(400, "declared_task must be a non-empty string or null.")
    
    src_path = os.path.join(d, "source.json")
    src = _read_json(src_path) or {}
    if declared_task is not None:
        src["declared_task"] = declared_task.strip()
    else:
        src.pop("declared_task", None)
    with open(src_path, "w", encoding="utf-8") as f:
        json.dump(src, f)
    return {"dataset_id": dataset_id, "declared_task": src.get("declared_task")}


def r_get_task_vocabulary(h, m, q):
    """Return available task names from federation jobs via the backend."""
    try:
        token = h.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        req = urllib.request.Request(
            f"{BACKEND_URL}/federation/jobs",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        jobs = data.get("jobs", [])
        tasks = sorted(set(j.get("task", "") for j in jobs if j.get("task")))
        return {"tasks": tasks}
    except Exception:
        return {"tasks": []}


def _submit_handoff(ws, model_id: str, job_id: str, round_id: str, token: str, api_url: str = BACKEND_URL) -> dict:
    """POSTs this run's federation handoff folder to the Module 9 round (backend /submit)."""
    d = ws.run_dir(model_id)
    handoff_dir = os.path.join(d, f"{model_id}_federation_handoff") if d else None
    if not handoff_dir or not os.path.isdir(handoff_dir):
        raise ApiError(404, "This run has no federation handoff to submit.")
    req = urllib.request.Request(
        f"{api_url}/federation/jobs/{job_id}/rounds/{round_id}/submit",
        data=json.dumps({"handoff_dir": handoff_dir}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            err = json.load(e).get("error")
        except Exception:
            err = None
        msg = err.get("message") if isinstance(err, dict) else err
        raise ApiError(502, f"Federation server refused the update: {msg or e.reason}")
    except urllib.error.URLError as e:
        raise ApiError(502, f"Federation server unreachable: {e.reason}")


def _ensure_base_model(ws, architecture: str, rnd: dict, token: str) -> None:
    """Downloads the round's canonical base checkpoint into this hospital's model store
    when it is not there yet. The Trainer verifies its checksum against the round
    before the first training step, so a corrupted or altered copy is refused."""
    from hospital_client.model_management.validation import ARTIFACT_FILENAME, METADATA_FILENAME
    model_id, version, round_id = str(rnd.get("base_model_id") or ""), rnd.get("base_model_version"), str(rnd.get("round_id") or "")
    if not (MODEL_ID_RE.match(model_id) and MODEL_ID_RE.match(round_id) and isinstance(version, int) and version >= 1):
        raise ApiError(400, "The round's base model reference is invalid.")
    dest = os.path.join(ws.trained, architecture, "models", model_id, f"v{version}")
    if os.path.isdir(dest):
        return
    os.makedirs(ws.tmp, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=ws.tmp)
    try:
        for name in (ARTIFACT_FILENAME, METADATA_FILENAME):
            req = urllib.request.Request(f"{BACKEND_URL}/federation/rounds/{round_id}/base-model/{name}",
                                         headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=300) as r, open(os.path.join(tmp, name), "wb") as f:
                shutil.copyfileobj(r, f)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(tmp, dest)
    except urllib.error.HTTPError as e:
        raise ApiError(502, f"Could not download the round's base model {model_id} v{version}: {e.reason}")
    except urllib.error.URLError as e:
        raise ApiError(502, f"Federation server unreachable: {e.reason}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def r_submit_handoff(h, m, q):
    """Retry: submit a finished federated run to the job/round recorded in its run.json."""
    model_id = m.group(1)
    if not MODEL_ID_RE.match(model_id):
        raise ApiError(400, "Invalid model id.")
    d = h.ws.run_dir(model_id)
    run = (_read_json(os.path.join(d, f"{model_id}_run.json")) if d else None) or {}
    job_id, round_id = run.get("job_id"), run.get("round_id")
    if not (job_id and round_id and MODEL_ID_RE.match(job_id) and MODEL_ID_RE.match(round_id)):
        raise ApiError(409, "This run did not record its federation job and round, so it cannot be resubmitted.")
    token = h.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return _submit_handoff(h.ws, model_id, job_id, round_id, token)


def build_federated_training(ws, job: dict, dataset_id: str, round_info: Optional[dict]):
    """The Module 8 training command of a federated run (manual Participate and Flower rounds share it).
    Returns (cmd, meta, model_id, run_dir). round_info: the round's id and base model, or None."""
    from hospital_client.federation.manifest_filter import create_job_manifests
    # Both become folder/file names below, and they arrive in request bodies.
    if not MODEL_ID_RE.match(str(job.get("federation_job_id") or "")) or not ARCH_RE.match(str(job.get("architecture") or "")):
        raise ApiError(400, "Invalid federation job id or architecture.")

    m5 = _require_preprocessed(ws, dataset_id)
    class_mapping = job.get("class_mapping", {})

    # Create run-isolated filtered manifests. They reference this hospital's own
    # files, so they live in its workspace, not in a folder shared by all hospitals.
    run_dir, counts = create_job_manifests(
        job_id=job["federation_job_id"],
        original_manifest_dir=m5,
        output_base_dir=ws.federation_runs,
        class_mapping=class_mapping
    )

    if counts.get("train", 0) == 0:
        raise ApiError(400, "Filtered training dataset has 0 samples for this job's classes.")

    dataset_name = _read_json(os.path.join(dataset_dir(ws, dataset_id), "source.json"))["name"]
    model_id = f"FED_{job['federation_job_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not MODEL_ID_RE.match(model_id):
        raise ApiError(400, "Federation job id is too long.")

    os.makedirs(ws.trained, exist_ok=True)
    round_id = (round_info or {}).get("round_id")
    meta = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "choice": "federated",
        "model_id": model_id,
        "job_id": job["federation_job_id"],
        "round_id": round_id,
        "run_dir": run_dir,
        "started_at": datetime.now().isoformat()
    }

    # Build the M8 train command pointing at the run-isolated directory
    # --canonical-mapping passes the job's class mapping to M7 via M8
    cmd = _module_cmd(
        "hospital_client.resource_training", "train",
        "--dataset", run_dir,
        "--output", ws.trained,
        "--model-id", model_id,
        "--architecture", job["architecture"],
        "--canonical-mapping", json.dumps(class_mapping),
        "--federation-job-architecture", job["architecture"],
        "--architecture-subdir",  # the round's canonical base model is read from trained/<architecture>/models
    )
    # Same held-out evaluation and plots as a local run, when the dataset has a test split.
    if counts.get("test", 0) > 0:
        cmd.extend(["--test", "--plots"])

    # Pass epochs if specified in training requirements
    reqs = job.get("training_requirements") or {}
    epochs = reqs.get("epochs")
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    # FedProx jobs set the proximal term weight; Module 9 rejects updates trained with another value.
    if reqs.get("proximal_mu"):
        cmd.extend(["--proximal-mu", str(reqs["proximal_mu"])])

    if round_info and round_info.get("base_model_id"):
        cmd.extend([
            "--canonical-base-model-id", round_info["base_model_id"],
            "--canonical-base-model-version", str(round_info["base_model_version"]),
            "--canonical-base-model-checksum", round_info["base_model_checksum"]
        ])
    return cmd, meta, model_id, run_dir


def r_federated_train(h, m, q):
    body = h.json_body()
    job = body.get("job")
    dataset_id = body.get("dataset_id")
    if not job or not dataset_id:
        raise ApiError(400, "job and dataset_id are required")
    ws = h.ws
    round_id = body.get("round_id")
    auth_token = body.get("auth_token")
    api_url = body.get("api_url")

    # Find the active round to extract canonical base model provenance
    active_round = next((r for r in job.get("rounds", []) if r["round_id"] == round_id), None)
    cmd, meta, model_id, _ = build_federated_training(ws, job, dataset_id, active_round or {"round_id": round_id})
    if active_round and active_round.get("base_model_id"):
        token = h.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        _ensure_base_model(ws, job["architecture"], active_round, token)

    def _finalize(j):
        res = training_run(ws, model_id)
        if (res.get("result") or {}).get("ready_for_federation") and auth_token and api_url and round_id:
            try:
                _submit_handoff(ws, model_id, job["federation_job_id"], round_id, auth_token, api_url)
                print("Automatically submitted federation handoff to M9.")
            except ApiError as e:
                print("Failed to auto-submit federation handoff to M9:", e)
        return res

    job_handle = start_job("train", cmd, _finalize, meta,
                           owner=ws.hospital_id, on_exit=_run_meta_writer(ws, model_id, meta))
    return 202, job_handle.to_dict()


# ---------------------------------------------------------------- Flower jobs

ADDRESS_RE = re.compile(r"^[A-Za-z0-9.\-]+:\d{2,5}$")
FLOWER_CALIBRATION_BATCHES = 50
FLOWER_GPU_WAIT_SECONDS = 3600   # calibration waits at most this long for another training on this machine
_flower = {}                      # (hospital_id, job_id) -> FlowerConnection
_flower_lock = threading.Lock()


def _save_received_base_model(ws, job: dict, parameters, round_info: dict) -> None:
    """Stores the model the Flower server sent as the round's base checkpoint, where the Trainer reads it
    (trained/<architecture>/models/<id>/v<version>). Refused unless its checksum is the round's."""
    from hospital_client.model_management.config import ModelConfig
    from hospital_client.model_management.parameters import get_parameters, set_parameters
    from hospital_client.model_management.registry import build_model, get_architecture_info
    from hospital_client.model_management.store import ModelStore
    from hospital_client.training.federation import _compute_parameters_checksum
    model_id, version = str(round_info.get("base_model_id") or ""), round_info.get("base_model_version")
    if not (MODEL_ID_RE.match(model_id) and isinstance(version, int) and version >= 1):
        raise ApiError(400, "The round's base model reference is invalid.")
    arch = job["architecture"]
    store = ModelStore(os.path.join(ws.trained, arch, "models"))
    if os.path.isdir(os.path.join(store.root, model_id, f"v{version}")):
        return
    config = ModelConfig(architecture=arch, num_classes=int(job.get("num_classes") or len(job.get("class_mapping") or {})),
                         input_size=tuple(get_architecture_info(arch).default_input_size))
    model = build_model(config)
    set_parameters(model, parameters)
    if _compute_parameters_checksum(get_parameters(model)) != round_info.get("base_model_checksum"):
        raise ApiError(409, f"The model received for {model_id} v{version} does not match the round's checksum; "
                            "it was not used.")
    store.save_checkpoint(model, config, model_id, status="trained", version=version)


def _flower_runner(ws, job: dict, dataset_id: str):
    """(run_training, calibrate) for a FlowerConnection: the same training job as Participate, then
    BatchNorm calibration on the run's training images."""
    last = {}

    def run_training(parameters, config, cancel, on_job):
        round_info = {k: config[k] for k in ("round_id", "base_model_id", "base_model_checksum")}
        round_info["base_model_version"] = int(config["base_model_version"])
        _save_received_base_model(ws, job, parameters, round_info)
        cmd, meta, model_id, run_dir = build_federated_training(ws, job, dataset_id, round_info)
        last["run_dir"] = run_dir
        # Hospitals sharing this machine also share its GPU: wait for it instead of failing the round.
        while True:
            try:
                handle = start_job("train", cmd, lambda j: training_run(ws, model_id), meta,
                                   owner=ws.hospital_id, on_exit=_run_meta_writer(ws, model_id, meta))
                break
            except ApiError as e:
                if e.status != 409:
                    raise
                if cancel.wait(2.0):
                    raise RuntimeError("The round closed while this hospital waited for the machine's GPU.")
        on_job(handle.id)
        while handle.status == "running":
            if cancel.wait(1.0):
                cancel_job(handle)
                break
        while handle.status == "running":
            time.sleep(0.5)
        if handle.status != "succeeded":
            raise RuntimeError(handle.error or f"Training {handle.status}.")
        from hospital_client.training.federation import FederationHandoff
        d = ws.run_dir(model_id)
        handoff_dir = os.path.join(d, f"{model_id}_federation_handoff") if d else None
        if not handoff_dir or not os.path.isdir(handoff_dir):
            raise RuntimeError("Training finished without a federation update (see the training log).")
        handoff = FederationHandoff.load(handoff_dir)
        return handoff.parameters, handoff.num_train_samples, handoff.to_dict()

    def calibrate(parameters, config):
        import torch
        from torch.utils.data import DataLoader
        from hospital_client.federation.bn_calibration import calibrate_batchnorm
        from hospital_client.model_management.config import ModelConfig
        from hospital_client.model_management.parameters import get_parameters, set_parameters
        from hospital_client.model_management.registry import build_model, get_architecture_info
        from hospital_client.training.dataset import ManifestImageDataset, load_manifest
        if not last.get("run_dir"):
            raise RuntimeError("No training run of this job on this machine to calibrate with.")
        deadline = time.time() + FLOWER_GPU_WAIT_SECONDS
        while any(j.kind == "train" and j.status == "running" for j in list(_jobs.values())):
            if time.time() > deadline:
                raise RuntimeError("A training run is still in progress on this machine; calibration gave up waiting.")
            time.sleep(2.0)
        arch = job["architecture"]
        cfg = ModelConfig(architecture=arch, num_classes=int(job.get("num_classes") or len(job["class_mapping"])),
                          input_size=tuple(get_architecture_info(arch).default_input_size), color_mode="RGB")
        model = build_model(cfg)
        set_parameters(model, parameters)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        train = load_manifest(os.path.join(last["run_dir"], "train_manifest.json"))
        dataset = ManifestImageDataset(train, job["class_mapping"], cfg.color_mode, tuple(cfg.input_size))
        loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
        calibrate_batchnorm(model, loader, FLOWER_CALIBRATION_BATCHES)
        return get_parameters(model.cpu()), len(dataset)

    return run_training, calibrate


def _backend_reporter(job_id: str, conn_ref: dict):
    """Tells the backend this hospital's step (and epoch); returns the round's {round_id, closing}."""
    def report(st):
        job = _jobs.get(st.get("training_job_id") or "")
        point = job.epochs[-1] if job and job.epochs else {}
        body = {"round_id": st.get("round_id"), "state": st.get("state"), "message": st.get("message"),
                "epoch": point.get("epoch"), "total_epochs": point.get("total_epochs")}
        token = conn_ref["conn"].client().get_properties({})["token"]
        req = urllib.request.Request(f"{BACKEND_URL}/federation/jobs/{job_id}/flower/report",
                                     data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    return report


def r_flower_connect(h, m, q):
    from hospital_client.federation.flower_client import FlowerConnection
    body = h.json_body()
    job, dataset_id, address = body.get("job") or {}, body.get("dataset_id"), str(body.get("address") or "")
    job_id = str(job.get("federation_job_id") or "")
    if not MODEL_ID_RE.match(job_id) or not ARCH_RE.match(str(job.get("architecture") or "")):
        raise ApiError(400, "Invalid federation job id or architecture.")
    if not ADDRESS_RE.match(address):
        raise ApiError(400, "Invalid Flower server address (expected host:port).")
    if not DATASET_ID_RE.match(dataset_id or ""):
        raise ApiError(400, "Invalid dataset id.")
    token = h.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    key = (h.ws.hospital_id, job_id)
    with _flower_lock:
        old = _flower.get(key)
        if old is not None and old.active():
            raise ApiError(409, "This hospital is already connected to this job's federation server.")
        run_training, calibrate = _flower_runner(h.ws, job, dataset_id)
        ref = {}
        conn = FlowerConnection(h.ws.hospital_id, job_id, address, token, run_training, calibrate,
                                report=_backend_reporter(job_id, ref))
        ref["conn"] = conn
        _flower[key] = conn
    conn.start()
    return 202, conn.status()


def r_flower_disconnect(h, m, q):
    job_id = str(h.json_body().get("job_id") or "")
    conn = _flower.get((h.ws.hospital_id, job_id))
    if conn is None:
        raise ApiError(404, "Not connected to this job.")
    conn.stop()
    return conn.status()


def r_flower_status(h, m, q):
    token = h.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    mine = [c for (hid, _), c in sorted(_flower.items()) if hid == h.ws.hospital_id]
    for c in mine:
        c.update_token(token)   # the operator's newest sign-in; tokens expire after an hour
    return {"connections": [c.status() for c in mine]}


ROUTES = [
    ("GET", re.compile(r"/api/ml/health"), r_health, True),
    ("GET", re.compile(r"/api/ml/resources"), r_resources, False),
    ("POST", re.compile(r"/api/ml/browse-folder"), r_browse, False),
    ("POST", re.compile(r"/api/ml/browse-csv"), r_browse_csv, False),
    ("POST", re.compile(r"/api/ml/metadata-csv"), r_metadata_csv, False),
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
    ("POST", re.compile(r"/api/ml/federation/evaluate-eligibility"), r_evaluate_eligibility, False),
    ("POST", re.compile(r"/api/ml/federation/participate"), r_federated_train, False),
    ("POST", re.compile(r"/api/ml/federation/submit/([A-Za-z0-9_\-]+)"), r_submit_handoff, False),
    ("POST", re.compile(r"/api/ml/datasets/([0-9a-f]{12})/set-task"), r_set_dataset_task, False),
    ("GET", re.compile(r"/api/ml/federation/task-vocabulary"), r_get_task_vocabulary, False),
    ("POST", re.compile(r"/api/ml/federation/flower/connect"), r_flower_connect, False),
    ("POST", re.compile(r"/api/ml/federation/flower/disconnect"), r_flower_disconnect, False),
    ("GET", re.compile(r"/api/ml/federation/flower/status"), r_flower_status, False),
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
                        self.ws = Workspace(authenticate(self.headers).get("hospital_id"))
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
    os.makedirs(WORK_DIR, exist_ok=True)
    server = make_server()
    print(f"MedFL local ML service on http://{HOST}:{PORT}/api/ml  (work dir: {WORK_DIR}/<hospital_id>)")
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
