"""
Hospital side of a Flower federation job (Module 9 integration, runs inside the local ML service).

A FlowerConnection keeps a Flower client connected to the job's Flower server in a background
thread. When the server asks this hospital to train, it runs the *same* training job the manual
Participate button starts (the local service supplies `run_training`), so the Training Monitor
follows it unchanged; when the server asks for calibration it re-measures BatchNorm statistics of
the new global model on this hospital's training images (`calibrate`). Raw images never leave.

Every step is visible in status(). While training, `report` (the local service's call to the
backend) is polled; when the server says the round is closing, training is cancelled and this
hospital's update for that round is skipped.
"""
import json
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import flwr as fl

# connecting → waiting → received_model → training → sending → waiting_for_others → calibrating → done
# (and missed / failed / disconnected)


class FlowerConnection:
    def __init__(self, hospital_id: str, job_id: str, address: str, token: str,
                 run_training: Callable, calibrate: Callable, report: Optional[Callable] = None,
                 poll_seconds: float = 5.0):
        self.hospital_id, self.job_id, self.address = hospital_id, job_id, address
        self._token = token
        self.run_training, self.calibrate, self.report, self.poll = run_training, calibrate, report, poll_seconds
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = None
        self._on_change = None   # test hook
        self._status = {"job_id": job_id, "state": "connecting", "phase": None, "fl_round": None, "total_rounds": None,
                        "round_id": None, "training_job_id": None, "message": "", "updated_at": _now()}

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────
    def start(self) -> None:
        def run():
            self._set(state="waiting", message="Connected to the federation server; waiting for the next round.")
            try:
                fl.client.start_client(server_address=self.address, client=self.client(), insecure=True)
                if not self._stopped:
                    self._set(state="disconnected", message="The federation server finished this job's rounds.")
            except Exception as e:  # server unreachable, connection lost
                if not self._stopped:
                    self._set(state="failed", message=f"Connection to the federation server failed: {e}")
        self._thread = threading.Thread(target=run, name=f"flower-{self.job_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # ponytail: Flower's start_client has no stop call; the thread ends when the server ends the run.
        self._stopped = True
        self._set(state="disconnected", message="Disconnected by this hospital.")

    def update_token(self, token: str) -> None:
        if token:
            with self._lock:
                self._token = token

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def active(self) -> bool:
        return not self._stopped and self.status()["state"] not in ("failed", "disconnected")

    def _set(self, **fields) -> None:
        with self._lock:
            self._status.update(fields, updated_at=_now())
            snapshot = dict(self._status)
        if self._on_change:
            self._on_change(snapshot)

    # ── Flower client ──────────────────────────────────────────────────────────────────────
    def client(self) -> fl.client.NumPyClient:
        conn = self

        class HospitalClient(fl.client.NumPyClient):
            def get_properties(self, config):
                with conn._lock:
                    return {"hospital_id": conn.hospital_id, "token": conn._token}

            def fit(self, parameters, config):
                if conn._stopped:
                    raise RuntimeError("This hospital disconnected from the federation job.")
                if config.get("phase") == "calibrate":
                    return conn._calibrate(parameters, config)
                return conn._train(parameters, config)

            def evaluate(self, parameters, config):
                return 0.0, 1, {}

        return HospitalClient()

    def _train(self, parameters, config):
        round_id = str(config["round_id"])
        self._set(state="received_model", phase="train", fl_round=int(config["fl_round"]),
                  total_rounds=int(config.get("total_rounds") or 0) or None, round_id=round_id,
                  training_job_id=None, message=f"Received the round {config['fl_round']} model.")
        cancel, finished = threading.Event(), threading.Event()
        if self.report:
            threading.Thread(target=self._watch_round, args=(round_id, cancel, finished), daemon=True).start()
        # Until the training job exists this hospital may be queued behind another training on the same GPU.
        self._set(state="training", message="Starting training; it waits while another training is using this machine's GPU.")
        try:
            params, n, metadata = self.run_training(parameters, config, cancel, lambda job_id: self._set(
                training_job_id=job_id, message="Training on this hospital's data."))
        except Exception as e:
            finished.set()
            if cancel.is_set():
                self._set(state="missed", message="The round closed before this hospital finished training; "
                                                   "its update for this round was skipped.")
                raise RuntimeError("Round closed before this hospital finished.") from e
            self._set(state="failed", message=str(e))
            raise
        finished.set()
        if cancel.is_set():
            self._set(state="missed", message="The round closed before this hospital finished training.")
            raise RuntimeError("Round closed before this hospital finished.")
        self._set(state="sending", message="Sending the trained weights to the federation server.")
        threading.Timer(2.0, self._after_send, args=(round_id,)).start()
        with self._lock:
            token = self._token
        return params, n, {"hospital_id": self.hospital_id, "token": token, "handoff_metadata": json.dumps(metadata)}

    def _after_send(self, round_id: str) -> None:
        st = self.status()
        if st["state"] == "sending" and st["round_id"] == round_id:
            self._set(state="waiting_for_others", message="Update sent; waiting for the other hospitals.")

    def _watch_round(self, round_id: str, cancel: threading.Event, finished: threading.Event) -> None:
        # Reports go on after training until the round moves on, so the backend also hears "update sent"
        # (the admin's sent count and Aggregate now depend on it). Only a still-training hospital cancels.
        while True:
            time.sleep(self.poll)
            st = self.status()
            if st["round_id"] != round_id or st["state"] not in ("training", "sending", "waiting_for_others"):
                return
            try:
                answer = self.report(st) or {}
            except Exception:
                continue  # backend briefly unreachable: keep training
            if answer.get("closing") and answer.get("round_id") == round_id and not finished.is_set():
                cancel.set()
                return

    def _calibrate(self, parameters, config):
        self._set(state="calibrating", phase="calibrate", message="Measuring BatchNorm statistics for the new global model.")
        try:
            params, n = self.calibrate(parameters, config)
        except Exception as e:
            self._set(state="failed", message=f"Calibration failed: {e}")
            raise
        self._set(state="done", phase=None, message=f"Round {config.get('fl_round')} complete.")
        return params, n, {"hospital_id": self.hospital_id}


def _now() -> str:
    return datetime.now().isoformat()
