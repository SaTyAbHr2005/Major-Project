"""Live status of a job's Flower server, written to a JSON file the backend serves to the admin and hospital UIs."""
import json
import os
import threading
import time

from federation.models import now_iso

MAX_EVENTS = 200


class StatusWriter:
    """Thread-safe; every change rewrites the file atomically. Never put tokens in here."""

    def __init__(self, path: str, **fields):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = {"state": "starting", "fl_round": 0, "total_rounds": None, "round_id": None, "phase": None,
                     "closing": False, "message": "", "pid": os.getpid(), "started_at": now_iso(),
                     "hospitals": {}, "events": [], **fields}
        self._write()

    def set(self, **fields) -> None:
        with self._lock:
            self.data.update(fields)
            self._write()

    def hospital(self, hospital_id: str, state: str, message: str = "") -> None:
        with self._lock:
            self.data["hospitals"][hospital_id] = {"state": state, "message": message, "updated_at": now_iso()}
            self._write()

    def event(self, type_: str, text: str, **ids) -> None:
        with self._lock:
            self.data["events"] = (self.data["events"] + [{"at": now_iso(), "type": type_, "text": text, **ids}])[-MAX_EVENTS:]
            self._write()

    def heartbeat(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.data["updated_at"] = now_iso()
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        for _ in range(20):   # Windows refuses the replace while a reader has the file open
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05)
