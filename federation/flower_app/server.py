"""
Flower server for one application job (protocol "flower").

Every federated round is two Flower rounds:
  odd  (train)     — the Module 9 round is opened (or reused), the approved hospitals that are connected
                     receive the round's base model, train, and send their weights back; each update is
                     rebuilt as a FederationHandoff and goes through the normal intake validation; the
                     round is aggregated by the Module 9 Aggregator.
  even (calibrate) — the hospitals whose updates were accepted re-measure BatchNorm statistics for the new
                     global model on their own training images; the averaged statistics are written into
                     the global model, then the job's round plan is applied (after_aggregation), which opens
                     the next round or finishes the job.
Progress goes to a StatusWriter. A round closes early when the admin asks (close file) or when its
collection deadline passes; hospitals still training see `closing` and give up their update.
"""
import datetime
import json
import os
import shutil
import threading
import time
from typing import Callable, Dict, Optional

import flwr as fl
from flwr.common import FitIns, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays

from federation import participation
from federation.aggregation import Aggregator
from federation.api_adapter import after_aggregation, find_job, job_plan, open_round
from federation.base_models import base_model_store
from federation.events import EventEmitter
from federation.flower_app.engine import average_bn_statistics, handoff_from_parameters, replace_global_model_parameters
from federation.flower_app.status import StatusWriter
from federation.intake import IntakeManager
from federation.models import RoundStatus, UpdateStatus, now_iso
from federation.rounds import RoundManager
from federation.security_hooks import M12SecurityHook
from federation.validation import UpdateValidator

from hospital_client.model_management.parameters import get_parameters
from hospital_client.training.federation import FederationHandoff

GRACE_SECONDS = 120   # updates in flight when a round closes still arrive before the Module 9 deadline
COLLECTING = ("OPEN", "RECEIVING")


def _utc(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _state_keys(architecture: str, num_classes: int):
    from hospital_client.model_management.config import ModelConfig
    from hospital_client.model_management.registry import build_model, get_architecture_info
    cfg = ModelConfig(architecture=architecture, num_classes=num_classes,
                      input_size=get_architecture_info(architecture).default_input_size, color_mode="RGB")
    return list(build_model(cfg).state_dict())


class AppStrategy(fl.server.strategy.Strategy):
    def __init__(self, storage, job_id: str, status: StatusWriter, verify_identity: Callable[[str, str], bool],
                 close_path: Optional[str] = None, eval_datasets_dir: str = "evaluation_datasets",
                 poll_seconds: float = 2.0, events: Optional[EventEmitter] = None):
        self.storage, self.job_id, self.status = storage, job_id, status
        self.verify_identity, self.close_path = verify_identity, close_path
        self.eval_datasets_dir, self.poll = eval_datasets_dir, poll_seconds
        self.events = events or EventEmitter()
        job = storage.get_job(job_id)
        self.keys = _state_keys(job.architecture, job.num_classes)
        self.done = False
        self.round = None
        self.chosen: Dict[str, object] = {}     # hospital_id -> ClientProxy for the current round
        self.accepted = []                       # hospital ids whose update was accepted this round
        self.gm_id = None
        self._closing = threading.Event()
        self._round_over = threading.Event()

    # ── Flower hooks ───────────────────────────────────────────────────────────────────────
    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters([])        # every round's model comes from its Module 9 base model

    def configure_fit(self, server_round, parameters, client_manager):
        if self.done:
            return []
        try:
            return self._configure_train(client_manager) if server_round % 2 else self._configure_calibrate()
        except Exception as e:  # never leave the UIs on a step that is no longer running
            self._fail(f"Round could not start: {e}")
            return []

    def aggregate_fit(self, server_round, results, failures):
        if self.done and not self.chosen:
            return None, {}
        try:
            return (self._aggregate_train(results, failures) if server_round % 2
                    else self._aggregate_calibrate(results, failures)), {}
        except Exception as e:
            self._fail(f"Round {self.round.round_number if self.round else '?'} failed: {e}")
            return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None                              # evaluation stays an admin action (or early stopping)

    # ── train phase ────────────────────────────────────────────────────────────────────────
    def _current_round(self):
        job = find_job(self.storage, self.job_id)
        rounds = [r for r in self.storage.list_rounds_for_job(self.job_id) if r.status.value in COLLECTING]
        if rounds:
            return max(rounds, key=lambda r: r.round_number)
        plan = job_plan(self.storage, job)
        if plan["finished"] or plan["next_round_number"] is None:
            return None
        n = plan["next_round_number"]
        hours = (job.get("aggregation_configuration") or {})["round_duration_hours"]
        deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours)
        rnd, _ = open_round(self.storage, {"job_id": self.job_id, "round_id": f"{self.job_id}_R{n}", "round_number": n,
                                           "deadline": deadline.isoformat(), "minimum_participants": job["minimum_participants"]})
        self.status.event("round_opened", f"Round {n} opened", round_id=rnd.round_id, round_number=n)
        return rnd

    def _identify(self, proxy) -> Optional[str]:
        try:
            props = proxy.get_properties(GetPropertiesIns({}), timeout=30, group_id=None).properties
        except Exception:
            return None
        hid, token = str(props.get("hospital_id", "")), str(props.get("token", ""))
        return hid if hid and self.verify_identity(hid, token) else None

    def _wait_for_hospitals(self, rnd, client_manager) -> Optional[dict]:
        deadline = _utc(rnd.deadline)
        job = self.storage.get_job(self.job_id)
        while True:
            chosen = {}
            for proxy in client_manager.all().values():
                hid = self._identify(proxy)
                if hid and hid not in chosen and participation.can_join(self.storage, job, hid):
                    chosen[hid] = proxy
            for hid in chosen:
                if self.status.data["hospitals"].get(hid, {}).get("state") not in ("connected",):
                    self.status.hospital(hid, "connected")
            if len(chosen) >= rnd.minimum_participants:
                return chosen
            if datetime.datetime.now(datetime.timezone.utc) >= deadline:
                return None
            self.status.set(message=f"Waiting for approved hospitals: {len(chosen)} of {rnd.minimum_participants} connected.")
            time.sleep(self.poll)

    def _configure_train(self, client_manager):
        rnd = self._current_round()
        if rnd is None:
            self._finish("Every planned round is complete.")
            return []
        self.round, self.accepted, self.gm_id = rnd, [], None
        plan = job_plan(self.storage, find_job(self.storage, self.job_id))
        self.status.set(state="waiting_for_hospitals", fl_round=rnd.round_number, total_rounds=plan["total_rounds"],
                        round_id=rnd.round_id, phase="train", closing=False, deadline=rnd.deadline,
                        minimum_participants=rnd.minimum_participants, message="")
        chosen = self._wait_for_hospitals(rnd, client_manager)
        if chosen is None:
            connected = len([h for h, v in self.status.data["hospitals"].items() if v["state"] == "connected"])
            rnd = self.storage.get_round(rnd.round_id)
            rnd.status, rnd.closed_at = RoundStatus.INSUFFICIENT_PARTICIPANTS, now_iso()
            self.storage.save_round(rnd)
            self._fail(f"Round {rnd.round_number} needed at least {rnd.minimum_participants} approved hospital(s) "
                       f"connected before its deadline; {connected} were.")
            return []
        self.chosen = chosen
        rnd = self.storage.get_round(rnd.round_id)
        for hid in chosen:
            if hid not in rnd.expected_participants:
                rnd.expected_participants.append(hid)
        self.storage.save_round(rnd)
        self.round = rnd

        model, _ = base_model_store(self.storage).load_checkpoint(rnd.base_model_id, rnd.base_model_version)
        job = self.storage.get_job(self.job_id)
        reqs = job.training_requirements or {}
        config = {"phase": "train", "job_id": self.job_id, "round_id": rnd.round_id, "fl_round": rnd.round_number,
                  "total_rounds": plan["total_rounds"] or 0, "base_model_id": rnd.base_model_id,
                  "base_model_version": rnd.base_model_version, "base_model_checksum": rnd.base_model_checksum}
        if reqs.get("epochs") is not None:
            config["epochs"] = int(reqs["epochs"])
        if reqs.get("proximal_mu"):
            config["proximal_mu"] = float(reqs["proximal_mu"])
        ins = FitIns(ndarrays_to_parameters(get_parameters(model)), config)
        for hid in chosen:
            self.status.hospital(hid, "training")
        self.status.set(state="training", message="")
        self.status.event("round_started", f"Round {rnd.round_number}: {len(chosen)} hospital(s) training",
                          round_id=rnd.round_id, hospitals=sorted(chosen))
        self._closing.clear()
        self._round_over.clear()
        threading.Thread(target=self._watch_closing, args=(rnd,), daemon=True).start()
        return [(proxy, ins) for proxy in chosen.values()]

    def _watch_closing(self, rnd) -> None:
        deadline = _utc(rnd.deadline)
        duration = (deadline - _utc(rnd.created_at)).total_seconds() if getattr(rnd, "created_at", None) else GRACE_SECONDS * 10
        collect_until = deadline - datetime.timedelta(seconds=min(GRACE_SECONDS, max(duration, 0) * 0.1))
        while not self._round_over.wait(self.poll):
            reason = None
            if self.close_path and os.path.exists(self.close_path):
                with open(self.close_path) as f:
                    if f.read().strip() == rnd.round_id:
                        reason = "the admin chose Aggregate now"
            if reason is None and datetime.datetime.now(datetime.timezone.utc) >= collect_until:
                reason = "its deadline arrived"
            if reason:
                self._closing.set()
                self.status.set(closing=True, message=f"Round {rnd.round_number} is closing: {reason}.")
                self.status.event("round_closing", f"Round {rnd.round_number} is closing: {reason}.", round_id=rnd.round_id)
                return

    def _hospital_of(self, proxy) -> Optional[str]:
        return next((h for h, p in self.chosen.items() if p.cid == proxy.cid), None)

    def _aggregate_train(self, results, failures):
        self._round_over.set()
        rnd = self.round
        intake = IntakeManager(self.storage, UpdateValidator(self.storage), M12SecurityHook(), self.events)
        incoming = os.path.join(self.storage.get_round_artifact_dir(self.job_id, rnd.round_id), "incoming")
        answered = set()
        for proxy, res in results:
            hid = self._hospital_of(proxy)
            m = res.metrics or {}
            if hid is None or m.get("hospital_id") != hid:
                continue
            answered.add(hid)
            if not self.verify_identity(hid, str(m.get("token", ""))):
                self._reject(hid, "The hospital's sign-in could not be verified (expired?); its update was not used.")
                continue
            try:
                d = handoff_from_parameters(parameters_to_ndarrays(res.parameters), json.loads(m["handoff_metadata"]),
                                            os.path.join(incoming, hid))
                upd = intake.receive_handoff(self.job_id, rnd.round_id, hid, d)
            except Exception as e:
                self._reject(hid, f"Update could not be read: {e}")
                continue
            if upd.status == UpdateStatus.ACCEPTED:
                self.accepted.append(hid)
                self.status.hospital(hid, "sent", "Update accepted")
            else:
                self._reject(hid, upd.rejection_reason or upd.status.value)
        shutil.rmtree(incoming, ignore_errors=True)
        for hid in self.chosen:
            if hid not in answered:
                state = "missed" if self._closing.is_set() else "failed"
                self.status.hospital(hid, state, "No update before the round closed" if state == "missed" else "Training failed")
        if self.close_path and os.path.exists(self.close_path):
            os.remove(self.close_path)

        self.status.set(state="aggregating", closing=False, message="")
        RoundManager(self.storage).check_deadline_and_participants(rnd.round_id, force_close=True)
        gm = Aggregator(self.storage, self.events).aggregate_round(self.job_id, rnd.round_id)
        self.gm_id = gm.global_model_id
        self.status.event("round_closed", f"Round {rnd.round_number} aggregated from {len(self.accepted)} update(s)",
                          round_id=rnd.round_id, participants=sorted(self.accepted))
        return ndarrays_to_parameters(FederationHandoff.load(gm.artifact_location).parameters)

    def _reject(self, hid: str, reason: str) -> None:
        self.status.hospital(hid, "rejected", reason)
        self.status.event("update_rejected", f"{hid}: {reason}", hospital_id=hid)

    # ── calibrate phase ────────────────────────────────────────────────────────────────────
    def _configure_calibrate(self):
        if not self.gm_id:
            return []
        gm = self.storage.get_global_model(self.gm_id)
        params = FederationHandoff.load(gm.artifact_location).parameters
        config = {"phase": "calibrate", "job_id": self.job_id, "round_id": self.round.round_id,
                  "fl_round": self.round.round_number, "global_model_id": self.gm_id}
        targets = [(self.chosen[h], FitIns(ndarrays_to_parameters(params), config)) for h in self.accepted]
        for h in self.accepted:
            self.status.hospital(h, "calibrating")
        self.status.set(state="calibrating", phase="calibrate")
        return targets

    def _aggregate_calibrate(self, results, failures):
        gm = self.storage.get_global_model(self.gm_id)
        params = FederationHandoff.load(gm.artifact_location).parameters
        stats = [(parameters_to_ndarrays(res.parameters), res.num_examples) for proxy, res in results
                 if self._hospital_of(proxy) in self.accepted]
        if stats:
            params = average_bn_statistics(params, self.keys, stats)
            replace_global_model_parameters(self.storage, self.gm_id, params)
        else:
            self.status.event("warning", f"{self.gm_id}: no hospital returned BatchNorm statistics; kept the averaged ones.")
        plan, next_round, warnings = after_aggregation(self.storage, self.events, self.job_id, self.gm_id, self.eval_datasets_dir)
        version = self.storage.get_global_model(self.gm_id).version
        self.status.event("global_model_created", f"Global model v{version} created", global_model_id=self.gm_id,
                          round_id=self.round.round_id)
        for w in warnings:
            self.status.event("warning", w)
        for h in self.accepted:
            self.status.hospital(h, "done")
        if next_round is not None:
            self.status.event("round_opened", f"Round {next_round.round_number} opened",
                              round_id=next_round.round_id, round_number=next_round.round_number)
        self.chosen = {}
        if plan["finished"]:
            self._finish(plan["reason"] or "Every planned round is complete.")
        else:
            self.status.set(state="waiting_for_hospitals", phase=None, message="")
        return ndarrays_to_parameters(params)

    # ── end states ─────────────────────────────────────────────────────────────────────────
    def _finish(self, reason: str) -> None:
        self.done = True
        self.status.set(state="finished", phase=None, closing=False, message=reason)
        self.status.event("run_finished", reason)

    def _fail(self, reason: str) -> None:
        self.done, self.chosen = True, {}
        self._round_over.set()
        self.status.set(state="failed", phase=None, closing=False, message=reason)
        self.status.event("round_failed", reason)


def serve(storage, job_id: str, address: str, status_path: str, verify_identity: Callable[[str, str], bool],
          close_path: Optional[str] = None, public_address: Optional[str] = None,
          eval_datasets_dir: str = "evaluation_datasets", poll_seconds: float = 2.0) -> dict:
    """Runs the job's remaining planned rounds over Flower; returns the final status."""
    job = find_job(storage, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} does not exist.")
    plan = job_plan(storage, job)
    if plan["protocol"] != "flower":
        raise ValueError(f"Job {job_id} uses the manual protocol, not Flower.")
    if plan["finished"]:
        raise ValueError(f"Job {job_id} is finished. {plan['reason']}")
    remaining = plan["total_rounds"] - plan["rounds_completed"]
    hours = job["aggregation_configuration"]["round_duration_hours"]
    status = StatusWriter(status_path, job_id=job_id, address=public_address or address,
                          total_rounds=plan["total_rounds"], minimum_participants=job["minimum_participants"])
    strategy = AppStrategy(storage, job_id, status, verify_identity, close_path, eval_datasets_dir, poll_seconds)
    stop = threading.Event()

    def beat():
        while not stop.wait(5):
            status.heartbeat()
    threading.Thread(target=beat, daemon=True).start()
    try:
        fl.server.start_server(server_address=address, strategy=strategy,
                               config=fl.server.ServerConfig(num_rounds=2 * remaining, round_timeout=hours * 3600 + 600))
    finally:
        stop.set()
    if status.data["state"] not in ("finished", "failed"):
        strategy._finish("All rounds ran.")
    return status.data
