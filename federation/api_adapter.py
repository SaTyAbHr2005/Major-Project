import json
import sys
import argparse
import dataclasses
from enum import Enum
import sqlite3

from federation.storage import FederationStorage
from federation.jobs import JobManager
from federation.rounds import RoundManager
from federation.events import EventEmitter
from federation.models import now_iso, JobStatus, RoundStatus
from federation.round_plan import plan_status
from federation import participation

# The backend starts this adapter once per request. Intake, aggregation, evaluation and promotion
# pull in PyTorch/torchvision (about 7 s to import), so they are imported only by the actions that
# use them; listing jobs and models stays fast.

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.value
        return super().default(o)

def get_all_jobs(storage):
    with sqlite3.connect(storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM federation_jobs")
        rows = cur.fetchall()
        jobs = []
        for row in rows:
            job = dict(row)
            job['class_mapping'] = json.loads(job['class_mapping'])
            job['expected_participants'] = json.loads(job['expected_participants'])
            job['training_requirements'] = json.loads(job['training_requirements'])
            job['aggregation_configuration'] = json.loads(job['aggregation_configuration'])
            jobs.append(job)
        return jobs

def get_job_rounds(storage, job_id):
    with sqlite3.connect(storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM federation_rounds WHERE federation_job_id = ?", (job_id,))
        rows = cur.fetchall()
        rounds = []
        for row in rows:
            r = dict(row)
            r['expected_participants'] = json.loads(r['expected_participants'])
            r['received_participants'] = json.loads(r['received_participants'])
            r['accepted_participants'] = json.loads(r['accepted_participants'])
            r['rejected_participants'] = json.loads(r['rejected_participants'])
            r['quarantined_participants'] = json.loads(r['quarantined_participants'])
            rounds.append(r)
        return rounds

def get_round_updates(storage, round_id):
    with sqlite3.connect(storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM federation_updates WHERE round_id = ?", (round_id,))
        rows = cur.fetchall()
        updates = []
        for row in rows:
            u = dict(row)
            if u['parameter_metadata']:
                u['parameter_metadata'] = json.loads(u['parameter_metadata'])
            if u['training_configuration']:
                u['training_configuration'] = json.loads(u['training_configuration'])
            if u['validation_result']:
                u['validation_result'] = json.loads(u['validation_result'])
            updates.append(u)
        return updates

def get_all_global_models(storage):
    with sqlite3.connect(storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM global_model_versions")
        rows = cur.fetchall()
        models = []
        for row in rows:
            m = dict(row)
            if m.get('source_update_ids'):
                m['source_update_ids'] = json.loads(m['source_update_ids'])
            if m.get('parameter_metadata'):
                m['parameter_metadata'] = json.loads(m['parameter_metadata'])
            if m.get('aggregation_metadata'):
                m['aggregation_metadata'] = json.loads(m['aggregation_metadata'])
            if m.get('class_mapping'):
                try:
                    m['class_mapping'] = json.loads(m['class_mapping'])
                except (json.JSONDecodeError, TypeError):
                    # Guard: if stored value is not JSON (e.g., a status string due to a
                    # past column-order bug), leave it as a string so the error is visible
                    # in the response but the listing does not crash.
                    pass
            
            round_id = m.get('round_id')
            if round_id:
                cur2 = conn.cursor()
                cur2.execute("SELECT participant_id FROM federation_updates WHERE round_id = ? AND status = ?", (round_id, "USED_IN_AGGREGATION"))
                m['participating_hospitals'] = [r['participant_id'] for r in cur2.fetchall()]
            else:
                m['participating_hospitals'] = []
                
            models.append(m)
        return models


def get_all_evaluations(storage):
    with sqlite3.connect(storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM evaluation_records")
        rows = cur.fetchall()
        evals = []
        for row in rows:
            e = dict(row)
            if e.get('per_class_metrics'):
                e['per_class_metrics'] = json.loads(e['per_class_metrics'])
            if e.get('confusion_matrix'):
                e['confusion_matrix'] = json.loads(e['confusion_matrix'])
            if e.get('inference_configuration'):
                e['inference_configuration'] = json.loads(e['inference_configuration'])
            evals.append(e)
        return evals

def get_hospital_updates(storage, hospital_id):
    with sqlite3.connect(storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM federation_updates WHERE participant_id = ?", (hospital_id,))
        rows = cur.fetchall()
        updates = []
        for row in rows:
            u = dict(row)
            if u['parameter_metadata']:
                u['parameter_metadata'] = json.loads(u['parameter_metadata'])
            if u['training_configuration']:
                u['training_configuration'] = json.loads(u['training_configuration'])
            if u['validation_result']:
                u['validation_result'] = json.loads(u['validation_result'])
            updates.append(u)
        return updates

def job_plan(storage, job):
    """Round plan status of a job (dict as returned by get_all_jobs)."""
    agg = job.get("aggregation_configuration") or {}
    rounds = get_job_rounds(storage, job["federation_job_id"])
    numbers = {r["round_id"]: r["round_number"] for r in rounds}
    metric = (agg.get("early_stopping") or {}).get("metric", "accuracy")
    evals = [e for e in get_all_evaluations(storage) if e.get("status") == "COMPLETED"]
    scores = []
    for m in get_all_global_models(storage):
        if m["federation_job_id"] == job["federation_job_id"] and m.get("round_id") in numbers:
            done = sorted((e for e in evals if e["global_model_id"] == m["global_model_id"]), key=lambda e: e.get("completed_at") or "")
            if done:
                scores.append((numbers[m["round_id"]], float(done[-1][metric])))
    return plan_status(agg, rounds, scores)


def job_participants(storage, job_id):
    """Join requests of a job (dicts), including hospitals approved before requests existed."""
    job = storage.get_job(job_id)
    return [dataclasses.asdict(p) for p in participation.list_participants(storage, job)] if job else []


def find_job(storage, job_id):
    return next((j for j in get_all_jobs(storage) if j["federation_job_id"] == job_id), None)


def open_round(storage, req):
    """Create and open a round (admin request or automatic next round). Returns (round, base_model_source)."""
    import datetime
    from federation.base_models import prepare_base_model
    job = storage.get_job(req["job_id"])
    if not job:
        raise ValueError(f"Job {req['job_id']} does not exist.")
    plan = job_plan(storage, find_job(storage, req["job_id"]))
    if plan["finished"]:
        raise ValueError(f"Job {job.federation_job_id} is finished and opens no more rounds. {plan['reason']}")
    total = plan["total_rounds"]
    if total is not None and req["round_number"] > total:
        raise ValueError(f"Job {job.federation_job_id} is planned for {total} round(s); round {req['round_number']} is beyond the plan.")
    # Intake marks an update EXPIRED when now > deadline (string compare), so the
    # deadline is required, must be in the future and is stored in one UTC format.
    try:
        deadline = datetime.datetime.fromisoformat(str(req.get("deadline", "")).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("A valid round deadline (ISO 8601 date and time) is required.")
    if deadline.tzinfo is None:
        raise ValueError("The round deadline must include a timezone.")
    deadline = deadline.astimezone(datetime.timezone.utc)
    if deadline <= datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("The round deadline must be in the future.")

    # Every hospital must start from the same checkpoint; unless the caller names one,
    # the round starts from the job's latest global model (or a fresh seed for round 1).
    base_source = "provided"
    base_id, base_version, base_checksum = req.get("base_model_id"), req.get("base_model_version", 1), req.get("base_model_checksum")
    if not base_id:
        job_models = [m for m in get_all_global_models(storage) if m["federation_job_id"] == job.federation_job_id]
        previous = max(job_models, key=lambda m: m["version"]) if job_models else None
        base_id, base_version, base_checksum = prepare_base_model(storage, job, previous)
        base_source = previous["global_model_id"] if previous else "seed"

    rm = RoundManager(storage)
    rm.create_round(
        federation_job_id=req["job_id"],
        round_id=req["round_id"],
        round_number=req["round_number"],
        expected_participants=req.get("expected_participants", []),
        minimum_participants=req.get("minimum_participants", job.minimum_participants),
        deadline=deadline.isoformat(),
        base_model_id=base_id,
        base_model_version=base_version,
        base_model_checksum=base_checksum
    )
    if job.status == JobStatus.CREATED:
        job.status = JobStatus.ACTIVE
        storage.save_job(job)
    return rm.open_round(req["round_id"]), base_source


def after_aggregation(storage, events, job_id, global_model_id, eval_datasets_dir):
    """Apply the job's round plan once a round produced a global model.

    Returns (plan, next_round, warnings). With early stopping the new model is evaluated first,
    because the stopping rule only looks at evaluated rounds.
    """
    import datetime
    warnings = []
    job_dict = find_job(storage, job_id)
    agg = job_dict.get("aggregation_configuration") or {}
    if agg.get("early_stopping"):
        try:
            from federation.evaluation import EvaluationManager
            EvaluationManager(storage, events, eval_datasets_dir=eval_datasets_dir).evaluate(global_model_id)
        except Exception as e:  # aggregation already succeeded; report and carry on
            warnings.append(f"Early stopping could not be checked: evaluating {global_model_id} failed ({e}).")
    plan = job_plan(storage, job_dict)
    job = storage.get_job(job_id)
    if plan["finished"]:
        job.status = JobStatus.COMPLETED
        storage.save_job(job)
        return plan, None, warnings
    if not plan["auto_continue"] or plan["next_round_number"] is None:
        return plan, None, warnings
    n = plan["next_round_number"]
    deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=agg["round_duration_hours"])
    try:
        rnd, _ = open_round(storage, {"job_id": job_id, "round_id": f"{job_id}_R{n}", "round_number": n,
                                      "deadline": deadline.isoformat(), "minimum_participants": job.minimum_participants})
    except Exception as e:
        warnings.append(f"Round {n} could not be opened automatically: {e}")
        return plan, None, warnings
    return job_plan(storage, find_job(storage, job_id)), rnd, warnings


def main():
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            print(json.dumps({"success": False, "error": "No input provided"}))
            return
            
        req = json.loads(input_data)
        action = req.get("action")
        storage = FederationStorage()
        events = EventEmitter()

        if action == "get_architectures":
            import hospital_client.model_management.registry as registry
            # Get catalog and convert to dict list
            archs = []
            for name in registry.list_architectures():
                info = registry.get_architecture_info(name)
                archs.append(info.to_dict())
            print(json.dumps({"success": True, "architectures": archs}, cls=EnhancedJSONEncoder))
            return

        if action == "list_jobs":
            jobs = get_all_jobs(storage)
            for j in jobs:
                j['rounds'] = get_job_rounds(storage, j['federation_job_id'])
                j['plan'] = job_plan(storage, j)
                j['participants'] = job_participants(storage, j['federation_job_id'])
            print(json.dumps({"success": True, "jobs": jobs}, cls=EnhancedJSONEncoder))
            
        elif action == "get_job_details":
            job_id = req.get("job_id")
            jobs = get_all_jobs(storage)
            job = next((j for j in jobs if j['federation_job_id'] == job_id), None)
            if not job:
                print(json.dumps({"success": False, "error": "Job not found"}))
                return
            job['rounds'] = get_job_rounds(storage, job_id)
            job['plan'] = job_plan(storage, job)
            job['participants'] = job_participants(storage, job_id)
            for r in job['rounds']:
                r['updates'] = get_round_updates(storage, r['round_id'])
            print(json.dumps({"success": True, "job": job}, cls=EnhancedJSONEncoder))

        elif action == "create_job":
            jm = JobManager(storage)
            job = jm.create_job(
                federation_job_id=req["job_id"],
                task=req["task"],
                task_type=req.get("task_type", "image_classification"),
                architecture=req["architecture"],
                num_classes=req["num_classes"],
                class_mapping=req.get("class_mapping", {}),
                expected_participants=req.get("expected_participants", []),
                minimum_participants=req.get("minimum_participants", 2),
                deadline=req.get("deadline", ""),
                training_requirements=req.get("training_requirements", {}),
                aggregation_configuration=req.get("aggregation_configuration", {})
            )
            print(json.dumps({"success": True, "job": job}, cls=EnhancedJSONEncoder))

        elif action == "create_round":
            rnd, base_source = open_round(storage, req)
            print(json.dumps({"success": True, "round": rnd, "base_model_source": base_source}, cls=EnhancedJSONEncoder))

        elif action == "get_base_model":
            from federation.base_models import base_model_dir, BASE_MODEL_FILES
            rnd = storage.get_round(req["round_id"])
            if not rnd or not rnd.base_model_id:
                raise ValueError(f"Round {req['round_id']} has no base model.")
            print(json.dumps({
                "success": True, "base_model_id": rnd.base_model_id, "base_model_version": rnd.base_model_version,
                "base_model_checksum": rnd.base_model_checksum,
                "dir": base_model_dir(storage, rnd.base_model_id, rnd.base_model_version), "files": list(BASE_MODEL_FILES),
            }))

        elif action == "aggregate":
            rm = RoundManager(storage)
            # force_close=True: Admin may close the round before the deadline
            # if the minimum accepted participant count has been reached.
            # All non-deadline checks (minimum_participants, participant validation,
            # security hooks, state guards) remain fully active.
            force_close = bool(req.get("force_close", False))
            rm.check_deadline_and_participants(req["round_id"], force_close=force_close)
            from federation.aggregation import Aggregator
            agg = Aggregator(storage, events)
            # Look up job_id from the round if not explicitly provided
            job_id = req.get("job_id")
            if not job_id:
                rnd = storage.get_round(req["round_id"])
                job_id = rnd.federation_job_id
            gm = agg.aggregate_round(job_id, req["round_id"])
            closed = storage.get_round(req["round_id"])
            plan, next_round, warnings = after_aggregation(
                storage, events, job_id, gm.global_model_id, req.get("eval_datasets_dir", "evaluation_datasets"))
            print(json.dumps({"success": True, "global_model": gm, "round": closed, "plan": plan,
                              "next_round": next_round, "warnings": warnings}, cls=EnhancedJSONEncoder))


        elif action == "list_global_models":
            models = get_all_global_models(storage)
            evals = get_all_evaluations(storage)
            # attach evals to models
            for m in models:
                m['evaluations'] = [e for e in evals if e['global_model_id'] == m['global_model_id']]
            print(json.dumps({"success": True, "models": models}, cls=EnhancedJSONEncoder))

        elif action == "evaluate":
            from federation.evaluation import EvaluationManager
            ev = EvaluationManager(storage, events, eval_datasets_dir=req.get("eval_datasets_dir", "evaluation_datasets"))
            record = ev.evaluate(req["global_model_id"])
            # A manual evaluation can trigger the job's early-stopping rule.
            gm = storage.get_global_model(req["global_model_id"])
            plan = job_plan(storage, find_job(storage, gm.federation_job_id))
            if plan["finished"]:
                job = storage.get_job(gm.federation_job_id)
                job.status = JobStatus.COMPLETED
                storage.save_job(job)
            print(json.dumps({"success": True, "evaluation": record, "plan": plan}, cls=EnhancedJSONEncoder))

        elif action == "promote":
            from federation.promotion import PromotionManager, PromotionPolicy
            pm = PromotionManager(storage, events, PromotionPolicy())
            pd = pm.evaluate_promotion(req["global_model_id"], req["eval_record_id"])
            print(json.dumps({"success": True, "decision": pd}, cls=EnhancedJSONEncoder))
            
        elif action == "get_hospital_status":
            hosp_id = req["hospital_id"]
            updates = get_hospital_updates(storage, hosp_id)
            print(json.dumps({"success": True, "updates": updates}, cls=EnhancedJSONEncoder))

        elif action == "request_participation":
            # Hospital asks to join a job, with the summary of its local eligibility check.
            job = storage.get_job(req["job_id"])
            if not job:
                raise ValueError(f"Job {req['job_id']} does not exist.")
            p = participation.request_participation(storage, job, req["participant_id"], req.get("eligibility"))
            print(json.dumps({"success": True, "participant": p}, cls=EnhancedJSONEncoder))

        elif action == "decide_participation":
            # Admin approves or rejects a hospital's request.
            job = storage.get_job(req["job_id"])
            if not job:
                raise ValueError(f"Job {req['job_id']} does not exist.")
            p = participation.decide_participation(storage, job, req["participant_id"], req.get("decision"),
                                                   req.get("reason", ""), req.get("admin_id", ""))
            events.emit("audit", action=f"federation_participant_{p.authorization_status.lower()}",
                        job_id=job.federation_job_id, hospital_id=p.hospital_identifier,
                        admin=req.get("admin_id", ""), timestamp=now_iso())
            print(json.dumps({"success": True, "participant": p}, cls=EnhancedJSONEncoder))

        elif action == "register_participant":
            # Joining a round: only hospitals the admin approved for the job (participation.py).
            job_id = req["job_id"]
            round_id = req["round_id"]
            participant_id = req["participant_id"]

            job = storage.get_job(job_id)
            if not job:
                print(json.dumps({"success": False, "error": "Job not found"}))
                return
            if not participation.can_join(storage, job, participant_id):
                p = participation.get_participant(storage, job, participant_id)
                state = {None: "has not requested to join", "PENDING": "is waiting for admin approval",
                         "REJECTED": "was not approved"}.get(p.authorization_status if p else None, "is not approved")
                print(json.dumps({"success": False, "error": f"Hospital {participant_id} {state} for job {job_id}; "
                                                           "it cannot join this job's rounds."}))
                return
            if participant_id not in job.expected_participants:
                job.expected_participants.append(participant_id)
                storage.save_job(job)

            if round_id:
                rnd = storage.get_round(round_id)
                if not rnd or rnd.federation_job_id != job_id:
                    print(json.dumps({"success": False, "error": "Round not found in this job"}))
                    return
                if participant_id not in rnd.expected_participants:
                    rnd.expected_participants.append(participant_id)
                    storage.save_round(rnd)
            print(json.dumps({"success": True}))

        elif action == "submit_update":
            job_id = req["job_id"]
            round_id = req["round_id"]
            participant_id = req["participant_id"]
            handoff_dir = req["handoff_dir"]
            
            from federation.intake import IntakeManager
            from federation.validation import UpdateValidator
            from federation.security_hooks import M12SecurityHook
            im = IntakeManager(storage, UpdateValidator(storage), M12SecurityHook(), events)
            try:
                update = im.receive_handoff(job_id, round_id, participant_id, handoff_dir)
                print(json.dumps({"success": True, "update": update}, cls=EnhancedJSONEncoder))
            except Exception as e:
                print(json.dumps({"success": False, "error": str(e)}))

        elif action == "delete_job":
            job_id = req["job_id"]
            admin = req.get("admin_id", "admin")
            job = storage.get_job(job_id)
            if not job:
                print(json.dumps({"success": False, "error": "Job not found"}))
                return
            
            rounds = get_job_rounds(storage, job_id)
            has_history = False
            for r in rounds:
                updates = get_round_updates(storage, r["round_id"])
                if r["status"] not in ["CREATED", "OPEN", "RECEIVING"] or any(u["status"] in ["ACCEPTED", "USED_IN_AGGREGATION"] for u in updates):
                    has_history = True
            
            if has_history:
                job.status = JobStatus.ARCHIVED
                storage.save_job(job)
                events.emit("audit", action="federation_job_archived", job_id=job_id, admin=admin, timestamp=now_iso())
                print(json.dumps({"success": True, "action_taken": "archived"}))
            else:
                for r in rounds:
                    storage.delete_round(r["round_id"])
                storage.delete_job(job_id)
                import shutil, os
                job_dir = storage.get_job_artifact_dir(job_id)
                if os.path.exists(job_dir):
                    shutil.rmtree(job_dir)
                events.emit("audit", action="federation_job_deleted", job_id=job_id, admin=admin, timestamp=now_iso())
                print(json.dumps({"success": True, "action_taken": "deleted"}))

        elif action == "delete_round":
            round_id = req["round_id"]
            admin = req.get("admin_id", "admin")
            r = storage.get_round(round_id)
            if not r:
                print(json.dumps({"success": False, "error": "Round not found"}))
                return
                
            updates = get_round_updates(storage, round_id)
            has_history = False
            if r.status not in ["CREATED", "OPEN", "RECEIVING"] or any(u["status"] in ["ACCEPTED", "USED_IN_AGGREGATION"] for u in updates):
                has_history = True
                
            if has_history:
                r.status = RoundStatus.CANCELLED
                storage.save_round(r)
                events.emit("audit", action="federation_round_cancelled", round_id=round_id, admin=admin, timestamp=now_iso())
                print(json.dumps({"success": True, "action_taken": "cancelled"}))
            else:
                storage.delete_round(round_id)
                import shutil, os
                # get_round_artifact_dir returns the updates dir, but we should just rm the round dir
                round_dir = os.path.join(storage.artifacts_dir, "jobs", r.federation_job_id, "rounds", round_id)
                if os.path.exists(round_dir):
                    shutil.rmtree(round_dir)
                events.emit("audit", action="federation_round_deleted", round_id=round_id, admin=admin, timestamp=now_iso())
                print(json.dumps({"success": True, "action_taken": "deleted"}))

        else:
            print(json.dumps({"success": False, "error": f"Unknown action: {action}"}))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    main()
