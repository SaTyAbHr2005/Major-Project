import os
import json
import sqlite3
import uuid
from typing import List, Optional, Dict, Any
from .models import (
    FederationJob, FederationParticipant, FederationRound,
    FederationUpdate, GlobalModelVersion, EvaluationRecord,
    PromotionDecision, JobStatus, RoundStatus, UpdateStatus,
    GlobalModelStatus, PromotionDecisionValue, now_iso
)

class FederationStorage:
    def __init__(self, db_path: str = "federation_metadata.db", artifacts_dir: str = "federation_artifacts"):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS federation_jobs (
                    federation_job_id TEXT PRIMARY KEY,
                    task TEXT,
                    task_type TEXT,
                    architecture TEXT,
                    num_classes INTEGER,
                    class_mapping TEXT,
                    expected_participants TEXT,
                    minimum_participants INTEGER,
                    deadline TEXT,
                    training_requirements TEXT,
                    aggregation_configuration TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS federation_participants (
                    participant_id TEXT PRIMARY KEY,
                    federation_job_id TEXT,
                    hospital_identifier TEXT,
                    authorization_status TEXT,
                    expected_status TEXT,
                    last_submission TEXT,
                    participation_status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS federation_rounds (
                    round_id TEXT PRIMARY KEY,
                    federation_job_id TEXT,
                    round_number INTEGER,
                    expected_participants TEXT,
                    minimum_participants INTEGER,
                    deadline TEXT,
                    base_model_id TEXT,
                    base_model_version INTEGER,
                    base_model_checksum TEXT,
                    status TEXT,
                    received_participants TEXT,
                    accepted_participants TEXT,
                    rejected_participants TEXT,
                    quarantined_participants TEXT,
                    created_at TEXT,
                    opened_at TEXT,
                    closed_at TEXT,
                    aggregation_started_at TEXT,
                    aggregation_completed_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS federation_updates (
                    update_id TEXT PRIMARY KEY,
                    federation_job_id TEXT,
                    round_id TEXT,
                    participant_id TEXT,
                    model_id TEXT,
                    architecture TEXT,
                    artifact_location TEXT,
                    artifact_checksum TEXT,
                    parameter_count INTEGER,
                    parameter_metadata TEXT,
                    num_train_samples INTEGER,
                    training_configuration TEXT,
                    validation_result TEXT,
                    rejection_reason TEXT,
                    status TEXT,
                    received_at TEXT,
                    validated_at TEXT,
                    used_in_aggregation_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS global_model_versions (
                    global_model_id TEXT PRIMARY KEY,
                    task TEXT,
                    task_type TEXT,
                    architecture TEXT,
                    version INTEGER,
                    artifact_location TEXT,
                    artifact_checksum TEXT,
                    artifact_size INTEGER,
                    parameter_count INTEGER,
                    parameter_metadata TEXT,
                    federation_job_id TEXT,
                    round_id TEXT,
                    source_update_ids TEXT,
                    aggregation_method TEXT,
                    aggregation_metadata TEXT,
                    class_mapping TEXT DEFAULT '{}',
                    schema_hash TEXT DEFAULT 'LEGACY',
                    status TEXT,
                    created_at TEXT
                )
            ''')
            
            # Migration for existing databases
            try:
                cursor.execute("ALTER TABLE global_model_versions ADD COLUMN class_mapping TEXT DEFAULT '{}'")
            except sqlite3.OperationalError:
                pass
            # Participation requests (added 2026-10-14)
            for column in ("eligibility TEXT DEFAULT '{}'", "decision_reason TEXT DEFAULT ''", "decided_by TEXT DEFAULT ''"):
                try:
                    cursor.execute(f"ALTER TABLE federation_participants ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass
            try:
                cursor.execute("ALTER TABLE global_model_versions ADD COLUMN schema_hash TEXT DEFAULT 'LEGACY'")
            except sqlite3.OperationalError:
                pass
            # Module 12 security updates migration
            for column in ("security_status TEXT DEFAULT 'ACCEPT'", "security_reason TEXT DEFAULT ''", "security_metrics TEXT DEFAULT '{}'"):
                try:
                    cursor.execute(f"ALTER TABLE federation_updates ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_records (
                    evaluation_id TEXT PRIMARY KEY,
                    global_model_id TEXT,
                    model_version INTEGER,
                    task TEXT,
                    architecture TEXT,
                    evaluation_dataset_id TEXT,
                    evaluation_dataset_version TEXT,
                    sample_count INTEGER,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    per_class_metrics TEXT,
                    confusion_matrix TEXT,
                    predictions_location TEXT,
                    metrics_location TEXT,
                    inference_configuration TEXT,
                    status TEXT,
                    started_at TEXT,
                    completed_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS promotion_decisions (
                    promotion_decision_id TEXT PRIMARY KEY,
                    candidate_global_model_id TEXT,
                    previous_main_global_model_id TEXT,
                    decision TEXT,
                    promotion_policy_version TEXT,
                    evaluation_record_id TEXT,
                    reason TEXT,
                    metric_summary TEXT,
                    created_at TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS federation_dp_ledger (
                    entry_id TEXT PRIMARY KEY,
                    federation_job_id TEXT,
                    round_id TEXT,
                    participant_id TEXT,
                    noise_multiplier REAL,
                    sample_rate REAL,
                    steps INTEGER,
                    delta REAL,
                    round_epsilon REAL,
                    cumulative_epsilon REAL,
                    created_at TEXT
                )
            ''')
            
            conn.commit()

    def get_job_artifact_dir(self, job_id: str) -> str:
        d = os.path.join(self.artifacts_dir, "jobs", job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def get_round_artifact_dir(self, job_id: str, round_id: str) -> str:
        d = os.path.join(self.artifacts_dir, "jobs", job_id, "rounds", round_id, "updates")
        os.makedirs(d, exist_ok=True)
        return d

    def get_global_model_dir(self, task: str, architecture: str, version: int) -> str:
        d = os.path.join(self.artifacts_dir, "global_models", task, architecture, f"v{version}")
        os.makedirs(d, exist_ok=True)
        return d

    # --- FederationJob ---
    def save_job(self, job: FederationJob):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO federation_jobs 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.federation_job_id, job.task, job.task_type, job.architecture,
                job.num_classes, json.dumps(job.class_mapping), json.dumps(job.expected_participants),
                job.minimum_participants, job.deadline, json.dumps(job.training_requirements),
                json.dumps(job.aggregation_configuration), job.status.value, job.created_at, job.updated_at
            ))

    def get_job(self, job_id: str) -> Optional[FederationJob]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM federation_jobs WHERE federation_job_id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                return FederationJob(
                    federation_job_id=row[0], task=row[1], task_type=row[2], architecture=row[3],
                    num_classes=row[4], class_mapping=json.loads(row[5]), expected_participants=json.loads(row[6]),
                    minimum_participants=row[7], deadline=row[8], training_requirements=json.loads(row[9]),
                    aggregation_configuration=json.loads(row[10]), status=JobStatus(row[11]), created_at=row[12], updated_at=row[13]
                )
        return None

    def delete_job(self, job_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM federation_jobs WHERE federation_job_id = ?', (job_id,))

    # --- FederationParticipant (join requests) ---
    _PARTICIPANT_COLUMNS = ("participant_id", "federation_job_id", "hospital_identifier", "authorization_status",
                            "expected_status", "last_submission", "participation_status", "created_at", "updated_at",
                            "eligibility", "decision_reason", "decided_by")

    def save_participant(self, p: FederationParticipant):
        values = {**{c: getattr(p, c) for c in self._PARTICIPANT_COLUMNS}, "eligibility": json.dumps(p.eligibility)}
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO federation_participants ({', '.join(self._PARTICIPANT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in self._PARTICIPANT_COLUMNS)})",
                [values[c] for c in self._PARTICIPANT_COLUMNS])

    def _participant(self, row) -> FederationParticipant:
        data = {c: row[c] for c in self._PARTICIPANT_COLUMNS}
        data["eligibility"] = json.loads(data["eligibility"] or "{}")
        data["decision_reason"] = data["decision_reason"] or ""
        data["decided_by"] = data["decided_by"] or ""
        return FederationParticipant(**data)

    def get_participant(self, participant_id: str) -> Optional[FederationParticipant]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM federation_participants WHERE participant_id = ?', (participant_id,)).fetchone()
            return self._participant(row) if row else None

    def list_participants(self, job_id: str) -> List[FederationParticipant]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('SELECT * FROM federation_participants WHERE federation_job_id = ? ORDER BY created_at',
                                (job_id,)).fetchall()
            return [self._participant(r) for r in rows]

    # --- FederationRound ---
    def list_rounds_for_job(self, job_id: str) -> List[FederationRound]:
        with sqlite3.connect(self.db_path) as conn:
            ids = [r[0] for r in conn.execute('SELECT round_id FROM federation_rounds WHERE federation_job_id = ?', (job_id,))]
        return [self.get_round(i) for i in ids]

    def save_round(self, r: FederationRound):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO federation_rounds 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                r.round_id, r.federation_job_id, r.round_number, json.dumps(r.expected_participants),
                r.minimum_participants, r.deadline, r.base_model_id, r.base_model_version, r.base_model_checksum,
                r.status.value,
                json.dumps(r.received_participants), json.dumps(r.accepted_participants),
                json.dumps(r.rejected_participants), json.dumps(r.quarantined_participants),
                r.created_at, r.opened_at, r.closed_at, r.aggregation_started_at, r.aggregation_completed_at
            ))

    def get_round(self, round_id: str) -> Optional[FederationRound]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM federation_rounds WHERE round_id = ?', (round_id,))
            row = cursor.fetchone()
            if row:
                return FederationRound(
                    round_id=row[0], federation_job_id=row[1], round_number=row[2],
                    expected_participants=json.loads(row[3]), minimum_participants=row[4], deadline=row[5],
                    base_model_id=row[6], base_model_version=row[7], base_model_checksum=row[8],
                    status=RoundStatus(row[9]), received_participants=json.loads(row[10]),
                    accepted_participants=json.loads(row[11]), rejected_participants=json.loads(row[12]),
                    quarantined_participants=json.loads(row[13]), created_at=row[14], opened_at=row[15],
                    closed_at=row[16], aggregation_started_at=row[17], aggregation_completed_at=row[18]
                )
        return None

    def delete_round(self, round_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM federation_rounds WHERE round_id = ?', (round_id,))
            
    # --- FederationUpdate ---
    def save_update(self, u: FederationUpdate):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO federation_updates 
                (update_id, federation_job_id, round_id, participant_id, model_id,
                 architecture, artifact_location, artifact_checksum, parameter_count,
                 parameter_metadata, num_train_samples, training_configuration,
                 validation_result, rejection_reason, status, received_at,
                 validated_at, used_in_aggregation_at, security_status, security_reason, security_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                u.update_id, u.federation_job_id, u.round_id, u.participant_id, u.model_id,
                u.architecture, u.artifact_location, u.artifact_checksum, u.parameter_count,
                json.dumps(u.parameter_metadata), u.num_train_samples, json.dumps(u.training_configuration),
                u.validation_result, u.rejection_reason, u.status.value, u.received_at, u.validated_at, u.used_in_aggregation_at,
                getattr(u, "security_status", "ACCEPT"), getattr(u, "security_reason", ""),
                json.dumps(getattr(u, "security_metrics", {}) or {})
            ))

    def get_update(self, update_id: str) -> Optional[FederationUpdate]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM federation_updates WHERE update_id = ?', (update_id,))
            row = cursor.fetchone()
            if row:
                keys = row.keys()
                return FederationUpdate(
                    update_id=row['update_id'], federation_job_id=row['federation_job_id'],
                    round_id=row['round_id'], participant_id=row['participant_id'],
                    model_id=row['model_id'], architecture=row['architecture'],
                    artifact_location=row['artifact_location'], artifact_checksum=row['artifact_checksum'],
                    parameter_count=row['parameter_count'],
                    parameter_metadata=json.loads(row['parameter_metadata']),
                    num_train_samples=row['num_train_samples'],
                    training_configuration=json.loads(row['training_configuration']),
                    validation_result=row['validation_result'],
                    rejection_reason=row['rejection_reason'],
                    security_status=row['security_status'] if 'security_status' in keys else "ACCEPT",
                    security_reason=row['security_reason'] if 'security_reason' in keys else "",
                    security_metrics=json.loads(row['security_metrics']) if 'security_metrics' in keys and row['security_metrics'] else {},
                    status=UpdateStatus(row['status']),
                    received_at=row['received_at'],
                    validated_at=row['validated_at'],
                    used_in_aggregation_at=row['used_in_aggregation_at']
                )
        return None

    def list_updates_for_round(self, round_id: str, status: Optional[UpdateStatus] = None) -> List[FederationUpdate]:
        updates = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cursor = conn.execute('SELECT * FROM federation_updates WHERE round_id = ? AND status = ?', (round_id, status.value))
            else:
                cursor = conn.execute('SELECT * FROM federation_updates WHERE round_id = ?', (round_id,))
            
            for row in cursor.fetchall():
                keys = row.keys()
                updates.append(FederationUpdate(
                    update_id=row['update_id'], federation_job_id=row['federation_job_id'],
                    round_id=row['round_id'], participant_id=row['participant_id'],
                    model_id=row['model_id'], architecture=row['architecture'],
                    artifact_location=row['artifact_location'], artifact_checksum=row['artifact_checksum'],
                    parameter_count=row['parameter_count'],
                    parameter_metadata=json.loads(row['parameter_metadata']),
                    num_train_samples=row['num_train_samples'],
                    training_configuration=json.loads(row['training_configuration']),
                    validation_result=row['validation_result'],
                    rejection_reason=row['rejection_reason'],
                    security_status=row['security_status'] if 'security_status' in keys else "ACCEPT",
                    security_reason=row['security_reason'] if 'security_reason' in keys else "",
                    security_metrics=json.loads(row['security_metrics']) if 'security_metrics' in keys and row['security_metrics'] else {},
                    status=UpdateStatus(row['status']),
                    received_at=row['received_at'],
                    validated_at=row['validated_at'],
                    used_in_aggregation_at=row['used_in_aggregation_at']
                ))
        return updates

    # --- GlobalModelVersion ---
    def save_global_model(self, m: GlobalModelVersion):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO global_model_versions 
                (global_model_id, task, task_type, architecture, version,
                 artifact_location, artifact_checksum, artifact_size, parameter_count,
                 parameter_metadata, federation_job_id, round_id,
                 source_update_ids, aggregation_method, aggregation_metadata,
                 class_mapping, schema_hash, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                m.global_model_id, m.task, m.task_type, m.architecture, m.version,
                m.artifact_location, m.artifact_checksum, m.artifact_size, m.parameter_count,
                json.dumps(m.parameter_metadata), m.federation_job_id, m.round_id,
                json.dumps(m.source_update_ids), m.aggregation_method, json.dumps(m.aggregation_metadata),
                json.dumps(m.class_mapping), m.schema_hash, m.status.value, m.created_at
            ))

            
    def get_global_model(self, global_model_id: str) -> Optional[GlobalModelVersion]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM global_model_versions WHERE global_model_id = ?', (global_model_id,))
            row = cursor.fetchone()
            if row:
                return GlobalModelVersion(
                    global_model_id=row['global_model_id'], task=row['task'], task_type=row['task_type'],
                    architecture=row['architecture'], version=row['version'],
                    artifact_location=row['artifact_location'], artifact_checksum=row['artifact_checksum'],
                    artifact_size=row['artifact_size'], parameter_count=row['parameter_count'],
                    parameter_metadata=json.loads(row['parameter_metadata']),
                    federation_job_id=row['federation_job_id'], round_id=row['round_id'],
                    source_update_ids=json.loads(row['source_update_ids']),
                    aggregation_method=row['aggregation_method'],
                    aggregation_metadata=json.loads(row['aggregation_metadata']),
                    class_mapping=json.loads(row['class_mapping']),
                    schema_hash=row['schema_hash'],
                    status=GlobalModelStatus(row['status']),
                    created_at=row['created_at']
                )
        return None

    def get_latest_global_model_version(self, task: str, architecture: str, schema_hash: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT MAX(version) FROM global_model_versions WHERE task = ? AND architecture = ? AND schema_hash = ?', (task, architecture, schema_hash))
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else 0

    def get_main_global_model(self, task: str, architecture: str, schema_hash: str) -> Optional[GlobalModelVersion]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM global_model_versions WHERE task = ? AND architecture = ? AND schema_hash = ? AND status = ?',
                (task, architecture, schema_hash, GlobalModelStatus.MAIN.value)
            )
            row = cursor.fetchone()
            if row:
                return GlobalModelVersion(
                    global_model_id=row['global_model_id'], task=row['task'], task_type=row['task_type'],
                    architecture=row['architecture'], version=row['version'],
                    artifact_location=row['artifact_location'], artifact_checksum=row['artifact_checksum'],
                    artifact_size=row['artifact_size'], parameter_count=row['parameter_count'],
                    parameter_metadata=json.loads(row['parameter_metadata']),
                    federation_job_id=row['federation_job_id'], round_id=row['round_id'],
                    source_update_ids=json.loads(row['source_update_ids']),
                    aggregation_method=row['aggregation_method'],
                    aggregation_metadata=json.loads(row['aggregation_metadata']),
                    class_mapping=json.loads(row['class_mapping']),
                    schema_hash=row['schema_hash'],
                    status=GlobalModelStatus(row['status']),
                    created_at=row['created_at']
                )
        return None

    # --- EvaluationRecord and PromotionDecision ---
    _EVALUATION_COLUMNS = ("evaluation_id", "global_model_id", "model_version", "task", "architecture",
                           "evaluation_dataset_id", "evaluation_dataset_version", "sample_count", "accuracy", "precision",
                           "recall", "f1", "per_class_metrics", "confusion_matrix", "predictions_location",
                           "metrics_location", "inference_configuration", "status", "started_at", "completed_at")
    _EVALUATION_JSON = ("per_class_metrics", "confusion_matrix", "inference_configuration")

    def save_evaluation_record(self, record: EvaluationRecord):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f'''
                INSERT OR REPLACE INTO evaluation_records ({', '.join(self._EVALUATION_COLUMNS)})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.evaluation_id, record.global_model_id, record.model_version, record.task, record.architecture,
                record.evaluation_dataset_id, record.evaluation_dataset_version, record.sample_count,
                record.accuracy, record.precision, record.recall, record.f1, json.dumps(record.per_class_metrics),
                json.dumps(record.confusion_matrix), record.predictions_location, record.metrics_location,
                json.dumps(record.inference_configuration), record.status, record.started_at, record.completed_at
            ))

    def _evaluation(self, row) -> EvaluationRecord:
        data = {c: row[c] for c in self._EVALUATION_COLUMNS}
        for c in self._EVALUATION_JSON:
            data[c] = json.loads(data[c]) if data[c] else ({} if c != "confusion_matrix" else [])
        return EvaluationRecord(**data)

    def get_evaluation_record(self, evaluation_id: str) -> Optional[EvaluationRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM evaluation_records WHERE evaluation_id = ?', (evaluation_id,)).fetchone()
            return self._evaluation(row) if row else None

    def get_latest_evaluation(self, global_model_id: str) -> Optional[EvaluationRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM evaluation_records WHERE global_model_id = ? ORDER BY completed_at DESC LIMIT 1',
                               (global_model_id,)).fetchone()
            return self._evaluation(row) if row else None

    def save_promotion_decision(self, pd: PromotionDecision):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO promotion_decisions
                (promotion_decision_id, candidate_global_model_id, previous_main_global_model_id, decision,
                 promotion_policy_version, evaluation_record_id, reason, metric_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pd.promotion_decision_id, pd.candidate_global_model_id, pd.previous_main_global_model_id,
                pd.decision.value, pd.promotion_policy_version, pd.evaluation_record_id, pd.reason,
                json.dumps(pd.metric_summary), pd.created_at
            ))

    # --- Differential Privacy Ledger (Module 10) ---
    def record_dp_round(
        self,
        federation_job_id: str,
        round_id: str,
        participant_id: str,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
        delta: float,
        round_epsilon: float,
        cumulative_epsilon: float,
    ) -> str:
        entry_id = f"DPL-{uuid.uuid4().hex[:10].upper()}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO federation_dp_ledger
                (entry_id, federation_job_id, round_id, participant_id,
                 noise_multiplier, sample_rate, steps, delta,
                 round_epsilon, cumulative_epsilon, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_id, federation_job_id, round_id, participant_id,
                noise_multiplier, sample_rate, steps, delta,
                round_epsilon, cumulative_epsilon, now_iso()
            ))
        return entry_id

    def get_dp_ledger(self, federation_job_id: str, participant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if participant_id:
                cursor = conn.execute(
                    'SELECT * FROM federation_dp_ledger WHERE federation_job_id = ? AND participant_id = ? ORDER BY created_at ASC',
                    (federation_job_id, participant_id)
                )
            else:
                cursor = conn.execute(
                    'SELECT * FROM federation_dp_ledger WHERE federation_job_id = ? ORDER BY created_at ASC',
                    (federation_job_id,)
                )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def compute_cumulative_dp_epsilon(self, federation_job_id: str, participant_id: str, delta: float) -> float:
        rows = self.get_dp_ledger(federation_job_id, participant_id)
        if not rows:
            return 0.0
        from hospital_client.training.dp import compute_rdp_epsilon
        history = [(float(r["noise_multiplier"]), float(r["sample_rate"]), int(r["steps"])) for r in rows]
        return compute_rdp_epsilon(history, delta)
