"""
Module 12: Byzantine and Malicious Model-Update Detection Hook.

Per-update checks (works with 2 hospitals):
1. NaN / Inf values in any parameter tensor -> REJECT
2. Sample count consistency across rounds for the same hospital -> FLAG / QUARANTINE
3. Parameter delta L2 norm bound (delta = submitted - round base):
   - Non-DP default: flag at 15.0x base norm, quarantine at 50.0x base norm
     (calibrated empirically from real training where max clean observed ratio was 12.25)
   - DP default: flag at 2.0x base norm, quarantine at 10.0x base norm
     (calibrated empirically from real DP training where max clean observed ratio was 0.169)
   - Configurable overrides via job training_requirements (m12_max_l2_norm, m12_max_l2_ratio, etc.)
4. Cosine similarity with prior global update direction:
   - Only runs when two previous global models exist (round >= 3).
   - Rounds 1-2 have no previous direction; norm-preserving sign flips in rounds 1-2
     are not detected by direction check.
   - In round 3+, if cosine < -0.1 (or m12_min_cosine_similarity) -> FLAG
5. Implausible reported metrics (loss < 0, accuracy not in [0, 1]) -> FLAG
"""
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from hospital_client.training.dp import is_dp_job
from .models import FederationJob, FederationRound, FederationUpdate, UpdateStatus

# Empirically calibrated default ratio bounds (||delta|| / ||base||)
# Observed in federation_metadata.db:
# Non-DP max observed clean ratio: 12.2543 (JOB_MAL_RES18_CLS2_V2_R2)
DEFAULT_NON_DP_FLAG_RATIO = 15.0
DEFAULT_NON_DP_QUARANTINE_RATIO = 50.0

# DP max observed clean ratio: 0.1692 (JOB_MAL_MNV3_CLS2_V2_R2)
DEFAULT_DP_FLAG_RATIO = 2.0
DEFAULT_DP_QUARANTINE_RATIO = 10.0

DEFAULT_MIN_COSINE_SIMILARITY = -0.1


@dataclass
class SecurityDecision:
    verdict: str  # "ACCEPT", "FLAG", "QUARANTINE", "REJECT"
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _load_base_parameters(storage, job: Optional[FederationJob], rnd: Optional[FederationRound]) -> Optional[List[np.ndarray]]:
    """Loads canonical base parameters for a round."""
    if storage is None or rnd is None:
        return None

    base_id = rnd.base_model_id
    base_ver = rnd.base_model_version

    # 1. Check if base model is a global model
    gm = storage.get_global_model(base_id)
    if gm is not None and os.path.exists(gm.artifact_location):
        try:
            from hospital_client.training.federation import FederationHandoff
            return FederationHandoff.load(gm.artifact_location).parameters
        except Exception:
            pass

    # 2. Check base_models store
    try:
        from federation.base_models import base_model_store
        from hospital_client.model_management.parameters import get_parameters
        store = base_model_store(storage)
        loaded, _ = store.load_checkpoint(base_id, base_ver, device="cpu")
        return get_parameters(loaded)
    except Exception:
        pass

    base_models_dir = os.path.join(storage.artifacts_dir, "base_models", base_id, f"v{base_ver}")
    sf_path = os.path.join(base_models_dir, "model.safetensors")
    if os.path.exists(sf_path) and job:
        try:
            from safetensors.numpy import load_file
            from hospital_client.model_management.config import ModelConfig
            from hospital_client.model_management.registry import build_model, get_architecture_info
            from hospital_client.training.dp import is_dp_job
            tensors = load_file(sf_path)
            norm_type = "group" if is_dp_job(job) else "batch"
            cfg = ModelConfig(
                architecture=job.architecture,
                num_classes=job.num_classes,
                input_size=tuple(get_architecture_info(job.architecture).default_input_size),
                norm_type=norm_type,
            )
            m = build_model(cfg)
            keys = list(m.state_dict().keys())
            if all(k in tensors for k in keys):
                return [tensors[k] for k in keys]
            return list(tensors.values())
        except Exception:
            pass

    # 3. Check global_models artifact directory directly
    if job:
        gm_dir = storage.get_global_model_dir(job.task, job.architecture, base_ver)
        if os.path.exists(gm_dir):
            try:
                from hospital_client.training.federation import FederationHandoff
                return FederationHandoff.load(gm_dir).parameters
            except Exception:
                pass

    return None


class M12SecurityHook:
    def __init__(self, storage=None):
        self.storage = storage

    def evaluate_update(
        self,
        update: FederationUpdate,
        handoff: Any,
        job: Optional[FederationJob] = None,
        rnd: Optional[FederationRound] = None,
    ) -> SecurityDecision:
        """
        Evaluate a submitted update for Byzantine/malicious properties.
        Returns a SecurityDecision dataclass with .verdict in:
        'ACCEPT', 'FLAG', 'QUARANTINE', 'REJECT'
        """
        reasons: List[str] = []
        metrics: Dict[str, Any] = {}
        verdicts: List[str] = []

        is_dp = False
        if job:
            is_dp = is_dp_job(job)
            if not is_dp:
                reqs = job.training_requirements or {}
                if reqs.get("dp_enabled") or (isinstance(reqs.get("differential_privacy"), dict) and reqs.get("differential_privacy", {}).get("enabled")):
                    is_dp = True
        metrics["is_dp"] = is_dp

        # ── 1. Check for NaN or Inf in parameters ───────────────────────────
        has_nan_inf = False
        for idx, p in enumerate(handoff.parameters):
            if np.isnan(p).any() or np.isinf(p).any():
                has_nan_inf = True
                break
        metrics["has_nan_inf"] = has_nan_inf
        if has_nan_inf:
            reasons.append("Parameter tensor contains NaN or Inf values")
            metrics["nan_inf_detected"] = True
            return SecurityDecision(verdict="REJECT", reasons=reasons, metrics=metrics)

        # ── 2. Sample count sanity and consistency across rounds ───────────
        samples = handoff.num_train_samples
        metrics["num_train_samples"] = samples
        if samples is None or not isinstance(samples, (int, np.integer)) or samples <= 0:
            verdicts.append("REJECT")
            reasons.append(f"Invalid non-positive sample count reported: {samples}")
        elif samples > 10_000_000:
            verdicts.append("FLAG")
            reasons.append(f"Implausibly large sample count reported: {samples}")

        # Check sample count consistency for this hospital across earlier rounds
        if self.storage and job and update.participant_id and samples > 0:
            try:
                import sqlite3
                conn_cm = self.storage._connect() if hasattr(self.storage, "_connect") else sqlite3.connect(self.storage.db_path)
                with conn_cm as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT num_train_samples, round_id FROM federation_updates "
                        "WHERE federation_job_id = ? AND participant_id = ? AND update_id != ? "
                        "AND status IN ('ACCEPTED', 'USED_IN_AGGREGATION') "
                        "ORDER BY received_at DESC",
                        (job.federation_job_id, update.participant_id, update.update_id)
                    ).fetchall()
                    for r in rows:
                        prior_n = r["num_train_samples"]
                        metrics["prior_train_samples"] = prior_n
                        if prior_n > 0 and samples != prior_n:
                            ratio = float(samples) / float(prior_n)
                            metrics["sample_change_ratio"] = ratio
                            if ratio >= 3.0 or ratio <= 0.333:
                                verdicts.append("QUARANTINE")
                                reasons.append(
                                    f"Sample count jumped drastically from prior round ({samples} vs {prior_n}, {ratio:.2f}x)"
                                )
                            else:
                                verdicts.append("FLAG")
                                reasons.append(
                                    f"Sample count changed from prior round ({samples} vs {prior_n})"
                                )
                            break
            except Exception as e:
                # Fallback gracefully if database is locked or unavailable
                pass

        # ── 3. Delta L2 norm check ──────────────────────────────────────────
        base_params = _load_base_parameters(self.storage, job, rnd)
        delta_vec = None
        base_vec = None

        if base_params is not None and len(base_params) == len(handoff.parameters):
            float_diffs = []
            base_floats = []
            for p, b in zip(handoff.parameters, base_params):
                if np.issubdtype(p.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
                    if p.shape != b.shape:
                        verdicts.append("REJECT")
                        reasons.append(f"Parameter shape mismatch with base model: {p.shape} vs {b.shape}")
                        float_diffs = []
                        break
                    float_diffs.append((p - b).ravel())
                    base_floats.append(b.ravel())

            if float_diffs:
                delta_vec = np.concatenate(float_diffs)
                base_vec = np.concatenate(base_floats)
                delta_norm = float(np.linalg.norm(delta_vec))
                base_norm = float(np.linalg.norm(base_vec))
                ratio = delta_norm / (base_norm + 1e-12)

                metrics["delta_l2_norm"] = delta_norm
                metrics["base_l2_norm"] = base_norm
                metrics["delta_to_base_ratio"] = ratio

                # Check job requirements for explicit overrides
                reqs = (job.training_requirements or {}) if job else {}
                override_max_norm = reqs.get("m12_max_l2_norm")
                override_quarantine_norm = reqs.get("m12_quarantine_l2_norm")

                if override_max_norm is not None:
                    flag_norm = float(override_max_norm)
                    quarantine_norm = float(override_quarantine_norm) if override_quarantine_norm is not None else flag_norm * 3.0
                    metrics["l2_flag_threshold_norm"] = flag_norm
                    metrics["l2_quarantine_threshold_norm"] = quarantine_norm

                    if delta_norm > quarantine_norm:
                        verdicts.append("QUARANTINE")
                        reasons.append(f"Delta L2 norm {delta_norm:.4f} exceeds quarantine threshold {quarantine_norm:.4f}")
                    elif delta_norm > flag_norm:
                        verdicts.append("FLAG")
                        reasons.append(f"Delta L2 norm {delta_norm:.4f} exceeds flag threshold {flag_norm:.4f}")
                else:
                    flag_ratio = float(reqs.get("m12_max_l2_ratio", DEFAULT_DP_FLAG_RATIO if is_dp else DEFAULT_NON_DP_FLAG_RATIO))
                    quarantine_ratio = float(reqs.get("m12_quarantine_l2_ratio", DEFAULT_DP_QUARANTINE_RATIO if is_dp else DEFAULT_NON_DP_QUARANTINE_RATIO))
                    metrics["l2_flag_threshold_ratio"] = flag_ratio
                    metrics["l2_quarantine_threshold_ratio"] = quarantine_ratio

                    if ratio > quarantine_ratio:
                        verdicts.append("QUARANTINE")
                        reasons.append(f"Delta-to-base ratio {ratio:.4f} exceeds quarantine threshold {quarantine_ratio:.4f}")
                    elif ratio > flag_ratio:
                        verdicts.append("FLAG")
                        reasons.append(f"Delta-to-base ratio {ratio:.4f} exceeds flag threshold {flag_ratio:.4f}")

        # ── 4. Cosine similarity with prior global update direction ─────────
        # As documented in Requirement 3: The cosine check ONLY runs when two previous
        # global models exist (round >= 3). Rounds 1-2 have no previous global direction,
        # so norm-preserving sign-flip attacks in rounds 1-2 cannot be detected by direction.
        round_num = rnd.round_number if rnd else 1
        metrics["round_number"] = round_num

        if self.storage and job and rnd and round_num >= 3 and delta_vec is not None:
            # We need two previous global models: G_{t-1} and G_{t-2}
            gm_prev1 = None
            gm_prev2 = None
            try:
                import sqlite3
                conn_cm = self.storage._connect() if hasattr(self.storage, "_connect") else sqlite3.connect(self.storage.db_path)
                with conn_cm as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT artifact_location, version FROM global_model_versions "
                        "WHERE federation_job_id = ? ORDER BY version DESC LIMIT 2",
                        (job.federation_job_id,)
                    ).fetchall()
                    if len(rows) >= 2:
                        gm_prev1 = rows[0]["artifact_location"]
                        gm_prev2 = rows[1]["artifact_location"]
            except Exception:
                pass

            if not (gm_prev1 and gm_prev2 and os.path.exists(gm_prev1) and os.path.exists(gm_prev2)):
                task = job.task
                arch = job.architecture
                gm_prev1 = self.storage.get_global_model_dir(task, arch, round_num - 1)
                gm_prev2 = self.storage.get_global_model_dir(task, arch, round_num - 2)

            if gm_prev1 and gm_prev2 and os.path.exists(gm_prev1) and os.path.exists(gm_prev2):
                try:
                    from hospital_client.training.federation import FederationHandoff
                    p_prev1 = FederationHandoff.load(gm_prev1).parameters
                    p_prev2 = FederationHandoff.load(gm_prev2).parameters
                    v1 = np.concatenate([p.ravel() for p in p_prev1 if np.issubdtype(p.dtype, np.floating)])
                    v2 = np.concatenate([p.ravel() for p in p_prev2 if np.issubdtype(p.dtype, np.floating)])
                    d_global_prev = v1 - v2
                    n_gprev = np.linalg.norm(d_global_prev)
                    n_curr = np.linalg.norm(delta_vec)

                    if n_gprev > 1e-12 and n_curr > 1e-12:
                        cos_sim = float(np.dot(delta_vec, d_global_prev) / (n_curr * n_gprev))
                        metrics["cosine_similarity"] = cos_sim
                        reqs = (job.training_requirements or {}) if job else {}
                        min_cos = float(reqs.get("m12_min_cosine_similarity", DEFAULT_MIN_COSINE_SIMILARITY))
                        metrics["min_cosine_threshold"] = min_cos

                        if cos_sim < min_cos:
                            verdicts.append("FLAG")
                            reasons.append(
                                f"Cosine similarity with prior global update direction is strongly opposite ({cos_sim:.4f} < {min_cos:.4f})"
                            )
                except Exception:
                    pass

        # ── 5. Implausible reported metrics ────────────────────────────────
        t_metrics = handoff.training_metrics or {}
        v_metrics = handoff.validation_metrics or {}

        for m_name, m_dict in (("training", t_metrics), ("validation", v_metrics)):
            if "loss" in m_dict:
                loss_val = m_dict["loss"]
                if loss_val is not None:
                    try:
                        floss = float(loss_val)
                        metrics[f"{m_name}_loss"] = floss
                        if not np.isfinite(floss) or floss < 0.0 or floss > 10000.0:
                            verdicts.append("FLAG")
                            reasons.append(f"Implausible {m_name} loss: {floss}")
                    except (ValueError, TypeError):
                        pass

            if "accuracy" in m_dict:
                acc_val = m_dict["accuracy"]
                if acc_val is not None:
                    try:
                        facc = float(acc_val)
                        metrics[f"{m_name}_accuracy"] = facc
                        if not np.isfinite(facc) or facc < 0.0 or facc > 1.0:
                            verdicts.append("FLAG")
                            reasons.append(f"Implausible {m_name} accuracy: {facc} (expected in [0, 1])")
                    except (ValueError, TypeError):
                        pass

        # ── 6. Determine final verdict by severity ─────────────────────────
        # Hierarchy: REJECT > QUARANTINE > FLAG > ACCEPT
        if "REJECT" in verdicts:
            final_verdict = "REJECT"
        elif "QUARANTINE" in verdicts:
            final_verdict = "QUARANTINE"
        elif "FLAG" in verdicts:
            final_verdict = "FLAG"
        else:
            final_verdict = "ACCEPT"

        return SecurityDecision(verdict=final_verdict, reasons=reasons, metrics=metrics)
