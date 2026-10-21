"""
Module 12: Byzantine and Malicious Model-Update Detection Hook.

Per-update checks (works with 2 hospitals):
1. NaN / Inf values in any parameter tensor -> REJECT
2. Sample count consistency across rounds for the same hospital -> FLAG / QUARANTINE
3. Parameter delta L2 norm bound (delta = submitted - round base) over TRAINABLE WEIGHTS ONLY:
   - Non-DP default: flag at 2.2x base norm (2.0x observed max 1.1031), quarantine at 5.5x base norm (5.0x observed max)
   - DP default: flag at 0.4x base norm (2.0x observed max 0.1944), quarantine at 1.0x base norm (5.0x observed max)
   - Both DP and non-DP use the exact same documented margin (2.0x for Flag, 5.0x for Quarantine).
   - Configurable overrides via job training_requirements (m12_max_l2_norm, m12_max_l2_ratio, etc.)
4. Cosine similarity with prior global update direction:
   - Only runs when two previous global models exist for this job in global_model_versions (round >= 3).
   - Rounds 1-2 have no previous direction; norm-preserving sign flips in rounds 1-2
     are not detected by direction check.
   - In round 3+, if cosine < -0.1 (or m12_min_cosine_similarity) -> FLAG
5. Implausible reported metrics (loss < 0, accuracy not in [0, 1]) -> FLAG

Invariant: Never fail open silently. Every skipped check records its reason in metrics["checks_skipped"].
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from hospital_client.training.dp import is_dp_job
from .models import FederationJob, FederationRound, FederationUpdate, UpdateStatus

logger = logging.getLogger(__name__)

# Empirically calibrated default ratio bounds (||delta|| / ||base||) over trainable weights:
# Observed in federation_metadata.db across 19 clean accepted updates:
# Non-DP: count=14, min=0.0796, max=1.1031 (JOB_MAL_RES18_CLS2_V2_R2, HOSP_999999), mean=0.4112
# DP:     count=5,  min=0.1079, max=0.1944 (JOB_MAL_MNV3_CLS2_V2_R1, HOSP_999999), mean=0.1553
#
# Same documented margin applied to both:
# Flag threshold: 2.0x observed max clean ratio (100% margin over natural training drift)
# Quarantine threshold: 5.0x observed max clean ratio
DEFAULT_NON_DP_FLAG_RATIO = 2.2
DEFAULT_NON_DP_QUARANTINE_RATIO = 5.5

DEFAULT_DP_FLAG_RATIO = 0.4
DEFAULT_DP_QUARANTINE_RATIO = 1.0

DEFAULT_MIN_COSINE_SIMILARITY = -0.1

_TRAINABLE_INDICES_CACHE: Dict[Tuple[str, int, str], List[int]] = {}


def _get_trainable_indices(architecture: str, num_classes: int, norm_type: str = "batch") -> Optional[List[int]]:
    """Returns parameter indices corresponding to trainable weights (excludes BatchNorm buffers, etc.)."""
    key = (architecture, num_classes, norm_type)
    if key in _TRAINABLE_INDICES_CACHE:
        return _TRAINABLE_INDICES_CACHE[key]
    try:
        from hospital_client.model_management.config import ModelConfig
        from hospital_client.model_management.registry import build_model, get_architecture_info
        info = get_architecture_info(architecture)
        cfg = ModelConfig(
            architecture=architecture,
            num_classes=num_classes,
            input_size=tuple(info.default_input_size),
            norm_type=norm_type,
        )
        m = build_model(cfg)
        sd_keys = list(m.state_dict().keys())
        trainable_names = {name for name, p in m.named_parameters() if p.requires_grad}
        indices = [i for i, k in enumerate(sd_keys) if k in trainable_names]
        _TRAINABLE_INDICES_CACHE[key] = indices
        return indices
    except Exception as e:
        logger.debug(f"Could not build architecture {architecture} for trainable indices: {e}")
        return None


@dataclass
class SecurityDecision:
    verdict: str  # "ACCEPT", "FLAG", "QUARANTINE", "REJECT"
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _load_base_parameters(
    storage, job: Optional[FederationJob], rnd: Optional[FederationRound]
) -> Tuple[Optional[List[np.ndarray]], Optional[str]]:
    """
    Loads canonical base parameters for a round.
    Returns (parameters, skip_reason). Never raises exceptions silently.
    """
    if storage is None:
        return None, "Storage instance not provided"
    if rnd is None:
        return None, "Round metadata not provided"

    base_id = rnd.base_model_id
    base_ver = rnd.base_model_version

    if not base_id:
        return None, "Round has no base_model_id specified"

    # 1. Check if base model is a global model
    try:
        gm = storage.get_global_model(base_id)
        if gm is not None and gm.artifact_location and os.path.exists(gm.artifact_location):
            from hospital_client.training.federation import FederationHandoff
            return FederationHandoff.load(gm.artifact_location).parameters, None
    except Exception as e:
        logger.debug(f"Failed loading base model from global_model {base_id}: {e}")

    # 2. Check base_models store via ModelStore
    try:
        from federation.base_models import base_model_store
        from hospital_client.model_management.parameters import get_parameters
        store = base_model_store(storage)
        loaded, _ = store.load_checkpoint(base_id, base_ver, device="cpu")
        return get_parameters(loaded), None
    except Exception as e:
        logger.debug(f"Failed loading base model from base_model_store {base_id} v{base_ver}: {e}")

    # 3. Fallback: direct safetensors in base_models folder
    base_models_dir = os.path.join(storage.artifacts_dir, "base_models", base_id, f"v{base_ver}")
    sf_path = os.path.join(base_models_dir, "model.safetensors")
    if os.path.exists(sf_path) and job:
        try:
            from safetensors.numpy import load_file
            from hospital_client.model_management.config import ModelConfig
            from hospital_client.model_management.registry import build_model, get_architecture_info
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
                return [tensors[k] for k in keys], None
            return list(tensors.values()), None
        except Exception as e:
            logger.debug(f"Failed loading safetensors directly for {base_id}: {e}")

    # 4. Check global_models artifact directory directly
    if job:
        try:
            gm_dir = storage.get_global_model_dir(job.task, job.architecture, base_ver)
            if os.path.exists(gm_dir):
                from hospital_client.training.federation import FederationHandoff
                return FederationHandoff.load(gm_dir).parameters, None
        except Exception as e:
            logger.debug(f"Failed loading from global_model_dir: {e}")

    return None, f"Base checkpoint '{base_id}' v{base_ver} not found in global models or base_models store"


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
        Never fails open silently: any check that cannot run records a reason in metrics['checks_skipped'].
        """
        reasons: List[str] = []
        metrics: Dict[str, Any] = {}
        verdicts: List[str] = []
        checks_skipped: Dict[str, str] = {}

        is_dp = False
        if job:
            is_dp = is_dp_job(job)
            if not is_dp:
                reqs = job.training_requirements or {}
                if reqs.get("dp_enabled") or (
                    isinstance(reqs.get("differential_privacy"), dict)
                    and reqs.get("differential_privacy", {}).get("enabled")
                ):
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
            metrics["checks_skipped"] = checks_skipped
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
        if not self.storage:
            checks_skipped["sample_consistency"] = "Storage not configured"
        elif not job:
            checks_skipped["sample_consistency"] = "Job metadata not available"
        elif not update.participant_id:
            checks_skipped["sample_consistency"] = "Participant identifier not available"
        elif samples is not None and samples > 0:
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
                checks_skipped["sample_consistency"] = f"Database query error: {e}"

        # ── 3. Delta L2 norm check over trainable weights only ──────────────
        base_params, base_load_err = _load_base_parameters(self.storage, job, rnd)
        delta_vec = None
        base_vec = None

        if base_params is None:
            checks_skipped["l2"] = base_load_err or "Base model parameters not found"
        elif len(base_params) != len(handoff.parameters):
            checks_skipped["l2"] = (
                f"Parameter tensor count mismatch between handoff ({len(handoff.parameters)}) "
                f"and base model ({len(base_params)})"
            )
        else:
            # Trainable weight indices filter
            norm_type = "group" if is_dp else "batch"
            arch = getattr(handoff, "architecture", None) or (job.architecture if job else None)
            num_classes = getattr(handoff, "num_classes", None) or (job.num_classes if job else None)
            trainable_indices = None
            if arch and num_classes:
                trainable_indices = _get_trainable_indices(arch, num_classes, norm_type)

            total_params = len(handoff.parameters)
            if trainable_indices is not None and max(trainable_indices, default=-1) < total_params:
                indices_to_check = trainable_indices
            else:
                indices_to_check = [
                    i for i, p in enumerate(handoff.parameters)
                    if np.issubdtype(p.dtype, np.floating)
                ]

            float_diffs = []
            base_floats = []
            shape_mismatch = False

            for idx in indices_to_check:
                p = handoff.parameters[idx]
                b = base_params[idx]
                if np.issubdtype(p.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
                    if p.shape != b.shape:
                        shape_mismatch = True
                        checks_skipped["l2"] = f"Shape mismatch at parameter index {idx}: {p.shape} vs {b.shape}"
                        verdicts.append("REJECT")
                        reasons.append(f"Parameter shape mismatch with base model: {p.shape} vs {b.shape}")
                        float_diffs = []
                        break
                    float_diffs.append((p - b).ravel())
                    base_floats.append(b.ravel())

            if shape_mismatch:
                pass
            elif not float_diffs:
                checks_skipped["l2"] = "No floating-point trainable parameters found"
            else:
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
                    quarantine_norm = (
                        float(override_quarantine_norm)
                        if override_quarantine_norm is not None
                        else flag_norm * 3.0
                    )
                    metrics["l2_flag_threshold_norm"] = flag_norm
                    metrics["l2_quarantine_threshold_norm"] = quarantine_norm

                    if delta_norm > quarantine_norm:
                        verdicts.append("QUARANTINE")
                        reasons.append(
                            f"Delta L2 norm {delta_norm:.4f} exceeds quarantine threshold {quarantine_norm:.4f}"
                        )
                    elif delta_norm > flag_norm:
                        verdicts.append("FLAG")
                        reasons.append(
                            f"Delta L2 norm {delta_norm:.4f} exceeds flag threshold {flag_norm:.4f}"
                        )
                else:
                    flag_ratio = float(
                        reqs.get(
                            "m12_max_l2_ratio",
                            DEFAULT_DP_FLAG_RATIO if is_dp else DEFAULT_NON_DP_FLAG_RATIO,
                        )
                    )
                    quarantine_ratio = float(
                        reqs.get(
                            "m12_quarantine_l2_ratio",
                            DEFAULT_DP_QUARANTINE_RATIO if is_dp else DEFAULT_NON_DP_QUARANTINE_RATIO,
                        )
                    )
                    metrics["l2_flag_threshold_ratio"] = flag_ratio
                    metrics["l2_quarantine_threshold_ratio"] = quarantine_ratio

                    if ratio > quarantine_ratio:
                        verdicts.append("QUARANTINE")
                        reasons.append(
                            f"Delta-to-base ratio {ratio:.4f} exceeds quarantine threshold {quarantine_ratio:.4f}"
                        )
                    elif ratio > flag_ratio:
                        verdicts.append("FLAG")
                        reasons.append(
                            f"Delta-to-base ratio {ratio:.4f} exceeds flag threshold {flag_ratio:.4f}"
                        )

        # ── 4. Cosine similarity with prior global update direction ─────────
        # Runs strictly when round >= 3 and job's own global models exist in global_model_versions.
        round_num = rnd.round_number if rnd else 1
        metrics["round_number"] = round_num

        if round_num < 3:
            checks_skipped["cosine"] = f"Fewer than 2 previous global models (round {round_num} < 3)"
        elif delta_vec is None:
            checks_skipped["cosine"] = "L2 delta vector not available"
        elif not self.storage or not job:
            checks_skipped["cosine"] = "Storage or job metadata not available"
        else:
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
                    else:
                        checks_skipped["cosine"] = (
                            f"Fewer than 2 previous global models found in database ({len(rows)} found for job)"
                        )
            except Exception as e:
                checks_skipped["cosine"] = f"DB error querying global models: {e}"

            if gm_prev1 and gm_prev2:
                if not (os.path.exists(gm_prev1) and os.path.exists(gm_prev2)):
                    checks_skipped["cosine"] = "Previous global model artifact files missing on disk"
                else:
                    try:
                        from hospital_client.training.federation import FederationHandoff
                        p_prev1 = FederationHandoff.load(gm_prev1).parameters
                        p_prev2 = FederationHandoff.load(gm_prev2).parameters
                        norm_type = "group" if is_dp else "batch"
                        arch = job.architecture
                        num_classes = job.num_classes
                        tr_idx = _get_trainable_indices(arch, num_classes, norm_type) if arch and num_classes else None

                        if tr_idx is not None and max(tr_idx, default=-1) < len(p_prev1):
                            v1_parts = [p_prev1[i].ravel() for i in tr_idx if np.issubdtype(p_prev1[i].dtype, np.floating)]
                            v2_parts = [p_prev2[i].ravel() for i in tr_idx if np.issubdtype(p_prev2[i].dtype, np.floating)]
                        else:
                            v1_parts = [p.ravel() for p in p_prev1 if np.issubdtype(p.dtype, np.floating)]
                            v2_parts = [p.ravel() for p in p_prev2 if np.issubdtype(p.dtype, np.floating)]

                        v1 = np.concatenate(v1_parts)
                        v2 = np.concatenate(v2_parts)
                        d_global_prev = v1 - v2
                        n_gprev = np.linalg.norm(d_global_prev)
                        n_curr = np.linalg.norm(delta_vec)

                        if n_gprev > 1e-12 and n_curr > 1e-12 and len(delta_vec) == len(d_global_prev):
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
                        else:
                            checks_skipped["cosine"] = "Zero vector norm or dimension mismatch in cosine similarity"
                    except Exception as e:
                        checks_skipped["cosine"] = f"Error computing cosine similarity: {e}"

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
                    except (ValueError, TypeError) as e:
                        checks_skipped[f"{m_name}_loss"] = f"Cannot parse metric {loss_val!r}: {e}"

            if "accuracy" in m_dict:
                acc_val = m_dict["accuracy"]
                if acc_val is not None:
                    try:
                        facc = float(acc_val)
                        metrics[f"{m_name}_accuracy"] = facc
                        if not np.isfinite(facc) or facc < 0.0 or facc > 1.0:
                            verdicts.append("FLAG")
                            reasons.append(f"Implausible {m_name} accuracy: {facc} (expected in [0, 1])")
                    except (ValueError, TypeError) as e:
                        checks_skipped[f"{m_name}_accuracy"] = f"Cannot parse metric {acc_val!r}: {e}"

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

        metrics["checks_skipped"] = checks_skipped
        return SecurityDecision(verdict=final_verdict, reasons=reasons, metrics=metrics)
