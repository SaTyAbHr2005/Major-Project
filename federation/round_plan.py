"""
Round plan of a federation job: which strategy it uses, how many rounds it runs,
when it stops early, and whether the next round opens by itself.

Stored in the job's existing JSON fields (no database schema change):

aggregation_configuration
    strategy              "fedavg" (default) or "fedprox"
    protocol              "manual" (default: admin and hospitals act per round) or "flower" (a Flower
                          server runs the planned rounds automatically; needs total_rounds and
                          round_duration_hours, the deadline of each round)
    total_rounds          integer >= 1. Absent = open-ended: the admin opens rounds until they stop.
    auto_continue         after a successful aggregation, open the next round automatically.
                          Defaults to True when total_rounds is set.
    round_duration_hours  deadline of an automatically opened round (hours from opening). Required
                          when auto_continue is on.
    early_stopping        optional {"metric": "accuracy" | "f1", "min_delta": >= 0, "patience": >= 1}.
                          Stops the job when the evaluated global model has not improved by at least
                          min_delta for `patience` rounds in a row. Only evaluated rounds count.

training_requirements
    proximal_mu           FedProx proximal-term weight (> 0, required for "fedprox"). Hospitals train
                          with it, and the existing training-requirement check rejects updates trained
                          with a different value. Server-side aggregation is the same weighted average.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

STRATEGIES = ("fedavg", "fedprox")
PROTOCOLS = ("manual", "flower")
EARLY_STOPPING_METRICS = ("accuracy", "f1")
# Rounds that produced a global model.
FINISHED_ROUND_STATUSES = {"GLOBAL_MODEL_CREATED", "EVALUATION_PENDING", "COMPLETED"}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_plan(aggregation_configuration: Dict[str, Any], training_requirements: Dict[str, Any]) -> None:
    """Raises ValueError naming the first invalid setting; never substitutes a default for a bad value."""
    agg = aggregation_configuration or {}
    reqs = training_requirements or {}

    proto = agg.get("protocol", "manual")
    if proto not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {list(PROTOCOLS)}, got {proto!r}")
    if proto == "flower":
        if agg.get("total_rounds") is None:
            raise ValueError("Flower jobs need total_rounds: the Flower server runs exactly that many rounds.")
        if agg.get("auto_continue") is False:
            raise ValueError("Flower jobs run their rounds automatically; auto_continue cannot be turned off.")

    strategy = agg.get("strategy", "fedavg")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {list(STRATEGIES)}, got {strategy!r}")
    mu = reqs.get("proximal_mu")
    if strategy == "fedprox":
        if not _is_number(mu) or mu <= 0:
            raise ValueError(f"FedProx needs training_requirements.proximal_mu > 0, got {mu!r}")
    elif mu not in (None, 0, 0.0):
        raise ValueError("proximal_mu is only used with strategy 'fedprox'.")

    dp_enabled = bool(
        reqs.get("dp_enabled")
        or (reqs.get("dp_noise_multiplier") is not None and float(reqs.get("dp_noise_multiplier", 0)) > 0)
    )
    if dp_enabled:
        if strategy == "fedprox" or mu not in (None, 0, 0.0):
            raise ValueError("Differential privacy cannot be combined with strategy 'fedprox' (FedProx proximal term).")
        sigma = reqs.get("dp_noise_multiplier")
        if not _is_number(sigma) or sigma <= 0:
            raise ValueError(f"dp_noise_multiplier must be a number > 0, got {sigma!r}")
        c = reqs.get("dp_max_grad_norm")
        if not _is_number(c) or c <= 0:
            raise ValueError(f"dp_max_grad_norm must be a number > 0, got {c!r}")
        delta = reqs.get("dp_target_delta")
        if not _is_number(delta) or not (0 < delta < 1):
            raise ValueError(f"dp_target_delta must be a number in (0, 1), got {delta!r}")
        target_eps = reqs.get("dp_target_epsilon")
        if target_eps is not None and (not _is_number(target_eps) or target_eps <= 0):
            raise ValueError(f"dp_target_epsilon must be a number > 0, got {target_eps!r}")
        logical_b = reqs.get("dp_logical_batch_size")
        if logical_b is not None and (not _is_int(logical_b) or logical_b < 1):
            raise ValueError(f"dp_logical_batch_size must be an integer >= 1, got {logical_b!r}")

    total = agg.get("total_rounds")
    if total is not None and (not _is_int(total) or total < 1):
        raise ValueError(f"total_rounds must be an integer >= 1, got {total!r}")

    if agg.get("auto_continue") and total is None:
        raise ValueError("auto_continue needs total_rounds, so the job knows when to stop.")
    if auto_continue(agg):
        hours = agg.get("round_duration_hours")
        if not _is_number(hours) or hours <= 0:
            raise ValueError(f"round_duration_hours must be a number > 0 when rounds open automatically, got {hours!r}")

    es = agg.get("early_stopping")
    if es is not None:
        if not isinstance(es, dict):
            raise ValueError("early_stopping must be an object with metric, min_delta and patience.")
        if es.get("metric", "accuracy") not in EARLY_STOPPING_METRICS:
            raise ValueError(f"early_stopping.metric must be one of {list(EARLY_STOPPING_METRICS)}, got {es.get('metric')!r}")
        delta = es.get("min_delta", 0.0)
        if not _is_number(delta) or delta < 0:
            raise ValueError(f"early_stopping.min_delta must be a number >= 0, got {delta!r}")
        patience = es.get("patience")
        if not _is_int(patience) or patience < 1:
            raise ValueError(f"early_stopping.patience must be an integer >= 1, got {patience!r}")


def protocol(agg: Dict[str, Any]) -> str:
    return (agg or {}).get("protocol", "manual")


def auto_continue(agg: Dict[str, Any]) -> bool:
    agg = agg or {}
    return bool(agg.get("auto_continue", agg.get("total_rounds") is not None))


def early_stopped(scores: List[float], min_delta: float, patience: int) -> bool:
    """True when the last `patience` scores each failed to beat the best earlier score by min_delta."""
    best: Optional[float] = None
    stale = 0
    for s in scores:
        if best is None or s >= best + min_delta:
            best, stale = s, 0
        else:
            stale += 1
    return stale >= patience


def plan_status(agg: Dict[str, Any], rounds: List[Dict[str, Any]], scores: List[Tuple[int, float]]) -> Dict[str, Any]:
    """
    rounds: the job's rounds as dicts with round_number and status.
    scores: (round_number, early-stopping metric of that round's evaluated global model).
    """
    agg = agg or {}
    total = agg.get("total_rounds")
    completed = sorted(r["round_number"] for r in rounds if str(r["status"]) in FINISHED_ROUND_STATUSES)
    next_number = max([0] + [r["round_number"] for r in rounds]) + 1

    finished, reason = False, ""
    es = agg.get("early_stopping")
    if es and early_stopped([v for _, v in sorted(scores)], es.get("min_delta", 0.0), es["patience"]):
        finished = True
        reason = (f"Early stopping: {es.get('metric', 'accuracy')} did not improve by {es.get('min_delta', 0.0)} "
                  f"for {es['patience']} evaluated round(s).")
    elif total is not None and len(completed) >= total:
        finished, reason = True, f"All {total} planned round(s) are complete."

    return {
        "strategy": agg.get("strategy", "fedavg"),
        "protocol": protocol(agg),
        "total_rounds": total,
        "rounds_completed": len(completed),
        # None when the job is finished or every planned round already exists.
        "next_round_number": None if finished or (total is not None and next_number > total) else next_number,
        "auto_continue": auto_continue(agg),
        "early_stopping": es,
        "scores": [{"round_number": n, "value": v} for n, v in sorted(scores)],
        "finished": finished,
        "reason": reason,
    }
