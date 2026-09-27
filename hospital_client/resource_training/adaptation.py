"""
Runtime resource adaptation for Module 8. Operates at RUN/RETRY granularity
around Module 7's Trainer.run() (runner.py) - Module 7's training loop is
never modified, so adaptation cannot intervene mid-epoch. This is an
explicit, documented limitation (see module8_readme.md).

Every adaptation is bounded (max_adaptation_attempts) and recorded. Model
architecture and device are NEVER silently changed here - only batch_size,
and only downward.
"""
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from hospital_client.resource_training.policy import ResourcePolicy
from hospital_client.training.config import TrainingConfig
from hospital_client.training.result import TrainingResult, TrainingStatus


@dataclass
class AdaptationEvent:
    timestamp: str
    reason: str
    old_batch_size: int
    new_batch_size: Optional[int]
    outcome: str  # "retrying" | "exhausted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "reason": self.reason,
            "old_batch_size": self.old_batch_size,
            "new_batch_size": self.new_batch_size,
            "outcome": self.outcome,
        }


def detect_cuda_oom(result: TrainingResult) -> bool:
    if result.status != TrainingStatus.TRAINING_FAILED:
        return False
    return any("out of memory" in e.lower() for e in result.errors)


def adapt_config_for_oom(config: TrainingConfig, policy: ResourcePolicy) -> Optional[TrainingConfig]:
    """
    Returns a copy of config with a reduced batch_size, or None if batch_size
    is already at (or would round to) the minimum of 1 - the caller must
    stop retrying in that case rather than loop indefinitely.
    """
    new_batch_size = max(1, int(config.batch_size * policy.batch_size_reduction_factor))
    if new_batch_size >= config.batch_size:
        return None
    adapted = copy.deepcopy(config)
    adapted.batch_size = new_batch_size
    return adapted


def make_event(reason: str, old_batch_size: int, new_batch_size: Optional[int], outcome: str) -> AdaptationEvent:
    return AdaptationEvent(
        timestamp=datetime.now(timezone.utc).isoformat(), reason=reason,
        old_batch_size=old_batch_size, new_batch_size=new_batch_size, outcome=outcome,
    )
