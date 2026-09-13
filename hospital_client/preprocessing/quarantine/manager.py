from typing import Dict, List, Any
from enum import Enum

class QualityStatus(Enum):
    VALID = "valid"
    LOW_QUALITY = "low_quality"
    REJECTED = "rejected"
    ERROR = "error"

class QuarantineManager:
    def __init__(self):
        self.accepted: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []
        self.review: List[Dict[str, Any]] = []

    def add_sample(self, file_path: str, status: QualityStatus, reason: str = "", metadata: Dict[str, Any] = None):
        record = {
            "file": file_path,
            "status": status.value,
            "reason": reason,
            "metadata": metadata or {}
        }
        if status == QualityStatus.VALID:
            self.accepted.append(record)
        elif status == QualityStatus.LOW_QUALITY:
            self.review.append(record)
        else:
            self.rejected.append(record)

    def get_accepted_paths(self) -> List[str]:
        return [r["file"] for r in self.accepted]

    def summary(self) -> Dict[str, int]:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "review": len(self.review),
            "total_discovered": len(self.accepted) + len(self.rejected) + len(self.review)
        }

