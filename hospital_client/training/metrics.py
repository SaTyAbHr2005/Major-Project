"""
Classification metrics. Reuses scikit-learn (already a project dependency
via Module 5) rather than reimplementing precision/recall/F1/confusion
matrix computation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass
class ClassificationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_order: List[str] = field(default_factory=list)
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "class_order": self.class_order,
            "sample_count": self.sample_count,
        }


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    loss: float,
    class_mapping: Dict[str, int],
) -> ClassificationMetrics:
    class_order = [label for label, _ in sorted(class_mapping.items(), key=lambda kv: kv[1])]
    labels = list(range(len(class_order)))

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    accuracy = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
    )

    per_class = {
        class_order[i]: {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i]),
        }
        for i in range(len(class_order))
    }

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist() if len(y_true_arr) else []

    return ClassificationMetrics(
        loss=float(loss),
        accuracy=accuracy,
        precision=float(precision_macro),
        recall=float(recall_macro),
        f1=float(f1_macro),
        per_class=per_class,
        confusion_matrix=cm,
        class_order=class_order,
        sample_count=len(y_true_arr),
    )
