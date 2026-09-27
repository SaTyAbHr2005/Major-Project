"""
Evaluation plots for Module 7: confusion matrix (counts + row-normalised),
one-vs-rest ROC curves and precision-recall curves, plus the AUC / average
precision numbers behind them.

Inputs are the y_true / class-probability arrays captured by Trainer for the
test set (Trainer.test_outputs), so the plots come from the same pass as
result.test_metrics. Only aggregate arrays are used - no images, paths or
patient data ever reach a plot or the summary JSON.

matplotlib is imported lazily so training itself never depends on it.
"""
import json
import os
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on environment
        raise RuntimeError("Plot generation requires matplotlib (pip install matplotlib).") from e
    return plt


def _curve_classes(y_true: np.ndarray, class_order: Sequence[str]) -> List[int]:
    """
    Classes a one-vs-rest curve is defined for: at least one positive AND one
    negative sample. Binary tasks plot only the positive class (index 1), since
    the class-0 curve is its mirror image.
    """
    n = len(class_order)
    if n == 2:
        candidates = [1]
    else:
        candidates = list(range(n))
    return [c for c in candidates if 0 < int((y_true == c).sum()) < len(y_true)]


def compute_curve_metrics(y_true, probs, class_order: Sequence[str]) -> Dict:
    """ROC AUC and average precision per class (one-vs-rest); classes with no valid curve are listed as skipped."""
    y = np.asarray(y_true)
    p = np.asarray(probs)
    valid = _curve_classes(y, class_order)
    per_class = {}
    for c in valid:
        pos = (y == c).astype(int)
        fpr, tpr, _ = roc_curve(pos, p[:, c])
        per_class[class_order[c]] = {
            "roc_auc": float(auc(fpr, tpr)),
            "average_precision": float(average_precision_score(pos, p[:, c])),
            "support": int(pos.sum()),
        }
    skipped = [class_order[c] for c in range(len(class_order))
               if c not in valid and not (len(class_order) == 2 and c == 0)]
    macro = None
    if per_class:
        macro = {
            "roc_auc": float(np.mean([v["roc_auc"] for v in per_class.values()])),
            "average_precision": float(np.mean([v["average_precision"] for v in per_class.values()])),
        }
    return {"per_class": per_class, "macro": macro, "skipped_classes": skipped, "sample_count": int(len(y))}


def _plot_confusion(plt, y, p, class_order, path, title_suffix):
    labels = list(range(len(class_order)))
    cm = confusion_matrix(y, p.argmax(axis=1), labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    size = max(5.0, 0.9 * len(class_order) + 3)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, size))
    for ax, data, fmt, title in ((axes[0], cm, "d", "Counts"), (axes[1], norm, ".2f", "Row-normalised (recall)")):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        ax.set_xticks(labels, class_order, rotation=45, ha="right")
        ax.set_yticks(labels, class_order)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thresh = data.max() / 2 if data.max() else 0
        for i in labels:
            for j in labels:
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion matrix - {title_suffix}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_curves(plt, y, p, class_order, curve_metrics, roc_path, pr_path, title_suffix):
    valid = [c for c in range(len(class_order)) if class_order[c] in curve_metrics["per_class"]]
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for c in valid:
            pos = (y == c).astype(int)
            m = curve_metrics["per_class"][class_order[c]]
            if kind == "roc":
                x_, y_, _ = roc_curve(pos, p[:, c])
                label = f"{class_order[c]} (AUC={m['roc_auc']:.3f}, n={m['support']})"
            else:
                y_, x_, _ = precision_recall_curve(pos, p[:, c])
                label = f"{class_order[c]} (AP={m['average_precision']:.3f}, n={m['support']})"
            ax.plot(x_, y_, label=label)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
            ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC (one-vs-rest) - {title_suffix}")
        else:
            ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall (one-vs-rest) - {title_suffix}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def generate_evaluation_plots(y_true, probs, class_order: Sequence[str], output_dir: str, model_id: str) -> Dict:
    """
    Writes into <output_dir>/<model_id>_plots/:
      confusion_matrix.png, roc_curves.png, pr_curves.png, curve_metrics.json
    ROC/PR files are skipped (and noted in curve_metrics.json) when no class has both
    positive and negative test samples. Returns {"directory", "files", "curve_metrics"}.
    """
    y = np.asarray(y_true)
    p = np.asarray(probs)
    if len(y) == 0:
        raise ValueError("Cannot generate evaluation plots: the test set produced no predictions.")

    plt = _plt()
    out = os.path.join(output_dir, f"{model_id}_plots")
    os.makedirs(out, exist_ok=True)
    suffix = f"{model_id}, test set, n={len(y)}"

    files = []
    cm_path = os.path.join(out, "confusion_matrix.png")
    _plot_confusion(plt, y, p, list(class_order), cm_path, suffix)
    files.append(cm_path)

    curve_metrics = compute_curve_metrics(y, p, class_order)
    if curve_metrics["per_class"]:
        roc_path, pr_path = os.path.join(out, "roc_curves.png"), os.path.join(out, "pr_curves.png")
        _plot_curves(plt, y, p, list(class_order), curve_metrics, roc_path, pr_path, suffix)
        files += [roc_path, pr_path]

    metrics_path = os.path.join(out, "curve_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(curve_metrics, f, indent=2)
    files.append(metrics_path)
    return {"directory": out, "files": files, "curve_metrics": curve_metrics}
