"""Classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(label_names))),
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "per_class": {},
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(label_names))),
        ).tolist(),
    }

    for idx, name in enumerate(label_names):
        metrics["per_class"][name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return metrics

