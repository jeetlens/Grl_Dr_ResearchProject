"""Publication-quality plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PUBLICATION_DPI = 600
MODEL_COLORS = {
    "Baseline": "#1f77b4",
    "GRL": "#d62728",
    "baseline": "#1f77b4",
    "grl": "#d62728",
}


def set_publication_style() -> None:
    """Apply a clean matplotlib style suitable for paper figures."""
    plt.rcParams.update(
        {
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    label_names: list[str],
    output_path: str | Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Save a publication-quality confusion matrix using matplotlib only."""
    set_publication_style()
    values = np.asarray(matrix, dtype=float)
    display_values = values.copy()
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        display_values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values),
            where=row_sums != 0,
        )

    figure, axis = plt.subplots(
        figsize=(max(6.5, len(label_names) * 0.75), 5.8)
    )
    image = axis.imshow(display_values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    threshold = display_values.max() / 2.0 if display_values.size else 0.0
    for row in range(display_values.shape[0]):
        for column in range(display_values.shape[1]):
            value = display_values[row, column]
            text = f"{value:.2f}" if normalize else f"{int(values[row, column])}"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    _save_figure(figure, output_path)


def plot_training_curves(
    history: list[dict],
    output_path: str | Path,
) -> None:
    """Save training/validation loss and accuracy curves."""
    set_publication_style()
    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    train_accuracy = [record["train_accuracy"] for record in history]
    val_accuracy = [record["val_accuracy"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, marker="o", label="Training loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_accuracy, marker="o", label="Training accuracy")
    axes[1].plot(epochs, val_accuracy, marker="o", label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    _save_figure(figure, output_path)


def plot_loss_curves(
    history: list[dict],
    output_path: str | Path,
) -> None:
    """Save GRL total and component loss curves."""
    set_publication_style()
    epochs = [record["epoch"] for record in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [record["train_loss"] for record in history],
        marker="o",
        label="Training total loss",
    )
    axis.plot(
        epochs,
        [record["val_loss"] for record in history],
        marker="o",
        label="Validation total loss",
    )
    if "train_classification_loss" in history[0]:
        axis.plot(
            epochs,
            [record["train_classification_loss"] for record in history],
            linestyle="--",
            label="Training classification loss",
        )
    if "train_domain_loss" in history[0]:
        axis.plot(
            epochs,
            [record["train_domain_loss"] for record in history],
            linestyle="--",
            label="Training domain loss",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("GRL Loss Curves")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, output_path)


def plot_accuracy_curves(
    history: list[dict],
    output_path: str | Path,
) -> None:
    """Save mental health and optional domain accuracy curves."""
    set_publication_style()
    epochs = [record["epoch"] for record in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [record["train_accuracy"] for record in history],
        marker="o",
        label="Training accuracy",
    )
    axis.plot(
        epochs,
        [record["val_accuracy"] for record in history],
        marker="o",
        label="Validation accuracy",
    )
    if "train_domain_accuracy" in history[0]:
        axis.plot(
            epochs,
            [record["train_domain_accuracy"] for record in history],
            linestyle="--",
            label="Training domain accuracy",
        )
    if "val_domain_accuracy" in history[0]:
        axis.plot(
            epochs,
            [record["val_domain_accuracy"] for record in history],
            linestyle="--",
            label="Validation domain accuracy",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("GRL Accuracy Curves")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_figure(figure, output_path)


def plot_history_metric(
    histories: dict[str, list[dict[str, Any]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot one training-history metric for one or more models."""
    set_publication_style()
    figure, axis = plt.subplots(figsize=(7.5, 5))

    for model_name, history in histories.items():
        if not history or metric_key not in history[0]:
            continue
        epochs = [record["epoch"] for record in history]
        values = [record[metric_key] for record in history]
        axis.plot(
            epochs,
            values,
            marker="o",
            linewidth=2,
            color=MODEL_COLORS.get(model_name),
            label=model_name,
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_path)


def plot_train_val_metric(
    histories: dict[str, list[dict[str, Any]]],
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Plot train and validation curves for one metric across models."""
    set_publication_style()
    figure, axis = plt.subplots(figsize=(7.5, 5))

    for model_name, history in histories.items():
        if not history or train_key not in history[0] or val_key not in history[0]:
            continue
        epochs = [record["epoch"] for record in history]
        color = MODEL_COLORS.get(model_name)
        axis.plot(
            epochs,
            [record[train_key] for record in history],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{model_name} train",
        )
        axis.plot(
            epochs,
            [record[val_key] for record in history],
            marker="s",
            linewidth=2,
            linestyle="--",
            color=color,
            label=f"{model_name} validation",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output_path)


def plot_macro_f1_comparison(
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot a focused macro-F1 comparison bar chart."""
    set_publication_style()
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    colors = [
        MODEL_COLORS.get(model_name, "#4c4c4c")
        for model_name in comparison_frame["model"]
    ]
    bars = axis.bar(
        comparison_frame["model"],
        comparison_frame["macro_f1"].astype(float),
        color=colors,
        width=0.55,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Macro F1")
    axis.set_title("Macro F1 Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    _save_figure(figure, output_path)


def plot_model_comparison(
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot a grouped comparison over all scalar model metrics."""
    set_publication_style()
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc",
    ]
    plot_frame = comparison_frame.set_index("model")[metric_columns].astype(float)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    plot_frame.T.plot(kind="bar", ax=axis, width=0.78)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_xlabel("Metric")
    axis.set_title("Baseline vs GRL")
    axis.legend(title="Model")
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    _save_figure(figure, output_path)


def plot_roc_curves(
    roc_curves: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Plot ROC curves for one or more models."""
    set_publication_style()
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6c6c6c", label="Chance")

    for curve in roc_curves:
        if curve.get("fpr") is None or curve.get("tpr") is None:
            continue
        label = f"{curve['model']} (AUC = {curve['auc']:.3f})"
        axis.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=2,
            color=MODEL_COLORS.get(curve["model"]),
            label=label,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    _save_figure(figure, output_path)


def _save_figure(figure, output_path: str | Path) -> None:
    """Save and close a matplotlib figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight")
    plt.close(figure)
