"""Generate publication-quality figures from saved experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import apply_overrides, load_config
from utils.paths import project_root, resolve_project_paths
from utils.plots import (
    plot_confusion_matrix,
    plot_history_metric,
    plot_macro_f1_comparison,
    plot_roc_curves,
    plot_train_val_metric,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate publication-quality figures.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root() / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values.",
    )
    return parser.parse_args()


def main() -> None:
    """Create all requested figures under outputs/figures."""
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    config = resolve_project_paths(config)

    output_dir = Path(config["training"]["output_dir"])
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(reports_dir)
    if histories:
        plot_history_metric(
            histories=histories,
            metric_key="train_loss",
            ylabel="Training Loss",
            title="Training Loss Curve",
            output_path=figures_dir / "training_loss_curve.png",
        )
        plot_history_metric(
            histories=histories,
            metric_key="val_loss",
            ylabel="Validation Loss",
            title="Validation Loss Curve",
            output_path=figures_dir / "validation_loss_curve.png",
        )
        plot_train_val_metric(
            histories=histories,
            train_key="train_accuracy",
            val_key="val_accuracy",
            ylabel="Accuracy",
            title="Accuracy Curve",
            output_path=figures_dir / "accuracy_curve.png",
        )

    comparison_path = reports_dir / "comparison.csv"
    if comparison_path.exists():
        comparison_frame = pd.read_csv(comparison_path)
        plot_macro_f1_comparison(
            comparison_frame=comparison_frame,
            output_path=figures_dir / "macro_f1_comparison.png",
        )

    metrics_path = reports_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        plot_confusion_matrices(metrics, figures_dir)
        if metrics.get("roc_curves"):
            plot_roc_curves(
                roc_curves=metrics["roc_curves"],
                output_path=figures_dir / "roc_curve.png",
            )


def load_histories(reports_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load baseline and GRL training histories if available."""
    history_files = {
        "Baseline": reports_dir / "baseline_training_history.json",
        "GRL": reports_dir / "grl_training_history.json",
    }
    histories = {}
    for model_name, path in history_files.items():
        if not path.exists():
            continue
        payload = load_json(path)
        history = payload.get("history", [])
        if history:
            histories[model_name] = history
    return histories


def plot_confusion_matrices(metrics: dict[str, Any], figures_dir: Path) -> None:
    """Regenerate baseline and GRL confusion matrix figures."""
    models = metrics.get("models", {})
    names = {
        "baseline": "Baseline",
        "grl": "GRL",
    }
    for model_key, model_name in names.items():
        model_metrics = models.get(model_key)
        if not model_metrics or "confusion_matrix" not in model_metrics:
            continue
        label_names = infer_label_names(model_metrics)
        plot_confusion_matrix(
            matrix=model_metrics["confusion_matrix"],
            label_names=label_names,
            output_path=figures_dir / f"{model_key}_confusion_matrix.png",
            title=f"{model_name} Confusion Matrix",
        )


def infer_label_names(model_metrics: dict[str, Any]) -> list[str]:
    """Infer label names from the classification report."""
    report = model_metrics.get("classification_report", {})
    excluded = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    label_names = [label for label in report if label not in excluded]
    if label_names:
        return label_names
    matrix_size = len(model_metrics["confusion_matrix"])
    return [str(index) for index in range(matrix_size)]


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()

