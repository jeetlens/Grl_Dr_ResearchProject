"""Evaluate and compare baseline and GRL DistilRoBERTa models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from datasets.data_module import build_dataloader
from datasets.mental_health_dataset import MentalHealthDataset, metadata_from_dict
from models.baseline import DistilRobertaClassifier
from models.grl_model import DistilRobertaGRLClassifier
from trainers.training_utils import move_batch_to_device, tensor_to_numpy
from utils.checkpoint import load_checkpoint
from utils.config import apply_overrides, load_config
from utils.io import ensure_output_dirs, save_json
from utils.logging import get_logger
from utils.paths import project_root, resolve_project_paths
from utils.plots import (
    plot_confusion_matrix,
    plot_macro_f1_comparison,
    plot_model_comparison,
    plot_roc_curves,
)
from utils.seed import resolve_device, set_seed


DEFAULT_BASELINE_CHECKPOINT = "outputs/checkpoints/baseline_model.pt"
DEFAULT_GRL_CHECKPOINT = "outputs/checkpoints/grl_model.pt"
COMPARISON_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "roc",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare baseline and GRL models.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root() / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=DEFAULT_BASELINE_CHECKPOINT,
        help="Path to outputs/checkpoints/baseline_model.pt.",
    )
    parser.add_argument(
        "--grl-checkpoint",
        type=str,
        default=DEFAULT_GRL_CHECKPOINT,
        help="Path to outputs/checkpoints/grl_model.pt.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. data.test_file=path/to/file.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate both trained checkpoints and save comparison artifacts."""
    args = parse_args()
    base_config = apply_overrides(load_config(args.config), args.override)
    base_config = resolve_project_paths(base_config)
    set_seed(int(base_config["project"]["seed"]))
    ensure_output_dirs(base_config["training"]["output_dir"])

    output_dir = Path(base_config["training"]["output_dir"])
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    logger = get_logger("evaluate_models", reports_dir / "comparison.log")
    device = resolve_device(base_config["project"].get("device", "auto"))
    logger.info("Using device: %s", device)

    model_specs = [
        {
            "name": "Baseline",
            "kind": "baseline",
            "checkpoint_path": _resolve_path(args.baseline_checkpoint),
        },
        {
            "name": "GRL",
            "kind": "grl",
            "checkpoint_path": _resolve_path(args.grl_checkpoint),
        },
    ]

    all_metrics = {}
    comparison_rows = []
    roc_curves = []
    for spec in model_specs:
        logger.info("Evaluating %s model.", spec["name"])
        result = evaluate_checkpoint(
            spec=spec,
            fallback_config=base_config,
            overrides=args.override,
            device=device,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
        )
        all_metrics[spec["kind"]] = result["metrics"]
        comparison_rows.append(result["comparison_row"])
        if result["roc_curve"] is not None:
            roc_curves.append(result["roc_curve"])

    comparison_frame = pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS)
    comparison_frame.to_csv(reports_dir / "comparison.csv", index=False)
    plot_model_comparison(
        comparison_frame=comparison_frame,
        output_path=figures_dir / "comparison.png",
    )
    plot_macro_f1_comparison(
        comparison_frame=comparison_frame,
        output_path=figures_dir / "macro_f1_comparison.png",
    )
    plot_roc_curves(
        roc_curves=roc_curves,
        output_path=figures_dir / "roc_curve.png",
    )

    save_json(
        {
            "models": all_metrics,
            "comparison": comparison_frame.to_dict(orient="records"),
            "roc_curves": roc_curves,
        },
        reports_dir / "metrics.json",
    )
    logger.info("Saved comparison to %s", reports_dir / "comparison.csv")


def evaluate_checkpoint(
    spec: dict[str, Any],
    fallback_config: dict[str, Any],
    overrides: list[str],
    device: torch.device,
    reports_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    """Evaluate one checkpoint and return metrics plus comparison row."""
    checkpoint_path = spec["checkpoint_path"]
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = resolve_project_paths(checkpoint.get("config", fallback_config))
    config = apply_overrides(config, overrides)
    metadata = metadata_from_dict(checkpoint["metadata"])
    tokenizer = load_tokenizer(
        checkpoint_path=checkpoint_path,
        config=config,
        model_kind=spec["kind"],
    )

    dataset = MentalHealthDataset(
        csv_path=config["data"]["test_file"],
        tokenizer=tokenizer,
        max_length=int(config["data"]["max_length"]),
        metadata=metadata,
        lowercase=bool(config["data"].get("lowercase", False)),
    )
    data_loader = build_dataloader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    model = build_model(
        model_kind=spec["kind"],
        config=config,
        metadata=metadata,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    labels, predictions, probabilities = predict(
        model=model,
        model_kind=spec["kind"],
        data_loader=data_loader,
        device=device,
        mixed_precision=bool(config["training"].get("mixed_precision", True)),
    )
    metrics = compute_metrics(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        label_names=metadata.label_names,
    )
    roc_curve_data = compute_roc_curve_data(
        labels=labels,
        probabilities=probabilities,
        label_names=metadata.label_names,
        model_name=spec["name"],
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["test_file"] = str(config["data"]["test_file"])
    metrics["roc_curve"] = roc_curve_data

    save_model_outputs(
        model_kind=spec["kind"],
        model_name=spec["name"],
        metrics=metrics,
        dataset=dataset,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        label_names=metadata.label_names,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )
    return {
        "metrics": metrics,
        "comparison_row": comparison_row(spec["name"], metrics),
        "roc_curve": roc_curve_data,
    }


def build_model(
    model_kind: str,
    config: dict[str, Any],
    metadata,
) -> torch.nn.Module:
    """Create the correct architecture for a checkpoint."""
    if model_kind == "grl":
        return DistilRobertaGRLClassifier(
            model_name=config["model"]["backbone"],
            num_labels=len(metadata.label_to_id),
            num_domains=len(metadata.domain_to_id or {"disabled": 0}),
            dropout=float(config["model"].get("dropout", 0.2)),
            grl_lambda=float(config.get("grl", {}).get("lambda_max", 0.1)),
            pretrained=False,
            domain_branch_enabled=metadata.domain_strategy is not None,
        )

    return DistilRobertaClassifier(
        model_name=config["model"]["backbone"],
        num_labels=len(metadata.label_to_id),
        dropout=float(config["model"].get("dropout", 0.2)),
        pretrained=False,
    )


def predict(
    model: torch.nn.Module,
    model_kind: str,
    data_loader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batched inference and return labels, predictions, probabilities."""
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    amp_enabled = mixed_precision and device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc=f"Evaluating {model_kind}",
            leave=False,
        ):
            batch = move_batch_to_device(batch, device)
            with autocast(enabled=amp_enabled):
                if model_kind == "grl":
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        lambda_value=0.0,
                    )
                else:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                logits = outputs["logits"]
                batch_probabilities = torch.softmax(logits, dim=-1)
                batch_predictions = torch.argmax(batch_probabilities, dim=-1)

            labels.append(tensor_to_numpy(batch["labels"]))
            predictions.append(tensor_to_numpy(batch_predictions))
            probabilities.append(tensor_to_numpy(batch_probabilities))

    return (
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Compute all requested classification metrics."""
    class_ids = list(range(len(label_names)))
    report = classification_report(
        labels,
        predictions,
        labels=class_ids,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    metrics.update(compute_roc(labels, probabilities, label_names))
    return metrics


def compute_roc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
) -> dict[str, float | None]:
    """Compute binary or multiclass ROC AUC when defined."""
    num_classes = len(label_names)
    if num_classes < 2:
        return {"roc": None}

    try:
        if num_classes == 2:
            return {
                "roc": float(roc_auc_score(labels, probabilities[:, 1])),
            }
        return {
            "roc": float(
                roc_auc_score(
                    labels,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                )
            )
        }
    except ValueError:
        return {"roc": None}


def compute_roc_curve_data(
    labels: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    model_name: str,
) -> dict[str, Any] | None:
    """Compute ROC curve coordinates for binary or multiclass settings."""
    num_classes = len(label_names)
    if num_classes < 2:
        return None

    try:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            curve_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(
                labels,
                classes=list(range(num_classes)),
            )
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), probabilities.ravel())
            curve_auc = auc(fpr, tpr)
    except ValueError:
        return None

    return {
        "model": model_name,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(curve_auc),
    }


def save_model_outputs(
    model_kind: str,
    model_name: str,
    metrics: dict[str, Any],
    dataset: MentalHealthDataset,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    label_names: list[str],
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    """Save per-model reports, confusion matrices, and predictions."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    report_frame = pd.DataFrame(metrics["classification_report"]).transpose()
    report_frame.to_csv(reports_dir / f"{model_kind}_classification_report.csv")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        label_names,
        figures_dir / f"{model_kind}_confusion_matrix.png",
        normalize=False,
        title=f"{model_name} Confusion Matrix",
    )

    prediction_rows: dict[str, Any] = {
        "model": model_name,
        "text": dataset.texts,
        "true_label": [label_names[index] for index in labels],
        "predicted_label": [label_names[index] for index in predictions],
        "confidence": probabilities.max(axis=1),
    }
    for index, label_name in enumerate(label_names):
        prediction_rows[f"prob_{label_name}"] = probabilities[:, index]

    pd.DataFrame(prediction_rows).to_csv(
        reports_dir / f"{model_kind}_predictions.csv",
        index=False,
    )


def comparison_row(model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Build one row for the comparison table."""
    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc": metrics["roc"],
    }


def load_tokenizer(
    checkpoint_path: Path,
    config: dict[str, Any],
    model_kind: str,
) -> AutoTokenizer:
    """Load the tokenizer saved during training or fall back to backbone."""
    tokenizer_name = "grl_tokenizer" if model_kind == "grl" else "tokenizer"
    tokenizer_candidates = [
        checkpoint_path.parent / tokenizer_name,
        checkpoint_path.parent / "tokenizer",
        checkpoint_path.parent.parent / tokenizer_name,
        checkpoint_path.parent.parent / "tokenizer",
    ]
    for tokenizer_path in tokenizer_candidates:
        if tokenizer_path.exists():
            return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    return AutoTokenizer.from_pretrained(config["model"]["backbone"], use_fast=True)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


if __name__ == "__main__":
    main()
