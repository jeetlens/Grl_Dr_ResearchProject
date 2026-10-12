"""Verify and regenerate core classification metrics from saved predictions."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

def load_predictions(model_kind):
    """Load predictions for a model."""
    csv_path = Path(f"outputs/reports/{model_kind}_predictions.csv")
    df = pd.read_csv(csv_path)
    return df

def compute_metrics(df):
    """Compute all classification metrics from predictions dataframe."""
    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    # Get class names in alphabetical order
    class_names = sorted(df["true_label"].unique())
    class_to_id = {cls: idx for idx, cls in enumerate(class_names)}

    # Convert to numeric
    y_true_numeric = np.array([class_to_id[label] for label in y_true])
    y_pred_numeric = np.array([class_to_id[label] for label in y_pred])

    # Get probability columns
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    class_names_from_probs = sorted([col.replace("prob_", "") for col in prob_cols])

    # Build probability matrix
    probabilities = df[[f"prob_{cls}" for cls in class_names_from_probs]].values

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true_numeric, y_pred_numeric)),
        "macro_precision": float(
            precision_score(y_true_numeric, y_pred_numeric, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_numeric, y_pred_numeric, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true_numeric, y_pred_numeric, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
        ),
        "weighted_precision": float(
            precision_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
        ),
        "num_samples": len(y_true),
    }

    # Try ROC-AUC
    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true_numeric, probabilities, average="macro", multi_class="ovr")
        )
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        metrics["roc_auc"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true_numeric, y_pred_numeric, labels=range(len(class_names)))
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    report = classification_report(
        y_true_numeric,
        y_pred_numeric,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    metrics["per_class"] = {}
    for cls in class_names:
        if cls in report:
            metrics["per_class"][cls] = {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }

    metrics["class_names"] = class_names

    return metrics

def save_metrics(model_kind, metrics):
    """Save metrics to CSV and JSON."""
    # Save JSON
    json_path = Path(f"results/core_performance/{model_kind}_metrics.json")
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {json_path}")

    # Save CSV for per-class metrics
    csv_path = Path(f"results/core_performance/{model_kind}_per_class_metrics.csv")
    rows = []
    for cls in metrics["class_names"]:
        row = {"class": cls}
        if cls in metrics["per_class"]:
            row.update(metrics["per_class"][cls])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

def generate_comparison_table(baseline_metrics, grl_metrics):
    """Generate comparison table."""
    rows = []

    # Baseline
    rows.append({
        "Model": "DistilRoBERTa (Baseline)",
        "Accuracy": f"{baseline_metrics['accuracy']:.4f}",
        "Macro Precision": f"{baseline_metrics['macro_precision']:.4f}",
        "Macro Recall": f"{baseline_metrics['macro_recall']:.4f}",
        "Macro F1": f"{baseline_metrics['macro_f1']:.4f}",
        "Weighted F1": f"{baseline_metrics['weighted_f1']:.4f}",
        "ROC-AUC": f"{baseline_metrics['roc_auc']:.4f}" if baseline_metrics['roc_auc'] else "N/A",
    })

    # GRL
    rows.append({
        "Model": "DistilRoBERTa + GRL",
        "Accuracy": f"{grl_metrics['accuracy']:.4f}",
        "Macro Precision": f"{grl_metrics['macro_precision']:.4f}",
        "Macro Recall": f"{grl_metrics['macro_recall']:.4f}",
        "Macro F1": f"{grl_metrics['macro_f1']:.4f}",
        "Weighted F1": f"{grl_metrics['weighted_f1']:.4f}",
        "ROC-AUC": f"{grl_metrics['roc_auc']:.4f}" if grl_metrics['roc_auc'] else "N/A",
    })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = Path("results/core_performance/model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save Markdown
    md_path = Path("results/core_performance/model_comparison.md")
    with md_path.open("w") as f:
        f.write("# Model Comparison: Baseline vs GRL\n\n")
        f.write("## Overall Metrics\n\n")
        f.write("| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']} | {row['Macro Precision']} | {row['Macro Recall']} | {row['Macro F1']} | {row['Weighted F1']} | {row['ROC-AUC']} |\n")
        f.write("\n")

        # Add per-class comparison
        f.write("## Per-Class Metrics (Baseline)\n\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cls in baseline_metrics["class_names"]:
            if cls in baseline_metrics["per_class"]:
                metrics = baseline_metrics["per_class"][cls]
                f.write(f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {int(metrics['support'])} |\n")

        f.write("\n## Per-Class Metrics (GRL)\n\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cls in grl_metrics["class_names"]:
            if cls in grl_metrics["per_class"]:
                metrics = grl_metrics["per_class"][cls]
                f.write(f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {int(metrics['support'])} |\n")

    print(f"Saved: {md_path}")

def main():
    """Generate all core performance metrics."""
    print("Loading predictions and computing metrics...")

    # Baseline
    print("\nProcessing Baseline model...")
    baseline_df = load_predictions("baseline")
    baseline_metrics = compute_metrics(baseline_df)
    save_metrics("baseline", baseline_metrics)

    # GRL
    print("\nProcessing GRL model...")
    grl_df = load_predictions("grl")
    grl_metrics = compute_metrics(grl_df)
    save_metrics("grl", grl_metrics)

    # Comparison
    print("\nGenerating comparison table...")
    generate_comparison_table(baseline_metrics, grl_metrics)

    print("\nCore performance metrics generation complete!")

if __name__ == "__main__":
    main()
