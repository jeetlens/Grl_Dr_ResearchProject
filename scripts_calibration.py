"""Compute calibration metrics from saved predictions."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

def load_predictions(model_kind):
    """Load predictions for a model."""
    csv_path = Path(f"outputs/reports/{model_kind}_predictions.csv")
    df = pd.read_csv(csv_path)
    return df

def compute_calibration_metrics(df):
    """Compute ECE, MCE, Brier score, and NLL from predictions."""
    y_true = df["true_label"].values

    # Get class names in alphabetical order
    class_names = sorted(df["true_label"].unique())
    class_to_id = {cls: idx for idx, cls in enumerate(class_names)}

    # Convert to numeric
    y_true_numeric = np.array([class_to_id[label] for label in y_true])

    # Get probability columns
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    class_names_from_probs = sorted([col.replace("prob_", "") for col in prob_cols])

    # Build probability matrix
    probabilities = df[[f"prob_{cls}" for cls in class_names_from_probs]].values

    metrics = {}

    # Brier Score (mean squared error between predicted probabilities and true labels)
    # For multiclass, we use one-hot encoding of true labels
    y_true_one_hot = np.zeros_like(probabilities)
    y_true_one_hot[np.arange(len(y_true_numeric)), y_true_numeric] = 1
    metrics["brier_score"] = float(brier_score_loss(y_true_one_hot.ravel(), probabilities.ravel()))

    # Negative Log-Likelihood (cross-entropy)
    metrics["nll"] = float(log_loss(y_true_numeric, probabilities))

    # Expected Calibration Error (ECE)
    # Bin predictions by confidence level and compute calibration gap
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = predictions == y_true_numeric

    ece = 0
    mce = 0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "prop_in_bin": float(prop_in_bin),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "count": int(in_bin.sum()),
            })

    metrics["ece"] = float(ece)
    metrics["mce"] = float(mce)
    metrics["ece_bins"] = bin_data
    metrics["n_bins"] = n_bins
    metrics["num_samples"] = len(y_true)

    # Mean confidence
    metrics["mean_confidence"] = float(np.mean(confidences))

    return metrics

def save_calibration_metrics(model_kind, metrics):
    """Save calibration metrics."""
    # Save JSON
    json_path = Path(f"results/calibration/{model_kind}_calibration_metrics.json")
    with json_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {json_path}")

    # Save CSV summary
    csv_path = Path(f"results/calibration/{model_kind}_calibration_summary.csv")
    summary_data = {
        "Metric": [
            "ECE (Expected Calibration Error)",
            "MCE (Maximum Calibration Error)",
            "Brier Score",
            "Negative Log-Likelihood (NLL)",
            "Mean Confidence",
            "Num Samples",
        ],
        "Value": [
            f"{metrics['ece']:.4f}",
            f"{metrics['mce']:.4f}",
            f"{metrics['brier_score']:.4f}",
            f"{metrics['nll']:.4f}",
            f"{metrics['mean_confidence']:.4f}",
            str(metrics['num_samples']),
        ],
    }
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

def generate_comparison(baseline_metrics, grl_metrics):
    """Generate comparison of calibration metrics."""
    comparison_data = {
        "Metric": [
            "ECE",
            "MCE",
            "Brier Score",
            "Negative Log-Likelihood",
            "Mean Confidence",
        ],
        "Baseline": [
            f"{baseline_metrics['ece']:.4f}",
            f"{baseline_metrics['mce']:.4f}",
            f"{baseline_metrics['brier_score']:.4f}",
            f"{baseline_metrics['nll']:.4f}",
            f"{baseline_metrics['mean_confidence']:.4f}",
        ],
        "GRL": [
            f"{grl_metrics['ece']:.4f}",
            f"{grl_metrics['mce']:.4f}",
            f"{grl_metrics['brier_score']:.4f}",
            f"{grl_metrics['nll']:.4f}",
            f"{grl_metrics['mean_confidence']:.4f}",
        ],
    }

    df = pd.DataFrame(comparison_data)

    # Save CSV
    csv_path = Path("results/calibration/calibration_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save Markdown
    md_path = Path("results/calibration/calibration_analysis.md")
    with md_path.open("w") as f:
        f.write("# Calibration Analysis\n\n")
        f.write("## Calibration Metrics Comparison\n\n")
        f.write("| Metric | Baseline | GRL |\n")
        f.write("|---|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Metric']} | {row['Baseline']} | {row['GRL']} |\n")
        f.write("\n")

        f.write("### Metric Definitions\n\n")
        f.write("- **ECE (Expected Calibration Error)**: Measures the gap between predicted confidence and actual accuracy across confidence bins (lower is better)\n")
        f.write("- **MCE (Maximum Calibration Error)**: Maximum calibration gap across all bins (lower is better)\n")
        f.write("- **Brier Score**: Mean squared error between predicted probabilities and true labels (lower is better)\n")
        f.write("- **NLL (Negative Log-Likelihood)**: Cross-entropy loss; measures how well predicted probabilities match true distribution (lower is better)\n")
        f.write("- **Mean Confidence**: Average predicted confidence across all samples\n\n")

        f.write("### Interpretation\n\n")
        f.write("Lower ECE, MCE, Brier Score, and NLL indicate better calibration. Models with high confidence predictions that are frequently incorrect are poorly calibrated.\n\n")

        # Add bin details
        f.write("## Baseline Model - Calibration by Confidence Bin\n\n")
        f.write("| Bin Range | Samples | Accuracy | Confidence | Gap |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for bin_data in baseline_metrics["ece_bins"]:
            gap = abs(bin_data["confidence"] - bin_data["accuracy"])
            f.write(f"| {bin_data['bin_lower']:.2f}-{bin_data['bin_upper']:.2f} | {bin_data['count']} | {bin_data['accuracy']:.4f} | {bin_data['confidence']:.4f} | {gap:.4f} |\n")

        f.write("\n## GRL Model - Calibration by Confidence Bin\n\n")
        f.write("| Bin Range | Samples | Accuracy | Confidence | Gap |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for bin_data in grl_metrics["ece_bins"]:
            gap = abs(bin_data["confidence"] - bin_data["accuracy"])
            f.write(f"| {bin_data['bin_lower']:.2f}-{bin_data['bin_upper']:.2f} | {bin_data['count']} | {bin_data['accuracy']:.4f} | {bin_data['confidence']:.4f} | {gap:.4f} |\n")

    print(f"Saved: {md_path}")

def main():
    """Generate calibration metrics."""
    print("Computing calibration metrics...")

    # Baseline
    print("\nProcessing Baseline model...")
    baseline_df = load_predictions("baseline")
    baseline_metrics = compute_calibration_metrics(baseline_df)
    save_calibration_metrics("baseline", baseline_metrics)

    # GRL
    print("\nProcessing GRL model...")
    grl_df = load_predictions("grl")
    grl_metrics = compute_calibration_metrics(grl_df)
    save_calibration_metrics("grl", grl_metrics)

    # Comparison
    print("\nGenerating calibration comparison...")
    generate_comparison(baseline_metrics, grl_metrics)

    print("\nCalibration metrics generation complete!")

if __name__ == "__main__":
    main()
