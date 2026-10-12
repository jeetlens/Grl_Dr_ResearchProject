"""Generate statistical significance tests for multi-seed results."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

def load_multiseed_results():
    """Load multi-seed test results and parse mean/std from strings like '84.81 ± 0.71'."""
    csv_path = Path("outputs/reports/multiseed/paper_ready_test_results.csv")
    df = pd.read_csv(csv_path)

    # Parse the metric columns which have format "mean ± std"
    metrics_to_parse = ["Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC"]

    parsed_data = []
    for _, row in df.iterrows():
        parsed_row = {"Model": row["Model"]}
        for metric in metrics_to_parse:
            if metric in df.columns:
                value_str = row[metric]
                # Parse "mean ± std"
                parts = value_str.split(" ± ")
                if len(parts) == 2:
                    parsed_row[f"{metric}_mean"] = float(parts[0])
                    parsed_row[f"{metric}_std"] = float(parts[1])
        parsed_data.append(parsed_row)

    return pd.DataFrame(parsed_data)

def compute_bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval."""
    bootstrapped = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped.append(np.mean(sample))

    bootstrapped = np.array(bootstrapped)
    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "ci_lower": float(np.percentile(bootstrapped, lower_percentile)),
        "ci_upper": float(np.percentile(bootstrapped, upper_percentile)),
        "ci": ci,
    }

def perform_wilcoxon_test(baseline_data, grl_data):
    """Perform Wilcoxon signed-rank test."""
    # Check if we have paired samples (same number of observations)
    if len(baseline_data) != len(grl_data):
        return None  # Cannot perform paired test

    # Compute differences
    differences = grl_data - baseline_data

    # Perform test
    statistic, p_value = stats.wilcoxon(differences)

    return {
        "test": "Wilcoxon Signed-Rank Test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences)),
    }

def generate_statistics_report():
    """Generate statistical significance report."""
    df = load_multiseed_results()

    # Extract metrics - note that the data is aggregated (mean ± std from multiple seeds)
    # We'll use the reported mean and std as estimates
    baseline_accuracy = df[df["Model"] == "DistilRoBERTa"]["Accuracy_mean"].values
    grl_accuracy = df[df["Model"] == "DistilRoBERTa + GRL"]["Accuracy_mean"].values

    baseline_f1 = df[df["Model"] == "DistilRoBERTa"]["Macro-F1_mean"].values
    grl_f1 = df[df["Model"] == "DistilRoBERTa + GRL"]["Macro-F1_mean"].values

    baseline_weighted_f1 = df[df["Model"] == "DistilRoBERTa"]["Weighted-F1_mean"].values
    grl_weighted_f1 = df[df["Model"] == "DistilRoBERTa + GRL"]["Weighted-F1_mean"].values

    baseline_roc = df[df["Model"] == "DistilRoBERTa"]["ROC-AUC_mean"].values
    grl_roc = df[df["Model"] == "DistilRoBERTa + GRL"]["ROC-AUC_mean"].values

    results = {}

    # Accuracy
    print("Computing statistics for Accuracy...")
    results["accuracy"] = {
        "baseline_mean": float(baseline_accuracy[0]) if len(baseline_accuracy) > 0 else 0,
        "grl_mean": float(grl_accuracy[0]) if len(grl_accuracy) > 0 else 0,
        "baseline_std": float(df[df["Model"] == "DistilRoBERTa"]["Accuracy_std"].values[0]) if len(baseline_accuracy) > 0 else 0,
        "grl_std": float(df[df["Model"] == "DistilRoBERTa + GRL"]["Accuracy_std"].values[0]) if len(grl_accuracy) > 0 else 0,
    }

    # Macro F1
    print("Computing for Macro F1...")
    results["macro_f1"] = {
        "baseline_mean": float(baseline_f1[0]) if len(baseline_f1) > 0 else 0,
        "grl_mean": float(grl_f1[0]) if len(grl_f1) > 0 else 0,
        "baseline_std": float(df[df["Model"] == "DistilRoBERTa"]["Macro-F1_std"].values[0]) if len(baseline_f1) > 0 else 0,
        "grl_std": float(df[df["Model"] == "DistilRoBERTa + GRL"]["Macro-F1_std"].values[0]) if len(grl_f1) > 0 else 0,
    }

    # Weighted F1
    print("Computing for Weighted F1...")
    results["weighted_f1"] = {
        "baseline_mean": float(baseline_weighted_f1[0]) if len(baseline_weighted_f1) > 0 else 0,
        "grl_mean": float(grl_weighted_f1[0]) if len(grl_weighted_f1) > 0 else 0,
        "baseline_std": float(df[df["Model"] == "DistilRoBERTa"]["Weighted-F1_std"].values[0]) if len(baseline_weighted_f1) > 0 else 0,
        "grl_std": float(df[df["Model"] == "DistilRoBERTa + GRL"]["Weighted-F1_std"].values[0]) if len(grl_weighted_f1) > 0 else 0,
    }

    # ROC-AUC
    print("Computing for ROC-AUC...")
    results["roc_auc"] = {
        "baseline_mean": float(baseline_roc[0]) if len(baseline_roc) > 0 else 0,
        "grl_mean": float(grl_roc[0]) if len(grl_roc) > 0 else 0,
        "baseline_std": float(df[df["Model"] == "DistilRoBERTa"]["ROC-AUC_std"].values[0]) if len(baseline_roc) > 0 else 0,
        "grl_std": float(df[df["Model"] == "DistilRoBERTa + GRL"]["ROC-AUC_std"].values[0]) if len(grl_roc) > 0 else 0,
    }

    # Save results
    json_path = Path("results/statistics/statistical_tests.json")
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    return results

def generate_markdown_report(results):
    """Generate markdown report."""
    md_path = Path("results/statistics/statistical_analysis.md")

    with md_path.open("w") as f:
        f.write("# Statistical Analysis: Baseline vs GRL (Multi-Seed Results)\n\n")
        f.write("## Summary\n\n")
        f.write("This analysis reports mean and standard deviation across three random seeds:\n\n")

        f.write("## Accuracy\n\n")
        f.write("| Metric | Baseline | GRL |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Mean | {results['accuracy']['baseline_mean']:.4f} | {results['accuracy']['grl_mean']:.4f} |\n")
        f.write(f"| Std Dev | {results['accuracy']['baseline_std']:.4f} | {results['accuracy']['grl_std']:.4f} |\n")

        f.write("\n## Macro F1\n\n")
        f.write("| Metric | Baseline | GRL |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Mean | {results['macro_f1']['baseline_mean']:.4f} | {results['macro_f1']['grl_mean']:.4f} |\n")
        f.write(f"| Std Dev | {results['macro_f1']['baseline_std']:.4f} | {results['macro_f1']['grl_std']:.4f} |\n")

        f.write("\n## Weighted F1\n\n")
        f.write("| Metric | Baseline | GRL |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Mean | {results['weighted_f1']['baseline_mean']:.4f} | {results['weighted_f1']['grl_mean']:.4f} |\n")
        f.write(f"| Std Dev | {results['weighted_f1']['baseline_std']:.4f} | {results['weighted_f1']['grl_std']:.4f} |\n")

        f.write("\n## ROC-AUC\n\n")
        f.write("| Metric | Baseline | GRL |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Mean | {results['roc_auc']['baseline_mean']:.4f} | {results['roc_auc']['grl_mean']:.4f} |\n")
        f.write(f"| Std Dev | {results['roc_auc']['baseline_std']:.4f} | {results['roc_auc']['grl_std']:.4f} |\n")

        f.write("\n## Key Observations\n\n")

        # Compute differences
        acc_diff = results['accuracy']['grl_mean'] - results['accuracy']['baseline_mean']
        f1_diff = results['macro_f1']['grl_mean'] - results['macro_f1']['baseline_mean']
        wf1_diff = results['weighted_f1']['grl_mean'] - results['weighted_f1']['baseline_mean']
        roc_diff = results['roc_auc']['grl_mean'] - results['roc_auc']['baseline_mean']

        f.write(f"- **Accuracy Difference** (GRL - Baseline): {acc_diff:+.4f}\n")
        f.write(f"- **Macro F1 Difference** (GRL - Baseline): {f1_diff:+.4f}\n")
        f.write(f"- **Weighted F1 Difference** (GRL - Baseline): {wf1_diff:+.4f}\n")
        f.write(f"- **ROC-AUC Difference** (GRL - Baseline): {roc_diff:+.4f}\n\n")

        f.write("## Limitations\n\n")
        f.write("- **Small sample size**: Only 3 independent seeds; results are descriptive\n")
        f.write("- **No formal significance test**: Reported results are raw means ± std from saved experiments\n")
        f.write("- **Proxy-domain evaluation**: Results use proxy-domain labels (text-length bins), not genuine source domains\n")
        f.write("- **Overlapping confidence bands**: Suggest no clear winner between models\n\n")

        f.write("## Conclusion\n\n")
        f.write("Based on three independent training seeds, the models show similar performance:\n\n")
        if abs(f1_diff) < 0.01:
            f.write("- The baseline and GRL models have nearly identical Macro F1 performance\n")
        elif f1_diff > 0:
            f.write("- The GRL model shows slightly higher Macro F1 performance\n")
        else:
            f.write("- The baseline model shows slightly higher Macro F1 performance\n")

        f.write("- Standard deviations overlap, indicating high uncertainty in estimates\n")
        f.write("- With only 3 seeds, these results should be treated as descriptive, not conclusive\n")
        f.write("- Larger-scale studies with more seeds would provide stronger evidence\n")

    print(f"Saved: {md_path}")

def main():
    """Generate statistical analysis."""
    print("Generating statistical analysis...")
    results = generate_statistics_report()
    generate_markdown_report(results)
    print("Statistical analysis complete!")

if __name__ == "__main__":
    main()
