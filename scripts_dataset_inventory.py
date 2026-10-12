"""Generate dataset inventory table from validation report."""

import json
from pathlib import Path
import pandas as pd

def load_validation_report():
    """Load the dataset validation report."""
    report_path = Path("outputs/reports/dataset_validation.json")
    with report_path.open("r") as f:
        return json.load(f)

def df_to_markdown(df):
    """Convert dataframe to markdown table."""
    headers = df.columns.tolist()
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line = "|" + "|".join(["---" if isinstance(df[col].iloc[0], str) else "---:" for col in headers]) + "|"

    rows = []
    for _, row in df.iterrows():
        row_line = "| " + " | ".join(str(v) for v in row) + " |"
        rows.append(row_line)

    return "\n".join([header_line, sep_line] + rows)

def generate_dataset_inventory():
    """Generate comprehensive dataset inventory table."""
    report = load_validation_report()
    splits = report["splits"]

    # Create inventory table
    rows = []
    for split_name, split_data in splits.items():
        rows.append({
            "Split": split_name.capitalize(),
            "File": f"data/both_{split_name}.csv",
            "Raw Rows": split_data["raw_row_count"],
            "Usable Rows": split_data["usable_row_count"],
            "Duplicates": split_data["duplicate_combined_texts"],
            "Missing Values": sum(split_data["missing_values"].values()),
        })

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = Path("results/dataset/dataset_inventory.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Save markdown
    md_path = Path("results/dataset/dataset_inventory.md")
    with md_path.open("w") as f:
        f.write("# Dataset Inventory\n\n")
        f.write(df_to_markdown(df))
        f.write("\n\n")

        # Add class distribution
        f.write("## Class Distribution\n\n")
        f.write("| Split | ADHD | Anxiety | Bipolar | Depression | PTSD | None | Total |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for split_name in ["train", "validation", "test"]:
            split_data = splits[split_name]
            classes = split_data["class_distribution"]
            total = sum(classes.values())
            f.write(f"| {split_name.capitalize()} | {classes.get('adhd', 0)} | {classes.get('anxiety', 0)} | {classes.get('bipolar', 0)} | {classes.get('depression', 0)} | {classes.get('ptsd', 0)} | {classes.get('none', 0)} | {total} |\n")

        # Add domain distribution
        f.write("\n## Proxy-Domain Distribution (Text Length)\n\n")
        f.write("| Split | Short (<=Median) | Long (>Median) | Total |\n")
        f.write("|---|---:|---:|---:|\n")
        for split_name in ["train", "validation", "test"]:
            split_data = splits[split_name]
            domains = split_data["generated_domain_distribution"]
            short = domains.get("short_or_equal_training_median", 0)
            long = domains.get("longer_than_training_median", 0)
            total = short + long
            f.write(f"| {split_name.capitalize()} | {short} | {long} | {total} |\n")

        # Add notes
        f.write("\n## Notes\n\n")
        f.write("- Proxy Domain: Text-length bins computed from training set median (135.0 words)\n")
        f.write("- Classes: 6 mental health categories (ADHD, Anxiety, Bipolar, Depression, PTSD, None)\n")
        f.write("- Training: 13,727 samples (with 3 duplicate combined texts)\n")
        f.write("- Validation: 1,488 samples (balanced, 248 per class)\n")
        f.write("- Test: 1,488 samples (balanced, 248 per class)\n")
        f.write("- Total: 16,703 unique samples\n")

    print(f"Saved: {md_path}")

if __name__ == "__main__":
    generate_dataset_inventory()
