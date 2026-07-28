"""Validate Murarka CSV files before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.domain_strategy import apply_domain_strategy
from datasets.mental_health_dataset import (
    MentalHealthDataset,
    prepare_dataframe,
)
from utils.config import apply_overrides, load_config
from utils.paths import project_root, resolve_project_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Murarka mental health dataset CSV files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root() / "configs" / "default.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of combined title/post examples to print per split.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values.",
    )
    return parser.parse_args()


def main() -> None:
    """Print validation diagnostics and save a JSON report."""
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    config = resolve_project_paths(config)
    data_config = config["data"]
    domain_config = config.get("domain", {})

    paths = {
        "train": Path(data_config["train_file"]),
        "validation": Path(data_config["val_file"]),
        "test": Path(data_config["test_file"]),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files:\n" + "\n".join(missing)
        )

    metadata = MentalHealthDataset.create_metadata_from_train(
        csv_path=paths["train"],
        text_column=data_config.get("text_column"),
        label_column=data_config.get("label_column"),
        domain_strategy_name=domain_config.get("strategy", "text_length_bins"),
        domain_column=data_config.get("domain_column"),
    )

    print_header("Domain Strategy")
    if metadata.domain_strategy is None:
        print("strategy: disable")
    else:
        print(f"strategy: {metadata.domain_strategy.strategy}")
        print(f"description: {metadata.domain_strategy.description}")
        print(f"domain_to_id: {metadata.domain_strategy.domain_to_id}")
        print(f"text_length_threshold: "
              f"{metadata.domain_strategy.text_length_threshold}")

    report: dict[str, Any] = {
        "domain_strategy": (
            None
            if metadata.domain_strategy is None
            else {
                "strategy": metadata.domain_strategy.strategy,
                "description": metadata.domain_strategy.description,
                "domain_to_id": metadata.domain_strategy.domain_to_id,
                "text_length_threshold": (
                    metadata.domain_strategy.text_length_threshold
                ),
                "length_unit": metadata.domain_strategy.length_unit,
            }
        ),
        "splits": {},
    }

    for split_name, path in paths.items():
        split_report = validate_split(
            split_name=split_name,
            path=path,
            metadata=metadata,
            examples=args.examples,
        )
        report["splits"][split_name] = split_report

    output_path = (
        Path(config["training"]["output_dir"])
        / "reports"
        / "dataset_validation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved validation report: {output_path}")


def validate_split(
    split_name: str,
    path: Path,
    metadata,
    examples: int,
) -> dict[str, Any]:
    """Print and return validation diagnostics for one split."""
    raw = pd.read_csv(path)
    prepared = prepare_dataframe(raw, metadata)

    print_header(f"{split_name.upper()} split")
    print(f"file: {path}")
    print(f"CSV headers: {list(raw.columns)}")
    print(f"raw row count: {len(raw)}")
    print(f"usable row count: {len(prepared)}")
    print("missing values:")
    print(raw.isna().sum().to_string())

    duplicate_count = int(prepared[metadata.text_column].duplicated().sum())
    print(f"duplicate combined texts: {duplicate_count}")

    class_distribution = (
        prepared[metadata.label_column].astype(str).value_counts().to_dict()
    )
    print("class distribution:")
    print(pd.Series(class_distribution).to_string())

    domain_distribution = generated_domain_distribution(prepared, metadata)
    print("generated domain distribution:")
    print(pd.Series(domain_distribution).to_string())

    print("combined title/post examples:")
    for example in prepared[metadata.text_column].head(examples).tolist():
        print(f"- {example[:300]}")

    return {
        "headers": list(raw.columns),
        "raw_row_count": int(len(raw)),
        "usable_row_count": int(len(prepared)),
        "missing_values": {
            column: int(value)
            for column, value in raw.isna().sum().to_dict().items()
        },
        "duplicate_combined_texts": duplicate_count,
        "class_distribution": {
            str(key): int(value)
            for key, value in class_distribution.items()
        },
        "generated_domain_distribution": domain_distribution,
        "examples": prepared[metadata.text_column].head(examples).tolist(),
    }


def generated_domain_distribution(
    dataframe: pd.DataFrame,
    metadata,
) -> dict[str, int]:
    """Return generated domain counts for the selected strategy."""
    if metadata.domain_strategy is None:
        return {"disabled": int(len(dataframe))}

    domains = apply_domain_strategy(
        dataframe=dataframe,
        metadata=metadata.domain_strategy,
        text_column=metadata.text_column,
        title_column=metadata.title_column,
    )
    id_to_domain = {
        domain_id: domain_name
        for domain_name, domain_id in (
            metadata.domain_strategy.domain_to_id or {}
        ).items()
    }
    series = pd.Series(domains).map(id_to_domain).value_counts()
    return {
        str(key): int(value)
        for key, value in series.to_dict().items()
    }


def print_header(title: str) -> None:
    """Print a readable section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


if __name__ == "__main__":
    main()
