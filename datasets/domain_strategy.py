"""Documented domain-label strategies for GRL experiments.

The Murarka ``both_*.csv`` files do not contain a genuine cross-dataset source
or domain column. Therefore, the default GRL experiment is explicitly labeled as
proxy-domain adversarial training, not cross-dataset domain adaptation.

Supported strategies:

- ``explicit_column``: use a genuine domain/source column when present.
- ``text_length_bins``: compute the median word length from the training split
  only; domain 0 means length <= training median, domain 1 means length >
  training median.
- ``title_presence``: domain 0 when title is empty, domain 1 when title exists.
- ``disable``: return no domain labels so GRL training runs without the
  adversarial branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SUPPORTED_DOMAIN_STRATEGIES = (
    "explicit_column",
    "text_length_bins",
    "title_presence",
    "disable",
)


@dataclass(frozen=True)
class DomainStrategyMetadata:
    """Serialized state needed to apply one domain strategy to all splits."""

    strategy: str
    description: str
    domain_to_id: dict[str, int] | None
    domain_column: str | None = None
    text_length_threshold: float | None = None
    length_unit: str | None = None


def build_domain_strategy(
    dataframe: pd.DataFrame,
    strategy_name: str,
    text_column: str,
    title_column: str | None,
    domain_column: str | None = None,
) -> DomainStrategyMetadata | None:
    """Build domain metadata from the training split only."""
    if strategy_name not in SUPPORTED_DOMAIN_STRATEGIES:
        raise ValueError(
            f"Unsupported domain strategy '{strategy_name}'. "
            f"Choose one of {SUPPORTED_DOMAIN_STRATEGIES}."
        )

    if strategy_name == "disable":
        return None

    if strategy_name == "explicit_column":
        if not domain_column:
            raise ValueError(
                "domain.strategy=explicit_column requires data.domain_column."
            )
        if domain_column not in dataframe.columns:
            raise ValueError(
                f"Domain column '{domain_column}' not found. Available columns: "
                f"{list(dataframe.columns)}"
            )
        domains = sorted(
            str(value)
            for value in dataframe[domain_column].dropna().unique().tolist()
        )
        return DomainStrategyMetadata(
            strategy=strategy_name,
            description="Genuine domain labels from an explicit CSV column.",
            domain_column=domain_column,
            domain_to_id={domain: index for index, domain in enumerate(domains)},
        )

    if strategy_name == "text_length_bins":
        lengths = word_lengths(dataframe[text_column])
        threshold = float(lengths.median())
        return DomainStrategyMetadata(
            strategy=strategy_name,
            description=(
                "Proxy-domain adversarial training using training-set median "
                "word length. This is not cross-dataset domain adaptation."
            ),
            domain_to_id={
                "short_or_equal_training_median": 0,
                "longer_than_training_median": 1,
            },
            text_length_threshold=threshold,
            length_unit="words",
        )

    if strategy_name == "title_presence":
        if title_column is None:
            raise ValueError("title_presence strategy requires a title column.")
        return DomainStrategyMetadata(
            strategy=strategy_name,
            description=(
                "Proxy-domain adversarial training based on title presence. "
                "This is not cross-dataset domain adaptation."
            ),
            domain_to_id={"title_empty": 0, "title_present": 1},
        )

    raise AssertionError(f"Unhandled domain strategy: {strategy_name}")


def apply_domain_strategy(
    dataframe: pd.DataFrame,
    metadata: DomainStrategyMetadata,
    text_column: str,
    title_column: str | None,
) -> list[int]:
    """Apply training-derived domain metadata to any split."""
    if metadata.strategy == "explicit_column":
        if metadata.domain_column is None:
            raise ValueError("Explicit domain strategy is missing domain_column.")
        raw_domains = dataframe[metadata.domain_column].astype(str).tolist()
        domain_to_id = metadata.domain_to_id or {}
        unknown = sorted(set(raw_domains) - set(domain_to_id))
        if unknown:
            raise ValueError(
                "Found domain values absent from training metadata: "
                f"{unknown}"
            )
        return [domain_to_id[value] for value in raw_domains]

    if metadata.strategy == "text_length_bins":
        if metadata.text_length_threshold is None:
            raise ValueError("text_length_bins strategy is missing threshold.")
        lengths = word_lengths(dataframe[text_column])
        return [
            0 if length <= metadata.text_length_threshold else 1
            for length in lengths.tolist()
        ]

    if metadata.strategy == "title_presence":
        if title_column is None:
            raise ValueError("title_presence strategy requires a title column.")
        titles = dataframe[title_column].fillna("").astype(str).str.strip()
        return [0 if title == "" else 1 for title in titles.tolist()]

    raise ValueError(f"Cannot apply unsupported strategy: {metadata.strategy}")


def word_lengths(text_series: pd.Series) -> pd.Series:
    """Count whitespace-delimited words in already prepared text."""
    return text_series.fillna("").astype(str).str.split().str.len()


def domain_metadata_to_dict(
    metadata: DomainStrategyMetadata,
) -> dict[str, Any]:
    """Serialize domain strategy metadata."""
    return {
        "strategy": metadata.strategy,
        "description": metadata.description,
        "domain_to_id": metadata.domain_to_id,
        "domain_column": metadata.domain_column,
        "text_length_threshold": metadata.text_length_threshold,
        "length_unit": metadata.length_unit,
    }


def domain_metadata_from_dict(data: dict[str, Any]) -> DomainStrategyMetadata:
    """Deserialize domain strategy metadata."""
    return DomainStrategyMetadata(
        strategy=data["strategy"],
        description=data["description"],
        domain_to_id=(
            {
                str(key): int(value)
                for key, value in data["domain_to_id"].items()
            }
            if data.get("domain_to_id") is not None
            else None
        ),
        domain_column=data.get("domain_column"),
        text_length_threshold=data.get("text_length_threshold"),
        length_unit=data.get("length_unit"),
    )

