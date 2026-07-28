"""Murarka mental health dataset handling.

The expected Murarka CSV schema contains:

- ``ID``
- ``title``
- ``post``
- ``class_name``
- ``class_id``

For model input, this module constructs ``combined_text`` as a safe
concatenation of ``title`` and ``post``. Rows with empty combined text are
rejected before tokenization. Domain labels are never fabricated from row index;
they must come from a documented strategy in ``datasets.domain_strategy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from datasets.domain_strategy import (
    DomainStrategyMetadata,
    apply_domain_strategy,
    build_domain_strategy,
    domain_metadata_from_dict,
    domain_metadata_to_dict,
)


TEXT_COLUMN_CANDIDATES = (
    "combined_text",
    "text",
    "post",
    "tweet",
    "content",
    "statement",
    "sentence",
    "body",
)

LABEL_COLUMN_CANDIDATES = (
    "class_name",
    "label",
    "class",
    "target",
    "category",
    "mental_health_label",
)

TITLE_COLUMN_CANDIDATES = ("title",)
POST_COLUMN_CANDIDATES = ("post", "text", "body", "content")
CLASS_ID_COLUMN_CANDIDATES = ("class_id",)
ID_COLUMN_CANDIDATES = ("ID", "id")


@dataclass(frozen=True)
class DatasetMetadata:
    """Shared dataset metadata for reproducible training and evaluation."""

    text_column: str
    label_column: str
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    title_column: str | None = None
    post_column: str | None = None
    id_column: str | None = None
    class_id_column: str | None = None
    domain_strategy: DomainStrategyMetadata | None = None

    @property
    def label_names(self) -> list[str]:
        """Return label names sorted by numeric class id."""
        return [
            self.id_to_label[index]
            for index in sorted(self.id_to_label)
        ]

    @property
    def domain_to_id(self) -> dict[str, int] | None:
        """Return domain mapping if a domain strategy is enabled."""
        if self.domain_strategy is None:
            return None
        return self.domain_strategy.domain_to_id


class MentalHealthDataset(Dataset):
    """Tokenized CSV-backed dataset for mental health classification."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        metadata: DatasetMetadata | None = None,
        text_column: str | None = None,
        label_column: str | None = None,
        domain_strategy_name: str = "text_length_bins",
        domain_column: str | None = None,
        lowercase: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase
        raw_dataframe = pd.read_csv(self.csv_path)

        if raw_dataframe.empty:
            raise ValueError(f"Dataset is empty: {self.csv_path}")

        self.metadata = metadata or self.create_metadata_from_train(
            csv_path=self.csv_path,
            text_column=text_column,
            label_column=label_column,
            domain_strategy_name=domain_strategy_name,
            domain_column=domain_column,
        )
        self.dataframe = prepare_dataframe(raw_dataframe, self.metadata)

        if self.dataframe.empty:
            raise ValueError(
                "No usable rows remain after rejecting empty combined text: "
                f"{self.csv_path}"
            )

        self.texts = self._prepare_texts()
        self.labels = self._encode_labels()
        self.domains = self._encode_domains()

    def __len__(self) -> int:
        """Return dataset size after empty text rows are removed."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a tokenized training example."""
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }
        if self.domains is not None:
            item["domain_labels"] = torch.tensor(
                self.domains[index],
                dtype=torch.long,
            )
        return item

    @classmethod
    def create_metadata_from_train(
        cls,
        csv_path: str | Path,
        text_column: str | None = None,
        label_column: str | None = None,
        domain_strategy_name: str = "text_length_bins",
        domain_column: str | None = None,
    ) -> DatasetMetadata:
        """Create metadata from the training split only."""
        train_path = Path(csv_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset file not found: {train_path}")

        raw_dataframe = pd.read_csv(train_path)
        if raw_dataframe.empty:
            raise ValueError(f"Dataset is empty: {train_path}")

        title_column = infer_optional_column(
            raw_dataframe,
            TITLE_COLUMN_CANDIDATES,
        )
        post_column = infer_optional_column(raw_dataframe, POST_COLUMN_CANDIDATES)
        inferred_text_column = infer_text_column(
            raw_dataframe,
            text_column=text_column,
            title_column=title_column,
            post_column=post_column,
        )
        inferred_label_column = label_column or infer_column(
            raw_dataframe,
            LABEL_COLUMN_CANDIDATES,
            purpose="label",
        )
        id_column = infer_optional_column(raw_dataframe, ID_COLUMN_CANDIDATES)
        class_id_column = infer_optional_column(
            raw_dataframe,
            CLASS_ID_COLUMN_CANDIDATES,
        )

        metadata = DatasetMetadata(
            text_column=inferred_text_column,
            label_column=inferred_label_column,
            label_to_id={},
            id_to_label={},
            title_column=title_column,
            post_column=post_column,
            id_column=id_column,
            class_id_column=class_id_column,
            domain_strategy=None,
        )
        prepared = prepare_dataframe(raw_dataframe, metadata)
        label_to_id, id_to_label = build_label_mapping(
            prepared,
            label_column=inferred_label_column,
            class_id_column=class_id_column,
        )
        domain_strategy = build_domain_strategy(
            dataframe=prepared,
            strategy_name=domain_strategy_name,
            text_column=inferred_text_column,
            title_column=title_column,
            domain_column=domain_column,
        )
        return DatasetMetadata(
            text_column=inferred_text_column,
            label_column=inferred_label_column,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            title_column=title_column,
            post_column=post_column,
            id_column=id_column,
            class_id_column=class_id_column,
            domain_strategy=domain_strategy,
        )

    def _prepare_texts(self) -> list[str]:
        texts = self.dataframe[self.metadata.text_column].astype(str).tolist()
        if self.lowercase:
            texts = [text.lower() for text in texts]
        return texts

    def _encode_labels(self) -> list[int]:
        label_to_id = self.metadata.label_to_id
        raw_labels = self.dataframe[self.metadata.label_column].astype(str).tolist()
        unknown = sorted(set(raw_labels) - set(label_to_id))
        if unknown:
            raise ValueError(
                f"Found labels in {self.csv_path} absent from training metadata: "
                f"{unknown}"
            )
        return [label_to_id[label] for label in raw_labels]

    def _encode_domains(self) -> list[int] | None:
        if self.metadata.domain_strategy is None:
            return None
        return apply_domain_strategy(
            dataframe=self.dataframe,
            metadata=self.metadata.domain_strategy,
            text_column=self.metadata.text_column,
            title_column=self.metadata.title_column,
        )


def prepare_dataframe(
    dataframe: pd.DataFrame,
    metadata: DatasetMetadata,
) -> pd.DataFrame:
    """Validate columns, create combined text, and reject empty text rows."""
    missing = [
        column
        for column in required_columns(metadata)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Available columns: "
            f"{list(dataframe.columns)}"
        )

    prepared = dataframe.copy()
    if metadata.title_column and metadata.post_column:
        prepared[metadata.text_column] = build_combined_text(
            prepared,
            title_column=metadata.title_column,
            post_column=metadata.post_column,
        )
    else:
        prepared[metadata.text_column] = (
            prepared[metadata.text_column].fillna("").astype(str).str.strip()
        )

    prepared = prepared[prepared[metadata.text_column].str.len() > 0].copy()
    prepared = prepared.reset_index(drop=True)
    return prepared


def build_combined_text(
    dataframe: pd.DataFrame,
    title_column: str,
    post_column: str,
) -> pd.Series:
    """Safely concatenate title and post exactly as required."""
    title = dataframe[title_column].fillna("").astype(str).str.strip()
    post = dataframe[post_column].fillna("").astype(str).str.strip()
    return (title + " " + post).str.strip()


def build_label_mapping(
    dataframe: pd.DataFrame,
    label_column: str,
    class_id_column: str | None,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build label mapping, preferring Murarka class_id order when available."""
    if class_id_column and class_id_column in dataframe.columns:
        pairs = (
            dataframe[[label_column, class_id_column]]
            .dropna()
            .drop_duplicates()
            .copy()
        )
        pairs[class_id_column] = pairs[class_id_column].astype(int)
        pairs = pairs.sort_values(class_id_column)
        label_to_id = {
            str(row[label_column]): index
            for index, (_, row) in enumerate(pairs.iterrows())
        }
    else:
        labels = sorted(
            str(label)
            for label in dataframe[label_column].dropna().unique().tolist()
        )
        label_to_id = {label: index for index, label in enumerate(labels)}

    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def infer_text_column(
    dataframe: pd.DataFrame,
    text_column: str | None,
    title_column: str | None,
    post_column: str | None,
) -> str:
    """Infer the text column, using combined title/post for Murarka files."""
    if text_column:
        return text_column
    if title_column and post_column:
        return "combined_text"
    return infer_column(dataframe, TEXT_COLUMN_CANDIDATES, purpose="text")


def required_columns(metadata: DatasetMetadata) -> list[str]:
    """Return raw CSV columns required by the selected metadata."""
    columns = [metadata.label_column]
    if metadata.title_column and metadata.post_column:
        columns.extend([metadata.title_column, metadata.post_column])
    else:
        columns.append(metadata.text_column)
    return sorted(set(columns))


def infer_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    purpose: str,
) -> str:
    """Infer a required column from common names."""
    normalized = {column.lower(): column for column in dataframe.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise ValueError(
        f"Could not infer {purpose} column. Available columns: "
        f"{list(dataframe.columns)}. Set it explicitly in the config."
    )


def infer_optional_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Infer an optional column from common names."""
    normalized = {column.lower(): column for column in dataframe.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def metadata_to_dict(metadata: DatasetMetadata) -> dict[str, Any]:
    """Serialize dataset metadata."""
    return {
        "text_column": metadata.text_column,
        "label_column": metadata.label_column,
        "label_to_id": metadata.label_to_id,
        "id_to_label": {
            str(key): value
            for key, value in metadata.id_to_label.items()
        },
        "title_column": metadata.title_column,
        "post_column": metadata.post_column,
        "id_column": metadata.id_column,
        "class_id_column": metadata.class_id_column,
        "domain_strategy": (
            domain_metadata_to_dict(metadata.domain_strategy)
            if metadata.domain_strategy is not None
            else None
        ),
    }


def metadata_from_dict(data: dict[str, Any]) -> DatasetMetadata:
    """Deserialize dataset metadata."""
    return DatasetMetadata(
        text_column=data["text_column"],
        label_column=data["label_column"],
        label_to_id={
            str(key): int(value)
            for key, value in data["label_to_id"].items()
        },
        id_to_label={
            int(key): str(value)
            for key, value in data["id_to_label"].items()
        },
        title_column=data.get("title_column"),
        post_column=data.get("post_column"),
        id_column=data.get("id_column"),
        class_id_column=data.get("class_id_column"),
        domain_strategy=(
            domain_metadata_from_dict(data["domain_strategy"])
            if data.get("domain_strategy") is not None
            else None
        ),
    )
