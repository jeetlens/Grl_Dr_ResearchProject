"""DataLoader construction for experiments."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.mental_health_dataset import (
    DatasetMetadata,
    MentalHealthDataset,
)


def build_tokenizer(model_name: str):
    """Load a HuggingFace tokenizer."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_datasets(
    config: dict,
    tokenizer,
    smoke_test: bool = False,
) -> tuple[
    MentalHealthDataset,
    MentalHealthDataset,
    MentalHealthDataset,
    DatasetMetadata,
]:
    """Build train, validation, and test datasets from config."""
    data_config = config["data"]
    domain_config = config.get("domain", {})

    metadata = MentalHealthDataset.create_metadata_from_train(
        data_config["train_file"],
        text_column=data_config.get("text_column"),
        label_column=data_config.get("label_column"),
        domain_strategy_name=domain_config.get("strategy", "text_length_bins"),
        domain_column=data_config.get("domain_column"),
    )

    common_kwargs = {
        "tokenizer": tokenizer,
        "max_length": int(data_config["max_length"]),
        "metadata": metadata,
        "lowercase": bool(data_config.get("lowercase", False)),
    }

    train_dataset = MentalHealthDataset(
        Path(data_config["train_file"]),
        **common_kwargs,
    )
    val_dataset = MentalHealthDataset(
        Path(data_config["val_file"]),
        **common_kwargs,
    )
    test_dataset = MentalHealthDataset(
        Path(data_config["test_file"]),
        **common_kwargs,
    )

    if smoke_test:
        train_dataset = subset_dataset(train_dataset, 100)
        val_dataset = subset_dataset(val_dataset, 50)

    return train_dataset, val_dataset, test_dataset, metadata


def build_dataloader(
    dataset: MentalHealthDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def subset_dataset(dataset: MentalHealthDataset, max_rows: int):
    """Return a deterministic first-N subset while preserving dataset fields."""
    from torch.utils.data import Subset

    return Subset(dataset, range(min(max_rows, len(dataset))))
