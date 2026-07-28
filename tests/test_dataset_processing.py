"""Unit tests for Murarka dataset preprocessing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mental_health_dataset import MentalHealthDataset  # noqa: E402


class SimpleTokenizer:
    """Small tokenizer stub for dataset unit tests."""

    def __call__(
        self,
        text,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        del padding, truncation, return_tensors
        token_ids = [min(len(token), 99) for token in text.split()]
        token_ids = token_ids[:max_length]
        attention = [1] * len(token_ids)
        pad_length = max_length - len(token_ids)
        token_ids += [0] * pad_length
        attention += [0] * pad_length
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
        }


class DatasetProcessingTest(unittest.TestCase):
    """Validate combined text, labels, and proxy-domain metadata."""

    def test_combined_text_and_text_length_domains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "both_train.csv"
            pd.DataFrame(
                [
                    {
                        "ID": 1,
                        "title": "Short title",
                        "post": "short post",
                        "class_name": "anxiety",
                        "class_id": 0,
                    },
                    {
                        "ID": 2,
                        "title": "",
                        "post": "this is a considerably longer post",
                        "class_name": "depression",
                        "class_id": 1,
                    },
                    {
                        "ID": 3,
                        "title": None,
                        "post": "   ",
                        "class_name": "anxiety",
                        "class_id": 0,
                    },
                ]
            ).to_csv(csv_path, index=False)

            metadata = MentalHealthDataset.create_metadata_from_train(
                csv_path=csv_path,
                label_column="class_name",
                domain_strategy_name="text_length_bins",
            )
            dataset = MentalHealthDataset(
                csv_path=csv_path,
                tokenizer=SimpleTokenizer(),
                max_length=8,
                metadata=metadata,
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.texts[0], "Short title short post")
            self.assertEqual(metadata.label_to_id["anxiety"], 0)
            self.assertEqual(metadata.label_to_id["depression"], 1)
            self.assertIsNotNone(metadata.domain_strategy)
            self.assertEqual(
                metadata.domain_strategy.strategy,
                "text_length_bins",
            )
            self.assertEqual(dataset.domains, [0, 1])


if __name__ == "__main__":
    unittest.main()

