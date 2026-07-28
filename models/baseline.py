"""DistilRoBERTa sequence-classification baseline."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class DistilRobertaClassifier(nn.Module):
    """Thin production wrapper around AutoModelForSequenceClassification."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        del dropout
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=self.config,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_config(
                self.config
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if outputs.loss is not None else None,
        }
