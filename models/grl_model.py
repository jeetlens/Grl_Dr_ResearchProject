"""DistilRoBERTa with a Gradient Reversal Layer.

The model has one shared text encoder and two prediction branches:

1. Mental health classification branch:
   DistilRoBERTa -> shared feature -> classification head.

2. Domain-adversarial branch:
   DistilRoBERTa -> shared feature -> gradient reversal -> domain classifier.

For the Murarka ``both_*.csv`` files, this is used as proxy-domain adversarial
training when ``domain.strategy=text_length_bins``. It is not cross-dataset
domain adaptation unless a genuine domain/source column is supplied.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.autograd import Function
from transformers import AutoConfig, AutoModel


DEFAULT_GRL_LAMBDA = 0.1


class GradientReversalFunction(Function):
    """Autograd function for gradient reversal.

    Forward pass:
        Returns the input unchanged.

    Backward pass:
        Multiplies incoming gradients by ``-lambda_value``. This makes the
        encoder maximize domain-classification loss while the domain classifier
        itself still minimizes it.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        """Return inputs unchanged and store lambda for backward."""
        ctx.lambda_value = lambda_value
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Reverse and scale gradients flowing into the shared encoder."""
        return -ctx.lambda_value * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper around ``GradientReversalFunction``.

    This layer behaves like an identity layer during inference/forward
    computation but reverses gradients during backpropagation.
    """

    def __init__(self, lambda_value: float = DEFAULT_GRL_LAMBDA) -> None:
        super().__init__()
        self.lambda_value = lambda_value

    def forward(
        self,
        inputs: torch.Tensor,
        lambda_value: float | None = None,
    ) -> torch.Tensor:
        """Apply gradient reversal with a fixed or call-time lambda."""
        active_lambda = self.lambda_value if lambda_value is None else lambda_value
        return GradientReversalFunction.apply(inputs, active_lambda)


class ClassificationHead(nn.Module):
    """Mental health classification head.

    Receives the shared DistilRoBERTa ``[CLS]`` feature and predicts the mental
    health class.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return mental health class logits."""
        return self.layers(features)


class DomainClassifier(nn.Module):
    """Domain classifier head.

    Receives gradient-reversed shared features and predicts the domain/source.
    Its gradients are reversed before reaching DistilRoBERTa.
    """

    def __init__(
        self,
        hidden_size: int,
        num_domains: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return domain class logits."""
        return self.layers(features)


class DistilRobertaGRLClassifier(nn.Module):
    """DistilRoBERTa encoder with classification and adversarial domain heads."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_domains: int,
        dropout: float = 0.2,
        grl_lambda: float = DEFAULT_GRL_LAMBDA,
        pretrained: bool = True,
        domain_branch_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.domain_branch_enabled = domain_branch_enabled
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                config=self.config,
            )
        else:
            self.encoder = AutoModel.from_config(self.config)
        hidden_size = self.config.hidden_size

        self.classification_head = ClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.gradient_reversal = (
            GradientReversalLayer(lambda_value=grl_lambda)
            if domain_branch_enabled
            else None
        )
        self.domain_classifier = (
            DomainClassifier(
                hidden_size=hidden_size,
                num_domains=num_domains,
                dropout=dropout,
            )
            if domain_branch_enabled
            else None
        )
        self.label_loss_fn = nn.CrossEntropyLoss()
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        lambda_value: float | None = None,
        domain_loss_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Run the full domain-adversarial forward pass.

        Returns:
            ``features``: shared DistilRoBERTa representation.
            ``logits``: mental health prediction logits.
            ``domain_logits``: domain/source prediction logits.
            ``label_loss``: optional mental health cross-entropy loss.
            ``domain_loss``: optional domain cross-entropy loss.
            ``loss``: optional combined loss.
        """
        features = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = self.classification_head(features)

        outputs = {
            "features": features,
            "logits": logits,
        }

        if labels is not None:
            outputs["label_loss"] = self.label_loss_fn(logits, labels)

        if self.domain_branch_enabled:
            if self.gradient_reversal is None or self.domain_classifier is None:
                raise RuntimeError("Domain branch is not initialized.")
            reversed_features = self.gradient_reversal(features, lambda_value)
            domain_logits = self.domain_classifier(reversed_features)
            outputs["domain_logits"] = domain_logits

        if domain_labels is not None and "domain_logits" in outputs:
            outputs["domain_loss"] = self.domain_loss_fn(
                outputs["domain_logits"],
                domain_labels,
            )

        if "label_loss" in outputs and "domain_loss" in outputs:
            outputs["loss"] = (
                outputs["label_loss"]
                + domain_loss_weight * outputs["domain_loss"]
            )
        elif "label_loss" in outputs:
            outputs["loss"] = outputs["label_loss"]

        return outputs

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenized text and return the shared ``[CLS]`` feature."""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return encoder_outputs.last_hidden_state[:, 0]


def compute_grl_lambda(
    progress: float,
    lambda_max: float = DEFAULT_GRL_LAMBDA,
    schedule: str = "constant",
) -> float:
    """Compute GRL lambda for optional training schedules.

    The requested default is a fixed ``lambda=0.1``. Linear and sigmoid schedules
    are retained for controlled ablation experiments.
    """
    clipped = min(max(progress, 0.0), 1.0)

    if schedule == "constant":
        return lambda_max
    if schedule == "linear":
        return lambda_max * clipped
    if schedule == "sigmoid":
        return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * clipped)) - 1.0)

    raise ValueError(
        f"Unknown GRL schedule '{schedule}'. "
        "Choose one of: constant, linear, sigmoid."
    )
