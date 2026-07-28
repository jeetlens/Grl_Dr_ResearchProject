"""Shared training helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move a mini-batch to device."""
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def logits_to_predictions(logits: torch.Tensor) -> np.ndarray:
    """Convert model logits to class predictions."""
    return torch.argmax(logits, dim=-1).detach().cpu().numpy()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Move a tensor to CPU NumPy."""
    return tensor.detach().cpu().numpy()

