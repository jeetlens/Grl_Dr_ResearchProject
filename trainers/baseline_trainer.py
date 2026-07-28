"""Training loop for the DistilRoBERTa baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from datasets.mental_health_dataset import metadata_to_dict
from trainers.training_utils import (
    logits_to_predictions,
    move_batch_to_device,
    tensor_to_numpy,
)
from utils.checkpoint import save_checkpoint
from utils.io import save_json
from utils.metrics import compute_classification_metrics
from utils.plots import plot_training_curves


class BaselineTrainer:
    """Trainer for supervised DistilRoBERTa fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        metadata,
        config: dict[str, Any],
        device: torch.device,
        run_dir: str | Path,
        logger,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metadata = metadata
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.training_config = config["training"]
        self.output_dir = Path(self.training_config["output_dir"])
        self.metric_name = self.training_config.get(
            "metric_for_best_model",
            "macro_f1",
        )

        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.amp_enabled = (
            bool(self.training_config.get("mixed_precision", True))
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=self.amp_enabled)

    def train(self) -> dict[str, Any]:
        """Train for configured epochs and checkpoint by validation macro-F1."""
        best_metric = -math.inf
        best_metrics: dict[str, Any] = {}
        patience_value = self.training_config.get("early_stopping_patience")
        patience = int(patience_value) if patience_value is not None else None
        bad_epochs = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, int(self.training_config["epochs"]) + 1):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, split_name="val")
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
            history.append(epoch_record)

            score = float(val_metrics[self.metric_name])
            report_payload = {
                "history": history,
                "best_metric": max(best_metric, score),
                "metric_for_best_model": self.metric_name,
            }
            self._save_training_artifacts(report_payload, history)

            self.logger.info(
                (
                    "Epoch %s | train_loss=%.5f | val_loss=%.5f | "
                    "train_acc=%.5f | val_acc=%.5f | val_%s=%.5f"
                ),
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["accuracy"],
                self.metric_name,
                score,
            )

            if score > best_metric:
                best_metric = score
                best_metrics = {
                    **val_metrics,
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                }
                bad_epochs = 0
                self._save_best_checkpoint(epoch, best_metrics)
            else:
                bad_epochs += 1

            if patience is not None and bad_epochs >= patience:
                self.logger.info(
                    "Early stopping after %s epochs without improvement.",
                    bad_epochs,
                )
                break

        return best_metrics

    def evaluate(self, data_loader, split_name: str) -> dict[str, Any]:
        """Evaluate the model on a split."""
        self.model.eval()
        losses = []
        predictions = []
        labels = []

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(
                data_loader,
                desc=f"Evaluating {split_name}",
                leave=False,
            ), start=1):
                if self._should_stop_eval(batch_index):
                    break
                batch = move_batch_to_device(batch, self.device)
                with autocast(enabled=self.amp_enabled):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = self.loss_fn(outputs["logits"], batch["labels"])

                losses.append(float(loss.detach().cpu()))
                predictions.append(logits_to_predictions(outputs["logits"]))
                labels.append(tensor_to_numpy(batch["labels"]))

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(labels)
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            self.metadata.label_names,
        )
        metrics["loss"] = float(np.mean(losses))
        return metrics

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        losses = []
        predictions = []
        labels = []
        accumulation_steps = int(
            self.training_config.get("gradient_accumulation_steps", 1)
        )
        max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.train_loader,
            desc=f"Training epoch {epoch}",
            leave=False,
        )
        for step, batch in enumerate(progress, start=1):
            if self._should_stop_train(step):
                break
            batch = move_batch_to_device(batch, self.device)
            with autocast(enabled=self.amp_enabled):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = self.loss_fn(
                    outputs["logits"],
                    batch["labels"],
                ) / accumulation_steps

            self.scaler.scale(loss).backward()

            if step % accumulation_steps == 0 or step == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            actual_loss = float(loss.detach().cpu()) * accumulation_steps
            losses.append(actual_loss)
            predictions.append(logits_to_predictions(outputs["logits"]))
            labels.append(tensor_to_numpy(batch["labels"]))
            progress.set_postfix(loss=f"{actual_loss:.4f}")

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(labels)
        return {
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(y_pred == y_true)),
        }

    def _should_stop_train(self, step: int) -> bool:
        max_batches = self.training_config.get("max_train_batches")
        return max_batches is not None and step > int(max_batches)

    def _should_stop_eval(self, step: int) -> bool:
        max_batches = self.training_config.get("max_eval_batches")
        return max_batches is not None and step > int(max_batches)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ("bias", "LayerNorm.weight")
        grouped_parameters = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if not any(token in name for token in no_decay)
                ],
                "weight_decay": float(self.training_config["weight_decay"]),
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if any(token in name for token in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            grouped_parameters,
            lr=float(self.training_config["learning_rate"]),
        )

    def _build_scheduler(self):
        accumulation_steps = int(
            self.training_config.get("gradient_accumulation_steps", 1)
        )
        update_steps_per_epoch = math.ceil(
            len(self.train_loader) / accumulation_steps
        )
        total_steps = update_steps_per_epoch * int(
            self.training_config["epochs"]
        )
        warmup_steps = int(
            total_steps * float(self.training_config.get("warmup_ratio", 0.0))
        )
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _save_training_artifacts(
        self,
        report_payload: dict[str, Any],
        history: list[dict[str, float | int]],
    ) -> None:
        save_json(report_payload, self.run_dir / "training_history.json")
        save_json(
            report_payload,
            self.output_dir / "reports" / "baseline_training_history.json",
        )
        plot_training_curves(
            history,
            self.output_dir / "figures" / "baseline_training_curves.png",
        )

    def _save_best_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        checkpoint_paths = (
            self.output_dir / "checkpoints" / "baseline_model.pt",
            self.run_dir / "best_model.pt",
        )
        for checkpoint_path in checkpoint_paths:
            save_checkpoint(
                checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
                metadata=metadata_to_dict(self.metadata),
            )

        self.logger.info(
            "Saved best baseline checkpoint to %s",
            checkpoint_paths[0],
        )
