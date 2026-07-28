"""Training loop for domain-adversarial DistilRoBERTa."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from datasets.mental_health_dataset import metadata_to_dict
from models.grl_model import compute_grl_lambda
from trainers.training_utils import (
    logits_to_predictions,
    move_batch_to_device,
    tensor_to_numpy,
)
from utils.checkpoint import save_checkpoint
from utils.io import save_json
from utils.metrics import compute_classification_metrics
from utils.plots import plot_accuracy_curves, plot_loss_curves


class GRLTrainer:
    """Trainer for DistilRoBERTa with a Gradient Reversal Layer."""

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
        self.grl_config = config.get("grl", {})
        self.output_dir = Path(self.training_config["output_dir"])
        self.metric_name = self.training_config.get(
            "metric_for_best_model",
            "macro_f1",
        )
        self.domain_loss_weight = float(
            self.grl_config.get("domain_loss_weight", 0.1)
        )
        self.domain_enabled = metadata.domain_strategy is not None

        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.amp_enabled = (
            bool(self.training_config.get("mixed_precision", True))
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.global_step = 0
        self.total_update_steps = self._total_update_steps()

    def train(self) -> dict[str, Any]:
        """Train the GRL model and save the best checkpoint by validation F1."""
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
                "train_classification_loss": train_metrics[
                    "classification_loss"
                ],
                "val_classification_loss": val_metrics["classification_loss"],
                "train_domain_loss": train_metrics["domain_loss"],
                "val_domain_loss": val_metrics["domain_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_domain_accuracy": train_metrics["domain_accuracy"],
                "val_domain_accuracy": val_metrics["domain_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "grl_lambda": train_metrics["grl_lambda"],
                "domain_loss_weight": self.domain_loss_weight,
            }
            history.append(epoch_record)

            score = float(val_metrics[self.metric_name])
            report_payload = {
                "history": history,
                "best_metric": max(best_metric, score),
                "metric_for_best_model": self.metric_name,
                "loss_formula": (
                    "total_loss = classification_loss "
                    f"+ {self.domain_loss_weight} * domain_loss"
                    if self.domain_enabled
                    else "total_loss = classification_loss"
                ),
                "experiment_label": self.config.get("domain", {}).get(
                    "experiment_label",
                    "proxy-domain adversarial training",
                ),
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
                    "domain_loss_weight": self.domain_loss_weight,
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
        """Evaluate mental health and domain predictions."""
        self.model.eval()
        total_losses = []
        classification_losses = []
        domain_losses = []
        predictions = []
        labels = []
        domain_predictions = []
        domain_labels = []

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
                        labels=batch["labels"],
                        domain_labels=batch.get("domain_labels"),
                        lambda_value=self._configured_lambda(),
                        domain_loss_weight=self.domain_loss_weight,
                    )

                total_losses.append(float(outputs["loss"].detach().cpu()))
                classification_losses.append(
                    float(outputs["label_loss"].detach().cpu())
                )
                if "domain_loss" in outputs:
                    domain_losses.append(float(outputs["domain_loss"].detach().cpu()))
                predictions.append(logits_to_predictions(outputs["logits"]))
                labels.append(tensor_to_numpy(batch["labels"]))
                if "domain_logits" in outputs and "domain_labels" in batch:
                    domain_predictions.append(
                        logits_to_predictions(outputs["domain_logits"])
                    )
                    domain_labels.append(tensor_to_numpy(batch["domain_labels"]))

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(labels)
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            self.metadata.label_names,
        )
        metrics["loss"] = float(np.mean(total_losses))
        metrics["classification_loss"] = float(np.mean(classification_losses))
        metrics["domain_loss"] = (
            float(np.mean(domain_losses))
            if domain_losses
            else 0.0
        )
        if domain_predictions and domain_labels:
            domain_pred = np.concatenate(domain_predictions)
            domain_true = np.concatenate(domain_labels)
            metrics["domain_accuracy"] = float(np.mean(domain_pred == domain_true))
        else:
            metrics["domain_accuracy"] = 0.0
        return metrics

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_losses = []
        classification_losses = []
        domain_losses = []
        lambdas = []
        predictions = []
        labels = []
        domain_predictions = []
        domain_labels = []
        accumulation_steps = int(
            self.training_config.get("gradient_accumulation_steps", 1)
        )
        max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))
        self.optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            self.train_loader,
            desc=f"GRL training epoch {epoch}",
            leave=False,
        )
        for step, batch in enumerate(progress, start=1):
            if self._should_stop_train(step):
                break
            batch = move_batch_to_device(batch, self.device)
            lambda_value = self._current_lambda()

            with autocast(enabled=self.amp_enabled):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    domain_labels=batch.get("domain_labels"),
                    lambda_value=lambda_value,
                    domain_loss_weight=self.domain_loss_weight,
                )
                loss = outputs["loss"] / accumulation_steps

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
                self.global_step += 1

            actual_loss = float(loss.detach().cpu()) * accumulation_steps
            total_losses.append(actual_loss)
            classification_losses.append(
                float(outputs["label_loss"].detach().cpu())
            )
            if "domain_loss" in outputs:
                domain_losses.append(float(outputs["domain_loss"].detach().cpu()))
            lambdas.append(lambda_value)
            predictions.append(logits_to_predictions(outputs["logits"]))
            labels.append(tensor_to_numpy(batch["labels"]))
            if "domain_logits" in outputs and "domain_labels" in batch:
                domain_predictions.append(
                    logits_to_predictions(outputs["domain_logits"])
                )
                domain_labels.append(tensor_to_numpy(batch["domain_labels"]))
            progress.set_postfix(
                loss=f"{actual_loss:.4f}",
                grl=f"{lambda_value:.3f}",
            )

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(labels)
        if domain_predictions and domain_labels:
            domain_pred = np.concatenate(domain_predictions)
            domain_true = np.concatenate(domain_labels)
            domain_accuracy = float(np.mean(domain_pred == domain_true))
        else:
            domain_accuracy = 0.0
        return {
            "loss": float(np.mean(total_losses)),
            "classification_loss": float(np.mean(classification_losses)),
            "domain_loss": float(np.mean(domain_losses)) if domain_losses else 0.0,
            "accuracy": float(np.mean(y_pred == y_true)),
            "domain_accuracy": domain_accuracy,
            "grl_lambda": float(np.mean(lambdas)),
        }

    def _should_stop_train(self, step: int) -> bool:
        max_batches = self.training_config.get("max_train_batches")
        return max_batches is not None and step > int(max_batches)

    def _should_stop_eval(self, step: int) -> bool:
        max_batches = self.training_config.get("max_eval_batches")
        return max_batches is not None and step > int(max_batches)

    def _current_lambda(self) -> float:
        progress = self.global_step / max(1, self.total_update_steps)
        return compute_grl_lambda(
            progress=progress,
            lambda_max=self._configured_lambda(),
            schedule=str(self.grl_config.get("lambda_schedule", "constant")),
        )

    def _configured_lambda(self) -> float:
        return float(self.grl_config.get("lambda_max", 0.1))

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
        total_steps = self._total_update_steps()
        warmup_steps = int(
            total_steps * float(self.training_config.get("warmup_ratio", 0.0))
        )
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _total_update_steps(self) -> int:
        accumulation_steps = int(
            self.training_config.get("gradient_accumulation_steps", 1)
        )
        update_steps_per_epoch = math.ceil(
            len(self.train_loader) / accumulation_steps
        )
        return update_steps_per_epoch * int(self.training_config["epochs"])

    def _save_training_artifacts(
        self,
        report_payload: dict[str, Any],
        history: list[dict[str, float | int]],
    ) -> None:
        save_json(report_payload, self.run_dir / "training_history.json")
        save_json(
            report_payload,
            self.output_dir / "reports" / "grl_training_history.json",
        )
        plot_loss_curves(
            history,
            self.output_dir / "figures" / "grl_loss_curves.png",
        )
        plot_accuracy_curves(
            history,
            self.output_dir / "figures" / "grl_accuracy_curves.png",
        )

    def _save_best_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        checkpoint_paths = (
            self.output_dir / "checkpoints" / "grl_model.pt",
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
            "Saved best GRL checkpoint to %s",
            checkpoint_paths[0],
        )
