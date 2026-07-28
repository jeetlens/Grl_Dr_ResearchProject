"""Train a supervised DistilRoBERTa baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets.data_module import (
    build_dataloader,
    build_datasets,
    build_tokenizer,
)
from models.baseline import DistilRobertaClassifier
from trainers.baseline_trainer import BaselineTrainer
from utils.config import apply_overrides, load_config, save_config
from utils.io import create_run_dir, ensure_output_dirs, save_json
from utils.logging import get_logger
from utils.paths import project_root, resolve_project_paths
from utils.seed import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DistilRoBERTa for mental health classification.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root() / "configs" / "default.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values, e.g. training.epochs=3.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run 100 train rows, 50 validation rows, one train batch, "
            "one validation batch, and save a smoke checkpoint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run baseline training."""
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    config = resolve_project_paths(config)
    config["model"]["architecture"] = "baseline"
    if args.smoke_test:
        config["training"]["output_dir"] = str(
            project_root() / "outputs" / "smoke_baseline"
        )
        config["training"]["experiment_name"] = "baseline_smoke_test"
        config["training"]["epochs"] = 1
        config["training"]["max_train_batches"] = 1
        config["training"]["max_eval_batches"] = 1

    set_seed(int(config["project"]["seed"]))
    ensure_output_dirs(config["training"]["output_dir"])
    run_dir = create_run_dir(
        config["training"]["output_dir"],
        config["training"]["experiment_name"],
    )
    logger = get_logger("train_baseline", run_dir / "train.log")
    save_config(config, run_dir / "config.yaml")

    logger.info("Run directory: %s", run_dir)
    logger.info("Loading tokenizer and datasets.")
    tokenizer = build_tokenizer(config["model"]["backbone"])
    train_dataset, val_dataset, _, metadata = build_datasets(
        config,
        tokenizer,
        smoke_test=args.smoke_test,
    )
    tokenizer.save_pretrained(run_dir / "tokenizer")
    tokenizer.save_pretrained(
        Path(config["training"]["output_dir"]) / "checkpoints" / "tokenizer"
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    device = resolve_device(config["project"].get("device", "auto"))
    logger.info("Using device: %s", device)
    model = DistilRobertaClassifier(
        model_name=config["model"]["backbone"],
        num_labels=len(metadata.label_to_id),
        dropout=float(config["model"]["dropout"]),
    )
    model.model.config.id2label = metadata.id_to_label
    model.model.config.label2id = metadata.label_to_id

    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        metadata=metadata,
        config=config,
        device=device,
        run_dir=run_dir,
        logger=logger,
    )
    best_metrics = trainer.train()
    save_json(best_metrics, Path(run_dir) / "best_val_metrics.json")
    logger.info("Best validation metrics: %s", best_metrics)


if __name__ == "__main__":
    main()
