"""File-system helpers for experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(output_dir: str | Path, experiment_name: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / "checkpoints" / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ensure_output_dirs(output_dir: str | Path) -> None:
    """Ensure standard output subdirectories exist."""
    root = Path(output_dir)
    for child in ("checkpoints", "figures", "reports"):
        (root / child).mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as indented JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

