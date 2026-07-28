"""Path normalization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def resolve_project_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve config paths relative to the project root."""
    root = project_root()
    resolved = dict(config)
    resolved["data"] = dict(config["data"])
    resolved["training"] = dict(config["training"])
    resolved["evaluation"] = dict(config.get("evaluation", {}))

    for key in ("train_file", "val_file", "test_file"):
        path = Path(resolved["data"][key])
        if not path.is_absolute():
            resolved["data"][key] = str(root / path)

    output_dir = Path(resolved["training"]["output_dir"])
    if not output_dir.is_absolute():
        resolved["training"]["output_dir"] = str(root / output_dir)

    checkpoint_path = resolved["evaluation"].get("checkpoint_path")
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            resolved["evaluation"]["checkpoint_path"] = str(root / path)

    return resolved

