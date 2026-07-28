"""Configuration loading and lightweight overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return config


def deep_update(
    base: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Recursively update a dictionary without mutating the input."""
    result = copy.deepcopy(base)

    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def apply_overrides(
    config: dict[str, Any],
    overrides: list[str] | None,
) -> dict[str, Any]:
    """Apply CLI overrides in dotted form, for example training.epochs=3."""
    if not overrides:
        return config

    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid override '{override}'. Expected key=value format."
            )
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        cursor = updated

        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]

        cursor[keys[-1]] = _parse_value(raw_value)

    return updated


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Write a YAML configuration snapshot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def _parse_value(value: str) -> Any:
    """Parse CLI override values into useful Python types."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value

