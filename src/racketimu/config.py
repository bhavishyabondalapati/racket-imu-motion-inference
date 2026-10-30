"""Load the YAML config, with optional overrides layered on top of the defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

REQUIRED_SECTIONS = ("paths", "seed", "imu", "sync", "segment", "reference", "train")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where values in `override` replace those in `base`, recursively.

    Nested dicts are merged key by key; any other value (lists included) is replaced whole.
    Neither input is modified.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the default config, then layer an optional YAML file and a dict of overrides.

    Example:
        >>> cfg = load_config(overrides={"segment": {"window_pre_s": 0.8}})
        >>> cfg["segment"]["window_pre_s"]
        0.8
    """
    config = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        config = deep_merge(config, _read_yaml(Path(path)))
    if overrides:
        config = deep_merge(config, overrides)

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise KeyError(f"Config is missing required sections: {missing}")
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return data
