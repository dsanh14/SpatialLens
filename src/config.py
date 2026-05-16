"""YAML config loading and validation for the SpatialLens Assist pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REQUIRED_TOP_LEVEL_KEYS = (
    "project",
    "video",
    "detection",
    "tracking",
    "motion",
    "outputs",
    "mock",
)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate a SpatialLens Assist YAML config.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed config dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required top-level keys are missing or the file is empty/malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            f"Expected a YAML file (see config.yaml in the repo root)."
        )

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config at '{path}' did not parse to a mapping. "
            f"Got type: {type(cfg).__name__}."
        )

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"Config at '{path}' is missing required top-level keys: {missing}. "
            f"Expected keys: {list(REQUIRED_TOP_LEVEL_KEYS)}."
        )

    return cfg
