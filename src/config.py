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

# Week 3 keys are optional. If a Weeks 1-2 era config is loaded without them,
# we backfill conservative defaults so the Week 3 pipeline can still run.
WEEK3_DEFAULTS: Dict[str, Any] = {
    "hazard": {
        "approach_score_threshold": 0.55,
        "crossing_threshold_frac_width": 0.08,
        "static_threshold_frac_diagonal": 0.04,
        "bbox_growth_threshold": 0.15,
        "bbox_shrink_threshold": -0.15,
        "flow_threshold": 1.0,
        "frame_diff_overlap_threshold": 0.08,
        "center_motion_threshold_frac": 0.05,
        "min_track_frames": 3,
        "near_miss_approach_margin": 0.25,
        "near_miss_crossing_fraction": 0.5,
        "use_depth_if_available": False,
    },
    "evaluation": {
        "label_file": "data/labels/hazard_labels.csv",
        "output_dir": "outputs/evaluation",
    },
    "alerts": {
        "save_alerts_json": True,
        "save_alerts_txt": True,
        "include_confidence": True,
    },
    "final_outputs": {
        "export_slide_assets": True,
        "export_demo_video": True,
    },
}


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

    for section, defaults in WEEK3_DEFAULTS.items():
        section_cfg = cfg.get(section)
        if not isinstance(section_cfg, dict):
            cfg[section] = dict(defaults)
            continue
        for k, v in defaults.items():
            section_cfg.setdefault(k, v)

    return cfg
