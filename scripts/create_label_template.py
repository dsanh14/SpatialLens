"""Create a manual-labeling CSV template for hazard evaluation (Week 3).

Usage:

    # Per-video template
    python scripts/create_label_template.py \
        --video-id bike_approaching_left \
        --config config.yaml

    # One combined template for every video that has hazard or track-feature CSVs
    python scripts/create_label_template.py --all --config config.yaml

The output CSVs contain ``predicted_label`` and ``evidence`` already
filled in so you can sanity-check your manual ``true_label`` decisions.
Fill in the empty ``true_label`` column with one of:

    approaching | crossing_left_to_right | crossing_right_to_left
    moving_away | static | uncertain

Then re-run the Week 3 pipeline (or the final pipeline) and metrics will
appear in ``outputs/evaluation/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

HAZARDS_DIR = Path("outputs/hazards")
TRACKS_DIR = Path("outputs/tracks")
LABELS_DIR = Path("data/labels")

TEMPLATE_COLUMNS = [
    "video_id",
    "track_id",
    "class_name",
    "predicted_label",
    "true_label",
    "notes",
    "first_frame",
    "last_frame",
    "evidence",
]


def _load_template_rows_for_video(video_id: str) -> pd.DataFrame:
    """Prefer the hazards CSV (has predicted_label + evidence). Fall back to
    track features when hazards haven't been generated yet."""
    hazards_csv = HAZARDS_DIR / f"{video_id}_hazards.csv"
    features_csv = TRACKS_DIR / f"{video_id}_track_features.csv"

    if hazards_csv.exists():
        df = pd.read_csv(hazards_csv)
        out = pd.DataFrame({
            "video_id": df["video_id"],
            "track_id": df["track_id"],
            "class_name": df["class_name"],
            "predicted_label": df["hazard_label"],
            "true_label": "",
            "notes": "",
            "first_frame": df.get("first_frame", -1),
            "last_frame": df.get("last_frame", -1),
            "evidence": df.get("evidence", ""),
        })
        return out

    if features_csv.exists():
        df = pd.read_csv(features_csv)
        out = pd.DataFrame({
            "video_id": df["video_id"],
            "track_id": df["track_id"],
            "class_name": df["class_name"],
            "predicted_label": "",  # no hazard classifier ran yet
            "true_label": "",
            "notes": "",
            "first_frame": df.get("first_frame", -1),
            "last_frame": df.get("last_frame", -1),
            "evidence": "",
        })
        return out

    return pd.DataFrame(columns=TEMPLATE_COLUMNS)


def _discover_video_ids() -> List[str]:
    ids: set = set()
    for p in HAZARDS_DIR.glob("*_hazards.csv"):
        ids.add(p.stem.replace("_hazards", ""))
    for p in TRACKS_DIR.glob("*_track_features.csv"):
        ids.add(p.stem.replace("_track_features", ""))
    return sorted(ids)


def write_template_for_video(video_id: str) -> Path:
    rows = _load_template_rows_for_video(video_id)
    out_dir = ensure_dir(LABELS_DIR)
    out_path = out_dir / f"{video_id}_hazard_label_template.csv"
    rows.to_csv(out_path, index=False)
    print(f"[label_template] {video_id}: {len(rows)} rows -> {out_path}")
    return out_path


def write_combined_template(video_ids: List[str]) -> Path:
    pieces: List[pd.DataFrame] = []
    for vid in video_ids:
        pieces.append(_load_template_rows_for_video(vid))
    combined = (pd.concat(pieces, ignore_index=True)
                if pieces else pd.DataFrame(columns=TEMPLATE_COLUMNS))
    out_dir = ensure_dir(LABELS_DIR)
    out_path = out_dir / "hazard_labels_template.csv"
    combined.to_csv(out_path, index=False)
    print(f"[label_template] combined: {len(combined)} rows -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Create hazard label templates.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="Single video_id to template.")
    g.add_argument("--all", action="store_true",
                   help="Make per-video templates AND one combined template "
                        "across all videos that have outputs.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    load_config(args.config)  # validate + apply Week 3 defaults

    if args.video_id:
        write_template_for_video(args.video_id)
        return

    video_ids = _discover_video_ids()
    if not video_ids:
        print("No hazards or track-features CSVs found. "
              "Run the Week 1-2 (or Week 3) pipeline first.")
        return
    for vid in video_ids:
        write_template_for_video(vid)
    write_combined_template(video_ids)


if __name__ == "__main__":
    main()
