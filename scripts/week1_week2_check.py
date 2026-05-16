"""Quick CLI sanity-check on the Week 1-2 outputs for a single ``video_id``.

This mirrors what ``notebooks/01_week1_week2_check.ipynb`` does — handy if
you don't have Jupyter handy.

Usage:
    python scripts/week1_week2_check.py --video-id bike_approaching_left
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DETECTIONS_DIR = Path("outputs/detections")
TRACKS_DIR = Path("outputs/tracks")
SUMMARIES_DIR = Path("outputs/summaries")
ANNOTATED_FRAMES_DIR = Path("outputs/annotated_frames")
PLOTS_DIR = Path("outputs/plots")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video-id", required=True)
    args = p.parse_args()
    vid = args.video_id

    det_csv = DETECTIONS_DIR / f"{vid}_detections.csv"
    feat_csv = TRACKS_DIR / f"{vid}_track_features.csv"
    summary_txt = SUMMARIES_DIR / f"{vid}_week1_week2_summary.txt"

    if not det_csv.exists() or not feat_csv.exists():
        print(f"Missing outputs for {vid!r}. Run the pipeline first.")
        sys.exit(1)

    det = pd.read_csv(det_csv)
    feats = pd.read_csv(feat_csv)

    print(f"\n=== Detections by class ({vid}) ===")
    if det.empty:
        print("(no detections)")
    else:
        print(det["class_name"].value_counts().to_string())

    print(f"\n=== Per-track motion features ({vid}) ===")
    if feats.empty:
        print("(no tracks)")
    else:
        cols = [
            "track_id", "class_name", "num_frames",
            "dx_total", "dy_total", "bbox_growth_ratio",
            "avg_flow_mag", "avg_frame_diff_overlap",
            "preliminary_motion_label",
            "preliminary_direction_label",
            "preliminary_scale_label",
        ]
        print(feats[cols].to_string(index=False))

    print("\n=== Annotated tracking frames available at ===")
    print(f"  {ANNOTATED_FRAMES_DIR / (vid + '_tracks')}")
    print("\n=== Plots ===")
    for name in (
        "detections_per_frame", "trajectories", "bbox_area", "motion_features",
    ):
        path = PLOTS_DIR / f"{vid}_{name}.png"
        flag = "OK " if path.exists() else "MISS"
        print(f"  [{flag}] {path}")

    if summary_txt.exists():
        print(f"\n=== Week 1-2 summary ({summary_txt}) ===\n")
        print(summary_txt.read_text())


if __name__ == "__main__":
    main()
