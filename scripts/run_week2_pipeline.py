"""Week 2 pipeline (assumes Week 1 detections already exist).

Usage:
    python scripts/run_week2_pipeline.py --video-id bike_approaching_left --config config.yaml

Steps:
    1. Load existing frames + detection CSV.
    2. Assign tracks.
    3. Compute frame-differencing motion overlap.
    4. Compute Farneback optical flow features.
    5. Re-save enhanced tracks CSV.
    6. Compute per-track motion features.
    7. Draw track annotations + tracking video.
    8. Generate plots.
    9. Save Week 1-2 summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotate import draw_tracks_on_frames, make_annotated_video  # noqa: E402
from src.config import load_config  # noqa: E402
from src.frame_diff import compute_object_motion_overlap  # noqa: E402
from src.motion_features import compute_track_motion_features  # noqa: E402
from src.optical_flow import compute_object_flow_features  # noqa: E402
from src.summarize import summarize_week1_week2  # noqa: E402
from src.tracking import assign_tracks  # noqa: E402
from src.utils import ensure_dir  # noqa: E402
from src.visualize import (  # noqa: E402
    plot_bbox_area_over_time,
    plot_detections_per_frame,
    plot_motion_features,
    plot_track_trajectories,
)

FRAMES_DIR = Path("data/frames")
DETECTIONS_DIR = Path("outputs/detections")
TRACKS_DIR = Path("outputs/tracks")
MOTION_DIR = Path("outputs/motion")
ANNOTATED_FRAMES_DIR = Path("outputs/annotated_frames")
ANNOTATED_VIDEOS_DIR = Path("outputs/annotated_videos")
PLOTS_DIR = Path("outputs/plots")
SUMMARIES_DIR = Path("outputs/summaries")


def _load_existing_frames(video_id: str) -> List[Path]:
    frame_dir = FRAMES_DIR / video_id
    if not frame_dir.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {frame_dir}. "
            f"Run the Week 1 pipeline first."
        )
    return sorted(frame_dir.glob("frame_*.jpg"))


def _load_existing_detections(video_id: str) -> pd.DataFrame:
    csv_path = DETECTIONS_DIR / f"{video_id}_detections.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Detections CSV not found: {csv_path}. "
            f"Run the Week 1 pipeline first."
        )
    df = pd.read_csv(csv_path)
    return df


def run_week2_for_video_id(video_id: str, cfg: dict) -> dict:
    print(f"\n=== Week 2 pipeline: {video_id} ===")
    frame_paths = _load_existing_frames(video_id)
    if not frame_paths:
        raise RuntimeError(f"No frames found for {video_id}.")
    detections_df = _load_existing_detections(video_id)

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    image_h, image_w = sample.shape[:2]

    tracks_df = assign_tracks(
        detections_df=detections_df,
        image_width=image_w,
        image_height=image_h,
        config=cfg,
        output_dir=TRACKS_DIR,
        video_id=video_id,
    )

    tracks_df = compute_object_motion_overlap(
        detections_df=tracks_df,
        frame_paths=[str(p) for p in frame_paths],
        config=cfg,
        output_dir=MOTION_DIR,
        video_id=video_id,
    )

    tracks_df = compute_object_flow_features(
        tracks_df=tracks_df,
        frame_paths=[str(p) for p in frame_paths],
        config=cfg,
        output_dir=MOTION_DIR,
        video_id=video_id,
        save_visualizations=True,
    )

    enhanced_tracks_csv = ensure_dir(TRACKS_DIR) / f"{video_id}_tracks.csv"
    tracks_df.to_csv(enhanced_tracks_csv, index=False)
    print(f"[week2] enhanced tracks -> {enhanced_tracks_csv}")

    features_df = compute_track_motion_features(
        tracks_df=tracks_df,
        config=cfg,
        image_width=image_w,
        image_height=image_h,
        output_dir=TRACKS_DIR,
        video_id=video_id,
    )

    track_frames_dir = ANNOTATED_FRAMES_DIR / f"{video_id}_tracks"
    track_frame_paths = draw_tracks_on_frames(
        frame_paths=[str(p) for p in frame_paths],
        tracks_df=tracks_df,
        detections_df=detections_df,
        output_dir=track_frames_dir,
    )
    tracking_video = ""
    if cfg["outputs"].get("save_annotated_video", True) and track_frame_paths:
        ensure_dir(ANNOTATED_VIDEOS_DIR)
        tracking_video = make_annotated_video(
            track_frame_paths,
            ANNOTATED_VIDEOS_DIR / f"{video_id}_tracks.mp4",
            fps=int(cfg["video"].get("annotated_video_fps", 2)),
        )

    plot_paths = {}
    if cfg["outputs"].get("save_plots", True):
        plot_paths["detections_per_frame"] = plot_detections_per_frame(
            detections_df, PLOTS_DIR / f"{video_id}_detections_per_frame.png"
        )
        plot_paths["trajectories"] = plot_track_trajectories(
            tracks_df, PLOTS_DIR / f"{video_id}_trajectories.png"
        )
        plot_paths["bbox_area"] = plot_bbox_area_over_time(
            tracks_df, PLOTS_DIR / f"{video_id}_bbox_area.png"
        )
        plot_paths["motion_features"] = plot_motion_features(
            features_df, PLOTS_DIR / f"{video_id}_motion_features.png"
        )

    summary_text = summarize_week1_week2(
        video_id=video_id,
        detections_df=detections_df,
        tracks_df=tracks_df,
        track_features_df=features_df,
        output_dir=SUMMARIES_DIR,
    )
    print("\n" + summary_text)

    return {
        "video_id": video_id,
        "tracks_csv": str(enhanced_tracks_csv),
        "track_features_csv": str(TRACKS_DIR / f"{video_id}_track_features.csv"),
        "tracking_video": tracking_video,
        "motion_dir": str(MOTION_DIR / video_id),
        "annotated_tracks_dir": str(track_frames_dir),
        "summary_txt": str(SUMMARIES_DIR / f"{video_id}_week1_week2_summary.txt"),
        "summary_json": str(SUMMARIES_DIR / f"{video_id}_week1_week2_summary.json"),
        **plot_paths,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run SpatialLens Assist Week 2 pipeline.")
    p.add_argument("--video-id", required=True,
                   help="Video id (matches frames dir + detections CSV prefix).")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)
    outputs = run_week2_for_video_id(args.video_id, cfg)

    print("\n=== Week 2 outputs ===")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
