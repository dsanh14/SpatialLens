"""End-to-end Weeks 1 + 2 pipeline for SpatialLens Assist.

This is the main script for the project.

Usage examples
--------------

Process a single video::

    python scripts/run_week1_week2_pipeline.py \
        --video data/raw_videos/example.mp4 \
        --config config.yaml

Process every video in ``data/raw_videos/``::

    python scripts/run_week1_week2_pipeline.py --all --config config.yaml

Run on synthetic mock videos (works today, no real videos required)::

    python scripts/run_week1_week2_pipeline.py --mock --config config.yaml

If no real videos exist and ``mock.enabled_if_no_videos`` is true in
``config.yaml``, mock videos are generated and processed automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotate import (  # noqa: E402
    draw_detections_on_frames,
    draw_tracks_on_frames,
    make_annotated_video,
)
from src.config import load_config  # noqa: E402
from src.detect_objects import run_detection  # noqa: E402
from src.extract_frames import extract_frames  # noqa: E402
from src.frame_diff import compute_object_motion_overlap  # noqa: E402
from src.mock_data import generate_all_mock_videos  # noqa: E402
from src.motion_features import compute_track_motion_features  # noqa: E402
from src.optical_flow import compute_object_flow_features  # noqa: E402
from src.summarize import summarize_week1_week2  # noqa: E402
from src.tracking import assign_tracks  # noqa: E402
from src.utils import ensure_dir, video_id_from_path  # noqa: E402
from src.visualize import (  # noqa: E402
    plot_bbox_area_over_time,
    plot_detections_per_frame,
    plot_motion_features,
    plot_track_trajectories,
)

RAW_VIDEO_DIR = Path("data/raw_videos")
MOCK_VIDEO_DIR = Path("data/mock_videos")
FRAMES_DIR = Path("data/frames")
DETECTIONS_DIR = Path("outputs/detections")
TRACKS_DIR = Path("outputs/tracks")
MOTION_DIR = Path("outputs/motion")
ANNOTATED_FRAMES_DIR = Path("outputs/annotated_frames")
ANNOTATED_VIDEOS_DIR = Path("outputs/annotated_videos")
PLOTS_DIR = Path("outputs/plots")
SUMMARIES_DIR = Path("outputs/summaries")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _discover_real_videos() -> List[Path]:
    if not RAW_VIDEO_DIR.exists():
        return []
    return sorted(
        p for p in RAW_VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def process_one(
    video_path: Path,
    cfg: dict,
    mock_scenario: Optional[str] = None,
) -> dict:
    """Run the full Weeks 1 + 2 pipeline on one video."""
    video_id = video_id_from_path(video_path)
    print(f"\n=== Pipeline: {video_id} (mock_scenario={mock_scenario}) ===")

    frame_dir = FRAMES_DIR / video_id
    frame_paths = extract_frames(
        video_path=video_path,
        output_dir=frame_dir,
        sample_fps=float(cfg["video"]["sample_fps"]),
        resize_width=int(cfg["video"]["resize_width"]),
        max_frames=int(cfg["video"]["max_frames"]),
    )
    if not frame_paths:
        raise RuntimeError(f"No frames extracted for {video_id}.")

    sample = cv2.imread(str(frame_paths[0]))
    image_h, image_w = sample.shape[:2]

    detections_df = run_detection(
        frame_paths=frame_paths,
        output_dir=DETECTIONS_DIR,
        config=cfg,
        video_id=video_id,
        mock_scenario=mock_scenario,
    )

    annotated_dir = ANNOTATED_FRAMES_DIR / video_id
    det_frames = draw_detections_on_frames(
        frame_paths=frame_paths,
        detections_df=detections_df,
        output_dir=annotated_dir,
    )
    det_video = ""
    if cfg["outputs"].get("save_annotated_video", True) and det_frames:
        ensure_dir(ANNOTATED_VIDEOS_DIR)
        det_video = make_annotated_video(
            det_frames,
            ANNOTATED_VIDEOS_DIR / f"{video_id}_detections.mp4",
            fps=int(cfg["video"].get("annotated_video_fps", 2)),
        )

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

    features_df = compute_track_motion_features(
        tracks_df=tracks_df,
        config=cfg,
        image_width=image_w,
        image_height=image_h,
        output_dir=TRACKS_DIR,
        video_id=video_id,
    )

    track_frames_dir = ANNOTATED_FRAMES_DIR / f"{video_id}_tracks"
    track_frames = draw_tracks_on_frames(
        frame_paths=[str(p) for p in frame_paths],
        tracks_df=tracks_df,
        detections_df=detections_df,
        output_dir=track_frames_dir,
    )
    tracking_video = ""
    if cfg["outputs"].get("save_annotated_video", True) and track_frames:
        ensure_dir(ANNOTATED_VIDEOS_DIR)
        tracking_video = make_annotated_video(
            track_frames,
            ANNOTATED_VIDEOS_DIR / f"{video_id}_tracks.mp4",
            fps=int(cfg["video"].get("annotated_video_fps", 2)),
        )

    plot_paths = {}
    if cfg["outputs"].get("save_plots", True):
        plot_paths["plot_detections_per_frame"] = plot_detections_per_frame(
            detections_df, PLOTS_DIR / f"{video_id}_detections_per_frame.png"
        )
        plot_paths["plot_trajectories"] = plot_track_trajectories(
            tracks_df, PLOTS_DIR / f"{video_id}_trajectories.png"
        )
        plot_paths["plot_bbox_area"] = plot_bbox_area_over_time(
            tracks_df, PLOTS_DIR / f"{video_id}_bbox_area.png"
        )
        plot_paths["plot_motion_features"] = plot_motion_features(
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

    # TODO(Week 3): apply final hazard classifier here using
    # `features_df` and emit final per-track hazard labels.

    return {
        "video_id": video_id,
        "num_frames": len(frame_paths),
        "num_detections": int(len(detections_df)),
        "num_tracks": int(features_df.shape[0]),
        "frames_dir": str(frame_dir),
        "detections_csv": str(DETECTIONS_DIR / f"{video_id}_detections.csv"),
        "tracks_csv": str(enhanced_tracks_csv),
        "track_features_csv": str(TRACKS_DIR / f"{video_id}_track_features.csv"),
        "annotated_frames_dir": str(annotated_dir),
        "annotated_tracks_dir": str(track_frames_dir),
        "detection_video": det_video,
        "tracking_video": tracking_video,
        "motion_dir": str(MOTION_DIR / video_id),
        "summary_txt": str(SUMMARIES_DIR / f"{video_id}_week1_week2_summary.txt"),
        "summary_json": str(SUMMARIES_DIR / f"{video_id}_week1_week2_summary.json"),
        **plot_paths,
    }


def _resolve_targets(
    args: argparse.Namespace,
    cfg: dict,
) -> List[Tuple[Path, Optional[str]]]:
    """Pick which (video_path, mock_scenario) pairs to process."""
    if args.mock:
        cfg["detection"]["backend"] = "mock"
        mock_paths = generate_all_mock_videos(MOCK_VIDEO_DIR, cfg)
        return [(Path(mp), Path(mp).stem) for mp in mock_paths]

    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            print(f"ERROR: video not found: {vp}", file=sys.stderr)
            sys.exit(1)
        return [(vp, None)]

    if args.all:
        videos = _discover_real_videos()
        if videos:
            return [(v, None) for v in videos]
        if cfg["mock"].get("enabled_if_no_videos", True):
            print("No real videos found; falling back to mock mode "
                  "(mock.enabled_if_no_videos=true).")
            cfg["detection"]["backend"] = "mock"
            mock_paths = generate_all_mock_videos(MOCK_VIDEO_DIR, cfg)
            return [(Path(mp), Path(mp).stem) for mp in mock_paths]
        print("No real videos found in data/raw_videos/. "
              "Use --mock or run scripts/generate_mock_videos.py.")
        sys.exit(0)

    # No flag passed: try real videos, otherwise fall back to mock if allowed.
    videos = _discover_real_videos()
    if videos:
        return [(v, None) for v in videos]
    if cfg["mock"].get("enabled_if_no_videos", True):
        print("No real videos found; running on mock videos "
              "(mock.enabled_if_no_videos=true).")
        cfg["detection"]["backend"] = "mock"
        mock_paths = generate_all_mock_videos(MOCK_VIDEO_DIR, cfg)
        return [(Path(mp), Path(mp).stem) for mp in mock_paths]
    print("No real videos found. Use --mock, --video, or --all "
          "(see scripts/run_week1_week2_pipeline.py --help).")
    sys.exit(0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run SpatialLens Assist Week 1 + Week 2 pipeline.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--video", help="Single video file to process.")
    g.add_argument("--all", action="store_true",
                   help="Process all videos in data/raw_videos/.")
    g.add_argument("--mock", action="store_true",
                   help="Generate + process synthetic mock videos.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)
    targets = _resolve_targets(args, cfg)

    all_outputs = []
    for vp, scenario in targets:
        all_outputs.append(process_one(vp, cfg, mock_scenario=scenario))

    print("\n=== Final outputs ===")
    for out in all_outputs:
        print(f"\n[{out['video_id']}] "
              f"frames={out['num_frames']}, "
              f"detections={out['num_detections']}, "
              f"tracks={out['num_tracks']}")
        for k, v in out.items():
            if k in {"video_id", "num_frames", "num_detections", "num_tracks"}:
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
