"""Week 1 pipeline: extract frames -> detect -> annotate -> summary.

Usage examples:
    python scripts/run_week1_pipeline.py --video data/raw_videos/example.mp4 --config config.yaml
    python scripts/run_week1_pipeline.py --all --config config.yaml
    python scripts/run_week1_pipeline.py --mock --config config.yaml

If no real videos are present and ``--mock`` is not given, prints a
helpful pointer instead of failing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotate import draw_detections_on_frames, make_annotated_video  # noqa: E402
from src.config import load_config  # noqa: E402
from src.detect_objects import run_detection  # noqa: E402
from src.extract_frames import extract_frames  # noqa: E402
from src.mock_data import generate_all_mock_videos  # noqa: E402
from src.utils import ensure_dir, video_id_from_path  # noqa: E402

RAW_VIDEO_DIR = Path("data/raw_videos")
MOCK_VIDEO_DIR = Path("data/mock_videos")
FRAMES_DIR = Path("data/frames")
DETECTIONS_DIR = Path("outputs/detections")
ANNOTATED_FRAMES_DIR = Path("outputs/annotated_frames")
ANNOTATED_VIDEOS_DIR = Path("outputs/annotated_videos")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _discover_real_videos() -> List[Path]:
    if not RAW_VIDEO_DIR.exists():
        return []
    return sorted(
        p for p in RAW_VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def run_week1_for_video(
    video_path: Path,
    cfg: dict,
    mock_scenario: Optional[str] = None,
) -> Tuple[str, dict]:
    """Run Week 1 pipeline on one video; return (video_id, output paths dict)."""
    video_id = video_id_from_path(video_path)
    print(f"\n=== Week 1 pipeline: {video_id} ===")

    frame_dir = FRAMES_DIR / video_id
    frame_paths = extract_frames(
        video_path=video_path,
        output_dir=frame_dir,
        sample_fps=float(cfg["video"]["sample_fps"]),
        resize_width=int(cfg["video"]["resize_width"]),
        max_frames=int(cfg["video"]["max_frames"]),
    )

    detections_df = run_detection(
        frame_paths=frame_paths,
        output_dir=DETECTIONS_DIR,
        config=cfg,
        video_id=video_id,
        mock_scenario=mock_scenario,
    )

    annotated_dir = ANNOTATED_FRAMES_DIR / video_id
    annotated_paths = draw_detections_on_frames(
        frame_paths=frame_paths,
        detections_df=detections_df,
        output_dir=annotated_dir,
    )

    video_out = ""
    if cfg["outputs"].get("save_annotated_video", True) and annotated_paths:
        video_out_path = ANNOTATED_VIDEOS_DIR / f"{video_id}_detections.mp4"
        ensure_dir(ANNOTATED_VIDEOS_DIR)
        video_out = make_annotated_video(
            annotated_paths,
            video_out_path,
            fps=int(cfg["video"].get("annotated_video_fps", 2)),
        )

    summary = {
        "video_id": video_id,
        "num_frames": len(frame_paths),
        "num_detections": int(len(detections_df)),
        "frames_dir": str(frame_dir),
        "detections_csv": str(DETECTIONS_DIR / f"{video_id}_detections.csv"),
        "annotated_frames_dir": str(annotated_dir),
        "annotated_video": video_out,
    }
    print(f"[week1] {video_id}: frames={summary['num_frames']}, "
          f"detections={summary['num_detections']}")
    return video_id, summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run SpatialLens Assist Week 1 pipeline.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--video", help="Single video file to process.")
    g.add_argument("--all", action="store_true",
                   help="Process all videos in data/raw_videos/.")
    g.add_argument("--mock", action="store_true",
                   help="Generate + process synthetic mock videos.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)

    targets: List[Tuple[Path, Optional[str]]] = []
    if args.mock:
        cfg["detection"]["backend"] = "mock"
        mock_paths = generate_all_mock_videos(MOCK_VIDEO_DIR, cfg)
        for mp in mock_paths:
            scenario = Path(mp).stem
            targets.append((Path(mp), scenario))
    elif args.video:
        vp = Path(args.video)
        if not vp.exists():
            print(f"ERROR: video not found: {vp}", file=sys.stderr)
            sys.exit(1)
        targets.append((vp, None))
    elif args.all:
        videos = _discover_real_videos()
        if not videos:
            print("No real videos found in data/raw_videos/.")
            print("Use --mock or run scripts/generate_mock_videos.py.")
            return
        for v in videos:
            targets.append((v, None))
    else:
        videos = _discover_real_videos()
        if not videos:
            print("No real videos found. Use --mock or run "
                  "scripts/generate_mock_videos.py.")
            return
        for v in videos:
            targets.append((v, None))

    results = []
    for vp, scenario in targets:
        results.append(run_week1_for_video(vp, cfg, mock_scenario=scenario))

    print("\n=== Week 1 outputs ===")
    for vid, summary in results:
        print(f"\n[{vid}]")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
