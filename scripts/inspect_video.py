"""Print metadata about a single video file.

Usage:
    python scripts/inspect_video.py --video data/raw_videos/example.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow `python scripts/inspect_video.py ...` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def inspect(video_path: str) -> int:
    path = Path(video_path)
    if not path.exists():
        print(f"[inspect] ERROR: video not found: {path}", file=sys.stderr)
        return 1
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[inspect] ERROR: could not open video: {path}", file=sys.stderr)
        return 1
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0

        print(f"Video:       {path}")
        print(f"  fps:         {fps:.3f}")
        print(f"  frame count: {frame_count}")
        print(f"  duration:    {duration:.2f} seconds")
        print(f"  width:       {width}")
        print(f"  height:      {height}")
    finally:
        cap.release()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Print video metadata.")
    p.add_argument("--video", required=True, help="Path to a video file.")
    args = p.parse_args()
    sys.exit(inspect(args.video))


if __name__ == "__main__":
    main()
