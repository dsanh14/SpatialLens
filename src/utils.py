"""Small shared helpers used across the SpatialLens Assist pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent

# BGR colors (OpenCV uses BGR) for consistent class coloring across frames.
CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 255, 0),       # green
    "bicycle": (0, 165, 255),    # orange
    "motorcycle": (0, 0, 255),   # red
    "skateboard": (255, 0, 255), # magenta
    "scooter": (255, 255, 0),    # cyan (unused by COCO YOLO but reserved)
}
DEFAULT_COLOR_BGR: Tuple[int, int, int] = (200, 200, 200)


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if it does not already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def video_id_from_path(video_path: str | Path) -> str:
    """Return a clean video identifier from a file path (basename without ext)."""
    return Path(video_path).stem


def color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Pick a stable BGR color for a class name."""
    return CLASS_COLORS_BGR.get(class_name, DEFAULT_COLOR_BGR)


def write_video_from_frames(
    frame_paths: Iterable[str | Path],
    output_video_path: str | Path,
    fps: int = 2,
) -> str:
    """Encode a sequence of image files into an mp4 video.

    Returns the output path as a string. Returns an empty string if no frames
    were provided (so callers can no-op gracefully).
    """
    frame_paths = [Path(p) for p in frame_paths]
    if not frame_paths:
        return ""

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    output_video_path = Path(output_video_path)
    ensure_dir(output_video_path.parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    try:
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            writer.write(img)
    finally:
        writer.release()
    return str(output_video_path)


def image_diagonal(width: int, height: int) -> float:
    """Length of the image diagonal in pixels."""
    return float((width**2 + height**2) ** 0.5)
