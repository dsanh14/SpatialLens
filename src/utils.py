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


# Codec fallback chain for cv2.VideoWriter. We try H.264 first because
# it's the only codec macOS QuickTime / Finder Quick Look / Cursor's
# built-in video preview all decode reliably; the older "mp4v" (MPEG-4
# Part 2) codec produces files that play in VLC but render as a solid
# color in QuickTime on modern macOS. The four-character codes are
# tried in order until VideoWriter.isOpened() returns True.
_VIDEO_FOURCC_FALLBACK = ("avc1", "H264", "mp4v")


def open_video_writer(
    output_path: str | Path,
    fps: int,
    width: int,
    height: int,
) -> Tuple[cv2.VideoWriter, str]:
    """Open a ``cv2.VideoWriter`` with H.264 preferred, MPEG-4 fallback.

    Returns ``(writer, fourcc_used)``. Raises ``RuntimeError`` if no
    codec in the fallback chain could be opened (extremely rare —
    indicates a broken OpenCV install).
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    last_err: str | None = None
    for code in _VIDEO_FOURCC_FALLBACK:
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height),
        )
        if writer.isOpened():
            return writer, code
        writer.release()
        last_err = code
    raise RuntimeError(
        f"cv2.VideoWriter could not be opened with any of "
        f"{_VIDEO_FOURCC_FALLBACK} for path={output_path}. "
        f"Last attempted fourcc={last_err!r}."
    )


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

    writer, _fourcc = open_video_writer(
        output_video_path, fps=fps, width=width, height=height,
    )
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
