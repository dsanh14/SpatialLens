"""Sample and resize frames from a video file.

Usage (CLI):

    python -m src.extract_frames \
        --video data/raw_videos/example.mp4 \
        --out data/frames/example \
        --fps 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from .utils import ensure_dir


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 2.0,
    resize_width: int = 960,
    max_frames: int = 80,
) -> List[str]:
    """Extract frames from a video at approximately ``sample_fps``.

    Frames are resized to ``resize_width`` (preserving aspect ratio) and
    saved as ``frame_0000.jpg`` ... in ``output_dir``.

    Parameters
    ----------
    video_path:
        Path to a video file readable by OpenCV.
    output_dir:
        Directory where extracted frames will be saved. Created if missing.
    sample_fps:
        Desired sampling rate. The actual stride is computed from the
        video's reported FPS (with a sensible fallback if unknown).
    resize_width:
        Target width for resized frames. Height is computed to preserve
        the aspect ratio. ``None`` or non-positive disables resizing.
    max_frames:
        Hard cap on how many frames are written.

    Returns
    -------
    list[str]
        Paths to the written frame images, in order.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not exist.
    RuntimeError
        If OpenCV cannot open the video.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if src_fps <= 0:
        # Some containers don't report FPS. Assume 30 as a safe fallback.
        src_fps = 30.0

    if sample_fps and sample_fps > 0:
        stride = max(int(round(src_fps / float(sample_fps))), 1)
    else:
        stride = 1

    print(
        f"[extract_frames] {video_path.name}: src_fps={src_fps:.2f}, "
        f"total_frames={total_frames}, stride={stride}, "
        f"sample_fps={sample_fps}, max_frames={max_frames}"
    )

    saved: List[str] = []
    frame_idx = 0
    saved_idx = 0
    pbar = tqdm(total=min(total_frames or 0, max_frames * stride) or None,
                desc=f"extracting {video_path.stem}", unit="f")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                if resize_width and resize_width > 0:
                    h, w = frame.shape[:2]
                    if w != resize_width:
                        new_h = int(round(h * (resize_width / float(w))))
                        frame = cv2.resize(
                            frame, (resize_width, new_h),
                            interpolation=cv2.INTER_AREA,
                        )
                out_path = out_dir / f"frame_{saved_idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved.append(str(out_path))
                saved_idx += 1
                if saved_idx >= max_frames:
                    break
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()

    print(f"[extract_frames] wrote {len(saved)} frames -> {out_dir}")
    return saved


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from a video.")
    p.add_argument("--video", required=True, help="Path to input video.")
    p.add_argument("--out", required=True, help="Output directory for frames.")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Approximate sampling FPS (default: 2).")
    p.add_argument("--resize-width", type=int, default=960,
                   help="Resize frames to this width (default: 960).")
    p.add_argument("--max-frames", type=int, default=80,
                   help="Maximum number of frames to write (default: 80).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.out,
        sample_fps=args.fps,
        resize_width=args.resize_width,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
