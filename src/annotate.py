"""Drawing utilities: overlay detections / tracks on frames and write videos."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .utils import color_for_class, ensure_dir, write_video_from_frames


def _draw_bbox_with_label(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    text_y = max(y1 - 6, th + 4)
    cv2.rectangle(
        img,
        (x1, text_y - th - 4),
        (x1 + tw + 4, text_y + baseline - 2),
        color,
        thickness=-1,
    )
    cv2.putText(
        img, label, (x1 + 2, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def draw_detections_on_frames(
    frame_paths: List[str | Path],
    detections_df: pd.DataFrame,
    output_dir: str | Path,
) -> List[str]:
    """Draw detection bboxes / class / confidence on each frame.

    Output frames are written to ``output_dir/frame_XXXX.jpg``.
    """
    out_dir = ensure_dir(output_dir)
    frame_paths = [Path(p) for p in frame_paths]

    by_path: Dict[str, pd.DataFrame] = {}
    if not detections_df.empty:
        by_path = {str(p): g for p, g in detections_df.groupby("frame_path")}

    annotated_paths: List[str] = []
    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        dets = by_path.get(str(fp))
        if dets is not None:
            for _, row in dets.iterrows():
                color = color_for_class(str(row["class_name"]))
                x1, y1 = int(row["x1"]), int(row["y1"])
                x2, y2 = int(row["x2"]), int(row["y2"])
                label = f"{row['class_name']} {float(row['confidence']):.2f}"
                _draw_bbox_with_label(img, x1, y1, x2, y2, label, color)
        out_path = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(out_path), img)
        annotated_paths.append(str(out_path))
    return annotated_paths


def _trajectory_history(
    tracks_df: pd.DataFrame,
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Return a dict mapping track_id -> list of (frame_idx, cx, cy) sorted by frame."""
    history: Dict[str, List[Tuple[int, int, int]]] = {}
    if tracks_df.empty or "track_id" not in tracks_df.columns:
        return history
    for tid, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        history[str(tid)] = [
            (int(r.frame_idx), int(r.cx), int(r.cy)) for r in g.itertuples()
        ]
    return history


def draw_tracks_on_frames(
    frame_paths: List[str | Path],
    tracks_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    output_dir: str | Path,
) -> List[str]:
    """Draw track bboxes + IDs + trajectory tails on each frame.

    The trajectory tail is the polyline of centroid positions for that
    track_id up to the current frame. ``detections_df`` is kept in the
    signature for symmetry with ``draw_detections_on_frames`` and may be
    used by future iterations (e.g. dimming unmatched detections).
    """
    out_dir = ensure_dir(output_dir)
    frame_paths = [Path(p) for p in frame_paths]
    history = _trajectory_history(tracks_df)

    by_frame_idx: Dict[int, pd.DataFrame] = {}
    if not tracks_df.empty:
        by_frame_idx = {int(f): g for f, g in tracks_df.groupby("frame_idx")}

    annotated_paths: List[str] = []
    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        rows = by_frame_idx.get(i)
        if rows is not None:
            for _, row in rows.iterrows():
                color = color_for_class(str(row["class_name"]))
                x1, y1 = int(row["x1"]), int(row["y1"])
                x2, y2 = int(row["x2"]), int(row["y2"])
                tid = str(row["track_id"])
                label = f"{tid}"
                _draw_bbox_with_label(img, x1, y1, x2, y2, label, color)

                pts = [(cx, cy) for (f_idx, cx, cy) in history.get(tid, [])
                       if f_idx <= i]
                if len(pts) >= 2:
                    poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [poly], isClosed=False,
                                  color=color, thickness=2)
                if pts:
                    cx, cy = pts[-1]
                    cv2.circle(img, (int(cx), int(cy)), 4, color, -1)

                # Optional motion vector: last two centroids -> arrow.
                if len(pts) >= 2:
                    (px, py), (cx, cy) = pts[-2], pts[-1]
                    if (px, py) != (cx, cy):
                        cv2.arrowedLine(
                            img, (int(px), int(py)), (int(cx), int(cy)),
                            (255, 255, 255), 2, tipLength=0.3,
                        )
        out_path = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(out_path), img)
        annotated_paths.append(str(out_path))
    return annotated_paths


def make_annotated_video(
    annotated_frame_paths: List[str | Path],
    output_video_path: str | Path,
    fps: int = 2,
) -> str:
    """Encode annotated frames into an mp4 file."""
    return write_video_from_frames(
        annotated_frame_paths, output_video_path, fps=fps
    )
