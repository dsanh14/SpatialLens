"""Per-track motion feature aggregation for Week 2.

These features are the inputs the Week 3 hazard classifier will use to
decide ``approaching`` / ``crossing left-to-right`` / etc. For Weeks 1-2,
we only emit *preliminary* labels (``static`` vs. ``moving``,
``growing`` / ``shrinking`` / ``stable``, simple direction) for sanity
checking and to drop into the slide deck.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import ensure_dir, image_diagonal

EPSILON = 1e-6

TRACK_FEATURE_COLUMNS = [
    "video_id",
    "track_id",
    "class_name",
    "first_frame",
    "last_frame",
    "num_frames",
    "start_cx", "start_cy",
    "end_cx", "end_cy",
    "total_displacement_px",
    "total_displacement_norm",
    "dx_total",
    "dy_total",
    "avg_bbox_area",
    "start_area",
    "end_area",
    "bbox_growth_ratio",
    "avg_flow_dx",
    "avg_flow_dy",
    "avg_flow_mag",
    "avg_global_flow_dx",
    "avg_global_flow_dy",
    "avg_global_flow_mag",
    "avg_frame_diff_overlap",
    "preliminary_motion_label",
    "preliminary_direction_label",
    "preliminary_scale_label",
]


def _label_motion(
    total_disp_norm: float,
    avg_flow_mag: float,
    static_thresh: float,
    flow_thresh: float,
) -> str:
    if total_disp_norm < static_thresh and avg_flow_mag < flow_thresh:
        return "static"
    return "moving"


def _label_direction(
    dx_total: float,
    horiz_thresh_px: float,
) -> str:
    if dx_total > horiz_thresh_px:
        return "left_to_right"
    if dx_total < -horiz_thresh_px:
        return "right_to_left"
    return "toward_center_or_away"


def _label_scale(
    bbox_growth_ratio: float,
    growth_thresh: float,
) -> str:
    if bbox_growth_ratio > growth_thresh:
        return "growing"
    if bbox_growth_ratio < -growth_thresh:
        return "shrinking"
    return "stable"


def compute_track_motion_features(
    tracks_df: pd.DataFrame,
    config: dict,
    image_width: int,
    image_height: int,
    output_dir: str | Path | None = None,
    video_id: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-track motion features and preliminary labels.

    The input ``tracks_df`` is expected to already contain (optional)
    flow + frame-diff columns:

    * ``flow_dx``, ``flow_dy``, ``flow_mag`` (from
      :mod:`src.optical_flow`)
    * ``frame_diff_overlap`` (from :mod:`src.frame_diff`)

    Missing columns are tolerated and treated as zero.
    """
    motion_cfg = config["motion"]
    static_thresh = float(motion_cfg["static_threshold_frac_diagonal"])
    flow_thresh = float(motion_cfg["flow_magnitude_threshold"])
    growth_thresh = float(motion_cfg["bbox_growth_threshold"])
    horiz_thresh_frac = float(motion_cfg["horizontal_motion_threshold_frac_width"])
    horiz_thresh_px = horiz_thresh_frac * float(image_width)
    img_diag = image_diagonal(image_width, image_height)

    if tracks_df.empty or "track_id" not in tracks_df.columns:
        empty = pd.DataFrame(
            {c: pd.Series(dtype="object") for c in TRACK_FEATURE_COLUMNS}
        )
        if output_dir and video_id:
            out = ensure_dir(output_dir) / f"{video_id}_track_features.csv"
            empty.to_csv(out, index=False)
        return empty

    df = tracks_df.copy()
    for col in (
        "flow_dx", "flow_dy", "flow_mag",
        "global_flow_dx", "global_flow_dy",
        "frame_diff_overlap",
    ):
        if col not in df.columns:
            df[col] = 0.0

    rows: List[Dict] = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        first = g.iloc[0]
        last = g.iloc[-1]
        start_cx, start_cy = float(first["cx"]), float(first["cy"])
        end_cx, end_cy = float(last["cx"]), float(last["cy"])
        dx_total = end_cx - start_cx
        dy_total = end_cy - start_cy
        total_disp_px = math.hypot(dx_total, dy_total)
        total_disp_norm = total_disp_px / img_diag if img_diag > 0 else 0.0
        start_area = float(first["area"])
        end_area = float(last["area"])
        avg_area = float(g["area"].mean())
        growth_ratio = (end_area - start_area) / (start_area + EPSILON)

        avg_flow_dx = float(g["flow_dx"].mean()) if len(g) else 0.0
        avg_flow_dy = float(g["flow_dy"].mean()) if len(g) else 0.0
        avg_flow_mag = float(g["flow_mag"].mean()) if len(g) else 0.0
        avg_g_dx = float(g["global_flow_dx"].mean()) if len(g) else 0.0
        avg_g_dy = float(g["global_flow_dy"].mean()) if len(g) else 0.0
        avg_g_mag = math.hypot(avg_g_dx, avg_g_dy)
        avg_fd_overlap = float(g["frame_diff_overlap"].mean()) if len(g) else 0.0

        motion_label = _label_motion(
            total_disp_norm, avg_flow_mag, static_thresh, flow_thresh
        )
        direction_label = _label_direction(dx_total, horiz_thresh_px)
        scale_label = _label_scale(growth_ratio, growth_thresh)

        rows.append({
            "video_id": str(first.get("video_id", video_id or "")),
            "track_id": str(tid),
            "class_name": str(first["class_name"]),
            "first_frame": int(first["frame_idx"]),
            "last_frame": int(last["frame_idx"]),
            "num_frames": int(len(g)),
            "start_cx": start_cx,
            "start_cy": start_cy,
            "end_cx": end_cx,
            "end_cy": end_cy,
            "total_displacement_px": total_disp_px,
            "total_displacement_norm": total_disp_norm,
            "dx_total": dx_total,
            "dy_total": dy_total,
            "avg_bbox_area": avg_area,
            "start_area": start_area,
            "end_area": end_area,
            "bbox_growth_ratio": growth_ratio,
            "avg_flow_dx": avg_flow_dx,
            "avg_flow_dy": avg_flow_dy,
            "avg_flow_mag": avg_flow_mag,
            "avg_global_flow_dx": avg_g_dx,
            "avg_global_flow_dy": avg_g_dy,
            "avg_global_flow_mag": avg_g_mag,
            "avg_frame_diff_overlap": avg_fd_overlap,
            "preliminary_motion_label": motion_label,
            "preliminary_direction_label": direction_label,
            "preliminary_scale_label": scale_label,
        })

    feats = pd.DataFrame(rows, columns=TRACK_FEATURE_COLUMNS)
    if output_dir and video_id:
        out = ensure_dir(output_dir) / f"{video_id}_track_features.csv"
        feats.to_csv(out, index=False)
        print(f"[motion_features] wrote {len(feats)} track features -> {out}")
    return feats


# TODO(Week 3): the final hazard classifier will combine
# (preliminary_motion_label, preliminary_direction_label,
# preliminary_scale_label, avg_flow_mag, avg_frame_diff_overlap,
# bbox_growth_ratio, dx_total) into the six target labels:
#   approaching / crossing_l2r / crossing_r2l / moving_away / static / uncertain.
