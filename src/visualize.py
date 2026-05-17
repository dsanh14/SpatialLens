"""Matplotlib plots + hazard-frame overlay for SpatialLens Assist.

The matplotlib plots are intentionally simple — they are dropped into the
Google Slides deck for the final report. The hazard-frame overlay
(:func:`draw_hazard_labels_on_frames`) reuses the OpenCV drawing helpers
in :mod:`src.annotate` so we don't duplicate bbox / trajectory code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # safe for headless / CI use

import matplotlib.pyplot as plt
import pandas as pd

from .annotate import make_annotated_video
from .utils import color_for_class, ensure_dir


def _save_fig(fig: "plt.Figure", output_path: str | Path) -> str:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return str(output_path)


def plot_detections_per_frame(
    detections_df: pd.DataFrame, output_path: str | Path
) -> str:
    """Bar chart of detection count per frame, split by class."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if detections_df.empty:
        ax.set_title("Detections per frame (no detections)")
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("count")
        return _save_fig(fig, output_path)

    pivot = (
        detections_df
        .pivot_table(index="frame_idx", columns="class_name",
                     values="confidence", aggfunc="count", fill_value=0)
        .sort_index()
    )
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Detections per frame")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("count")
    ax.legend(title="class", fontsize=8)
    return _save_fig(fig, output_path)


def plot_track_trajectories(
    tracks_df: pd.DataFrame, output_path: str | Path
) -> str:
    """Plot centroid trajectories for each track_id."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if tracks_df.empty or "track_id" not in tracks_df.columns:
        ax.set_title("Track trajectories (no tracks)")
        ax.set_xlabel("cx (px)")
        ax.set_ylabel("cy (px)")
        ax.invert_yaxis()
        return _save_fig(fig, output_path)

    for tid, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        ax.plot(g["cx"], g["cy"], marker="o", linewidth=1.5,
                label=str(tid))
    ax.set_title("Track trajectories (image coords)")
    ax.set_xlabel("cx (px)")
    ax.set_ylabel("cy (px)")
    ax.invert_yaxis()  # image y axis grows downward
    ax.legend(fontsize=8, loc="best")
    return _save_fig(fig, output_path)


def plot_bbox_area_over_time(
    tracks_df: pd.DataFrame, output_path: str | Path
) -> str:
    """Plot bbox area vs. frame_idx for each track_id.

    Helpful for visually separating "approaching" (growing) from
    "moving away" (shrinking).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if tracks_df.empty or "track_id" not in tracks_df.columns:
        ax.set_title("Bbox area over time (no tracks)")
        ax.set_xlabel("frame_idx")
        ax.set_ylabel("bbox area (px^2)")
        return _save_fig(fig, output_path)

    for tid, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        ax.plot(g["frame_idx"], g["area"], marker="o",
                linewidth=1.5, label=str(tid))
    ax.set_title("Bbox area over time per track")
    ax.set_xlabel("frame_idx")
    ax.set_ylabel("bbox area (px^2)")
    ax.legend(fontsize=8, loc="best")
    return _save_fig(fig, output_path)


def plot_motion_features(
    track_features_df: pd.DataFrame, output_path: str | Path
) -> str:
    """Summary scatter: bbox_growth_ratio vs. dx_total, colored by class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if track_features_df.empty:
        ax.set_title("Per-track motion features (none)")
        ax.set_xlabel("dx_total (px)")
        ax.set_ylabel("bbox_growth_ratio")
        return _save_fig(fig, output_path)

    classes = sorted(track_features_df["class_name"].dropna().unique())
    for cls in classes:
        sub = track_features_df[track_features_df["class_name"] == cls]
        ax.scatter(sub["dx_total"], sub["bbox_growth_ratio"], s=80,
                   label=cls, alpha=0.85)
        for _, r in sub.iterrows():
            ax.annotate(str(r["track_id"]),
                        (r["dx_total"], r["bbox_growth_ratio"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title("Per-track motion features: bbox growth vs. horizontal displacement")
    ax.set_xlabel("dx_total (px)  [positive = rightward]")
    ax.set_ylabel("bbox_growth_ratio  [positive = growing]")
    ax.legend(title="class", fontsize=8, loc="best")
    return _save_fig(fig, output_path)


# --------------------------------------------------------------------- #
# Week 3 — hazard plots
# --------------------------------------------------------------------- #

_HAZARD_COLOR = {
    "approaching": "#d62728",
    "crossing_left_to_right": "#1f77b4",
    "crossing_right_to_left": "#9467bd",
    "moving_away": "#2ca02c",
    "static": "#7f7f7f",
    "uncertain": "#ff7f0e",
}


def plot_hazard_timeline(
    hazards_df: pd.DataFrame, output_path: str | Path
) -> str:
    """Bar chart: one bar per track, colored by hazard label.

    The bar height is the approach_score so it doubles as a quick visual
    cue for "how strong is this hazard signal".
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if hazards_df is None or hazards_df.empty:
        ax.set_title("Hazard timeline (no tracks)")
        ax.set_xlabel("track_id")
        ax.set_ylabel("approach_score")
        return _save_fig(fig, output_path)

    df = hazards_df.copy()
    df = df.sort_values(["hazard_label", "track_id"])
    colors = [_HAZARD_COLOR.get(str(l), "#cccccc") for l in df["hazard_label"]]
    bars = ax.bar(df["track_id"].astype(str), df["approach_score"], color=colors)
    for bar, label in zip(bars, df["hazard_label"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                str(label), ha="center", va="bottom", fontsize=7, rotation=20)
    ax.set_ylim(0, max(1.05, float(df["approach_score"].max()) + 0.2))
    ax.set_title("Hazard timeline per track")
    ax.set_xlabel("track_id")
    ax.set_ylabel("approach_score")
    ax.tick_params(axis="x", rotation=30)
    return _save_fig(fig, output_path)


def plot_approach_scores(
    hazards_df: pd.DataFrame, output_path: str | Path
) -> str:
    """Sorted bar chart of approach scores per track."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if hazards_df is None or hazards_df.empty:
        ax.set_title("Approach scores (no tracks)")
        ax.set_xlabel("track_id")
        ax.set_ylabel("approach_score")
        return _save_fig(fig, output_path)

    df = hazards_df.sort_values("approach_score", ascending=False)
    colors = [_HAZARD_COLOR.get(str(l), "#cccccc") for l in df["hazard_label"]]
    ax.bar(df["track_id"].astype(str), df["approach_score"], color=colors)
    ax.axhline(0.55, color="black", linestyle="--", linewidth=1,
               label="approach_score_threshold (0.55)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Approach scores per track (sorted)")
    ax.set_xlabel("track_id")
    ax.set_ylabel("approach_score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    return _save_fig(fig, output_path)


def plot_confusion_matrix(
    confusion_df: pd.DataFrame, output_path: str | Path,
    title: str = "Hazard confusion matrix (true vs predicted)",
) -> str:
    """Heatmap of a confusion matrix dataframe (rows=true, cols=pred)."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if confusion_df is None or confusion_df.empty:
        ax.set_title(f"{title} — no data")
        return _save_fig(fig, output_path)

    mat = confusion_df.values
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_xticklabels(confusion_df.columns, rotation=30, ha="right")
    ax.set_yticklabels(confusion_df.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    thresh = mat.max() / 2.0 if mat.max() else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat[i, j]),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_fig(fig, output_path)


# --------------------------------------------------------------------- #
# Week 3 — per-frame hazard overlay
# --------------------------------------------------------------------- #


def _draw_hazard_label_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: str,
    class_name: str,
    hazard_label: str,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw a thick bbox + class/track + a prominent hazard banner."""
    is_approaching = hazard_label == "approaching"
    thickness = 4 if is_approaching else 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

    # Top label: class + track id (small, always present).
    top = f"{class_name} {track_id}"
    (tw, th), _ = cv2.getTextSize(top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_y = max(y1 - 6, th + 4)
    cv2.rectangle(img, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 2),
                  color_bgr, thickness=-1)
    cv2.putText(img, top, (x1 + 3, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Hazard label banner (bigger; bright for "APPROACHING").
    if is_approaching:
        banner = "APPROACHING"
        font_scale = 0.95
        font_thick = 3
        banner_color = (0, 0, 255)  # bright red BGR
    else:
        banner = hazard_label.replace("_", " ").upper()
        font_scale = 0.6
        font_thick = 2
        banner_color = color_bgr

    (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, font_thick)
    bx1 = x1
    by1 = max(y2 + 4, bh + 6)
    bx2 = min(x1 + bw + 12, img.shape[1] - 1)
    by2 = by1 + bh + 10
    cv2.rectangle(img, (bx1, by1), (bx2, by2), banner_color, thickness=-1)
    cv2.putText(img, banner, (bx1 + 6, by2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thick, cv2.LINE_AA)


def draw_hazard_labels_on_frames(
    frame_paths: List[str | Path],
    tracks_df: pd.DataFrame,
    hazards_df: pd.DataFrame,
    output_dir: str | Path,
) -> List[str]:
    """Overlay bbox + track_id + hazard label + trajectory on each frame.

    Approaching tracks get a thicker bbox and a bright red ``APPROACHING``
    banner so they pop in the demo video.
    """
    out_dir = ensure_dir(output_dir)
    frame_paths = [Path(p) for p in frame_paths]

    hazard_by_tid: Dict[str, str] = {}
    if hazards_df is not None and not hazards_df.empty:
        for _, r in hazards_df.iterrows():
            hazard_by_tid[str(r["track_id"])] = str(r["hazard_label"])

    history: Dict[str, List[Tuple[int, int, int]]] = {}
    by_frame_idx: Dict[int, pd.DataFrame] = {}
    if tracks_df is not None and not tracks_df.empty:
        for tid, g in tracks_df.groupby("track_id"):
            g = g.sort_values("frame_idx")
            history[str(tid)] = [
                (int(r.frame_idx), int(r.cx), int(r.cy)) for r in g.itertuples()
            ]
        by_frame_idx = {int(f): g for f, g in tracks_df.groupby("frame_idx")}

    annotated_paths: List[str] = []
    for i, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        rows = by_frame_idx.get(i)
        if rows is not None:
            for _, row in rows.iterrows():
                tid = str(row["track_id"])
                class_name = str(row["class_name"])
                color = color_for_class(class_name)
                x1, y1 = int(row["x1"]), int(row["y1"])
                x2, y2 = int(row["x2"]), int(row["y2"])
                hazard_label = hazard_by_tid.get(tid, "uncertain")
                _draw_hazard_label_box(
                    img, x1, y1, x2, y2, tid, class_name,
                    hazard_label, color,
                )
                pts = [(cx, cy) for (f_idx, cx, cy) in history.get(tid, [])
                       if f_idx <= i]
                if len(pts) >= 2:
                    poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [poly], isClosed=False,
                                  color=color, thickness=2)
                    (px, py), (cx, cy) = pts[-2], pts[-1]
                    if (px, py) != (cx, cy):
                        cv2.arrowedLine(img, (int(px), int(py)),
                                        (int(cx), int(cy)),
                                        (255, 255, 255), 2, tipLength=0.3)
        out_path = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(out_path), img)
        annotated_paths.append(str(out_path))
    return annotated_paths


def make_hazard_annotated_video(
    annotated_frame_paths: List[str | Path],
    output_video_path: str | Path,
    fps: int = 2,
) -> str:
    """Encode hazard-annotated frames into the final demo mp4."""
    return make_annotated_video(annotated_frame_paths, output_video_path, fps=fps)
