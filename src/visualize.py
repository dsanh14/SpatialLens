"""Matplotlib plots for SpatialLens Assist Week 1-2 outputs.

These plots are intentionally simple — they are mainly meant to be dropped
into the Google Slides deck for the final report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe for headless / CI use

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


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
