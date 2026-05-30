#!/usr/bin/env python3
"""Generate publication-style figures for the CS131 progress report PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "reports" / "progress_report" / "figures"
EVAL_JSON = ROOT / "outputs" / "evaluation" / "all_videos_evaluation_summary.json"
HAZARDS_DIR = ROOT / "outputs" / "hazards"
ANNOTATED = FIG_DIR / "annotated_example.jpg"

# Matplotlib style tuned to look like a research paper (OpenJarvis-like).
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

PALETTE = {
    "approaching": "#c0392b",
    "crossing": "#2980b9",
    "moving_away": "#27ae60",
    "uncertain": "#e67e22",
    "static": "#7f8c8d",
    "accent": "#2c3e50",
    "light": "#ecf0f1",
}


def _panel_label(ax, label: str) -> None:
    ax.text(
        0.02, 0.98, label, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.8),
    )


def _draw_pipeline_panel(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Pipeline", pad=6)

    stages = [
        ("Raw\nvideo", "#d5e8f7"),
        ("YOLO\nv8n", "#c8e6c9"),
        ("IoU +\ncentroid\ntracker", "#ffe0b2"),
        ("Motion\nfeatures", "#e1bee7"),
        ("Rule\ncascade", "#ffcdd2"),
        ("Assistive\nalerts", "#cfd8dc"),
    ]
    xs = np.linspace(0.6, 9.4, len(stages))
    y = 5.2
    w, h = 1.15, 2.2
    for x, (label, color) in zip(xs, stages):
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=0.8, edgecolor="#666666", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5)

    for i in range(len(xs) - 1):
        ax.annotate(
            "", xy=(xs[i + 1] - w / 2 - 0.05, y),
            xytext=(xs[i] + w / 2 + 0.05, y),
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.2),
        )

    ax.text(
        5.0, 1.6,
        "Ego-motion compensation on optical flow + frame diff\n"
        "Evidence strings on every hazard label",
        ha="center", va="center", fontsize=7.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["light"],
                  edgecolor="#cccccc"),
    )
    _panel_label(ax, "(a)")


def _draw_metrics_panel(ax, agg: dict) -> None:
    ax.set_title("Evaluation summary", pad=6)
    sel = agg.get("selective_accuracy") or {}
    labels = ["Decidable\n(≥3 frames)", "Overall\n(all tracks)"]
    vals = [
        100 * float(sel.get("decidable_accuracy", 0)),
        100 * float(agg.get("overall_accuracy", 0)),
    ]
    colors = ["#1a6f3c", "#5d6d7e"]
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black",
                  linewidth=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.axhline(90, color="#999999", linestyle="--", linewidth=0.8, alpha=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.text(
        0.98, 0.98,
        f"Macro F1 = {agg.get('macro_f1', 0):.3f}\n"
        f"approaching P/R = "
        f"{agg.get('approaching_precision', 0):.2f}/"
        f"{agg.get('approaching_recall', 0):.2f}\n"
        f"n = {agg.get('num_evaluated_tracks', 0)} evaluated tracks",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc"),
    )
    _panel_label(ax, "(b)")


def _draw_ablation_panel(ax) -> None:
    ax.set_title("Ablations", pad=6)
    names = ["v1 baseline", "+ frame gap", "− strong_crossing", "final"]
    vals = [67.2, 82.5, 78.9, 82.5]
    colors = ["#bdc3c7", "#85c1e9", "#f5b7b1", "#1a6f3c"]
    bars = ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Overall accuracy (%)")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    _panel_label(ax, "(c)")


def _draw_annotated_panel(ax) -> None:
    ax.axis("off")
    ax.set_title("Qualitative output", pad=6)
    if ANNOTATED.exists():
        img = Image.open(ANNOTATED)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "annotated_example.jpg\nnot found",
                ha="center", va="center", transform=ax.transAxes)
    _panel_label(ax, "(d)")


def build_figure1_overview(agg: dict, out_path: Path) -> str:
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(7.5, 2.45))
        gs = GridSpec(1, 4, width_ratios=[1.35, 0.95, 0.95, 1.05], wspace=0.42)
        _draw_pipeline_panel(fig.add_subplot(gs[0, 0]))
        _draw_metrics_panel(fig.add_subplot(gs[0, 1]), agg)
        _draw_ablation_panel(fig.add_subplot(gs[0, 2]))
        _draw_annotated_panel(fig.add_subplot(gs[0, 3]))
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    return str(out_path)


def _confusion_matrix_df(agg: dict) -> pd.DataFrame:
    cm = agg.get("confusion_matrix", {})
    labels = agg.get("labels") or list(cm.keys())
    rows = []
    for true_lab in labels:
        row = cm.get(true_lab, {})
        rows.append({pred: int(row.get(pred, 0)) for pred in labels})
    return pd.DataFrame(rows, index=labels, columns=labels)


def _draw_confusion_panel(ax, agg: dict) -> None:
    cm_df = _confusion_matrix_df(agg)
    mat = cm_df.values.astype(float)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(5, mat.max()))
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    short = {
        "approaching": "approach",
        "crossing_left_to_right": "cross L→R",
        "crossing_right_to_left": "cross R→L",
        "moving_away": "away",
        "static": "static",
        "uncertain": "uncertain",
    }
    ax.set_xticklabels([short.get(c, c) for c in cm_df.columns],
                       rotation=35, ha="right")
    ax.set_yticklabels([short.get(r, r) for r in cm_df.index])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix", pad=6)
    thresh = mat.max() / 2 if mat.max() else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8,
                        color="white" if val > thresh else "black")
    _panel_label(ax, "(a)")


def _load_all_hazards() -> pd.DataFrame:
    frames = []
    for path in sorted(HAZARDS_DIR.glob("*_hazards.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _draw_uncertain_panel(ax, hazards: pd.DataFrame) -> None:
    ax.set_title("Uncertain abstention reasons", pad=6)
    if hazards.empty or "uncertain_reason" not in hazards.columns:
        ax.text(0.5, 0.5, "No uncertain-reason data", ha="center", va="center",
                transform=ax.transAxes)
        _panel_label(ax, "(b)")
        return

    unc = hazards[hazards["hazard_label"] == "uncertain"].copy()
    reasons = (
        unc["uncertain_reason"].fillna("(unknown)").replace("", "(unknown)")
        .value_counts().sort_values(ascending=True)
    )
    colors = {
        "short_track": "#7f8c8d",
        "near_approaching": "#c0392b",
        "near_crossing_left_to_right": "#e67e22",
        "near_crossing_right_to_left": "#e67e22",
        "near_moving_away": "#8e44ad",
        "conflicting_cues": "#2980b9",
        "low_signal": "#27ae60",
    }
    bar_colors = [colors.get(r, "#bdc3c7") for r in reasons.index]
    bars = ax.barh(reasons.index, reasons.values, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    total = int(reasons.sum())
    for bar, count in zip(bars, reasons.values):
        pct = 100 * count / total if total else 0
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(count)} ({pct:.0f}%)", va="center", fontsize=7.5)
    ax.set_xlabel("Tracks")
    ax.set_xlim(0, reasons.max() * 1.25)
    _panel_label(ax, "(b)")


def build_figure2_evaluation(agg: dict, out_path: Path) -> str:
    hazards = _load_all_hazards()
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(7.5, 2.5))
        gs = GridSpec(1, 2, width_ratios=[1.05, 0.95], wspace=0.35)
        _draw_confusion_panel(fig.add_subplot(gs[0, 0]), agg)
        _draw_uncertain_panel(fig.add_subplot(gs[0, 1]), hazards)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    return str(out_path)


def build_figure3_qualitative(out_path: Path) -> str:
    traj = FIG_DIR / "trajectories_example.png"
    annot = FIG_DIR / "annotated_example.jpg"
    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(7.5, 2.8))
        gs = GridSpec(1, 2, width_ratios=[1.0, 1.05], wspace=0.12)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        for ax, path, title, label in (
            (ax0, traj, "Track trajectories (IMG 4974)", "(a)"),
            (ax1, annot, "Hazard overlay (IMG 4976)", "(b)"),
        ):
            ax.axis("off")
            ax.set_title(title, pad=6)
            if path.exists():
                ax.imshow(Image.open(path))
            else:
                ax.text(0.5, 0.5, f"{path.name}\nnot found", ha="center",
                        va="center", transform=ax.transAxes)
            _panel_label(ax, label)
        fig.savefig(out_path, facecolor="white")
        plt.close(fig)
    return str(out_path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not EVAL_JSON.exists():
        raise SystemExit(f"Missing evaluation summary: {EVAL_JSON}")
    with EVAL_JSON.open(encoding="utf-8") as f:
        agg = json.load(f)

    f1 = build_figure1_overview(agg, FIG_DIR / "figure1_overview.png")
    f2 = build_figure2_evaluation(agg, FIG_DIR / "figure2_evaluation.png")
    f3 = build_figure3_qualitative(FIG_DIR / "figure3_qualitative.png")
    print(f"Wrote {f1}")
    print(f"Wrote {f2}")
    print(f"Wrote {f3}")


if __name__ == "__main__":
    main()
