#!/usr/bin/env python3
"""Generate one PNG per idea for the CS131 progress report (no composite panels)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "reports" / "progress_report" / "figures"
EVAL_JSON = ROOT / "outputs" / "evaluation" / "all_videos_evaluation_summary.json"
HAZARDS_DIR = ROOT / "outputs" / "hazards"
ANNOTATED = FIG_DIR / "annotated_example.jpg"
TRAJECTORIES = FIG_DIR / "trajectories_example.png"

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
}

STAGE_COLORS = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#F3E5F5", "#FFEBEE", "#ECEFF1"]
STAGE_EDGES = "#455A64"

_SHORT = {
    "approaching": "Approach",
    "crossing_left_to_right": "Cross L->R",
    "crossing_right_to_left": "Cross R->L",
    "moving_away": "Away",
    "static": "Static",
    "uncertain": "Uncertain",
}

_REASON_SHORT = {
    "short_track": "Short track",
    "low_signal": "Low signal",
    "near_approaching": "Near approaching",
    "near_crossing_left_to_right": "Near cross L->R",
    "near_crossing_right_to_left": "Near cross R->L",
    "near_moving_away": "Near moving away",
    "conflicting_cues": "Conflicting cues",
}


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return str(path)


def _style_axes(ax, grid_y: bool = False, grid_x: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_y:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)
    if grid_x:
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)


def save_fig_pipeline(out: Path) -> str:
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.8, 1.85))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 5.5)
        ax.axis("off")

        stages = [
            ("Raw video", STAGE_COLORS[0]),
            ("YOLOv8n", STAGE_COLORS[1]),
            ("Track", STAGE_COLORS[2]),
            ("Motion", STAGE_COLORS[3]),
            ("Classify", STAGE_COLORS[4]),
            ("Alerts", STAGE_COLORS[5]),
        ]
        xs = np.linspace(1.0, 11.0, len(stages))
        y, bw, bh = 3.0, 1.5, 0.95
        for x, (name, fill) in zip(xs, stages):
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - bw / 2, y - bh / 2), bw, bh,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                linewidth=1.0, edgecolor=STAGE_EDGES, facecolor=fill,
            ))
            ax.text(x, y, name, ha="center", va="center", fontsize=10,
                    fontweight="medium")
        for i in range(len(xs) - 1):
            ax.annotate(
                "", xy=(xs[i + 1] - bw / 2 - 0.06, y),
                xytext=(xs[i] + bw / 2 + 0.06, y),
                arrowprops=dict(arrowstyle="-|>", color=STAGE_EDGES, lw=1.3),
            )
        ax.text(
            6.0, 0.85,
            "Ego-motion compensation (median flow + ECC)  ·  "
            "Evidence string on every hazard label",
            ha="center", fontsize=8.5, color="#546E7A",
        )
        return _save(fig, out)


def save_fig_accuracy(agg: dict, out: Path) -> str:
    with plt.rc_context(PAPER_RC):
        sel = agg.get("selective_accuracy") or {}
        n_eval = int(agg.get("num_evaluated_tracks", 57))
        n_dec = int(sel.get("num_decidable_tracks", 30))
        vals = [
            100 * float(sel.get("decidable_accuracy", 0)),
            100 * float(agg.get("overall_accuracy", 0)),
        ]

        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        x = np.arange(2)
        bars = ax.bar(
            x, vals, width=0.42, color=["#2E7D32", "#546E7A"],
            edgecolor="#212121", linewidth=0.8, zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Decidable", "Overall"], fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel("Accuracy (%)", labelpad=6)
        _style_axes(ax, grid_y=True)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, min(val + 3, 97),
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#212121",
            )

        # Sub-labels under each bar (axes coords) — no overlap with tick text.
        ax.text(0, -0.11, f"tracks $\\geq$3 frames\n($n={n_dec}$)",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#546E7A", linespacing=1.15)
        ax.text(1, -0.11, f"all matched tracks\n($n={n_eval}$)",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color="#546E7A", linespacing=1.15)

        fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.22)
        return _save(fig, out)


def save_fig_ablations(out: Path) -> str:
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        names = ["v1 baseline", "+ frame gap", "w/o strong crossing", "Final"]
        vals = [67.2, 82.5, 78.9, 82.5]
        colors = ["#B0BEC5", "#64B5F6", "#EF9A9A", "#2E7D32"]
        y = np.arange(len(names))
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="#212121", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 96)
        ax.set_xlabel("Overall accuracy (%)")
        ax.set_title("Ablation study", pad=10)
        ax.invert_yaxis()
        _style_axes(ax, grid_x=True)
        for i, val in enumerate(vals):
            ax.text(val + 1.2, i, f"{val:.1f}%", va="center", fontweight="bold")
        return _save(fig, out)


def _confusion_matrix_df(agg: dict) -> pd.DataFrame:
    cm = agg.get("confusion_matrix", {})
    labels = agg.get("labels") or list(cm.keys())
    rows = []
    for true_lab in labels:
        row = cm.get(true_lab, {})
        rows.append({pred: int(row.get(pred, 0)) for pred in labels})
    return pd.DataFrame(rows, index=labels, columns=labels)


def save_fig_confusion(agg: dict, out: Path) -> str:
    with plt.rc_context(PAPER_RC):
        cm_df = _confusion_matrix_df(agg)
        mat = cm_df.values.astype(float)
        n = len(cm_df)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        vmax = max(float(mat.max()), 1.0)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([_SHORT.get(c, c) for c in cm_df.columns],
                           rotation=42, ha="right")
        ax.set_yticklabels([_SHORT.get(r, r) for r in cm_df.index])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (11 videos)", pad=10)
        thresh = vmax * 0.55
        for i in range(n):
            for j in range(n):
                val = int(mat[i, j])
                if val == 0:
                    continue
                ax.text(j, i, str(val), ha="center", va="center", fontsize=10,
                        fontweight="bold",
                        color="white" if val > thresh else "#212121")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Count", fontsize=9)
        return _save(fig, out)


def _load_all_hazards() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(HAZARDS_DIR.glob("*_hazards.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_fig_uncertain_reasons(out: Path) -> str:
    hazards = _load_all_hazards()
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        unc = hazards[hazards["hazard_label"] == "uncertain"]
        reasons = (
            unc["uncertain_reason"].fillna("unknown").replace("", "unknown")
            .value_counts().sort_values(ascending=True)
        )
        labels = [_REASON_SHORT.get(str(r), str(r)) for r in reasons.index]
        counts = reasons.values
        total = int(counts.sum())
        colors = ["#78909C" if "Short" in lb else "#66BB6A" for lb in labels]
        y = np.arange(len(labels))
        ax.barh(y, counts, height=0.55, color=colors, edgecolor="#212121", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Tracks")
        ax.set_title("Why predictions are uncertain", pad=10)
        ax.set_xlim(0, max(counts) * 1.25 + 1)
        _style_axes(ax, grid_x=True)
        for i, count in enumerate(counts):
            pct = 100 * count / total if total else 0
            ax.text(count + 0.5, i, f"{int(count)} ({pct:.0f}%)", va="center")
        return _save(fig, out)


def save_fig_hazard_overlay(out: Path) -> str:
    """Copy the annotated demo frame as its own figure file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ANNOTATED.exists():
        raise SystemExit(f"Missing {ANNOTATED}")
    img = Image.open(ANNOTATED)
    # Reasonable width for single-column LaTeX (~3.4 in at 200 dpi)
    max_w = 680
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.Resampling.LANCZOS)
    img.save(out, quality=92)
    return str(out)


def save_fig_trajectories(out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    if TRAJECTORIES.exists():
        shutil.copy2(TRAJECTORIES, out)
        return str(out)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.text(0.5, 0.5, "Run pipeline to generate trajectories", ha="center",
                transform=ax.transAxes)
        return _save(fig, out)


def main() -> None:
    if not EVAL_JSON.exists():
        raise SystemExit(f"Missing {EVAL_JSON}")
    with EVAL_JSON.open(encoding="utf-8") as f:
        agg = json.load(f)

    outputs = {
        "fig_pipeline.png": save_fig_pipeline(FIG_DIR / "fig_pipeline.png"),
        "fig_accuracy.png": save_fig_accuracy(agg, FIG_DIR / "fig_accuracy.png"),
        "fig_ablations.png": save_fig_ablations(FIG_DIR / "fig_ablations.png"),
        "fig_confusion_matrix.png": save_fig_confusion(
            agg, FIG_DIR / "fig_confusion_matrix.png"),
        "fig_uncertain_reasons.png": save_fig_uncertain_reasons(
            FIG_DIR / "fig_uncertain_reasons.png"),
        "fig_hazard_overlay.jpg": save_fig_hazard_overlay(
            FIG_DIR / "fig_hazard_overlay.jpg"),
        "fig_trajectories.png": save_fig_trajectories(
            FIG_DIR / "fig_trajectories.png"),
    }
    for name, path in outputs.items():
        print(f"Wrote {name} -> {path}")


if __name__ == "__main__":
    main()
