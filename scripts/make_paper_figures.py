"""Publication-quality result figures for the final report.

Produces:
  - figures/confusion_matrix.png : row-normalized confusion matrix
    (recall heat) with raw counts annotated, clean typography.
  - figures/ablation_bars.png    : horizontal bar chart of the
    isolated accuracy drop when each component is removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager  # noqa: F401

FIG = Path("reports/final_report/figures")
EVAL = Path("outputs/evaluation")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
})

CLASS_SHORT = {
    "approaching": "appr.",
    "crossing_left_to_right": "cross L→R",
    "crossing_right_to_left": "cross R→L",
    "moving_away": "mov.~away",
    "static": "static",
    "uncertain": "uncert.",
}


def confusion_matrix_figure() -> None:
    with open(EVAL / "all_videos_evaluation_summary.json") as f:
        data = json.load(f)
    labels = data["labels"]
    cm = pd.DataFrame(data["confusion_matrix"]).T
    cm = cm.reindex(index=labels, columns=labels).fillna(0).astype(int)
    counts = cm.values
    row_sums = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, np.maximum(row_sums, 1))

    short = [CLASS_SHORT[c].replace("~", " ") for c in labels]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1.0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(short, fontsize=8.5)
    ax.set_xlabel("Predicted label", fontsize=10, labelpad=6)
    ax.set_ylabel("True label", fontsize=10, labelpad=6)

    # Minor grid lines between cells.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            c = counts[i, j]
            if c == 0:
                txt, color = "·", "#bbbbbb"
            else:
                pct = norm[i, j] * 100
                txt = f"{c}\n{pct:.0f}%"
                color = "white" if norm[i, j] > 0.55 else "#1a1a1a"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=color, linespacing=0.95)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (recall)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    ax.set_title("Hazard confusion matrix (11 videos, 51 tracks)",
                 fontsize=10.5, pad=8)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote confusion_matrix.png")


def ablation_bar_figure() -> None:
    df = pd.read_csv(EVAL / "ablations.csv")
    full = df[df["variant"] == "full system"]["raw_acc"].iloc[0]
    # Isolated single-component removals only (skip 'both off').
    rows = [
        ("Appearance re-ID stitching", "-appearance_reid"),
        ("2-frame crossing salvage", "-2f_crossing_salvage"),
        ("2-frame approach salvage", "-2f_approach_salvage"),
        ("Trajectory-reversal rule", "-trajectory_reversal"),
        ("Soft moving-away guard", "-soft_moving_away"),
        ("Slow-approach allowance", "-slow_approach"),
        ("Same-frame NMS", "-same_frame_nms"),
    ]
    names, drops = [], []
    for label, key in rows:
        acc = df[df["variant"] == key]["raw_acc"]
        if len(acc):
            names.append(label)
            drops.append((full - acc.iloc[0]) * 100)
    order = np.argsort(drops)
    names = [names[i] for i in order]
    drops = [drops[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    colors = plt.cm.YlOrBr(np.linspace(0.45, 0.85, len(drops)))
    bars = ax.barh(names, drops, color=colors, edgecolor="#7a4a00",
                   linewidth=0.6, height=0.66)
    for bar, d in zip(bars, drops):
        ax.text(d + 0.12, bar.get_y() + bar.get_height() / 2,
                f"$-${d:.1f}", va="center", ha="left", fontsize=8.5)
    ax.set_xlabel("Accuracy drop when removed (pts, raw labels)",
                  fontsize=9.5)
    ax.set_xlim(0, max(drops) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8.8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Isolated contribution of each component",
                 fontsize=10.5, pad=6)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote ablation_bars.png")


if __name__ == "__main__":
    confusion_matrix_figure()
    ablation_bar_figure()
