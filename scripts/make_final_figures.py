"""Generate all final-report figures and LaTeX tables from existing data.

Run from the repo root:

    python scripts/make_final_figures.py

Outputs land in:

    reports/final_report/figures/final/   *.pdf and *.png
    reports/final_report/tables/          *.tex

Inputs (read-only):
    data/labels/hazard_labels.csv
    data/labels/hazard_labels_dedup.csv
    outputs/tracks/<vid>_track_features.csv
    outputs/tracks/<vid>_tracks.csv
    outputs/hazards/<vid>_hazards.csv
    outputs/evaluation/all_videos_evaluation_summary.json
    outputs/evaluation/ablations.csv
    outputs/evaluation/per_video_breakdown.csv
    data/frames/<vid>/frame_XXXX.jpg
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = ROOT / "data" / "labels" / "hazard_labels.csv"
LABELS_DEDUP_CSV = ROOT / "data" / "labels" / "hazard_labels_dedup.csv"
TRACKS_DIR = ROOT / "outputs" / "tracks"
HAZARDS_DIR = ROOT / "outputs" / "hazards"
EVAL_DIR = ROOT / "outputs" / "evaluation"
FRAMES_ROOT = ROOT / "data" / "frames"

FIG_DIR = ROOT / "reports" / "final_report" / "figures" / "final"
TBL_DIR = ROOT / "reports" / "final_report" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "approaching",
    "crossing_left_to_right",
    "crossing_right_to_left",
    "moving_away",
    "static",
    "uncertain",
]
CLASS_SHORT = {
    "approaching": "approaching",
    "crossing_left_to_right": "cross L→R",
    "crossing_right_to_left": "cross R→L",
    "moving_away": "moving away",
    "static": "static",
    "uncertain": "uncertain",
}
CLASS_COLOR = {
    "approaching": "#d62728",
    "crossing_left_to_right": "#1f77b4",
    "crossing_right_to_left": "#9467bd",
    "moving_away": "#2ca02c",
    "static": "#7f7f7f",
    "uncertain": "#ff7f0e",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 130,
})


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    return pd.read_csv(LABELS_CSV)


def load_tracks() -> pd.DataFrame:
    """Join labels with per-track features + hazards for every video."""
    labels = load_labels()
    feats, hazards = [], []
    for vid in sorted(labels["video_id"].unique()):
        f = TRACKS_DIR / f"{vid}_track_features.csv"
        h = HAZARDS_DIR / f"{vid}_hazards.csv"
        if f.exists():
            feats.append(pd.read_csv(f))
        if h.exists():
            hazards.append(pd.read_csv(h))
    feats = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    hazards = pd.concat(hazards, ignore_index=True) if hazards else pd.DataFrame()

    df = labels.merge(
        feats, on=["video_id", "track_id"], how="left",
        suffixes=("", "_feat"),
    )
    df = df.merge(
        hazards[[
            "video_id", "track_id", "hazard_label", "uncertain_reason",
            "approach_score", "crossing_score", "image_width", "image_height",
            "evidence",
        ]],
        on=["video_id", "track_id"], how="left",
    )
    # Signed normalised horizontal displacement (dx_norm) - used by rule cascade
    diag = np.sqrt(df["image_width"].fillna(960) ** 2
                   + df["image_height"].fillna(1707) ** 2)
    df["dx_norm"] = df["dx_total"].fillna(0) / diag
    df["correct"] = df["true_label"] == df["hazard_label"]
    return df


def load_eval() -> dict:
    with open(EVAL_DIR / "all_videos_evaluation_summary.json") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Figure: feature-space scatter (dx_norm vs bbox_growth_ratio)
# --------------------------------------------------------------------------
def plot_feature_space(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["dx_norm", "bbox_growth_ratio"]).copy()
    # Clip extreme growth values for plot readability (a couple of tracks
    # grow >10x; we annotate that in the caption rather than warp the axes).
    growth_clip = sub["bbox_growth_ratio"].clip(lower=-1.2, upper=4.0)
    dx_clip = sub["dx_norm"].clip(lower=-0.9, upper=0.9)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for cls in CLASSES:
        m = sub["true_label"] == cls
        if not m.any():
            continue
        # correct = filled, wrong = hollow ring
        correct = m & sub["correct"]
        wrong = m & ~sub["correct"]
        if correct.any():
            ax.scatter(
                dx_clip[correct], growth_clip[correct],
                s=55, color=CLASS_COLOR[cls], alpha=0.85,
                edgecolor="#222", linewidth=0.5,
                label=f"{CLASS_SHORT[cls]} ({int(m.sum())})",
            )
        if wrong.any():
            ax.scatter(
                dx_clip[wrong], growth_clip[wrong],
                s=70, facecolor="white", edgecolor=CLASS_COLOR[cls],
                linewidth=1.8, marker="o",
            )

    # Rule-cascade thresholds (signed dx_norm and growth)
    for x in (-0.50, -0.08, 0.08, 0.50):
        ax.axvline(x, color="#888", linestyle="--", linewidth=0.7, zorder=0)
    for y in (-0.15, 0.15):
        ax.axhline(y, color="#888", linestyle="--", linewidth=0.7, zorder=0)

    ax.text(-0.88, -1.1, "← strong cross R→L  |  cross R→L  |  approaching / moving-away  |  cross L→R  |  strong cross L→R →",
            fontsize=6.8, color="#555", ha="left", va="bottom")
    ax.text(0.87, 0.18, "$r > 0.15$: approaching", fontsize=7.0,
            color="#555", ha="right", va="bottom")
    ax.text(0.87, -0.17, "$r < -0.15$: moving away", fontsize=7.0,
            color="#555", ha="right", va="top")

    ax.axhline(0, color="#bbb", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#bbb", linewidth=0.5, zorder=0)
    ax.set_xlabel("Signed normalized horizontal displacement  "
                  r"$dx_\mathrm{norm}$")
    ax.set_ylabel(r"Bounding-box growth ratio  $r$")
    ax.set_title("Motion feature space coloured by ground-truth label")
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-1.2, 4.2)

    # Two-column legend below plot
    leg = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
        frameon=False, handletextpad=0.4, columnspacing=1.2,
    )
    # Note about hollow markers
    ax.text(
        0.02, 0.97,
        "filled = correct prediction\nhollow = misclassified",
        transform=ax.transAxes, fontsize=7.8, va="top",
        bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.3"),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    out = FIG_DIR / "feature_space_scatter.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure: confusion matrix (counts + recall heat)
# --------------------------------------------------------------------------
def plot_confusion_matrix(eval_summary: dict) -> None:
    labels = eval_summary["labels"]
    cm = pd.DataFrame(eval_summary["confusion_matrix"]).T
    cm = cm.reindex(index=labels, columns=labels).fillna(0).astype(int)
    counts = cm.values
    rowsum = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, np.maximum(rowsum, 1))

    short = [CLASS_SHORT[c] for c in labels]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label", labelpad=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            c = counts[i, j]
            if c == 0:
                ax.text(j, i, ".", ha="center", va="center",
                        color="#bbb", fontsize=10)
            else:
                color = "white" if norm[i, j] > 0.55 else "#1a1a1a"
                ax.text(j, i, f"{c}\n{norm[i, j]*100:.0f}%",
                        ha="center", va="center", fontsize=9.2,
                        color=color, linespacing=0.95)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized recall")
    ax.set_title("Confusion matrix — 57 evaluated tracks")
    fig.tight_layout()
    out = FIG_DIR / "confusion_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure: ablation study (overall + decidable bars)
# --------------------------------------------------------------------------
def plot_ablation() -> None:
    df = pd.read_csv(EVAL_DIR / "ablations.csv")
    full_raw = df.loc[df.variant == "full system", "raw_acc"].iloc[0]
    full_ded = df.loc[df.variant == "full system", "dedup_acc"].iloc[0]

    rows = [
        ("Appearance re-ID stitching", "-appearance_reid"),
        ("2-frame crossing salvage",   "-2f_crossing_salvage"),
        ("2-frame approach salvage",   "-2f_approach_salvage"),
        ("Trajectory-reversal rule",   "-trajectory_reversal"),
        ("Soft moving-away guard",     "-soft_moving_away"),
        ("Slow-approach allowance",    "-slow_approach"),
        ("Same-frame NMS",             "-same_frame_nms"),
        ("NMS + re-ID both off",       "-nms -reid (both off)"),
    ]
    names, drops_raw, drops_ded = [], [], []
    for label, key in rows:
        r = df[df.variant == key]
        if r.empty:
            continue
        names.append(label)
        drops_raw.append((full_raw - r.raw_acc.iloc[0]) * 100)
        drops_ded.append((full_ded - r.dedup_acc.iloc[0]) * 100)
    # Sort by raw drop ascending so largest bar at top
    order = np.argsort(drops_raw)
    names = [names[i] for i in order]
    drops_raw = [drops_raw[i] for i in order]
    drops_ded = [drops_ded[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    y = np.arange(len(names))
    h = 0.38
    ax.barh(y - h / 2, drops_raw, h, color="#d97a3a",
            edgecolor="#7a4a00", linewidth=0.6,
            label="raw labels (n=57)")
    ax.barh(y + h / 2, drops_ded, h, color="#5a8cb0",
            edgecolor="#22506e", linewidth=0.6,
            label="consolidated (n=51)")
    for yi, d in zip(y - h / 2, drops_raw):
        ax.text(d + 0.12, yi, f"−{d:.1f}",
                va="center", fontsize=8.2)
    for yi, d in zip(y + h / 2, drops_ded):
        ax.text(d + 0.12, yi, f"−{d:.1f}",
                va="center", fontsize=8.2, color="#22506e")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Accuracy drop when removed (percentage points)")
    ax.set_xlim(0, max(drops_raw + drops_ded) * 1.18)
    ax.set_title("Ablation: isolated contribution of each component "
                 f"(full system = {full_raw*100:.1f}% raw / "
                 f"{full_ded*100:.1f}% consolidated)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "ablation_study.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure: class distribution
# --------------------------------------------------------------------------
def plot_class_distribution(df_labels: pd.DataFrame) -> None:
    counts = df_labels["true_label"].value_counts().reindex(CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = [CLASS_COLOR[c] for c in CLASSES]
    bars = ax.bar([CLASS_SHORT[c] for c in CLASSES], counts.values,
                  color=colors, edgecolor="#333", linewidth=0.6)
    for bar, c in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.4,
                str(int(c)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# ground-truth tracks")
    ax.set_title("Ground-truth class distribution (61 manually labeled tracks)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    out = FIG_DIR / "class_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure: uncertainty breakdown by sub-reason
# --------------------------------------------------------------------------
def plot_uncertainty_breakdown(df: pd.DataFrame) -> None:
    pred_uncert = df[df["hazard_label"] == "uncertain"].copy()
    reasons = pred_uncert["uncertain_reason"].fillna("other").value_counts()
    # Group sub-reasons into a small taxonomy
    pretty = {
        "short_track": "short track\n(<3 frames)",
        "low_signal":  "low motion\nsignal",
        "near_approaching": "near approaching\nthreshold",
        "ambiguous_crossing": "ambiguous\ncrossing",
        "other": "other",
    }
    keys = list(reasons.index)
    vals = list(reasons.values)
    labels = [pretty.get(k, k.replace("_", " ")) for k in keys]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.85, len(vals)))
    bars = ax.bar(labels, vals, color=colors, edgecolor="#7a4a00",
                  linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(int(v)),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# uncertain predictions")
    ax.set_title("Why the system abstains: uncertain-reason breakdown")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18 if vals else 1)
    fig.tight_layout()
    out = FIG_DIR / "uncertainty_breakdown.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure: temporal motion strip for an approaching example
# --------------------------------------------------------------------------
def _draw_bbox_on(ax, img_bgr, x1, y1, x2, y2, color, label, lw=3.0):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               edgecolor=color, linewidth=lw, fill=False))
    if label:
        ax.text(x1, y1 - 8, label, color="white", fontsize=8,
                bbox=dict(facecolor=color, edgecolor="none", pad=2,
                          boxstyle="round,pad=0.2"))
    ax.axis("off")


def plot_temporal_strip(df: pd.DataFrame) -> None:
    """Pick an approaching track that exists in tracks.csv with >=4 frames."""
    target_vid, target_tid = "IMG_4972", "person_1"  # 15-frame approaching person
    tracks = pd.read_csv(TRACKS_DIR / f"{target_vid}_tracks.csv")
    sub = tracks[tracks["track_id"] == target_tid].sort_values("frame_idx")
    if sub.empty:
        print(f"warn: temporal strip skipped, {target_vid}/{target_tid} not in tracks")
        return
    n = len(sub)
    pick_idx = np.linspace(0, n - 1, 4).round().astype(int)
    chosen = sub.iloc[pick_idx]

    feats = pd.read_csv(TRACKS_DIR / f"{target_vid}_track_features.csv")
    fr = feats[feats["track_id"] == target_tid].iloc[0]
    hazards = pd.read_csv(HAZARDS_DIR / f"{target_vid}_hazards.csv")
    hz = hazards[hazards["track_id"] == target_tid].iloc[0]

    # Centroid trail across the full track (drawn on all panels)
    trail = sub[["cx", "cy"]].values
    color = CLASS_COLOR["approaching"]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.8))
    for ax, (_, row) in zip(axes, chosen.iterrows()):
        img = cv2.imread(str(ROOT / row["frame_path"]))
        if img is None:
            ax.set_title("(frame missing)")
            ax.axis("off"); continue
        # Crop to roughly bbox + context
        h, w = img.shape[:2]
        margin_x, margin_y = int(w * 0.05), int(h * 0.05)
        crop = img[margin_y:h - margin_y, margin_x:w - margin_x]
        scale_x = crop.shape[1] / w
        scale_y = crop.shape[0] / h
        # Coordinates in crop space
        bx1 = (row["x1"] - margin_x)
        by1 = (row["y1"] - margin_y)
        bx2 = (row["x2"] - margin_x)
        by2 = (row["y2"] - margin_y)
        _draw_bbox_on(ax, crop, bx1, by1, bx2, by2, color,
                      label=f"{target_tid}  area={int(row['area'])}")
        # Trail (already in original coords) -> shift to crop coords
        tx = trail[:, 0] - margin_x
        ty = trail[:, 1] - margin_y
        ax.plot(tx, ty, color="yellow", linewidth=1.6, alpha=0.85)
        ax.scatter(tx[0], ty[0], color="white", s=22, zorder=3,
                   edgecolor="black", linewidth=0.6)
        ax.scatter(row["cx"] - margin_x, row["cy"] - margin_y,
                   color=color, s=46, zorder=4,
                   edgecolor="white", linewidth=0.9)
        # Arrow from start to current centroid
        ax.annotate(
            "", xy=(row["cx"] - margin_x, row["cy"] - margin_y),
            xytext=(trail[0, 0] - margin_x, trail[0, 1] - margin_y),
            arrowprops=dict(arrowstyle="->", color="yellow",
                            lw=1.2, alpha=0.9),
        )
        ax.set_title(f"frame {int(row['frame_idx'])}", fontsize=10)

    growth = float(fr["bbox_growth_ratio"])
    dx_total = float(fr["dx_total"])
    diag = math.hypot(hz["image_width"], hz["image_height"])
    dx_norm = dx_total / diag
    flow_mag = float(fr["avg_flow_mag"])
    evidence = (
        f"Evidence — bbox growth $r$ = {growth:+.2f}, "
        f"$dx_\\mathrm{{norm}}$ = {dx_norm:+.2f}, "
        f"flow mag = {flow_mag:.1f}, approach score = {hz['approach_score']:.2f}  "
        f"$\\Rightarrow$  label = {hz['hazard_label'].upper()}"
    )
    fig.suptitle(
        f"{target_vid} / {target_tid}: cyclist approaching the camera "
        f"— {n} frames, ground truth = approaching",
        fontsize=11, y=1.02,
    )
    fig.text(0.5, -0.03, evidence, ha="center", va="top", fontsize=9.5)
    fig.tight_layout()
    out = FIG_DIR / "temporal_strip.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Figure: qualitative 2x2 (approaching / crossing / moving-away / uncertain)
# --------------------------------------------------------------------------
def plot_qualitative_panel(df: pd.DataFrame) -> None:
    # Pick one good representative per panel from the labeled set.
    # Each entry: (video, track, panel_title, panel_subtitle).
    panels = [
        ("IMG_4972", "person_1",
         "Approaching success",
         "15-frame cyclist; growth +21.98, dx_norm = −0.15"),
        ("IMG_4981", "person_3",
         "Crossing R→L success",
         "11-frame pedestrian; large dx_norm, low growth"),
        ("IMG_4972", "person_2",
         "Moving-away success",
         "17-frame pedestrian; r < 0, shrinking bbox"),
        ("IMG_4982", "bicycle_1",
         "Conservative abstention (static)",
         "1-frame parked-bicycle detection → uncertain"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for ax, (vid, tid, title, subtitle), letter in zip(
        axes.flatten(), panels, letters,
    ):
        tracks = pd.read_csv(TRACKS_DIR / f"{vid}_tracks.csv")
        sub = tracks[tracks["track_id"] == tid].sort_values("frame_idx")
        if sub.empty:
            ax.set_title(f"{letter} {vid}/{tid}: not in tracks", fontsize=9)
            ax.axis("off"); continue
        # Use last frame for visual; overlay trail.
        last = sub.iloc[-1]
        img = cv2.imread(str(ROOT / last["frame_path"]))
        if img is None:
            ax.set_title(f"{letter} {vid}/{tid}: frame missing", fontsize=9)
            ax.axis("off"); continue
        # Trim a margin to keep aspect manageable.
        h, w = img.shape[:2]
        mx, my = int(w * 0.04), int(h * 0.04)
        crop = img[my:h - my, mx:w - mx]
        bx1, by1 = last["x1"] - mx, last["y1"] - my
        bx2, by2 = last["x2"] - mx, last["y2"] - my
        hz_df = pd.read_csv(HAZARDS_DIR / f"{vid}_hazards.csv")
        hz_row = hz_df[hz_df["track_id"] == tid]
        if hz_row.empty:
            pred = "unknown"
        else:
            pred = hz_row.iloc[0]["hazard_label"]
        true = df[(df.video_id == vid) & (df.track_id == tid)]
        true_label = true.iloc[0]["true_label"] if not true.empty else "?"
        color = CLASS_COLOR.get(true_label, "#888")
        _draw_bbox_on(ax, crop, bx1, by1, bx2, by2, color,
                      label=f"{tid}", lw=2.6)
        trail = sub[["cx", "cy"]].values
        ax.plot(trail[:, 0] - mx, trail[:, 1] - my,
                color="yellow", linewidth=1.4, alpha=0.85)
        ax.scatter(trail[0, 0] - mx, trail[0, 1] - my,
                   s=22, color="white", edgecolor="black", linewidth=0.6, zorder=3)
        ax.set_title(f"{letter} {title}", fontsize=10, pad=4)
        ax.text(0.0, -0.04, subtitle, transform=ax.transAxes,
                fontsize=8.0, va="top", family="monospace")
        ax.text(0.0, -0.10, f"true = {true_label}   pred = {pred}",
                transform=ax.transAxes, fontsize=8.0, va="top",
                family="monospace", color="dimgray")
    fig.tight_layout()
    out = FIG_DIR / "qualitative_panel.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def write_table(path: Path, body: str) -> None:
    path.write_text(body)
    print(f"wrote {path}")


def table_dataset_summary(df_labels: pd.DataFrame) -> None:
    n_videos = df_labels["video_id"].nunique()
    n_tracks = len(df_labels)
    eval_summary = load_eval()
    n_eval = eval_summary["num_evaluated_tracks"]
    n_dec = eval_summary["selective_accuracy"]["num_decidable_tracks"]
    body = (
        "\\begin{tabular}{lcccp{0.35\\linewidth}}\n"
        "\\toprule\n"
        "Split & Videos & Labeled tracks & Used quantitatively? & Notes \\\\\n"
        "\\midrule\n"
        f"Controlled campus videos    & {n_videos} & {n_tracks} & "
        "Yes & 2\\,fps, 960\\,px wide, hand-held phone \\\\\n"
        f"Matched evaluated tracks    & {n_videos} & {n_eval} & "
        "Yes & after track--label matching \\\\\n"
        f"Decidable subset ($n\\!\\ge\\!3$) & {n_videos} & {n_dec} & "
        "Yes & tracks with $\\geq 3$ frames \\\\\n"
        "Uncontrolled qualitative    & 2 & --- & "
        "No & natural-walk clips, demo only \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    write_table(TBL_DIR / "dataset_summary.tex", body)


def table_class_distribution(df_labels: pd.DataFrame) -> None:
    counts = df_labels["true_label"].value_counts().reindex(CLASSES).fillna(0)
    total = int(counts.sum())
    rows = []
    notes = {
        "approaching": "primary safety-critical class",
        "crossing_left_to_right": "limited support",
        "crossing_right_to_left": "limited support",
        "moving_away": "non-hazard motion",
        "static": r"only 2 examples $\Rightarrow$ high variance",
        "uncertain": "deliberate abstention label",
    }
    for c in CLASSES:
        n = int(counts[c])
        pct = 100 * n / total if total else 0
        rows.append(
            f"\\texttt{{{c.replace('_', '\\_')}}} & {n} & {pct:.1f}\\% & "
            f"{notes[c]} \\\\"
        )
    body = (
        "\\begin{tabular}{lccl}\n\\toprule\n"
        "Class & $n$ & \\% & Note \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\midrule\n"
        f"\\textbf{{Total}} & \\textbf{{{total}}} & 100.0\\% & 11 controlled videos \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    write_table(TBL_DIR / "class_distribution.tex", body)


def table_per_class_metrics() -> None:
    e = load_eval()
    pcm = e["per_class_metrics"]
    rows = []
    for c in CLASSES:
        m = pcm[c]
        rows.append(
            f"\\texttt{{{c.replace('_', '\\_')}}} & {m['precision']:.2f} & "
            f"{m['recall']:.2f} & {m['f1']:.2f} & {m['support']} \\\\"
        )
    decidable = e["selective_accuracy"]["decidable_accuracy"] * 100
    acc = e["overall_accuracy"] * 100
    macro = e["macro_f1"]
    body = (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Class & Precision & Recall & F1 & $n$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\midrule\n"
        f"\\textbf{{Overall accuracy}} & \\multicolumn{{4}}{{c}}{{{acc:.1f}\\% on 57 evaluated tracks}} \\\\\n"
        f"\\textbf{{Decidable accuracy ($n\\!\\geq\\!3$)}} & "
        f"\\multicolumn{{4}}{{c}}{{{decidable:.1f}\\% on 30 tracks}} \\\\\n"
        f"\\textbf{{Macro F1}} & \\multicolumn{{4}}{{c}}{{{macro:.2f}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    write_table(TBL_DIR / "per_class_metrics.tex", body)


def table_ablation() -> None:
    df = pd.read_csv(EVAL_DIR / "ablations.csv")
    pretty = {
        "full system": ("Full system",
                        "all components on"),
        "-appearance_reid": (r"$-$ Appearance re-ID",
                              "no fragment stitching"),
        "-2f_crossing_salvage": (r"$-$ 2-frame crossing salvage",
                                  "no n=2 crossing rule"),
        "-2f_approach_salvage": (r"$-$ 2-frame approach salvage",
                                  "no n=2 approach rule"),
        "-trajectory_reversal": (r"$-$ Trajectory-reversal rule",
                                  "no reversal heuristic"),
        "-soft_moving_away": (r"$-$ Soft moving-away",
                                "tight $r$ threshold"),
        "-slow_approach": (r"$-$ Slow-approach allowance",
                            "no $r{>}0$ slow-approach"),
        "-same_frame_nms": (r"$-$ Same-frame NMS",
                              "no intra-frame dedupe"),
        "-nms -reid (both off)": (r"$-$ NMS and re-ID (both off)",
                                    "fragments preserved"),
    }
    rows = []
    for _, r in df.iterrows():
        key = r["variant"]
        if key not in pretty:
            continue
        name, change = pretty[key]
        rows.append(
            f"{name} & {change} & "
            f"{r['raw_acc']*100:.1f}\\% & {r['dedup_acc']*100:.1f}\\% & "
            f"{r['raw_macro_f1']:.2f} \\\\"
        )
    body = (
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Configuration & Change & Raw acc. & Cons. acc. & Macro F1 \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_table(TBL_DIR / "ablation_study.tex", body)


def table_error_taxonomy() -> None:
    rows = [
        ("Short-track fragmentation",
         "tracker links only 1--2 frames at 2\\,fps",
         "IMG\\_4982 bicycle\\_1",
         "higher fps; learned re-ID embedding"),
        ("Static abstention",
         "only 2 static examples, both 1--2 frame fragments",
         "IMG\\_4982 bicycle\\_2",
         "explicit \\texttt{stationary} branch with depth cue"),
        ("Crossing $\\to$ uncertain",
         "fast crossing object fragments before $|dx_\\mathrm{norm}|$ accumulates",
         "IMG\\_4977 skateboard",
         "denser sampling; salvage at $n{=}1$"),
        ("Moving-away $\\to$ approaching",
         "monotone bbox growth despite physical recession (no depth cue)",
         "IMG\\_4973 person\\_4",
         "monocular depth (MiDaS)"),
        ("Trajectory-reversal miss",
         "person approached then turned; $\\rho{<}1.5$",
         "IMG\\_4973 person\\_2",
         "softer reversal threshold; second-half flow direction"),
    ]
    body = (
        "\\begin{tabular}{p{0.19\\linewidth}p{0.30\\linewidth}p{0.18\\linewidth}p{0.23\\linewidth}}\n"
        "\\toprule\n"
        "Error type & Cause & Example & Proposed fix \\\\\n"
        "\\midrule\n"
        + "\n".join(f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    write_table(TBL_DIR / "error_taxonomy.tex", body)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df_labels = load_labels()
    df = load_tracks()
    eval_summary = load_eval()

    plot_feature_space(df)
    plot_confusion_matrix(eval_summary)
    plot_ablation()
    plot_class_distribution(df_labels)
    plot_uncertainty_breakdown(df)
    plot_temporal_strip(df)
    plot_qualitative_panel(df)

    table_dataset_summary(df_labels)
    table_class_distribution(df_labels)
    table_per_class_metrics()
    table_ablation()
    table_error_taxonomy()


if __name__ == "__main__":
    main()
