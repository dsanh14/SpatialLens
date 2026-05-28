"""Generate the 4-panel failure-case figure for the final report.

Each panel shows one wrong prediction: the cropped frame around the
bbox, the bbox, the true vs predicted labels, and a short snippet of
the classifier's evidence string. The panels are picked to span the
two error modes that survived the final system:

    (a) Trajectory ambiguity   (IMG_4973/person_2, person_4)
    (b) 1-frame static / parked (IMG_4982/bicycle_1, bicycle_2)

Writes reports/final_report/figures/failure_cases.png.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = "data/labels/hazard_labels_dedup.csv"
HAZARDS_DIR = Path("outputs/hazards")
TRACKS_DIR = Path("outputs/tracks")
OUT = Path("reports/final_report/figures/failure_cases.png")

# Four manually selected failure panels (the only 4 errors against
# the consolidated label set, picked to span both surviving error
# modes). Order: trajectory cases first, then 1-frame static.
PANELS = [
    ("IMG_4973", "person_2", "moving_away $\\to$ uncertain",
     "9-frame track, area oscillating $\\sim$1100-1600 px$^2$ ",
     "(no clear shrinkage, low_signal abstention)"),
    ("IMG_4973", "person_4", "moving_away $\\to$ approaching",
     "11-frame track, monotone bbox growth (start->end +2.27)",
     "(reversal rule needs peak/end $>$1.5; here 1.17)"),
    ("IMG_4982", "bicycle_1", "static $\\to$ uncertain",
     "1-frame detection of parked bicycle near camera",
     "(short_track abstain: parked vs moving cannot be told)"),
    ("IMG_4982", "bicycle_2", "static $\\to$ uncertain",
     "1-frame detection of parked bicycle (smaller, mid-frame)",
     "(same root cause as panel (c))"),
]


def _load_pred(vid: str, tid: str) -> dict:
    df = pd.read_csv(HAZARDS_DIR / f"{vid}_hazards.csv")
    row = df[df["track_id"] == tid].iloc[0]
    return row.to_dict()


def _load_bbox(vid: str, tid: str) -> tuple:
    df = pd.read_csv(TRACKS_DIR / f"{vid}_tracks.csv")
    sub = df[df["track_id"] == tid].sort_values("frame_idx")
    middle = sub.iloc[len(sub) // 2]
    return (
        str(middle["frame_path"]),
        float(middle["x1"]), float(middle["y1"]),
        float(middle["x2"]), float(middle["y2"]),
    )


def _crop_around_bbox(img, x1, y1, x2, y2, pad_frac: float = 0.6):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    return img[cy1:cy2, cx1:cx2], (cx1, cy1)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.6))
    fig.subplots_adjust(wspace=0.05, hspace=0.32)
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (vid, tid, title, line1, line2), letter in zip(
        axes.flatten(), PANELS, panel_letters
    ):
        frame_path, x1, y1, x2, y2 = _load_bbox(vid, tid)
        img = cv2.imread(frame_path)
        if img is None:
            ax.set_title(f"{letter} {vid}/{tid} (frame missing)")
            ax.axis("off")
            continue
        crop, (cx1, cy1) = _crop_around_bbox(img, x1, y1, x2, y2, pad_frac=0.7)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        # Draw the bbox in crop coordinates.
        bx1, by1 = x1 - cx1, y1 - cy1
        bx2, by2 = x2 - cx1, y2 - cy1
        ax.add_patch(plt.Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            edgecolor="red", linewidth=2.2, fill=False,
        ))
        ax.set_title(f"{letter} {vid}/{tid}: {title}",
                     fontsize=9, pad=2)
        ax.text(0.0, -0.06, line1, transform=ax.transAxes,
                fontsize=7.5, va="top", family="monospace")
        ax.text(0.0, -0.13, line2, transform=ax.transAxes,
                fontsize=7.5, va="top", family="monospace",
                color="dimgray")
        ax.axis("off")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
