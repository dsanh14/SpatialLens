"""Appearance-based track re-identification (v3).

The IoU+centroid tracker in :mod:`src.tracking` fragments objects whose
detections cover large pixel distances between consecutive sampled
frames (fast-moving objects on 2 fps footage) or briefly disappear from
YOLO. This module is a *post-tracker* pass that runs two cleanup steps:

1. **Same-frame NMS** — when YOLO emits multiple overlapping same-class
   detections inside the same frame (one bike → three boxes), drop the
   lower-confidence boxes whose IoU with a kept box exceeds a threshold.

2. **Conservative appearance stitching** — merge fragmented tracks using
   an HSV color histogram as the re-ID signal. The gates are deliberately
   tight: at least one of the two tracks must be a single-frame fragment,
   and the merge must form either a contiguous prefix/suffix (one track
   strictly precedes the other) of an existing track. Both spatial
   distance and appearance similarity gates apply on top.

The result is a per-frame tracks DataFrame with the same schema, where
fragments that pass all gates have been relabeled to share the
earliest-starting track's ``track_id``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .tracking import bbox_iou
from .utils import image_diagonal


# ---------- same-frame NMS ----------------------------------------------- #

def _same_frame_nms(
    df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Drop overlapping same-class detections within each frame."""
    if df.empty or "confidence" not in df.columns:
        return df
    keep_idx: List = []
    for _, frame_group in df.groupby("frame_idx", sort=True):
        for _cls, cls_group in frame_group.groupby("class_name"):
            ordered = cls_group.sort_values("confidence", ascending=False)
            kept_boxes: List[Tuple[float, float, float, float]] = []
            for idx, row in ordered.iterrows():
                box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
                if any(bbox_iou(box, kb) >= iou_threshold for kb in kept_boxes):
                    continue
                kept_boxes.append(box)
                keep_idx.append(idx)
    return df.loc[sorted(keep_idx)].reset_index(drop=True)


# ---------- appearance descriptor ---------------------------------------- #

def _hsv_hist(crop: np.ndarray) -> Optional[np.ndarray]:
    """8x8 (H,S) histogram for a bbox crop, min-max normalized."""
    if crop is None or crop.size == 0:
        return None
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _crop_bbox(image: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    h, w = image.shape[:2]
    xi1 = max(0, min(w - 1, int(round(x1))))
    yi1 = max(0, min(h - 1, int(round(y1))))
    xi2 = max(xi1 + 1, min(w, int(round(x2))))
    yi2 = max(yi1 + 1, min(h, int(round(y2))))
    return image[yi1:yi2, xi1:xi2]


def _appearance_similarity(h1, h2) -> float:
    """Pearson correlation between two histograms, clipped to [0, 1]."""
    if h1 is None or h2 is None:
        return 0.0
    return max(0.0, float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)))


# ---------- conservative stitching --------------------------------------- #

def _stitch_fragments(
    tracks_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    max_gap_frames: int = 2,
    max_dist_frac: float = 0.45,
    max_area_ratio: float = 5.0,
    appearance_threshold: float = 0.45,
    fragment_threshold: int = 1,
) -> pd.DataFrame:
    """Merge fragmented same-class tracks using appearance.

    Only allows a merge when **at least one** of the two candidate tracks
    has length <= ``fragment_threshold`` (default 1 frame). This protects
    every track of length >= 2 from being absorbed by another long
    track. Combined with the appearance gate this keeps stitching from
    overwriting any track that the 2-frame salvage rules have already
    classified.

    The candidate B must start strictly *after* A's last frame within
    ``max_gap_frames``, must be within ``max_dist_frac * diagonal`` of
    A's last centroid, and bbox-area ratio must not exceed
    ``max_area_ratio``. Appearance correlation must clear
    ``appearance_threshold``.
    """
    if tracks_df is None or tracks_df.empty:
        return tracks_df
    if "frame_path" not in tracks_df.columns:
        return tracks_df

    img_diag = image_diagonal(image_width, image_height)
    max_dist = max_dist_frac * img_diag

    frame_cache: Dict[str, np.ndarray] = {}
    desc_cache: Dict[int, Optional[np.ndarray]] = {}

    def descriptor_for(row_idx: int, row: pd.Series) -> Optional[np.ndarray]:
        if row_idx in desc_cache:
            return desc_cache[row_idx]
        fp = str(row["frame_path"])
        img = frame_cache.get(fp)
        if img is None:
            img = cv2.imread(fp)
            if img is None:
                desc_cache[row_idx] = None
                return None
            frame_cache[fp] = img
        crop = _crop_bbox(img, row["x1"], row["y1"], row["x2"], row["y2"])
        desc = _hsv_hist(crop)
        desc_cache[row_idx] = desc
        return desc

    work = tracks_df.copy().reset_index(drop=True)

    while True:
        summaries: Dict[str, Dict] = {}
        for tid, g in work.groupby("track_id"):
            g_sorted = g.sort_values("frame_idx")
            first = g_sorted.iloc[0]
            last = g_sorted.iloc[-1]
            summaries[str(tid)] = {
                "class_name": str(first["class_name"]),
                "num_frames": int(len(g_sorted)),
                "first_frame": int(first["frame_idx"]),
                "last_frame": int(last["frame_idx"]),
                "first_cx": float(first["cx"]),
                "first_cy": float(first["cy"]),
                "last_cx": float(last["cx"]),
                "last_cy": float(last["cy"]),
                "first_area": float(first["area"]),
                "last_area": float(last["area"]),
                "first_row_idx": int(g_sorted.index[0]),
                "last_row_idx": int(g_sorted.index[-1]),
                "first_row": first,
                "last_row": last,
            }

        merged_any = False
        ordered_tids = sorted(
            summaries.keys(),
            key=lambda t: (summaries[t]["first_frame"], t),
        )

        for a_tid in ordered_tids:
            if a_tid not in summaries:
                continue
            a = summaries[a_tid]
            a_desc = descriptor_for(a["last_row_idx"], a["last_row"])

            best: Optional[Tuple[float, str]] = None
            for b_tid, b in summaries.items():
                if b_tid == a_tid:
                    continue
                if b["class_name"] != a["class_name"]:
                    continue
                # Both tracks must be <= fragment_threshold frames long.
                # The 2-frame salvage rules in the classifier already make
                # confident calls on 2-frame tracks, so absorbing a 2f
                # track (or being absorbed by one) risks turning a correct
                # salvage call into a wrong rule-cascade call.
                if (a["num_frames"] > fragment_threshold
                        or b["num_frames"] > fragment_threshold):
                    continue
                gap = b["first_frame"] - a["last_frame"]
                if gap < 1 or gap > max_gap_frames:
                    continue
                dist = math.hypot(
                    b["first_cx"] - a["last_cx"],
                    b["first_cy"] - a["last_cy"],
                )
                if dist > max_dist:
                    continue
                ratio = max(
                    b["first_area"] / max(a["last_area"], 1.0),
                    a["last_area"] / max(b["first_area"], 1.0),
                )
                if ratio > max_area_ratio:
                    continue
                b_desc = descriptor_for(b["first_row_idx"], b["first_row"])
                sim = _appearance_similarity(a_desc, b_desc)
                if sim < appearance_threshold:
                    continue
                score = -sim * 1000.0 + gap * 10.0 + dist / max(img_diag, 1.0)
                if best is None or score < best[0]:
                    best = (score, b_tid)

            if best is not None:
                _, b_tid = best
                work.loc[work["track_id"] == b_tid, "track_id"] = a_tid
                merged_any = True
                break
        if not merged_any:
            break
    return work.reset_index(drop=True)


def clean_tracks(
    tracks_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    nms_iou: float = 0.5,
    stitch_gap: int = 2,
    stitch_dist_frac: float = 0.45,
    stitch_area_ratio: float = 5.0,
    appearance_threshold: float = 0.45,
    fragment_threshold: int = 1,
) -> pd.DataFrame:
    """Apply same-frame NMS + conservative appearance stitching."""
    if tracks_df is None or tracks_df.empty:
        return tracks_df
    deduped = _same_frame_nms(tracks_df, iou_threshold=nms_iou)
    stitched = _stitch_fragments(
        deduped,
        image_width=image_width,
        image_height=image_height,
        max_gap_frames=stitch_gap,
        max_dist_frac=stitch_dist_frac,
        max_area_ratio=stitch_area_ratio,
        appearance_threshold=appearance_threshold,
        fragment_threshold=fragment_threshold,
    )
    return stitched
