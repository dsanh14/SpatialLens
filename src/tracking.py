"""Simple IoU + centroid tracker for Week 2 (with an optional ByteTrack backend).

The default tracker is per-class (a person detection cannot inherit a
bicycle's track_id), and matches each new detection to an existing
active track using a weighted combination of IoU and inverse
normalized centroid distance. It is intentionally light — not
ByteTrack/SORT — so the project's Weeks 1-2 deliverable shows a
from-scratch tracker that the Week 3 hazard classifier consumes.

For ablation comparisons, the public entry point :func:`assign_tracks`
dispatches to :mod:`src.tracking_bytetrack` when
``tracking.backend == "bytetrack"`` in the config. Both backends emit
the same DataFrame schema, so all downstream stages are unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .utils import ensure_dir, image_diagonal

TRACK_OUTPUT_COLUMNS = [
    "video_id",
    "frame_idx",
    "frame_path",
    "class_name",
    "confidence",
    "x1", "y1", "x2", "y2",
    "cx", "cy",
    "area",
    "track_id",
    "track_age",
    "matched_score",
]


def bbox_iou(boxA, boxB) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def centroid_distance(rowA, rowB) -> float:
    """Euclidean distance between two rows that have ``cx`` and ``cy``."""
    return math.hypot(float(rowA["cx"]) - float(rowB["cx"]),
                      float(rowA["cy"]) - float(rowB["cy"]))


def normalize_distance(distance: float, image_diag: float) -> float:
    """Normalize a pixel distance by image diagonal, clipped to [0, 1]."""
    if image_diag <= 0:
        return 0.0
    return float(max(0.0, min(1.0, distance / image_diag)))


class _Track:
    """Internal state of one active track."""

    __slots__ = ("track_id", "class_name", "last_frame_idx",
                 "last_box", "last_cx", "last_cy", "age")

    def __init__(
        self,
        track_id: str,
        class_name: str,
        frame_idx: int,
        box: Tuple[float, float, float, float],
        cx: float,
        cy: float,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.last_frame_idx = frame_idx
        self.last_box = box
        self.last_cx = cx
        self.last_cy = cy
        self.age = 1


def assign_tracks(
    detections_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    config: dict,
    output_dir: Optional[str | Path] = None,
    video_id: Optional[str] = None,
) -> pd.DataFrame:
    """Assign track IDs to detections.

    Dispatches to the IoU+centroid tracker (default) or the ByteTrack
    backend based on ``config["tracking"]["backend"]`` (one of
    ``"iou_centroid"`` or ``"bytetrack"``; defaults to
    ``"iou_centroid"``). Both backends emit the same DataFrame schema.

    Parameters
    ----------
    detections_df:
        Detection rows from :func:`src.detect_objects.run_detection`.
    image_width, image_height:
        Frame dimensions, used to normalize the centroid distance
        (IoU+centroid backend only).
    config:
        Parsed config dict; uses ``tracking.iou_threshold``,
        ``tracking.centroid_distance_threshold_frac``, and
        ``tracking.max_frame_gap`` for the default backend, and
        ``tracking.bytetrack.*`` for the ByteTrack backend.
    output_dir, video_id:
        If both are provided, the result is also written to
        ``{output_dir}/{video_id}_tracks.csv``.

    Returns
    -------
    pandas.DataFrame
        Original detection columns plus ``track_id``, ``track_age``,
        ``matched_score``.
    """
    cfg = config["tracking"]
    backend = str(cfg.get("backend", "iou_centroid")).lower()
    if backend == "bytetrack":
        # Lazy import keeps `supervision` an optional dependency for
        # users who only ever run the default backend.
        from .tracking_bytetrack import assign_tracks_bytetrack
        return assign_tracks_bytetrack(
            detections_df=detections_df,
            image_width=image_width,
            image_height=image_height,
            config=config,
            output_dir=output_dir,
            video_id=video_id,
        )
    if backend != "iou_centroid":
        print(f"[tracking][warn] unknown backend={backend!r}; "
              f"falling back to 'iou_centroid'.")

    iou_thresh = float(cfg["iou_threshold"])
    centroid_frac = float(cfg["centroid_distance_threshold_frac"])
    max_gap = int(cfg["max_frame_gap"])

    img_diag = image_diagonal(image_width, image_height)
    centroid_thresh_px = centroid_frac * img_diag

    if detections_df.empty:
        out_cols = TRACK_OUTPUT_COLUMNS
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in out_cols})
        if output_dir and video_id:
            out = ensure_dir(output_dir) / f"{video_id}_tracks.csv"
            empty.to_csv(out, index=False)
        return empty

    df = detections_df.copy().sort_values(["frame_idx"]).reset_index(drop=True)

    active_tracks: List[_Track] = []
    next_class_counter: Dict[str, int] = {}
    track_ids: List[str] = []
    track_ages: List[int] = []
    scores: List[float] = []

    for frame_idx, frame_group in df.groupby("frame_idx", sort=True):
        # Drop tracks that have been gone for too long.
        active_tracks = [
            t for t in active_tracks
            if (frame_idx - t.last_frame_idx) <= max_gap
        ]

        # Build candidate scores between each (track, detection) pair of the
        # same class, then greedily assign.
        det_indices = list(frame_group.index)
        assignments: Dict[int, Tuple[Optional[_Track], float]] = {
            di: (None, 0.0) for di in det_indices
        }
        candidates: List[Tuple[float, int, _Track]] = []  # (score, det_idx, track)

        for di in det_indices:
            d = df.loc[di]
            d_box = (float(d.x1), float(d.y1), float(d.x2), float(d.y2))
            for trk in active_tracks:
                if trk.class_name != d["class_name"]:
                    continue
                iou = bbox_iou(d_box, trk.last_box)
                dist = math.hypot(
                    float(d["cx"]) - trk.last_cx,
                    float(d["cy"]) - trk.last_cy,
                )
                dist_norm = normalize_distance(dist, img_diag)
                score = 0.7 * iou + 0.3 * (1.0 - dist_norm)
                # Only allow this candidate if either iou or centroid gate passes.
                if iou > iou_thresh or dist < centroid_thresh_px:
                    candidates.append((score, di, trk))

        # Greedy assign in order of best score.
        candidates.sort(key=lambda t: t[0], reverse=True)
        used_tracks: set = set()
        for score, di, trk in candidates:
            if assignments[di][0] is not None:
                continue
            if id(trk) in used_tracks:
                continue
            assignments[di] = (trk, score)
            used_tracks.add(id(trk))

        # Apply assignments, create new tracks for unmatched detections.
        for di in det_indices:
            d = df.loc[di]
            d_box = (float(d.x1), float(d.y1), float(d.x2), float(d.y2))
            trk, score = assignments[di]
            if trk is None:
                cls = str(d["class_name"])
                next_class_counter[cls] = next_class_counter.get(cls, 0) + 1
                tid = f"{cls}_{next_class_counter[cls]}"
                new_trk = _Track(
                    track_id=tid,
                    class_name=cls,
                    frame_idx=int(frame_idx),
                    box=d_box,
                    cx=float(d["cx"]),
                    cy=float(d["cy"]),
                )
                active_tracks.append(new_trk)
                track_ids.append(tid)
                track_ages.append(1)
                scores.append(0.0)
            else:
                trk.last_frame_idx = int(frame_idx)
                trk.last_box = d_box
                trk.last_cx = float(d["cx"])
                trk.last_cy = float(d["cy"])
                trk.age += 1
                track_ids.append(trk.track_id)
                track_ages.append(trk.age)
                scores.append(float(score))

    df = df.copy()
    df["track_id"] = track_ids
    df["track_age"] = track_ages
    df["matched_score"] = scores
    df = df[[c for c in TRACK_OUTPUT_COLUMNS if c in df.columns]]

    if output_dir and video_id:
        out = ensure_dir(output_dir) / f"{video_id}_tracks.csv"
        df.to_csv(out, index=False)
        print(f"[tracking] wrote {len(df)} track rows -> {out}")

    return df


# TODO(Week 3): replace this simple tracker with ByteTrack / SORT if track
# stability becomes a bottleneck for the hazard classifier.
