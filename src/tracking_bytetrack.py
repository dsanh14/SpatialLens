"""ByteTrack backend for SpatialLens (opt-in, via `supervision.ByteTrack`).

This module is an *alternative* tracker we can swap in via the
``tracking.backend: "bytetrack"`` config flag. The default backend
(:mod:`src.tracking`) is the from-scratch IoU + centroid matcher used
throughout Weeks 1-2; we keep that as the default so the project's
"what I built" story stays intact.

When ByteTrack is enabled, it replaces the IoU+centroid matching step
only — detection and all downstream Week 2/3 stages (motion features,
hazard classification, alerts, evaluation) are unchanged.

Tracker semantics that differ from the IoU+centroid backend:

* ByteTrack ignores detections that never get associated with a
  confirmed track (e.g. one-off low-confidence flickers). Those rows
  are dropped from the output rather than getting a fresh track_id,
  which is the standard ByteTrack behavior and tends to give cleaner
  per-track motion features. (The IoU+centroid tracker, by contrast,
  starts a new track for every unmatched detection.)
* IDs are global integers in ByteTrack's native form; we relabel them
  to the same ``{class}_{n}`` string convention used by the IoU+centroid
  tracker so that downstream code (and slide assets) doesn't need any
  changes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .tracking import TRACK_OUTPUT_COLUMNS
from .utils import ensure_dir


def _import_bytetrack():
    """Import supervision.ByteTrack lazily with a clear error message."""
    try:
        import supervision as sv  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ByteTrack backend requires the 'supervision' package. "
            "Install it with: pip install supervision  "
            "(or `pip install -r requirements.txt`)."
        ) from e
    return sv


def assign_tracks_bytetrack(
    detections_df: pd.DataFrame,
    image_width: int,
    image_height: int,
    config: dict,
    output_dir: Optional[str | Path] = None,
    video_id: Optional[str] = None,
) -> pd.DataFrame:
    """Assign track IDs using `supervision.ByteTrack`.

    Drop-in replacement for :func:`src.tracking.assign_tracks` — same
    input/output schema, same CSV write behavior. See the module
    docstring for the semantic differences.
    """
    sv = _import_bytetrack()

    cfg = config["tracking"]
    bt_cfg = cfg.get("bytetrack", {}) or {}

    # Suppress supervision's deprecation FutureWarning (it's still
    # functional through v0.30; we pin <0.30 in requirements.txt).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        tracker = sv.ByteTrack(
            track_activation_threshold=float(
                bt_cfg.get("track_activation_threshold", 0.25)),
            lost_track_buffer=int(
                bt_cfg.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(
                bt_cfg.get("minimum_matching_threshold", 0.8)),
            frame_rate=int(bt_cfg.get("frame_rate", 30)),
        )

    if detections_df.empty:
        empty = pd.DataFrame(
            {c: pd.Series(dtype="object") for c in TRACK_OUTPUT_COLUMNS}
        )
        if output_dir and video_id:
            out = ensure_dir(output_dir) / f"{video_id}_tracks.csv"
            empty.to_csv(out, index=False)
        return empty

    df = detections_df.copy().sort_values("frame_idx").reset_index(drop=True)

    # ByteTrack uses integer class_ids; build a stable string<->int map.
    classes = sorted(df["class_name"].unique())
    cls_to_id = {c: i for i, c in enumerate(classes)}

    # ByteTrack assigns global integer IDs (e.g. 1, 2, 3, ...). We
    # relabel them to per-class strings ({class}_{n}) so the output
    # schema matches the IoU+centroid backend. The mapping is stable
    # over the whole video because we accumulate it as we go.
    bt_id_to_label: Dict[int, str] = {}
    per_class_counter: Dict[str, int] = {}
    track_age: Dict[str, int] = {}

    accepted_rows: list = []  # list of (orig_index, track_id_str, age)

    for frame_idx, frame_group in df.groupby("frame_idx", sort=True):
        xyxy = frame_group[["x1", "y1", "x2", "y2"]].values.astype(float)
        conf = frame_group["confidence"].values.astype(float)
        cls_int = frame_group["class_name"].map(cls_to_id).values.astype(int)

        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out = tracker.update_with_detections(det)

        if len(out) == 0:
            continue

        # Match each output row back to its input row by exact xyxy
        # coordinates (supervision preserves coordinate values).
        in_xyxy = xyxy
        for j in range(len(out)):
            bt_id = int(out.tracker_id[j])
            out_box = out.xyxy[j]
            # Find the input row in this frame whose coords match.
            match = np.where(np.all(in_xyxy == out_box, axis=1))[0]
            if len(match) == 0:
                # Should not happen with the current supervision
                # implementation, but stay defensive.
                continue
            orig_pos = int(match[0])
            orig_idx = int(frame_group.index[orig_pos])

            if bt_id not in bt_id_to_label:
                cls = str(frame_group.iloc[orig_pos]["class_name"])
                per_class_counter[cls] = per_class_counter.get(cls, 0) + 1
                tid_str = f"{cls}_{per_class_counter[cls]}"
                bt_id_to_label[bt_id] = tid_str
                track_age[tid_str] = 0

            tid_str = bt_id_to_label[bt_id]
            track_age[tid_str] += 1
            accepted_rows.append((orig_idx, tid_str, track_age[tid_str]))

    if not accepted_rows:
        # ByteTrack rejected every detection — return an empty frame
        # in the expected schema rather than crashing downstream.
        out_df = df.iloc[0:0].copy()
        out_df["track_id"] = pd.Series(dtype=str)
        out_df["track_age"] = pd.Series(dtype=int)
        out_df["matched_score"] = pd.Series(dtype=float)
        out_df = out_df[[c for c in TRACK_OUTPUT_COLUMNS if c in out_df.columns]]
        if output_dir and video_id:
            out = ensure_dir(output_dir) / f"{video_id}_tracks.csv"
            out_df.to_csv(out, index=False)
            print(f"[tracking-bytetrack] wrote 0 track rows -> {out}")
        return out_df

    accepted_df = pd.DataFrame(
        accepted_rows, columns=["__orig_idx", "track_id", "track_age"]
    )
    kept = df.loc[accepted_df["__orig_idx"].values].reset_index(drop=True)
    kept["track_id"] = accepted_df["track_id"].values
    kept["track_age"] = accepted_df["track_age"].values
    # ByteTrack doesn't surface a single match score; use detection
    # confidence as a reasonable per-row proxy for the existing column.
    kept["matched_score"] = kept["confidence"].astype(float).values

    kept = kept[[c for c in TRACK_OUTPUT_COLUMNS if c in kept.columns]]

    if output_dir and video_id:
        out = ensure_dir(output_dir) / f"{video_id}_tracks.csv"
        kept.to_csv(out, index=False)
        print(f"[tracking-bytetrack] wrote {len(kept)} track rows "
              f"({len(bt_id_to_label)} unique tracks) -> {out}")

    return kept
