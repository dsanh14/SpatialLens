"""Final hazard classification for SpatialLens Assist (Week 3).

Consumes the per-track motion features produced by
:mod:`src.motion_features` and assigns one of six hazard labels per track:

    - approaching
    - crossing_left_to_right
    - crossing_right_to_left
    - moving_away
    - static
    - uncertain

The classifier is intentionally **rule-based and explainable** — no model
is trained. Each hazard row carries an ``evidence`` string that documents
why the label was chosen, which is the main computer vision contribution
for the final report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import ensure_dir

EPSILON = 1e-6

HAZARD_LABELS = (
    "approaching",
    "crossing_left_to_right",
    "crossing_right_to_left",
    "moving_away",
    "static",
    "uncertain",
)

HAZARD_COLUMNS = [
    "video_id",
    "track_id",
    "class_name",
    "hazard_label",
    "approach_score",
    "crossing_score",
    "bbox_growth_ratio",
    "total_displacement_norm",
    "avg_flow_dx",
    "avg_flow_dy",
    "avg_flow_mag",
    "avg_frame_diff_overlap",
    "dx_total",
    "dy_total",
    "start_cx",
    "start_cy",
    "first_frame",
    "last_frame",
    "image_width",
    "image_height",
    "confidence",
    "evidence",
]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Pull a numeric field from a row, tolerating NaN / missing columns."""
    if key not in row:
        return default
    val = row[key]
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


class HazardClassifier:
    """Rule-based hazard classifier with explanation strings.

    Reads thresholds from ``config['hazard']`` and applies them to per-track
    motion features. Per the spec, the classifier prioritizes labels in this
    order:

        1. ``static`` if there's no real motion evidence
        2. ``approaching`` if approach_score >= threshold AND bbox grew
        3. ``crossing_left_to_right`` / ``crossing_right_to_left`` if dx is
           strong horizontally AND bbox did not grow strongly
        4. ``moving_away`` if bbox shrank past the shrink threshold
        5. ``uncertain`` otherwise
    """

    def __init__(self, config: dict) -> None:
        haz = config.get("hazard", {})
        self.approach_threshold = float(haz.get("approach_score_threshold", 0.55))
        self.crossing_frac = float(haz.get("crossing_threshold_frac_width", 0.08))
        self.static_diag_frac = float(haz.get("static_threshold_frac_diagonal", 0.04))
        self.growth_threshold = float(haz.get("bbox_growth_threshold", 0.15))
        self.shrink_threshold = float(haz.get("bbox_shrink_threshold", -0.15))
        self.flow_threshold = float(haz.get("flow_threshold", 1.0))
        self.fd_overlap_threshold = float(
            haz.get("frame_diff_overlap_threshold", 0.08)
        )
        self.center_motion_frac = float(
            haz.get("center_motion_threshold_frac", 0.05)
        )
        self.use_depth = bool(haz.get("use_depth_if_available", False))

    # ------------------------------------------------------------------ #
    # Scoring components — each returns a value in [0, 1].
    # ------------------------------------------------------------------ #
    def _bbox_growth_score(self, growth_ratio: float) -> float:
        if growth_ratio <= 0 or self.growth_threshold <= 0:
            return 0.0
        return _clamp(growth_ratio / self.growth_threshold)

    def _center_motion_score(
        self,
        start_cx: float,
        start_cy: float,
        end_cx: float,
        end_cy: float,
        image_width: Optional[float],
        image_height: Optional[float],
    ) -> float:
        """1.0 if the centroid moved toward the image center or downward.

        Downward motion in image coordinates often correlates with an
        object getting closer in a forward-facing walkway camera (it
        moves into the lower portion of the frame as it nears the camera).
        """
        if image_width is None or image_height is None:
            return 0.0
        cx_center = image_width / 2.0
        cy_lower = image_height * 0.55  # slightly below midline
        moved_toward_cx = abs(end_cx - cx_center) < abs(start_cx - cx_center)
        moved_downward = (end_cy - start_cy) > self.center_motion_frac * image_height
        if moved_toward_cx or moved_downward:
            return 1.0
        return 0.0 if cy_lower < 0 else 0.0

    def _frame_diff_score(self, avg_fd_overlap: float) -> float:
        if self.fd_overlap_threshold <= 0:
            return 0.0
        return _clamp(avg_fd_overlap / self.fd_overlap_threshold)

    def _flow_score(self, avg_flow_mag: float) -> float:
        if self.flow_threshold <= 0:
            return 0.0
        return _clamp(avg_flow_mag / self.flow_threshold)

    # ------------------------------------------------------------------ #
    # Confidence bucketing.
    # ------------------------------------------------------------------ #
    def _confidence(self, label: str, approach_score: float,
                    dx_norm: float, growth_ratio: float,
                    moved: bool) -> str:
        if label == "static":
            return "high" if not moved else "medium"
        if label == "approaching":
            margin = approach_score - self.approach_threshold
            if margin >= 0.15 and growth_ratio >= 1.5 * self.growth_threshold:
                return "high"
            if margin >= 0.05:
                return "medium"
            return "low"
        if label in ("crossing_left_to_right", "crossing_right_to_left"):
            ratio = abs(dx_norm) / max(self.crossing_frac, EPSILON)
            if ratio >= 2.0:
                return "high"
            if ratio >= 1.25:
                return "medium"
            return "low"
        if label == "moving_away":
            if growth_ratio <= 1.5 * self.shrink_threshold:
                return "high"
            return "medium"
        return "low"  # uncertain

    # ------------------------------------------------------------------ #
    # Per-row classification.
    # ------------------------------------------------------------------ #
    def classify_track(
        self,
        row: pd.Series,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Classify one row from the track-features dataframe.

        Returns a dict with all columns listed in :data:`HAZARD_COLUMNS`.
        ``image_width`` / ``image_height``, when provided, enable
        normalized dx and the center-motion score. When omitted, the
        classifier falls back to absolute dx in pixels — labels are still
        produced, but crossing tests degrade gracefully.
        """
        growth_ratio = _safe_float(row, "bbox_growth_ratio")
        total_disp_norm = _safe_float(row, "total_displacement_norm")
        avg_flow_mag = _safe_float(row, "avg_flow_mag")
        avg_fd_overlap = _safe_float(row, "avg_frame_diff_overlap")
        dx_total = _safe_float(row, "dx_total")
        dy_total = _safe_float(row, "dy_total")
        avg_flow_dx = _safe_float(row, "avg_flow_dx")
        avg_flow_dy = _safe_float(row, "avg_flow_dy")
        start_cx = _safe_float(row, "start_cx")
        start_cy = _safe_float(row, "start_cy")
        end_cx = _safe_float(row, "end_cx")
        end_cy = _safe_float(row, "end_cy")

        moved = (
            total_disp_norm > self.static_diag_frac
            or avg_flow_mag > self.flow_threshold
            or avg_fd_overlap > self.fd_overlap_threshold
        )

        # Normalize horizontal displacement against width when available.
        if image_width and image_width > 0:
            dx_norm = dx_total / float(image_width)
        else:
            dx_norm = 0.0  # without width we cannot compare to crossing_frac

        # Approach score is a weighted blend of the four cues.
        s_growth = self._bbox_growth_score(growth_ratio)
        s_center = self._center_motion_score(
            start_cx, start_cy, end_cx, end_cy, image_width, image_height
        )
        s_fd = self._frame_diff_score(avg_fd_overlap)
        s_flow = self._flow_score(avg_flow_mag)
        approach_score = (
            0.45 * s_growth
            + 0.25 * s_center
            + 0.20 * s_fd
            + 0.10 * s_flow
        )
        crossing_score = abs(dx_norm) if image_width else 0.0

        # Label resolution in priority order.
        evidence_parts: List[str] = []
        if not moved:
            label = "static"
            evidence_parts.append(
                f"low motion (disp_norm={total_disp_norm:.3f}, "
                f"flow_mag={avg_flow_mag:.2f}, "
                f"frame_diff_overlap={avg_fd_overlap:.2f})"
            )
        elif (approach_score >= self.approach_threshold
              and growth_ratio > self.growth_threshold):
            label = "approaching"
            evidence_parts.append(
                f"bbox grew by {growth_ratio:.2f} "
                f"and approach_score={approach_score:.2f}"
            )
            if s_center > 0:
                evidence_parts.append("centroid moved toward image center / lower half")
            if avg_flow_mag > self.flow_threshold:
                evidence_parts.append(f"flow_mag={avg_flow_mag:.2f}")
        elif (image_width and dx_norm > self.crossing_frac
              and growth_ratio <= self.growth_threshold):
            label = "crossing_left_to_right"
            evidence_parts.append(
                f"dx={dx_total:.1f}px ({dx_norm*100:.1f}% of width) "
                f"with bbox_growth={growth_ratio:.2f} below threshold"
            )
        elif (image_width and dx_norm < -self.crossing_frac
              and growth_ratio <= self.growth_threshold):
            label = "crossing_right_to_left"
            evidence_parts.append(
                f"dx={dx_total:.1f}px ({dx_norm*100:.1f}% of width) "
                f"with bbox_growth={growth_ratio:.2f} below threshold"
            )
        elif growth_ratio < self.shrink_threshold:
            label = "moving_away"
            evidence_parts.append(
                f"bbox shrank: growth_ratio={growth_ratio:.2f} below "
                f"{self.shrink_threshold:.2f}"
            )
        else:
            label = "uncertain"
            evidence_parts.append(
                f"no rule fired strongly "
                f"(approach_score={approach_score:.2f}, "
                f"dx_norm={dx_norm:.3f}, growth_ratio={growth_ratio:.2f})"
            )

        if self.use_depth:
            evidence_parts.append("note: depth backend enabled in config but "
                                  "not implemented in this project")

        confidence = self._confidence(
            label, approach_score, dx_norm, growth_ratio, moved
        )
        track_id = str(row.get("track_id", ""))
        class_name = str(row.get("class_name", ""))
        evidence = f"{track_id} classified as {label} because " + "; ".join(evidence_parts) + "."

        return {
            "video_id": str(row.get("video_id", "")),
            "track_id": track_id,
            "class_name": class_name,
            "hazard_label": label,
            "approach_score": float(approach_score),
            "crossing_score": float(crossing_score),
            "bbox_growth_ratio": growth_ratio,
            "total_displacement_norm": total_disp_norm,
            "avg_flow_dx": avg_flow_dx,
            "avg_flow_dy": avg_flow_dy,
            "avg_flow_mag": avg_flow_mag,
            "avg_frame_diff_overlap": avg_fd_overlap,
            "dx_total": dx_total,
            "dy_total": dy_total,
            "start_cx": start_cx,
            "start_cy": start_cy,
            "first_frame": int(_safe_float(row, "first_frame", -1)),
            "last_frame": int(_safe_float(row, "last_frame", -1)),
            "image_width": float(image_width) if image_width else 0.0,
            "image_height": float(image_height) if image_height else 0.0,
            "confidence": confidence,
            "evidence": evidence,
        }

    def classify_all(
        self,
        track_features_df: pd.DataFrame,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> pd.DataFrame:
        """Classify every row in a track-features dataframe."""
        if track_features_df is None or track_features_df.empty:
            return pd.DataFrame(columns=HAZARD_COLUMNS)
        rows: List[Dict[str, Any]] = []
        for _, row in track_features_df.iterrows():
            rows.append(self.classify_track(
                row, image_width=image_width, image_height=image_height,
            ))
        return pd.DataFrame(rows, columns=HAZARD_COLUMNS)


def run_hazard_classification(
    track_features_df: pd.DataFrame,
    config: dict,
    video_id: str,
    output_dir: str | Path,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> pd.DataFrame:
    """Convenience entry point used by the pipeline scripts.

    Runs :class:`HazardClassifier` and writes
    ``<video_id>_hazards.csv`` + ``<video_id>_hazards.json`` into
    ``output_dir``. Returns the dataframe.
    """
    clf = HazardClassifier(config)
    df = clf.classify_all(
        track_features_df,
        image_width=image_width,
        image_height=image_height,
    )
    out_dir = ensure_dir(output_dir)
    csv_path = out_dir / f"{video_id}_hazards.csv"
    json_path = out_dir / f"{video_id}_hazards.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
    print(f"[hazard] wrote {len(df)} hazard rows -> {csv_path}")
    return df
