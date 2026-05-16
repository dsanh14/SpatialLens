"""Unit tests for per-track motion feature aggregation."""

from __future__ import annotations

import pandas as pd

from src.motion_features import compute_track_motion_features

CONFIG = {
    "motion": {
        "static_threshold_frac_diagonal": 0.04,
        "bbox_growth_threshold": 0.15,
        "horizontal_motion_threshold_frac_width": 0.08,
        "flow_magnitude_threshold": 1.0,
        "frame_diff_threshold": 25,
        "morphology_kernel_size": 5,
    }
}

IMG_W, IMG_H = 960, 540


def _track_row(
    frame_idx: int,
    track_id: str,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    flow_dx: float = 0.0,
    flow_dy: float = 0.0,
    flow_mag: float = 0.0,
    frame_diff_overlap: float = 0.0,
) -> dict:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "video_id": "unit",
        "frame_idx": frame_idx,
        "frame_path": f"frame_{frame_idx:04d}.jpg",
        "class_name": class_name,
        "confidence": 0.9,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": cx, "cy": cy,
        "area": float((x2 - x1) * (y2 - y1)),
        "track_id": track_id,
        "track_age": frame_idx + 1,
        "matched_score": 1.0,
        "flow_dx": flow_dx,
        "flow_dy": flow_dy,
        "flow_mag": flow_mag,
        "frame_diff_overlap": frame_diff_overlap,
    }


def test_growing_bbox_labeled_growing():
    """Bbox area doubles -> growth_ratio ~= 1.0 -> 'growing'."""
    rows = [
        _track_row(0, "bicycle_1", "bicycle", 100, 100, 200, 200,
                   flow_mag=2.0),
        _track_row(1, "bicycle_1", "bicycle", 95, 95, 245, 245,
                   flow_mag=2.0),
        _track_row(2, "bicycle_1", "bicycle", 90, 90, 290, 290,
                   flow_mag=2.0),
    ]
    df = pd.DataFrame(rows)
    feats = compute_track_motion_features(df, CONFIG, IMG_W, IMG_H)
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["bbox_growth_ratio"] > 0.15
    assert row["preliminary_scale_label"] == "growing"


def test_shrinking_bbox_labeled_shrinking():
    rows = [
        _track_row(0, "person_1", "person", 100, 100, 400, 500,
                   flow_mag=2.0),
        _track_row(1, "person_1", "person", 150, 150, 350, 450,
                   flow_mag=2.0),
        _track_row(2, "person_1", "person", 200, 200, 300, 400,
                   flow_mag=2.0),
    ]
    df = pd.DataFrame(rows)
    feats = compute_track_motion_features(df, CONFIG, IMG_W, IMG_H)
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["bbox_growth_ratio"] < -0.15
    assert row["preliminary_scale_label"] == "shrinking"


def test_static_object_labeled_static():
    """Same bbox every frame, no flow -> 'static'."""
    rows = [
        _track_row(0, "person_1", "person", 400, 200, 500, 400),
        _track_row(1, "person_1", "person", 400, 200, 500, 400),
        _track_row(2, "person_1", "person", 400, 200, 500, 400),
    ]
    df = pd.DataFrame(rows)
    feats = compute_track_motion_features(df, CONFIG, IMG_W, IMG_H)
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["preliminary_motion_label"] == "static"
    assert row["preliminary_scale_label"] == "stable"


def test_left_to_right_direction_label():
    """Centroid moves right by > 8% of image width -> left_to_right."""
    rows = [
        _track_row(0, "skateboard_1", "skateboard", 100, 200, 200, 300,
                   flow_mag=3.0),
        _track_row(1, "skateboard_1", "skateboard", 400, 200, 500, 300,
                   flow_mag=3.0),
        _track_row(2, "skateboard_1", "skateboard", 700, 200, 800, 300,
                   flow_mag=3.0),
    ]
    df = pd.DataFrame(rows)
    feats = compute_track_motion_features(df, CONFIG, IMG_W, IMG_H)
    assert len(feats) == 1
    assert feats.iloc[0]["preliminary_direction_label"] == "left_to_right"


def test_empty_tracks_returns_empty_features():
    feats = compute_track_motion_features(pd.DataFrame(), CONFIG, IMG_W, IMG_H)
    assert feats.empty
