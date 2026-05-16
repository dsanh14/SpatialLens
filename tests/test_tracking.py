"""Unit tests for the simple IoU + centroid tracker."""

from __future__ import annotations

import pandas as pd

from src.tracking import assign_tracks, bbox_iou, centroid_distance

CONFIG = {
    "tracking": {
        "iou_threshold": 0.25,
        "centroid_distance_threshold_frac": 0.15,
        "max_frame_gap": 2,
        "min_track_length": 2,
    }
}

IMG_W, IMG_H = 960, 540


def _det(
    frame_idx: int,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> dict:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "video_id": "unit",
        "frame_idx": frame_idx,
        "frame_path": f"frame_{frame_idx:04d}.jpg",
        "class_name": class_name,
        "confidence": confidence,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "cx": cx, "cy": cy,
        "area": float((x2 - x1) * (y2 - y1)),
    }


def test_bbox_iou_basic():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    iou = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    # intersection 5x5=25, union 100+100-25=175 -> 25/175 ~= 0.142857
    assert abs(iou - 25.0 / 175.0) < 1e-6


def test_centroid_distance_basic():
    a = {"cx": 0.0, "cy": 0.0}
    b = {"cx": 3.0, "cy": 4.0}
    assert centroid_distance(a, b) == 5.0


def test_same_object_keeps_same_track_id():
    """An object that moves a little frame-to-frame should keep its ID."""
    rows = [
        _det(0, "person", 100, 100, 200, 300),
        _det(1, "person", 105, 102, 205, 302),
        _det(2, "person", 112, 105, 212, 305),
        _det(3, "person", 120, 108, 220, 308),
    ]
    df = pd.DataFrame(rows)
    tracks = assign_tracks(df, IMG_W, IMG_H, CONFIG)
    assert tracks["track_id"].nunique() == 1, (
        f"Expected 1 track, got {tracks['track_id'].unique()}"
    )


def test_far_objects_get_different_track_ids():
    """Two persons on opposite sides of the frame must not merge."""
    rows = [
        _det(0, "person", 50, 50, 150, 250),
        _det(0, "person", 800, 50, 900, 250),
        _det(1, "person", 55, 52, 155, 252),
        _det(1, "person", 805, 55, 905, 255),
    ]
    df = pd.DataFrame(rows)
    tracks = assign_tracks(df, IMG_W, IMG_H, CONFIG)
    assert tracks["track_id"].nunique() == 2


def test_different_classes_do_not_match():
    """A person and a bicycle in similar positions must not share a track."""
    rows = [
        _det(0, "person", 100, 100, 200, 300),
        _det(1, "bicycle", 105, 102, 205, 302),
        _det(2, "person", 110, 105, 210, 305),
        _det(3, "bicycle", 115, 108, 215, 308),
    ]
    df = pd.DataFrame(rows)
    tracks = assign_tracks(df, IMG_W, IMG_H, CONFIG)
    person_tracks = tracks[tracks["class_name"] == "person"]["track_id"].unique()
    bicycle_tracks = tracks[tracks["class_name"] == "bicycle"]["track_id"].unique()
    assert len(person_tracks) == 1
    assert len(bicycle_tracks) == 1
    assert set(person_tracks).isdisjoint(set(bicycle_tracks))


def test_empty_detections_returns_empty_tracks():
    empty = pd.DataFrame(columns=[
        "video_id", "frame_idx", "frame_path", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "cx", "cy", "area",
    ])
    tracks = assign_tracks(empty, IMG_W, IMG_H, CONFIG)
    assert tracks.empty
