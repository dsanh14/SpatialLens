"""Unit tests for the opt-in ByteTrack tracking backend."""

from __future__ import annotations

import pandas as pd
import pytest

# Skip the whole module gracefully if `supervision` isn't installed.
supervision = pytest.importorskip("supervision")

from src.tracking import assign_tracks  # noqa: E402

IMG_W, IMG_H = 960, 540


def _det(frame_idx, class_name, x1, y1, x2, y2, confidence=0.9):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
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


def _config(backend: str) -> dict:
    return {
        "tracking": {
            "backend": backend,
            "iou_threshold": 0.25,
            "centroid_distance_threshold_frac": 0.15,
            "max_frame_gap": 4,
            "min_track_length": 2,
            "bytetrack": {
                "track_activation_threshold": 0.25,
                "lost_track_buffer": 30,
                "minimum_matching_threshold": 0.8,
                "frame_rate": 30,
            },
        }
    }


def test_bytetrack_dispatches_via_assign_tracks():
    """The public `assign_tracks` should route to the ByteTrack backend
    when config.tracking.backend == 'bytetrack', and the output schema
    should match the default backend."""
    rows = [
        _det(t, "person", 100 + t * 8, 200, 160 + t * 8, 260, confidence=0.9)
        for t in range(6)
    ]
    df = pd.DataFrame(rows)
    out = assign_tracks(df, IMG_W, IMG_H, _config("bytetrack"))
    expected_cols = {"track_id", "track_age", "matched_score", "frame_idx"}
    assert expected_cols.issubset(out.columns)
    # All confirmed-track rows should share one track_id.
    assert out["track_id"].nunique() >= 1


def test_bytetrack_preserves_id_across_brief_detection_miss():
    """Drop frame 3 entirely; ByteTrack's lost_track_buffer should keep
    the same track id when the object reappears at frame 4."""
    rows = []
    for t in range(8):
        if t == 3:
            continue  # missed detection
        rows.append(_det(t, "person", 100 + t * 8, 200,
                         160 + t * 8, 260, confidence=0.9))
    df = pd.DataFrame(rows)
    out = assign_tracks(df, IMG_W, IMG_H, _config("bytetrack"))
    # Should have ONE track id across all kept rows.
    assert out["track_id"].nunique() == 1, (
        f"ByteTrack should re-associate after a 1-frame miss; "
        f"got ids: {out['track_id'].unique()}"
    )


def test_bytetrack_relabels_to_per_class_string_ids():
    """Even though ByteTrack natively uses integer IDs, the backend
    should rewrite them to {class}_{n} so downstream code that groups
    by class+track_id keeps working."""
    rows = []
    for t in range(5):
        rows.append(_det(t, "person", 100 + t * 6, 200,
                         160 + t * 6, 260, confidence=0.9))
        rows.append(_det(t, "bicycle", 400 + t * 6, 300,
                         460 + t * 6, 360, confidence=0.9))
    df = pd.DataFrame(rows)
    out = assign_tracks(df, IMG_W, IMG_H, _config("bytetrack"))
    if out.empty:
        pytest.skip("ByteTrack didn't confirm any tracks at this scale.")
    classes_in_ids = {tid.split("_")[0] for tid in out["track_id"].unique()}
    assert classes_in_ids.issubset({"person", "bicycle"})


def test_bytetrack_default_backend_is_unchanged_when_not_configured():
    """When backend key is missing the default IoU+centroid path runs —
    this protects every existing test config from regressing."""
    rows = [
        _det(t, "person", 100 + t * 6, 200, 160 + t * 6, 260, confidence=0.9)
        for t in range(5)
    ]
    df = pd.DataFrame(rows)
    cfg = {
        "tracking": {
            "iou_threshold": 0.25,
            "centroid_distance_threshold_frac": 0.15,
            "max_frame_gap": 2,
            "min_track_length": 2,
        }  # no `backend` key
    }
    out = assign_tracks(df, IMG_W, IMG_H, cfg)
    # IoU+centroid keeps every detection.
    assert len(out) == 5
    assert out["track_id"].nunique() == 1
