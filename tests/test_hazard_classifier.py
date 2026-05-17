"""Unit tests for the Week 3 hazard classifier."""

from __future__ import annotations

import pandas as pd

from src.hazard_classifier import HAZARD_LABELS, HazardClassifier

CONFIG = {
    "hazard": {
        "approach_score_threshold": 0.55,
        "crossing_threshold_frac_width": 0.08,
        "static_threshold_frac_diagonal": 0.04,
        "bbox_growth_threshold": 0.15,
        "bbox_shrink_threshold": -0.15,
        "flow_threshold": 1.0,
        "frame_diff_overlap_threshold": 0.08,
        "center_motion_threshold_frac": 0.05,
        "use_depth_if_available": False,
    }
}

IMG_W, IMG_H = 960, 540


def _feature_row(**overrides) -> pd.Series:
    base = dict(
        video_id="unit",
        track_id="object_1",
        class_name="bicycle",
        first_frame=0,
        last_frame=10,
        num_frames=11,
        start_cx=200.0, start_cy=400.0,
        end_cx=500.0, end_cy=350.0,
        total_displacement_px=300.0,
        total_displacement_norm=0.30,
        dx_total=300.0,
        dy_total=-50.0,
        avg_bbox_area=20000.0,
        start_area=5000.0,
        end_area=30000.0,
        bbox_growth_ratio=5.0,
        avg_flow_dx=1.5,
        avg_flow_dy=0.5,
        avg_flow_mag=2.0,
        avg_frame_diff_overlap=0.2,
        preliminary_motion_label="moving",
        preliminary_direction_label="left_to_right",
        preliminary_scale_label="growing",
    )
    base.update(overrides)
    return pd.Series(base)


def test_classifier_returns_known_label_set():
    clf = HazardClassifier(CONFIG)
    out = clf.classify_track(_feature_row(), image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] in HAZARD_LABELS


def test_growing_moving_object_is_approaching():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="bicycle_1",
        class_name="bicycle",
        bbox_growth_ratio=2.5,
        start_area=4000.0,
        end_area=14000.0,
        avg_flow_mag=3.0,
        avg_frame_diff_overlap=0.25,
        total_displacement_norm=0.30,
        dx_total=80.0,  # small horizontal so it doesn't look like a cross
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "approaching"
    assert out["approach_score"] >= 0.55
    assert "approaching" in out["evidence"]
    assert out["confidence"] in {"medium", "high"}


def test_strong_positive_dx_no_growth_is_left_to_right():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="skateboard_1",
        class_name="skateboard",
        start_cx=100.0, start_cy=270.0,
        end_cx=900.0, end_cy=270.0,
        dx_total=800.0,
        dy_total=0.0,
        bbox_growth_ratio=0.0,
        start_area=15000.0,
        end_area=15000.0,
        avg_flow_mag=3.0,
        avg_frame_diff_overlap=0.2,
        total_displacement_norm=0.7,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "crossing_left_to_right"


def test_strong_negative_dx_no_growth_is_right_to_left():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_1",
        class_name="person",
        start_cx=900.0, start_cy=300.0,
        end_cx=100.0, end_cy=300.0,
        dx_total=-800.0,
        dy_total=0.0,
        bbox_growth_ratio=0.05,
        start_area=15000.0,
        end_area=15750.0,
        avg_flow_mag=2.0,
        avg_frame_diff_overlap=0.15,
        total_displacement_norm=0.6,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "crossing_right_to_left"


def test_shrinking_bbox_is_moving_away():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_2",
        class_name="person",
        start_cx=500.0, start_cy=300.0,
        end_cx=510.0, end_cy=220.0,
        dx_total=10.0,
        dy_total=-80.0,
        bbox_growth_ratio=-0.6,
        start_area=20000.0,
        end_area=8000.0,
        avg_flow_mag=1.5,
        avg_frame_diff_overlap=0.10,
        total_displacement_norm=0.10,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "moving_away"


def test_low_motion_is_static():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="bicycle_2",
        class_name="bicycle",
        start_cx=400.0, start_cy=300.0,
        end_cx=400.0, end_cy=300.0,
        dx_total=0.0,
        dy_total=0.0,
        bbox_growth_ratio=0.0,
        start_area=10000.0,
        end_area=10000.0,
        avg_flow_mag=0.1,
        avg_frame_diff_overlap=0.0,
        total_displacement_norm=0.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "static"
    assert out["confidence"] == "high"


def test_classify_all_handles_empty():
    clf = HazardClassifier(CONFIG)
    df = clf.classify_all(pd.DataFrame())
    assert df.empty
    assert "hazard_label" in df.columns


def test_classify_all_returns_one_row_per_track():
    clf = HazardClassifier(CONFIG)
    df = pd.DataFrame([
        _feature_row(track_id="a", class_name="bicycle",
                     bbox_growth_ratio=2.0, avg_flow_mag=3.0,
                     dx_total=50.0, total_displacement_norm=0.25),
        _feature_row(track_id="b", class_name="person",
                     bbox_growth_ratio=0.0, avg_flow_mag=0.0,
                     dx_total=0.0, total_displacement_norm=0.0),
    ])
    out = clf.classify_all(df, image_width=IMG_W, image_height=IMG_H)
    assert len(out) == 2
    assert set(out["track_id"]) == {"a", "b"}
