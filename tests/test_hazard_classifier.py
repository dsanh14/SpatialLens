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


# --------------------------------------------------------------------- #
# uncertain_reason taxonomy — every uncertain row must explain itself.
# --------------------------------------------------------------------- #

def test_non_uncertain_rows_have_empty_uncertain_reason():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        bbox_growth_ratio=2.5, avg_flow_mag=3.0,
        avg_frame_diff_overlap=0.25, total_displacement_norm=0.30,
        dx_total=80.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "approaching"
    assert out["uncertain_reason"] == ""


def test_short_track_with_motion_is_uncertain_short_track():
    """A track with only 2 detections + clear flow should be marked as
    uncertain/short_track instead of being force-classified."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="skateboard_1",
        num_frames=2,
        first_frame=9, last_frame=10,
        bbox_growth_ratio=0.0,
        dx_total=10.0,
        total_displacement_norm=0.02,
        avg_flow_mag=14.0,
        avg_frame_diff_overlap=0.20,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "short_track"
    assert "only 2 frames" in out["evidence"]
    assert out["confidence"] == "low"


def test_one_frame_track_with_no_motion_is_uncertain_not_static():
    """Regression test for the IMG_4973/person_1 manual-review finding.

    A 1-frame track cannot reliably be classified as `static`, because
    "not moving" requires at least two observations to establish.
    `short_track` must win over `static` for sub-min_track_frames
    tracks regardless of measured motion (which is trivially zero with
    a single sample anyway).
    """
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        num_frames=1, first_frame=5, last_frame=5,
        bbox_growth_ratio=0.0,
        dx_total=0.0,
        total_displacement_norm=0.0,
        avg_flow_mag=0.0,
        avg_frame_diff_overlap=0.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "short_track"
    # Evidence string should explain that we can't *confirm* it's static
    # from a single frame, not just regurgitate "motion was detected".
    assert "cannot confirm" in out["evidence"].lower()


def test_long_track_with_no_motion_still_static():
    """The static rule still fires for genuine multi-frame stillness —
    we only widened the short_track guard for sub-min_track_frames
    tracks, not all of them."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        num_frames=11, first_frame=0, last_frame=10,
        bbox_growth_ratio=0.0,
        dx_total=0.0,
        total_displacement_norm=0.0,
        avg_flow_mag=0.0,
        avg_frame_diff_overlap=0.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "static"
    assert out["uncertain_reason"] == ""


def test_near_approaching_signal_is_explained():
    """approach_score ~ 0.45 (just below 0.55) + tiny growth -> near_approaching."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_2",
        bbox_growth_ratio=0.05,  # below growth_threshold 0.15
        start_area=10000.0, end_area=10500.0,
        avg_flow_mag=2.0, avg_frame_diff_overlap=0.05,
        dx_total=20.0, total_displacement_norm=0.15,
        start_cx=480.0, end_cx=470.0,  # no center motion benefit
        start_cy=300.0, end_cy=305.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "near_approaching"
    assert "approach_score" in out["evidence"]


def test_near_crossing_left_to_right_is_explained():
    """dx_norm between 50% and 100% of crossing_frac (0.08) -> near_crossing_l2r."""
    clf = HazardClassifier(CONFIG)
    # crossing_frac=0.08, image_width=960 -> threshold dx is 76.8 px.
    # Use 50 px (5.2% of width) to land in the near-miss band (4%-8%).
    row = _feature_row(
        track_id="person_3",
        dx_total=50.0,
        bbox_growth_ratio=0.0,
        start_area=10000.0, end_area=10000.0,
        avg_flow_mag=2.0, avg_frame_diff_overlap=0.10,
        total_displacement_norm=0.10,
        start_cx=400.0, end_cx=450.0,
        start_cy=300.0, end_cy=300.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "near_crossing_left_to_right"
    assert "left-to-right" in out["evidence"]


def test_near_crossing_right_to_left_is_explained():
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_4",
        dx_total=-50.0,
        bbox_growth_ratio=0.0,
        start_area=10000.0, end_area=10000.0,
        avg_flow_mag=2.0, avg_frame_diff_overlap=0.10,
        total_displacement_norm=0.10,
        start_cx=500.0, end_cx=450.0,
        start_cy=300.0, end_cy=300.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "near_crossing_right_to_left"
    assert "right-to-left" in out["evidence"]


def test_near_moving_away_is_explained():
    """bbox shrunk but not below -0.15 -> near_moving_away."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_5",
        bbox_growth_ratio=-0.08,  # negative but not below -0.15
        start_area=10000.0, end_area=9200.0,
        dx_total=5.0,
        avg_flow_mag=2.0, avg_frame_diff_overlap=0.10,
        total_displacement_norm=0.10,
        start_cx=480.0, end_cx=481.0,
        start_cy=300.0, end_cy=295.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "near_moving_away"


def test_low_signal_uncertain_is_explained():
    """Motion present (flow) but every directional cue is zero -> low_signal."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="motorcycle_1",
        bbox_growth_ratio=0.0,
        start_area=10000.0, end_area=10000.0,
        dx_total=0.0, dy_total=0.0,
        avg_flow_mag=20.0,  # high flow keeps `moved` true
        avg_frame_diff_overlap=0.05,
        total_displacement_norm=0.0,
        start_cx=480.0, end_cx=480.0,
        start_cy=300.0, end_cy=300.0,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "uncertain"
    assert out["uncertain_reason"] == "low_signal"


def test_uncertain_reason_column_present_in_classify_all():
    clf = HazardClassifier(CONFIG)
    df = pd.DataFrame([
        _feature_row(track_id="a", num_frames=2, bbox_growth_ratio=0.0,
                     dx_total=5.0, avg_flow_mag=10.0,
                     total_displacement_norm=0.05),
    ])
    out = clf.classify_all(df, image_width=IMG_W, image_height=IMG_H)
    assert "uncertain_reason" in out.columns
    assert out["uncertain_reason"].iloc[0] == "short_track"


# ---------------------------------------------------------------------------
# v2 rule-cascade regression tests
#
# The cascade was reorganized after labeling 61 tracks across 11 real-world
# videos surfaced two systematic failure modes (priority-order conflicts):
#
#   (A) Strong lateral motion + bbox growth was labeled `approaching`
#       because the approach rule fired first in v1. Example tracks:
#       IMG_4978/person_1 (g=+1.72, |dx_norm|=0.68 -> true: crossing_R2L)
#       IMG_4981/person_3 (g=+1.92, |dx_norm|=0.80 -> true: crossing_R2L)
#
#   (B) Bbox shrinking + lateral motion was labeled `crossing` because
#       crossing fired before moving_away in v1. Examples:
#       IMG_4980/person_1 (g=-0.24, |dx|=0.15 -> true: moving_away)
#       IMG_4980/person_2 (g=-0.71, |dx|=0.54 -> true: moving_away)
#       IMG_4981/person_4 (g=-0.47, |dx|=0.15 -> true: moving_away)
#
# v2: moving_away is checked before crossing; strong_crossing fires
# before approaching when |dx_norm| > strong_crossing_threshold (0.50,
# the midpoint of an empirically observed gap between true-approaching
# tracks max |dx_norm|=0.43 and true-crossing tracks min |dx_norm|=0.68).
# ---------------------------------------------------------------------------


def test_v2_strong_lateral_overrides_approaching_to_crossing():
    """IMG_4978/person_1 regression: g=+1.72 + |dx_norm|=0.68 -> crossing_R2L."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_1",
        class_name="person",
        # ~68% of width=960 leftward
        start_cx=820.0, end_cx=170.0,
        dx_total=-650.0, dy_total=0.0,
        # bbox grew ~2.7x (g=+1.72)
        start_area=66000.0, end_area=180000.0,
        bbox_growth_ratio=1.72,
        avg_flow_mag=14.0, avg_frame_diff_overlap=0.47,
        total_displacement_norm=0.33,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "crossing_right_to_left"
    assert "strong-crossing threshold" in out["evidence"]


def test_v2_strong_positive_lateral_overrides_approaching():
    """IMG_4981/person_3 regression: g=+1.92 + dx_norm=+0.80 -> crossing_L2R.
    (Sign-mirrored from IMG_4981/person_3, which was negative dx.)"""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_3",
        class_name="person",
        start_cx=100.0, end_cx=870.0,
        dx_total=770.0, dy_total=0.0,
        start_area=10000.0, end_area=29000.0,
        bbox_growth_ratio=1.92,
        avg_flow_mag=14.0, avg_frame_diff_overlap=0.4,
        total_displacement_norm=0.4,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "crossing_left_to_right"


def test_v2_shrinking_bbox_with_lateral_motion_is_moving_away():
    """IMG_4980/person_2 regression: g=-0.71 + |dx_norm|=0.54 -> moving_away.

    v1 would have called this crossing because crossing was checked
    before moving_away in the cascade."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_2",
        class_name="person",
        # large leftward lateral motion (54% of width)
        start_cx=910.0, end_cx=390.0,
        dx_total=-520.0, dy_total=0.0,
        # but bbox SHRANK strongly (-71%)
        start_area=55000.0, end_area=16000.0,
        bbox_growth_ratio=-0.71,
        avg_flow_mag=6.0, avg_frame_diff_overlap=0.25,
        total_displacement_norm=0.4,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "moving_away"
    # Evidence should explicitly call out that lateral motion was present
    # but was overridden by the shrinkage signal -- otherwise reviewers
    # won't know what tipped the decision.
    assert "lateral motion" in out["evidence"].lower()


def test_v2_shrinking_bbox_with_mild_lateral_is_moving_away():
    """IMG_4980/person_1 + IMG_4981/person_4 regression:
    moderate shrinkage + mild lateral (just over the crossing threshold)
    should still be moving_away, not crossing."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_1",
        class_name="person",
        # 15% lateral - over crossing_frac=0.08 but under strong=0.50
        start_cx=430.0, end_cx=575.0,
        dx_total=145.0, dy_total=20.0,
        # moderate shrinkage (g=-0.24)
        start_area=166000.0, end_area=126000.0,
        bbox_growth_ratio=-0.24,
        avg_flow_mag=4.0, avg_frame_diff_overlap=0.2,
        total_displacement_norm=0.2,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "moving_away"


def test_v2_approaching_with_modest_lateral_still_approaching():
    """Regression: an approaching pedestrian whose dx is in the 0.20-0.43
    "gray zone" must still be classified as approaching. This is the
    range where an over-eager strong_crossing threshold (e.g. 0.20) would
    incorrectly flip 8 currently-correct labels in the eval set."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="bicycle_1",
        class_name="bicycle",
        # 31% of width - in the gray zone
        start_cx=290.0, end_cx=580.0,
        dx_total=290.0, dy_total=-50.0,
        # strong bbox growth (g=+21.0)
        start_area=2000.0, end_area=44000.0,
        bbox_growth_ratio=21.0,
        avg_flow_mag=14.5, avg_frame_diff_overlap=0.45,
        total_displacement_norm=0.35,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    assert out["hazard_label"] == "approaching"


def test_v2_strong_crossing_threshold_is_configurable():
    """The new threshold can be raised to 999 to disable strong_crossing
    entirely (useful for ablation studies and to revert to v1 behavior
    when combined with the original cascade order)."""
    cfg = {**CONFIG, "hazard": {
        **CONFIG["hazard"],
        "strong_crossing_threshold_frac_width": 999.0,  # effectively off
    }}
    clf = HazardClassifier(cfg)
    # Same inputs as test_v2_strong_lateral_overrides_approaching_to_crossing
    row = _feature_row(
        track_id="person_1",
        start_cx=820.0, end_cx=170.0,
        dx_total=-650.0,
        start_area=66000.0, end_area=180000.0,
        bbox_growth_ratio=1.72,
        avg_flow_mag=14.0, avg_frame_diff_overlap=0.47,
        total_displacement_norm=0.33,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    # With strong_crossing disabled, approach rule wins again (v1
    # behavior) because g=+1.72 > growth_threshold and approach_score
    # is high.
    assert out["hazard_label"] == "approaching"


def test_v2_evidence_mentions_strong_crossing_threshold():
    """The evidence string must explain *which* rule fired so that the
    decision can be audited from the CSV without re-running the
    classifier."""
    clf = HazardClassifier(CONFIG)
    row = _feature_row(
        track_id="person_x",
        start_cx=820.0, end_cx=170.0,
        dx_total=-650.0,
        start_area=66000.0, end_area=180000.0,
        bbox_growth_ratio=1.72,
        avg_flow_mag=14.0, avg_frame_diff_overlap=0.47,
        total_displacement_norm=0.33,
    )
    out = clf.classify_track(row, image_width=IMG_W, image_height=IMG_H)
    ev = out["evidence"].lower()
    assert "strong-crossing" in ev or "exceeds strong-crossing" in ev
    assert "50%" in ev  # the threshold percent
