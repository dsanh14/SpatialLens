"""Unit tests for assistive alert generation."""

from __future__ import annotations

import pandas as pd

from src.alerts import (
    generate_alert_for_track,
    generate_alerts,
    generate_video_summary,
    save_alerts,
)

IMG_W = 960


def _hazard_row(**overrides) -> pd.Series:
    base = dict(
        video_id="unit",
        track_id="bicycle_1",
        class_name="bicycle",
        hazard_label="approaching",
        approach_score=0.85,
        crossing_score=0.0,
        bbox_growth_ratio=1.5,
        total_displacement_norm=0.25,
        avg_flow_dx=1.0,
        avg_flow_dy=0.5,
        avg_flow_mag=2.5,
        avg_frame_diff_overlap=0.2,
        dx_total=80.0,
        dy_total=-30.0,
        start_cx=120.0,  # left side
        start_cy=300.0,
        first_frame=0,
        last_frame=23,
        image_width=IMG_W,
        image_height=540,
        confidence="high",
        evidence="bicycle_1 classified as approaching.",
    )
    base.update(overrides)
    return pd.Series(base)


def test_approaching_bicycle_on_left_says_from_the_left():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="approaching", start_cx=80.0, image_width=IMG_W
    ))
    assert msg == "Bicycle approaching from the left."


def test_approaching_on_right_says_from_the_right():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="approaching", start_cx=900.0, image_width=IMG_W
    ))
    assert msg == "Bicycle approaching from the right."


def test_approaching_in_center_says_directly_ahead():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="approaching", start_cx=480.0, image_width=IMG_W
    ))
    assert msg == "Bicycle approaching directly ahead."


def test_crossing_left_to_right_alert():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="crossing_left_to_right", class_name="skateboard"
    ))
    assert msg == "Skateboard crossing left-to-right."


def test_crossing_right_to_left_alert():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="crossing_right_to_left", class_name="person"
    ))
    assert msg == "Person crossing right-to-left."


def test_static_alert_says_no_immediate_hazard():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="static", class_name="bicycle"
    ))
    assert "no immediate motion hazard" in msg
    assert "bicycle" in msg.lower()


def test_moving_away_alert():
    msg = generate_alert_for_track(_hazard_row(
        hazard_label="moving_away", class_name="person"
    ))
    assert msg == "Person moving away."


def test_uncertain_alert():
    msg = generate_alert_for_track(_hazard_row(hazard_label="uncertain"))
    assert "uncertain" in msg.lower()


def test_generate_alerts_returns_one_per_row():
    df = pd.DataFrame([
        _hazard_row(track_id="bicycle_1", hazard_label="approaching"),
        _hazard_row(track_id="person_1", hazard_label="static",
                    class_name="person"),
    ])
    alerts = generate_alerts(df, config={"alerts": {"include_confidence": True}})
    assert len(alerts) == 2
    assert alerts[0]["alert"].startswith("Bicycle approaching")
    assert "static" in alerts[1]["alert"].lower()
    assert "confidence" in alerts[0]


def test_video_summary_mentions_moving_count():
    df = pd.DataFrame([
        _hazard_row(track_id="bicycle_1", hazard_label="approaching"),
        _hazard_row(track_id="person_1", hazard_label="static",
                    class_name="person"),
    ])
    alerts = generate_alerts(df, config={})
    line = generate_video_summary(alerts, df)
    assert "1 moving object" in line
    assert "approaching" in line


def test_save_alerts_writes_json_and_txt(tmp_path):
    df = pd.DataFrame([_hazard_row()])
    alerts = generate_alerts(df, config={})
    paths = save_alerts(
        alerts=alerts, hazards_df=df, video_id="unit_vid",
        output_dir=tmp_path, config={"alerts": {
            "save_alerts_json": True,
            "save_alerts_txt": True,
            "include_confidence": True,
        }},
    )
    assert "json" in paths and "txt" in paths
    assert (tmp_path / "unit_vid_alerts.json").exists()
    assert (tmp_path / "unit_vid_alerts.txt").exists()
