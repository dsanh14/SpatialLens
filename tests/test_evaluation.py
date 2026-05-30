"""Unit tests for hazard evaluation metrics."""

from __future__ import annotations

import math

import pandas as pd

from src.evaluation import evaluate_hazards, load_labels


def _pred(video_id: str, track_id: str, label: str) -> dict:
    return {
        "video_id": video_id,
        "track_id": track_id,
        "class_name": "bicycle",
        "hazard_label": label,
    }


def _label(video_id: str, track_id: str, label: str) -> dict:
    return {
        "video_id": video_id,
        "track_id": track_id,
        "class_name": "bicycle",
        "true_label": label,
    }


def test_perfect_accuracy_and_f1_when_predictions_match():
    preds = pd.DataFrame([
        _pred("v1", "a", "approaching"),
        _pred("v1", "b", "crossing_left_to_right"),
        _pred("v1", "c", "static"),
    ])
    labels = pd.DataFrame([
        _label("v1", "a", "approaching"),
        _label("v1", "b", "crossing_left_to_right"),
        _label("v1", "c", "static"),
    ])
    m = evaluate_hazards(preds, labels)
    assert m["overall_accuracy"] == 1.0
    assert m["approaching_precision"] == 1.0
    assert m["approaching_recall"] == 1.0
    assert m["approaching_f1"] == 1.0
    assert m["num_evaluated_tracks"] == 3


def test_partial_accuracy_and_precision_recall():
    preds = pd.DataFrame([
        _pred("v1", "a", "approaching"),  # TP
        _pred("v1", "b", "approaching"),  # FP (true=crossing_l2r)
        _pred("v1", "c", "static"),       # TN-ish for approaching
        _pred("v1", "d", "crossing_left_to_right"),  # FN for approaching
    ])
    labels = pd.DataFrame([
        _label("v1", "a", "approaching"),
        _label("v1", "b", "crossing_left_to_right"),
        _label("v1", "c", "static"),
        _label("v1", "d", "approaching"),
    ])
    m = evaluate_hazards(preds, labels)
    # 2 of 4 correct -> 0.5
    assert math.isclose(m["overall_accuracy"], 0.5)
    # Approaching: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    assert math.isclose(m["approaching_precision"], 0.5)
    assert math.isclose(m["approaching_recall"], 0.5)
    assert math.isclose(m["approaching_f1"], 0.5)


def test_no_labels_returns_skip_message():
    preds = pd.DataFrame([_pred("v1", "a", "approaching")])
    m = evaluate_hazards(preds, pd.DataFrame())
    assert "message" in m
    assert m["num_labeled_tracks"] == 0


def test_no_predictions_returns_skip_message():
    labels = pd.DataFrame([_label("v1", "a", "approaching")])
    m = evaluate_hazards(pd.DataFrame(), labels)
    assert "message" in m
    assert m["num_predicted_tracks"] == 0


def test_load_labels_missing_file_returns_empty(tmp_path):
    out = load_labels(tmp_path / "does_not_exist.csv")
    assert out.empty
    assert list(out.columns) == [
        "video_id", "track_id", "class_name", "true_label",
    ]


def test_load_labels_drops_blank_true_label(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "video_id,track_id,class_name,true_label,notes\n"
        "v1,a,bicycle,approaching,n1\n"
        "v1,b,bicycle,,n2\n"
        "v1,c,bicycle,static,n3\n"
    )
    df = load_labels(csv)
    assert len(df) == 2
    assert set(df["track_id"]) == {"a", "c"}


def test_selective_accuracy_excludes_short_tracks():
    # 2 long tracks (both correct) + 2 short tracks (both wrong abstentions).
    preds = pd.DataFrame([
        {**_pred("v1", "a", "approaching"), "num_frames": 8},
        {**_pred("v1", "b", "moving_away"), "num_frames": 5},
        {**_pred("v1", "c", "uncertain"), "num_frames": 1},
        {**_pred("v1", "d", "uncertain"), "num_frames": 2},
    ])
    labels = pd.DataFrame([
        _label("v1", "a", "approaching"),
        _label("v1", "b", "moving_away"),
        _label("v1", "c", "crossing_left_to_right"),
        _label("v1", "d", "approaching"),
    ])
    m = evaluate_hazards(preds, labels, min_track_frames=3)
    # Overall: 2/4 correct.
    assert math.isclose(m["overall_accuracy"], 0.5)
    sel = m["selective_accuracy"]
    assert sel is not None
    assert sel["num_decidable_tracks"] == 2
    assert sel["num_short_tracks"] == 2
    # Decidable subset (>=3 frames) is fully correct.
    assert math.isclose(sel["decidable_accuracy"], 1.0)


def test_selective_accuracy_absent_without_num_frames():
    preds = pd.DataFrame([_pred("v1", "a", "approaching")])
    labels = pd.DataFrame([_label("v1", "a", "approaching")])
    m = evaluate_hazards(preds, labels)
    assert m["selective_accuracy"] is None


def test_confusion_matrix_has_correct_shape_and_orientation():
    preds = pd.DataFrame([
        _pred("v1", "a", "crossing_left_to_right"),
        _pred("v1", "b", "static"),
    ])
    labels = pd.DataFrame([
        _label("v1", "a", "approaching"),
        _label("v1", "b", "static"),
    ])
    m = evaluate_hazards(preds, labels)
    cm = m["confusion_matrix"]
    # Orientation is documented as rows=true_label, cols=predicted_label.
    assert m["confusion_matrix_orientation"].startswith("rows=true_label")
    assert len(cm) == 6
    for row in cm.values():
        assert len(row) == 6
    # Mismatched track 'a' should land at (true=approaching, pred=crossing_left_to_right).
    assert cm["approaching"]["crossing_left_to_right"] == 1
    # Matched track 'b' should land on the diagonal.
    assert cm["static"]["static"] == 1
