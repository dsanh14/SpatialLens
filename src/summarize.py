"""Human-readable summaries (Week 1-2 + Week 3 final) per video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .utils import ensure_dir


def _detections_by_class_text(detections_df: pd.DataFrame) -> str:
    if detections_df.empty:
        return "  (no detections)"
    counts = detections_df["class_name"].value_counts().sort_index()
    return "\n".join(f"  {cls}: {n}" for cls, n in counts.items())


def _tracks_text(track_features_df: pd.DataFrame) -> str:
    if track_features_df.empty:
        return "  (no tracks)"
    lines = []
    for _, r in track_features_df.iterrows():
        lines.append(
            f"  {r['track_id']}: "
            f"{r['preliminary_motion_label']}, "
            f"{r['preliminary_scale_label']}, "
            f"dir={r['preliminary_direction_label']}, "
            f"dx={r['dx_total']:.1f} px, "
            f"bbox_growth={r['bbox_growth_ratio']:.2f}, "
            f"flow_mag={r['avg_flow_mag']:.2f}, "
            f"frames={int(r['num_frames'])}"
        )
    return "\n".join(lines)


def _preliminary_note(track_features_df: pd.DataFrame) -> str:
    if track_features_df.empty:
        return "  No tracks were generated. Check detection backend and frame count."
    notes = []
    for _, r in track_features_df.iterrows():
        if (r["preliminary_motion_label"] == "moving"
                and r["preliminary_scale_label"] == "growing"):
            notes.append(
                f"  {r['track_id']} may be approaching the camera "
                f"(bbox grows over time, growth_ratio={r['bbox_growth_ratio']:.2f})."
            )
        elif (r["preliminary_motion_label"] == "moving"
                and r["preliminary_scale_label"] == "shrinking"):
            notes.append(
                f"  {r['track_id']} may be moving away "
                f"(bbox shrinks, growth_ratio={r['bbox_growth_ratio']:.2f})."
            )
        elif r["preliminary_direction_label"] == "left_to_right":
            notes.append(
                f"  {r['track_id']} appears to cross left-to-right "
                f"(dx={r['dx_total']:.1f} px)."
            )
        elif r["preliminary_direction_label"] == "right_to_left":
            notes.append(
                f"  {r['track_id']} appears to cross right-to-left "
                f"(dx={r['dx_total']:.1f} px)."
            )
        elif r["preliminary_motion_label"] == "static":
            notes.append(f"  {r['track_id']} appears static.")
    if not notes:
        return "  No strong preliminary motion patterns detected."
    notes.append(
        "  NOTE: these are *preliminary* observations from Week 2 motion "
        "features. Final hazard classification will be implemented in Week 3."
    )
    return "\n".join(notes)


def summarize_week1_week2(
    video_id: str,
    detections_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    track_features_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> str:
    """Produce a Week 1-2 text + JSON summary for a single video.

    Returns the rendered text. If ``output_dir`` is given, also writes
    ``<video_id>_week1_week2_summary.txt`` and ``.json``.
    """
    if not detections_df.empty:
        num_frames = int(detections_df["frame_idx"].nunique())
    elif not tracks_df.empty:
        num_frames = int(tracks_df["frame_idx"].nunique())
    else:
        num_frames = 0

    num_detections = int(len(detections_df))
    detected_classes = sorted(detections_df["class_name"].unique()) \
        if not detections_df.empty else []
    num_tracks = int(track_features_df["track_id"].nunique()) \
        if not track_features_df.empty else 0

    text = (
        "SpatialLens Assist Week 1-2 Summary\n"
        f"Video: {video_id}\n"
        f"Frames processed: {num_frames}\n"
        f"Total detections: {num_detections}\n"
        f"Detected classes: {', '.join(detected_classes) if detected_classes else '(none)'}\n"
        "\n"
        "Detections by class:\n"
        f"{_detections_by_class_text(detections_df)}\n"
        "\n"
        f"Tracks ({num_tracks}):\n"
        f"{_tracks_text(track_features_df)}\n"
        "\n"
        "Preliminary interpretation (Week 2 features, not final hazard labels):\n"
        f"{_preliminary_note(track_features_df)}\n"
    )

    payload = {
        "video_id": video_id,
        "frames_processed": num_frames,
        "total_detections": num_detections,
        "detected_classes": detected_classes,
        "detections_by_class": (
            detections_df["class_name"].value_counts().to_dict()
            if not detections_df.empty else {}
        ),
        "num_tracks": num_tracks,
        "tracks": track_features_df.to_dict(orient="records")
            if not track_features_df.empty else [],
    }

    if output_dir is not None:
        out_dir = ensure_dir(output_dir)
        txt_path = out_dir / f"{video_id}_week1_week2_summary.txt"
        json_path = out_dir / f"{video_id}_week1_week2_summary.json"
        txt_path.write_text(text, encoding="utf-8")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[summary] wrote {txt_path} and {json_path}")

    return text


# ----------------------------------------------------------------------- #
# Week 3 — final summary including hazard labels + alerts
# ----------------------------------------------------------------------- #


def _hazards_text(hazards_df: pd.DataFrame) -> str:
    if hazards_df is None or hazards_df.empty:
        return "  (no hazards)"
    lines = []
    for _, r in hazards_df.iterrows():
        lines.append(
            f"  {r['track_id']}: {r['hazard_label']}, "
            f"approach_score={float(r['approach_score']):.2f}, "
            f"confidence={r['confidence']}"
        )
    return "\n".join(lines)


def _evidence_bullets(hazards_df: pd.DataFrame) -> List[str]:
    if hazards_df is None or hazards_df.empty:
        return []
    bullets: List[str] = []
    for _, r in hazards_df.iterrows():
        growth = float(r.get("bbox_growth_ratio", 0.0) or 0.0)
        dx = float(r.get("dx_total", 0.0) or 0.0)
        flow_mag = float(r.get("avg_flow_mag", 0.0) or 0.0)
        if abs(growth) > 0.1:
            bullets.append(
                f"  {r['track_id']} bbox grew by {growth:.2f}" if growth > 0
                else f"  {r['track_id']} bbox shrank by {abs(growth):.2f}"
            )
        if abs(dx) > 30:
            direction = "rightward" if dx > 0 else "leftward"
            bullets.append(f"  {r['track_id']} trajectory moved {direction} "
                           f"by {abs(dx):.0f}px")
        if flow_mag > 1.0:
            bullets.append(
                f"  {r['track_id']} optical flow magnitude {flow_mag:.2f} "
                f"(above 1.0 threshold)"
            )
    return bullets


def summarize_final(
    video_id: str,
    detections_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    track_features_df: pd.DataFrame,
    hazards_df: pd.DataFrame,
    alerts: Optional[List[Dict]] = None,
    video_summary: Optional[str] = None,
    evaluation: Optional[dict] = None,
    output_dir: str | Path | None = None,
) -> str:
    """Compose the project's final per-video summary (Week 3 deliverable)."""
    alerts = alerts or []
    detected_classes = sorted(detections_df["class_name"].unique()) \
        if not detections_df.empty else []
    num_tracks = int(track_features_df["track_id"].nunique()) \
        if not track_features_df.empty else 0

    hazards_lines = _hazards_text(hazards_df)
    alerts_lines = "\n".join(f"  - {a['alert']}" for a in alerts) \
        if alerts else "  (no alerts)"
    evidence_lines = "\n".join(_evidence_bullets(hazards_df)) \
        or "  (no notable cues)"

    eval_block = ""
    if evaluation and "overall_accuracy" in evaluation:
        eval_block = (
            "\nEvaluation (vs. manual labels):\n"
            f"  overall_accuracy={evaluation['overall_accuracy']:.2f}, "
            f"approaching_precision={evaluation['approaching_precision']:.2f}, "
            f"approaching_recall={evaluation['approaching_recall']:.2f}, "
            f"approaching_f1={evaluation['approaching_f1']:.2f}, "
            f"macro_f1={evaluation['macro_f1']:.2f}\n"
        )
    elif evaluation and "message" in evaluation:
        eval_block = f"\nEvaluation: {evaluation['message']}\n"

    text = (
        "SpatialLens Assist Final Summary\n"
        f"Video: {video_id}\n"
        f"Detected objects: {', '.join(detected_classes) if detected_classes else '(none)'}\n"
        f"Tracked objects: {num_tracks}\n"
        "\n"
        "Hazards:\n"
        f"{hazards_lines}\n"
        "\n"
        "Alerts:\n"
        f"{alerts_lines}\n"
    )
    if video_summary:
        text += f"\nVideo summary line:\n  {video_summary}\n"
    text += (
        "\nMain evidence:\n"
        f"{evidence_lines}\n"
        f"{eval_block}"
    )

    payload = {
        "video_id": video_id,
        "detected_classes": detected_classes,
        "num_tracks": num_tracks,
        "hazards": hazards_df.to_dict(orient="records")
            if hazards_df is not None and not hazards_df.empty else [],
        "alerts": alerts,
        "video_summary": video_summary,
        "evaluation": evaluation,
    }

    if output_dir is not None:
        out_dir = ensure_dir(output_dir)
        txt_path = out_dir / f"{video_id}_final_summary.txt"
        json_path = out_dir / f"{video_id}_final_summary.json"
        txt_path.write_text(text, encoding="utf-8")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[summary] wrote {txt_path} and {json_path}")

    return text
