"""Assistive alert generation (Week 3).

Turns hazard rows into short text strings that could (in a deployed
system) be spoken aloud to a blind or low-vision pedestrian. This is
not a production navigation tool — see README's Limitations section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import ensure_dir

CLASS_DISPLAY = {
    "person": "Person",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
    "skateboard": "Skateboard",
    "scooter": "Scooter",
}


def _display_name(class_name: str) -> str:
    return CLASS_DISPLAY.get(str(class_name).lower(), str(class_name).capitalize())


def _side_from_cx(start_cx: float, image_width: float) -> str:
    """Return ``"left"``, ``"center"``, ``"right"``, or ``""`` if unknown."""
    if image_width is None or image_width <= 0:
        return ""
    if start_cx < image_width / 3.0:
        return "left"
    if start_cx > 2.0 * image_width / 3.0:
        return "right"
    return "center"


UNCERTAIN_ALERT_BY_REASON = {
    "short_track":
        "{obj} detected briefly; not enough frames to judge motion.",
    "near_approaching":
        "{obj} possibly approaching, but the signal is weak.",
    "near_crossing_left_to_right":
        "{obj} possibly crossing left-to-right; signal is weak.",
    "near_crossing_right_to_left":
        "{obj} possibly crossing right-to-left; signal is weak.",
    "near_moving_away":
        "{obj} may be moving away, but the signal is weak.",
    "conflicting_cues":
        "{obj} is moving but direction is ambiguous.",
    "low_signal":
        "{obj} motion detected but too weak to classify.",
}


def generate_alert_for_track(row: pd.Series) -> str:
    """Build a short assistive alert string for a single hazard row."""
    label = str(row.get("hazard_label", "uncertain"))
    obj = _display_name(row.get("class_name", "object"))
    image_width = float(row.get("image_width", 0.0) or 0.0)
    start_cx = float(row.get("start_cx", 0.0) or 0.0)
    side = _side_from_cx(start_cx, image_width)

    if label == "approaching":
        if side in ("left", "right"):
            return f"{obj} approaching from the {side}."
        if side == "center":
            return f"{obj} approaching directly ahead."
        return f"{obj} approaching."
    if label == "crossing_left_to_right":
        return f"{obj} crossing left-to-right."
    if label == "crossing_right_to_left":
        return f"{obj} crossing right-to-left."
    if label == "moving_away":
        return f"{obj} moving away."
    if label == "static":
        return f"Static {obj.lower()} detected; no immediate motion hazard."
    # Uncertain — pick the reason-specific phrasing if one is available.
    reason = str(row.get("uncertain_reason", "")).strip()
    template = UNCERTAIN_ALERT_BY_REASON.get(reason)
    if template:
        return template.format(obj=obj)
    return f"{obj} motion uncertain."


def _hazard_priority(label: str) -> int:
    """Lower number = higher priority in the video-level summary line."""
    return {
        "approaching": 0,
        "crossing_left_to_right": 1,
        "crossing_right_to_left": 1,
        "moving_away": 2,
        "uncertain": 3,
        "static": 4,
    }.get(label, 5)


def generate_video_summary(
    alerts: List[Dict],
    hazards_df: pd.DataFrame,
) -> str:
    """Compose a single concise sentence describing the whole video."""
    if not alerts or hazards_df is None or hazards_df.empty:
        return "No tracked objects detected."

    moving_mask = hazards_df["hazard_label"].isin([
        "approaching", "crossing_left_to_right",
        "crossing_right_to_left", "moving_away",
    ])
    num_moving = int(moving_mask.sum())

    ordered = sorted(
        alerts,
        key=lambda a: _hazard_priority(a["hazard_label"]),
    )
    phrases = []
    for a in ordered:
        label = a["hazard_label"]
        track_id = a["track_id"]
        if label == "approaching":
            phrases.append(f"{track_id} is approaching")
        elif label == "crossing_left_to_right":
            phrases.append(f"{track_id} is crossing left-to-right")
        elif label == "crossing_right_to_left":
            phrases.append(f"{track_id} is crossing right-to-left")
        elif label == "moving_away":
            phrases.append(f"{track_id} is moving away")
        elif label == "static":
            phrases.append(f"{track_id} is static")
        else:
            phrases.append(f"{track_id} motion uncertain")

    head = (
        f"Detected {num_moving} moving object{'s' if num_moving != 1 else ''} "
        f"out of {len(alerts)} tracked."
    )
    if phrases:
        return head + " " + "; ".join(phrases) + "."
    return head


def generate_alerts(
    hazards_df: pd.DataFrame,
    config: dict | None = None,
) -> List[Dict]:
    """Convert each hazard row to an alert dict.

    Each dict contains: ``track_id``, ``class_name``, ``hazard_label``,
    ``alert``, and (if configured) ``confidence``.
    """
    cfg = (config or {}).get("alerts", {})
    include_conf = bool(cfg.get("include_confidence", True))

    alerts: List[Dict] = []
    if hazards_df is None or hazards_df.empty:
        return alerts
    for _, row in hazards_df.iterrows():
        rec = {
            "track_id": str(row.get("track_id", "")),
            "class_name": str(row.get("class_name", "")),
            "hazard_label": str(row.get("hazard_label", "uncertain")),
            "alert": generate_alert_for_track(row),
        }
        reason = str(row.get("uncertain_reason", "")).strip()
        if reason:
            rec["uncertain_reason"] = reason
        if include_conf:
            rec["confidence"] = str(row.get("confidence", "low"))
        alerts.append(rec)
    return alerts


def save_alerts(
    alerts: List[Dict],
    hazards_df: pd.DataFrame,
    video_id: str,
    output_dir: str | Path,
    config: dict | None = None,
) -> Dict[str, str]:
    """Write the JSON + TXT alert files for a single video.

    Returns a dict with ``json``, ``txt``, ``video_summary`` keys (paths /
    text). Honors ``alerts.save_alerts_json`` / ``alerts.save_alerts_txt``.
    """
    cfg = (config or {}).get("alerts", {})
    save_json = bool(cfg.get("save_alerts_json", True))
    save_txt = bool(cfg.get("save_alerts_txt", True))

    out_dir = ensure_dir(output_dir)
    summary_line = generate_video_summary(alerts, hazards_df)
    paths: Dict[str, str] = {"video_summary": summary_line}

    if save_json:
        payload = {
            "video_id": video_id,
            "video_summary": summary_line,
            "alerts": alerts,
        }
        json_path = out_dir / f"{video_id}_alerts.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        paths["json"] = str(json_path)

    if save_txt:
        txt_path = out_dir / f"{video_id}_alerts.txt"
        lines = [f"SpatialLens Assist — Alerts ({video_id})", "=" * 40, ""]
        for a in alerts:
            conf = f"  [{a['confidence']}]" if "confidence" in a else ""
            lines.append(f"- {a['alert']}{conf}")
        lines.append("")
        lines.append("Video summary:")
        lines.append(f"  {summary_line}")
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        paths["txt"] = str(txt_path)

    return paths
