"""Mock data generation for SpatialLens Assist.

Because real campus videos are not yet available, this module synthesizes
short videos containing simple colored rectangles + text labels that mimic
the four motion scenarios we want to test the Week 1-2 pipeline on:

- ``bike_approaching_left``           — bbox grows + moves toward center
- ``scooter_crossing_left_to_right``  — bbox moves horizontally, constant size
- ``person_walking_away``             — bbox shrinks + moves upward
- ``static_nonhazard``                — bbox stays put

It also exposes :func:`generate_mock_detections_for_frames`, which produces
deterministic detection rows for these synthetic scenarios so the rest of the
pipeline can run end-to-end without YOLO weights or any model download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from .utils import color_for_class, ensure_dir

SCENARIO_TO_CLASS: Dict[str, str] = {
    "bike_approaching_left": "bicycle",
    "scooter_crossing_left_to_right": "skateboard",  # COCO has no scooter
    "person_walking_away": "person",
    "static_nonhazard": "person",
}

# Background color for synthetic frames (BGR). A neutral gray that contrasts
# with the bright class colors used for the moving rectangles.
BACKGROUND_BGR: Tuple[int, int, int] = (40, 40, 40)


def _scenario_bbox(
    scenario: str,
    frame_idx: int,
    num_frames: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the moving rectangle at ``frame_idx``."""
    t = frame_idx / max(num_frames - 1, 1)  # 0..1

    if scenario == "bike_approaching_left":
        start_w, end_w = 70, 260
        start_h, end_h = 50, 180
        w = int(start_w + (end_w - start_w) * t)
        h = int(start_h + (end_h - start_h) * t)
        start_cx = int(width * 0.18)
        end_cx = int(width * 0.50)
        start_cy = int(height * 0.65)
        end_cy = int(height * 0.55)
        cx = int(start_cx + (end_cx - start_cx) * t)
        cy = int(start_cy + (end_cy - start_cy) * t)
        x1, y1 = cx - w // 2, cy - h // 2
        return x1, y1, x1 + w, y1 + h

    if scenario == "scooter_crossing_left_to_right":
        w, h = 140, 110
        start_cx = int(width * 0.10)
        end_cx = int(width * 0.90)
        cy = int(height * 0.55)
        cx = int(start_cx + (end_cx - start_cx) * t)
        x1, y1 = cx - w // 2, cy - h // 2
        return x1, y1, x1 + w, y1 + h

    if scenario == "person_walking_away":
        start_w, end_w = 180, 70
        start_h, end_h = 260, 110
        w = int(start_w + (end_w - start_w) * t)
        h = int(start_h + (end_h - start_h) * t)
        start_cx, end_cx = int(width * 0.50), int(width * 0.52)
        start_cy, end_cy = int(height * 0.70), int(height * 0.45)
        cx = int(start_cx + (end_cx - start_cx) * t)
        cy = int(start_cy + (end_cy - start_cy) * t)
        x1, y1 = cx - w // 2, cy - h // 2
        return x1, y1, x1 + w, y1 + h

    if scenario == "static_nonhazard":
        w, h = 120, 220
        cx = int(width * 0.50)
        cy = int(height * 0.55)
        x1, y1 = cx - w // 2, cy - h // 2
        return x1, y1, x1 + w, y1 + h

    raise ValueError(f"Unknown mock scenario: {scenario!r}")


def _draw_frame(
    scenario: str,
    frame_idx: int,
    num_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one mock frame for ``scenario``."""
    img = np.full((height, width, 3), BACKGROUND_BGR, dtype=np.uint8)

    # Light "ground" gradient so motion is more visible to optical flow.
    for y in range(int(height * 0.7), height):
        shade = int(60 + (y - height * 0.7) / (height * 0.3) * 40)
        img[y, :] = (shade, shade, shade)

    class_name = SCENARIO_TO_CLASS[scenario]
    color = color_for_class(class_name)
    x1, y1, x2, y2 = _scenario_bbox(scenario, frame_idx, num_frames, width, height)

    # Clip to image bounds defensively.
    x1c, y1c = max(x1, 0), max(y1, 0)
    x2c, y2c = min(x2, width - 1), min(y2, height - 1)
    if x2c > x1c and y2c > y1c:
        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), color, thickness=-1)
        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), (255, 255, 255), thickness=2)

    label = f"{scenario}  [{class_name}]  f{frame_idx:03d}"
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def generate_mock_video(
    scenario: str,
    output_path: str | Path,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> str:
    """Generate a single synthetic video for ``scenario`` and write it as mp4.

    Returns the output path as a string.
    """
    if scenario not in SCENARIO_TO_CLASS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            f"Known scenarios: {sorted(SCENARIO_TO_CLASS)}"
        )

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")
    try:
        for i in range(num_frames):
            frame = _draw_frame(scenario, i, num_frames, width, height)
            writer.write(frame)
    finally:
        writer.release()
    return str(output_path)


def generate_all_mock_videos(output_dir: str | Path, config: dict) -> List[str]:
    """Generate all mock videos listed in ``config['mock']['scenarios']``."""
    mock_cfg = config["mock"]
    width = int(mock_cfg["width"])
    height = int(mock_cfg["height"])
    num_frames = int(mock_cfg["num_frames"])
    fps = int(config["video"].get("annotated_video_fps", 2))

    out_dir = ensure_dir(output_dir)
    paths: List[str] = []
    for scenario in mock_cfg["scenarios"]:
        out_path = out_dir / f"{scenario}.mp4"
        print(f"[mock] generating {scenario} -> {out_path}")
        paths.append(generate_mock_video(
            scenario=scenario,
            output_path=out_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
        ))
    return paths


def generate_mock_detections_for_frames(
    frame_paths: List[str | Path],
    scenario: str,
    output_dir: str | Path,
    video_id: str | None = None,
) -> List[dict]:
    """Produce deterministic mock detections matching the synthetic objects.

    The bbox geometry is recomputed from the same scenario rules used when
    drawing the mock videos, then proportionally rescaled to the actual size
    of each extracted frame (which may differ from the source video size if
    the user changed ``video.resize_width``).

    Returns a list of detection dicts and also writes CSV + JSON files into
    ``output_dir``.
    """
    if scenario not in SCENARIO_TO_CLASS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            f"Known scenarios: {sorted(SCENARIO_TO_CLASS)}"
        )

    frame_paths = [Path(p) for p in frame_paths]
    if not frame_paths:
        return []

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    frame_h, frame_w = first.shape[:2]
    class_name = SCENARIO_TO_CLASS[scenario]
    n = len(frame_paths)

    rows: List[dict] = []
    for idx, fp in enumerate(frame_paths):
        x1, y1, x2, y2 = _scenario_bbox(scenario, idx, n, frame_w, frame_h)
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = float((x2 - x1) * (y2 - y1))
        rows.append({
            "video_id": video_id or scenario,
            "frame_idx": idx,
            "frame_path": str(fp),
            "class_name": class_name,
            "confidence": 0.99,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "cx": cx,
            "cy": cy,
            "area": area,
        })

    out_dir = ensure_dir(output_dir)
    vid = video_id or scenario
    csv_path = out_dir / f"{vid}_detections.csv"
    json_path = out_dir / f"{vid}_detections.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return rows
