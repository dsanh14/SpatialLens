"""Object detection backends for SpatialLens Assist.

Two backends are supported (selected by ``config['detection']['backend']``):

* ``"yolo"`` — Ultralytics YOLO (downloaded on first run). Used on real
  videos and any user-provided weights via ``detection.model_name``.
* ``"mock"`` — deterministic detections computed from the synthetic mock
  scenario geometry. Lets the rest of the Week 1-2 pipeline run with no
  model weights and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import pandas as pd
from tqdm import tqdm

from .mock_data import generate_mock_detections_for_frames
from .utils import ensure_dir

DETECTION_COLUMNS = [
    "video_id",
    "frame_idx",
    "frame_path",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "area",
]


def _empty_detections_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical detection columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in DETECTION_COLUMNS})


def _save_detections(
    df: pd.DataFrame,
    output_dir: str | Path,
    video_id: str,
    save_csv: bool = True,
    save_json: bool = True,
) -> None:
    out_dir = ensure_dir(output_dir)
    if save_csv:
        df.to_csv(out_dir / f"{video_id}_detections.csv", index=False)
    if save_json:
        records = df.to_dict(orient="records")
        with (out_dir / f"{video_id}_detections.json").open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)


def _run_yolo_detection(
    frame_paths: List[str | Path],
    config: dict,
    video_id: str,
) -> pd.DataFrame:
    """Run Ultralytics YOLO on each frame and return a DataFrame."""
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only when not installed
        raise RuntimeError(
            "Ultralytics is not installed. Install it via `pip install ultralytics`, "
            "or switch detection.backend to 'mock' in config.yaml."
        ) from e

    det_cfg = config["detection"]
    model_name = det_cfg["model_name"]
    conf_thresh = float(det_cfg["confidence_threshold"])
    target_classes = set(det_cfg["target_classes"])

    print(f"[detect] loading YOLO model: {model_name}")
    model = YOLO(model_name)

    model_class_names = (
        model.names if isinstance(model.names, dict)
        else {i: n for i, n in enumerate(model.names)}
    )
    available = set(model_class_names.values())
    missing = target_classes - available
    if missing:
        print(
            f"[detect][warning] target classes not present in model.names "
            f"(continuing without them): {sorted(missing)}. "
            f"COCO YOLO does not include a dedicated 'scooter' class — see README."
        )

    rows: List[dict] = []
    for fp in tqdm(frame_paths, desc=f"yolo {video_id}", unit="f"):
        fp = Path(fp)
        results = model.predict(source=str(fp), conf=conf_thresh, verbose=False)
        if not results:
            continue
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        try:
            frame_idx = int(fp.stem.split("_")[-1])
        except (ValueError, IndexError):
            frame_idx = -1
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, cls_ids):
            class_name = model_class_names.get(int(cid), str(int(cid)))
            if class_name not in target_classes:
                continue
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            area = float((x2 - x1) * (y2 - y1))
            rows.append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "frame_path": str(fp),
                "class_name": class_name,
                "confidence": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": cx,
                "cy": cy,
                "area": area,
            })

    if not rows:
        return _empty_detections_df()
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def _run_mock_detection(
    frame_paths: List[str | Path],
    config: dict,
    video_id: str,
    mock_scenario: Optional[str],
) -> pd.DataFrame:
    """Generate deterministic mock detections for a scenario."""
    scenario = mock_scenario or video_id
    out_tmp = ensure_dir(Path("outputs") / "detections")
    rows = generate_mock_detections_for_frames(
        frame_paths=frame_paths,
        scenario=scenario,
        output_dir=out_tmp,
        video_id=video_id,
    )
    if not rows:
        return _empty_detections_df()
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def run_detection(
    frame_paths: List[str | Path],
    output_dir: str | Path,
    config: dict,
    video_id: Optional[str] = None,
    mock_scenario: Optional[str] = None,
) -> pd.DataFrame:
    """Run object detection on a list of frames and persist CSV/JSON outputs.

    Parameters
    ----------
    frame_paths:
        Ordered list of frame image paths produced by :mod:`extract_frames`.
    output_dir:
        Directory where ``<video_id>_detections.csv`` and ``.json`` will be
        written.
    config:
        Parsed config dict (see :mod:`src.config`).
    video_id:
        Optional stable identifier for this video. If omitted, attempts to
        infer from the parent directory of the first frame.
    mock_scenario:
        Scenario name to use when ``config['detection']['backend'] == 'mock'``.
        Defaults to ``video_id``.

    Returns
    -------
    pandas.DataFrame
        Detection rows with the canonical Week 1 schema.
    """
    if video_id is None:
        if frame_paths:
            video_id = Path(frame_paths[0]).parent.name
        else:
            video_id = "unknown"

    backend = config["detection"]["backend"]
    if backend not in {"yolo", "mock"}:
        raise ValueError(
            f"Unknown detection backend {backend!r}. Expected 'yolo' or 'mock'."
        )

    if not frame_paths:
        print(f"[detect] no frames provided for video_id={video_id}; "
              f"writing empty detections file.")
        df = _empty_detections_df()
    elif backend == "yolo":
        df = _run_yolo_detection(frame_paths, config, video_id)
    else:
        df = _run_mock_detection(frame_paths, config, video_id, mock_scenario)

    _save_detections(
        df=df,
        output_dir=output_dir,
        video_id=video_id,
        save_csv=bool(config["detection"].get("save_csv", True)),
        save_json=bool(config["detection"].get("save_json", True)),
    )
    print(f"[detect] backend={backend} video_id={video_id} "
          f"-> {len(df)} detections")
    return df


# TODO(Week 3): hazard classifier head will consume per-track motion features
# downstream — detection logic itself does not change in Week 3.
