"""Frame differencing to estimate raw motion regions.

Used both as a sanity check / visualization and as a per-bbox feature
(``frame_diff_overlap``: fraction of the bbox area that overlaps the
binary motion mask).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import ensure_dir


def compute_frame_difference(
    prev_frame_path: str | Path,
    curr_frame_path: str | Path,
    threshold: int = 25,
    kernel_size: int = 5,
) -> np.ndarray:
    """Return a binary motion mask between two consecutive frames.

    Steps: grayscale -> ``cv2.absdiff`` -> threshold -> morphological
    open + close to clean speckle and fill small holes.
    """
    prev = cv2.imread(str(prev_frame_path), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(curr_frame_path), cv2.IMREAD_GRAYSCALE)
    if prev is None or curr is None:
        raise RuntimeError(
            f"Could not read frames: prev={prev_frame_path} "
            f"curr={curr_frame_path}"
        )
    if prev.shape != curr.shape:
        prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]))

    diff = cv2.absdiff(prev, curr)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)

    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_object_motion_overlap(
    detections_df: pd.DataFrame,
    frame_paths: List[str | Path],
    config: dict,
    output_dir: str | Path | None = None,
    video_id: str | None = None,
) -> pd.DataFrame:
    """Add a ``frame_diff_overlap`` column to detections.

    For each detection at frame ``t``, computes the binary motion mask
    between frame ``t-1`` and frame ``t``, then measures the fraction of
    the bbox area that overlaps the motion mask. Frame 0 (no previous
    frame) is assigned ``0.0``.

    If ``output_dir`` is given, the motion masks are also written as
    ``frame_diff_XXXX.jpg`` images for visualization.
    """
    motion_cfg = config["motion"]
    threshold = int(motion_cfg["frame_diff_threshold"])
    kernel_size = int(motion_cfg["morphology_kernel_size"])

    df = detections_df.copy()
    if df.empty:
        df["frame_diff_overlap"] = pd.Series(dtype=float)
        return df

    frame_paths = [Path(p) for p in frame_paths]
    path_to_idx = {str(p): i for i, p in enumerate(frame_paths)}

    save_dir: Path | None = None
    if output_dir is not None and video_id is not None:
        save_dir = ensure_dir(Path(output_dir) / video_id)

    # Compute and (optionally) save one mask per (t-1, t) pair.
    masks: Dict[int, np.ndarray] = {}
    for t in tqdm(range(1, len(frame_paths)),
                  desc=f"frame_diff {video_id or ''}", unit="f"):
        mask = compute_frame_difference(
            frame_paths[t - 1], frame_paths[t],
            threshold=threshold, kernel_size=kernel_size,
        )
        masks[t] = mask
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"frame_diff_{t:04d}.jpg"), mask)

    overlaps: List[float] = []
    for _, row in df.iterrows():
        fp = str(row["frame_path"])
        t = path_to_idx.get(fp, int(row["frame_idx"]))
        mask = masks.get(t)
        if mask is None:
            overlaps.append(0.0)
            continue
        h, w = mask.shape[:2]
        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(w, int(row["x2"]))
        y2 = min(h, int(row["y2"]))
        if x2 <= x1 or y2 <= y1:
            overlaps.append(0.0)
            continue
        sub = mask[y1:y2, x1:x2]
        bbox_area = float((x2 - x1) * (y2 - y1))
        motion_area = float(np.count_nonzero(sub))
        overlaps.append(motion_area / bbox_area if bbox_area > 0 else 0.0)

    df["frame_diff_overlap"] = overlaps
    return df
