"""Dense optical flow (Farneback) and per-bbox motion summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import ensure_dir


def compute_optical_flow(
    prev_frame_path: str | Path,
    curr_frame_path: str | Path,
) -> np.ndarray:
    """Return Farneback dense optical flow ``flow`` of shape ``(H, W, 2)``.

    ``flow[..., 0]`` is dx and ``flow[..., 1]`` is dy, where positive dx
    is rightward and positive dy is downward (image convention).
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
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow


def _flow_to_visualization(flow: np.ndarray) -> np.ndarray:
    """Encode a flow field as an HSV image for visualization."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = np.uint8(ang * 180.0 / np.pi / 2.0)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_object_flow_features(
    tracks_df: pd.DataFrame,
    frame_paths: List[str | Path],
    config: dict,
    output_dir: str | Path | None = None,
    video_id: str | None = None,
    save_visualizations: bool = True,
) -> pd.DataFrame:
    """Add ``flow_dx`` / ``flow_dy`` / ``flow_mag`` columns to each track row.

    For each detection at frame ``t``, the median dx / dy and mean
    magnitude of the Farneback flow inside the bbox are computed. Frame 0
    (no previous frame) is assigned ``0.0`` (not NaN) so downstream
    aggregations stay numeric.
    """
    df = tracks_df.copy()
    if df.empty:
        df["flow_dx"] = pd.Series(dtype=float)
        df["flow_dy"] = pd.Series(dtype=float)
        df["flow_mag"] = pd.Series(dtype=float)
        return df

    frame_paths = [Path(p) for p in frame_paths]
    path_to_idx = {str(p): i for i, p in enumerate(frame_paths)}

    save_dir: Path | None = None
    if save_visualizations and output_dir is not None and video_id is not None:
        save_dir = ensure_dir(Path(output_dir) / video_id)

    flow_cache: Dict[int, np.ndarray] = {}
    for t in tqdm(range(1, len(frame_paths)),
                  desc=f"optical_flow {video_id or ''}", unit="f"):
        try:
            flow = compute_optical_flow(frame_paths[t - 1], frame_paths[t])
        except RuntimeError as e:
            print(f"[optical_flow][warn] {e}; setting flow=zeros for frame {t}.")
            sample = cv2.imread(str(frame_paths[t]))
            if sample is None:
                continue
            flow = np.zeros((sample.shape[0], sample.shape[1], 2), dtype=np.float32)
        flow_cache[t] = flow
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"flow_{t:04d}.jpg"),
                        _flow_to_visualization(flow))

    dxs: List[float] = []
    dys: List[float] = []
    mags: List[float] = []
    for _, row in df.iterrows():
        fp = str(row["frame_path"])
        t = path_to_idx.get(fp, int(row["frame_idx"]))
        flow = flow_cache.get(t)
        if flow is None:
            dxs.append(0.0)
            dys.append(0.0)
            mags.append(0.0)
            continue
        h, w = flow.shape[:2]
        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(w, int(row["x2"]))
        y2 = min(h, int(row["y2"]))
        if x2 <= x1 or y2 <= y1:
            dxs.append(0.0)
            dys.append(0.0)
            mags.append(0.0)
            continue
        sub = flow[y1:y2, x1:x2]
        if sub.size == 0:
            dxs.append(0.0)
            dys.append(0.0)
            mags.append(0.0)
            continue
        sub_dx = sub[..., 0]
        sub_dy = sub[..., 1]
        mag = np.sqrt(sub_dx**2 + sub_dy**2)
        dxs.append(float(np.median(sub_dx)))
        dys.append(float(np.median(sub_dy)))
        mags.append(float(np.mean(mag)))

    df["flow_dx"] = dxs
    df["flow_dy"] = dys
    df["flow_mag"] = mags
    return df
