"""Dense optical flow (Farneback) and per-bbox motion summaries.

Optionally subtracts the *global* (camera ego-motion) flow from each
per-object flow so the per-bbox numbers reflect motion *relative to the
background* rather than absolute pixel motion. This is enabled via
``motion.ego_motion_compensation`` (default True) and is important on
hand-held footage where the whole frame is panning.

The global translation is the per-frame median of the dense flow field.
The median is robust to small moving foreground objects (which take up
only a small fraction of the frame) and is dominated by the background.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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


def estimate_global_flow(flow: np.ndarray) -> Tuple[float, float]:
    """Return the per-frame ``(dx, dy)`` ego-motion estimate from a flow field.

    Uses the median over the entire flow array. Because the median is
    robust to a small fraction of outlier pixels (the moving foreground
    objects), the result is dominated by the static background and is a
    good first-order estimate of camera translation between the two
    frames.
    """
    if flow is None or flow.size == 0:
        return 0.0, 0.0
    return float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))


def per_bbox_flow_stats(
    flow: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *,
    global_dx: float = 0.0,
    global_dy: float = 0.0,
) -> Tuple[float, float, float]:
    """Return ``(median_dx, median_dy, mean_mag)`` inside a bbox region.

    If ``global_dx`` / ``global_dy`` are provided, the camera ego-motion
    is subtracted *per pixel* before computing the per-bbox summaries —
    so the returned numbers describe object motion **relative to the
    background**, not absolute pixel motion in the image plane.
    """
    h, w = flow.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    sub = flow[y1:y2, x1:x2]
    if sub.size == 0:
        return 0.0, 0.0, 0.0
    sub_dx = sub[..., 0] - global_dx
    sub_dy = sub[..., 1] - global_dy
    mag = np.sqrt(sub_dx ** 2 + sub_dy ** 2)
    return float(np.median(sub_dx)), float(np.median(sub_dy)), float(np.mean(mag))


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
        df["global_flow_dx"] = pd.Series(dtype=float)
        df["global_flow_dy"] = pd.Series(dtype=float)
        return df

    motion_cfg = config.get("motion", {}) if isinstance(config, dict) else {}
    compensate = bool(motion_cfg.get("ego_motion_compensation", True))

    frame_paths = [Path(p) for p in frame_paths]
    path_to_idx = {str(p): i for i, p in enumerate(frame_paths)}

    save_dir: Path | None = None
    if save_visualizations and output_dir is not None and video_id is not None:
        save_dir = ensure_dir(Path(output_dir) / video_id)

    flow_cache: Dict[int, np.ndarray] = {}
    global_flow: Dict[int, Tuple[float, float]] = {}
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
        gdx, gdy = estimate_global_flow(flow) if compensate else (0.0, 0.0)
        global_flow[t] = (gdx, gdy)
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"flow_{t:04d}.jpg"),
                        _flow_to_visualization(flow))

    dxs: List[float] = []
    dys: List[float] = []
    mags: List[float] = []
    g_dxs: List[float] = []
    g_dys: List[float] = []
    for _, row in df.iterrows():
        fp = str(row["frame_path"])
        t = path_to_idx.get(fp, int(row["frame_idx"]))
        flow = flow_cache.get(t)
        gdx, gdy = global_flow.get(t, (0.0, 0.0))
        g_dxs.append(gdx)
        g_dys.append(gdy)
        if flow is None:
            dxs.append(0.0)
            dys.append(0.0)
            mags.append(0.0)
            continue
        dx, dy, mag = per_bbox_flow_stats(
            flow,
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
            global_dx=gdx, global_dy=gdy,
        )
        dxs.append(dx)
        dys.append(dy)
        mags.append(mag)

    df["flow_dx"] = dxs
    df["flow_dy"] = dys
    df["flow_mag"] = mags
    df["global_flow_dx"] = g_dxs
    df["global_flow_dy"] = g_dys
    return df
