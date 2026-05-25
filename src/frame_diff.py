"""Frame differencing to estimate raw motion regions.

Used both as a sanity check / visualization and as a per-bbox feature
(``frame_diff_overlap``: fraction of the bbox area that overlaps the
binary motion mask).

When ``motion.ego_motion_compensation`` is enabled (default True), the
previous frame is translation-aligned to the current frame using
``cv2.findTransformECC`` *before* the absolute difference is computed.
This removes most of the false-positive motion mask area caused by the
camera panning, so the per-bbox overlap measures genuine object motion
rather than image-plane shift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import ensure_dir

# ECC alignment hyperparameters. 30 iterations is plenty for pure
# translation on small (< 50 px) global shifts; we cap it to keep total
# runtime under a few seconds even on long videos. Affine convergence
# needs a few more iterations because 6 parameters take longer than 2.
_ECC_ITERS_TRANSLATION = 30
_ECC_ITERS_AFFINE = 50
_ECC_EPS = 1e-4

_ECC_MOTION_TYPES = {
    "translation": cv2.MOTION_TRANSLATION,
    "affine": cv2.MOTION_AFFINE,
}


def estimate_ego_motion_ecc(
    prev_gray: np.ndarray, curr_gray: np.ndarray,
    *,
    motion_model: str = "translation",
) -> np.ndarray:
    """Estimate the ego-motion warp from ``prev`` to ``curr`` via ECC.

    Returns a 2x3 warp matrix ``W`` such that
    ``cv2.warpAffine(prev, W, (w, h)) ~= curr``.

    ``motion_model="translation"`` is fast and stable but only handles
    pure pan (2 parameters, ``tx`` and ``ty``).
    ``motion_model="affine"`` is a 6-parameter fit that additionally
    captures rotation, scale (zoom), and shear — useful for hand-held
    footage with sharp turns or rapid forward motion. It's slower and a
    bit less robust on low-texture frames, so we fall back to identity
    if the solver fails to converge.

    Note on argument order: empirically,
    ``findTransformECC(template, input, warp)`` returns a warp such that
    ``warpAffine(template, warp) ~= input``. To get a warp that maps
    ``prev`` onto ``curr`` we therefore pass ``template=prev``,
    ``input=curr`` — not the reverse.
    """
    mt = _ECC_MOTION_TYPES.get(motion_model.lower(), cv2.MOTION_TRANSLATION)
    iters = (
        _ECC_ITERS_AFFINE if mt == cv2.MOTION_AFFINE else _ECC_ITERS_TRANSLATION
    )
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
                iters, _ECC_EPS)
    try:
        _cc, warp = cv2.findTransformECC(
            prev_gray, curr_gray, warp,
            motionType=mt,
            criteria=criteria,
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)
    return warp


def estimate_translation_ecc(
    prev_gray: np.ndarray, curr_gray: np.ndarray,
) -> tuple[float, float]:
    """Backwards-compatible translation-only ECC wrapper.

    Kept for tests and external callers that only need the pure pan
    estimate. New code should call :func:`estimate_ego_motion_ecc`,
    which can optionally return an affine warp.
    """
    warp = estimate_ego_motion_ecc(
        prev_gray, curr_gray, motion_model="translation",
    )
    return float(warp[0, 2]), float(warp[1, 2])


def _warp_is_significant(warp: np.ndarray) -> bool:
    """Whether an ECC warp is meaningfully different from identity.

    We skip the warp step (and the corresponding interpolation cost)
    when the estimated motion is sub-pixel and the linear part is
    essentially the identity, since the resulting `prev` would be
    pixel-identical to the unwarped one.
    """
    tx, ty = warp[0, 2], warp[1, 2]
    a, b = warp[0, 0], warp[0, 1]
    c, d = warp[1, 0], warp[1, 1]
    linear_dev = abs(a - 1.0) + abs(d - 1.0) + abs(b) + abs(c)
    return abs(tx) > 0.5 or abs(ty) > 0.5 or linear_dev > 1e-3


def compute_frame_difference(
    prev_frame_path: str | Path,
    curr_frame_path: str | Path,
    threshold: int = 25,
    kernel_size: int = 5,
    ego_motion_compensation: bool = True,
    ego_motion_model: str = "translation",
) -> np.ndarray:
    """Return a binary motion mask between two consecutive frames.

    Steps: grayscale -> (optional) ECC alignment of the previous frame
    onto the current frame using the requested motion model
    (``"translation"`` or ``"affine"``) -> ``cv2.absdiff`` -> threshold
    -> morphological open + close.
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

    if ego_motion_compensation:
        warp = estimate_ego_motion_ecc(
            prev, curr, motion_model=ego_motion_model,
        )
        if _warp_is_significant(warp):
            h, w = curr.shape[:2]
            prev = cv2.warpAffine(
                prev, warp, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

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
    ego_comp = bool(motion_cfg.get("ego_motion_compensation", True))
    ego_model = str(motion_cfg.get("ego_motion_model", "translation")).lower()
    if ego_model not in _ECC_MOTION_TYPES:
        print(f"[frame_diff][warn] unknown ego_motion_model={ego_model!r}; "
              f"falling back to 'translation'.")
        ego_model = "translation"

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
            ego_motion_compensation=ego_comp,
            ego_motion_model=ego_model,
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
