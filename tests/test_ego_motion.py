"""Unit tests for ego-motion compensation in optical flow and frame diff.

These tests verify the math of the helpers directly (using synthetic
flow fields and synthetic frame pairs) so they do not depend on the
specific behavior of Farneback or ECC on real images.
"""

from __future__ import annotations

import numpy as np

from src.frame_diff import (
    compute_frame_difference,
    estimate_ego_motion_ecc,
    estimate_translation_ecc,
)
from src.optical_flow import estimate_global_flow, per_bbox_flow_stats


# --------------------------------------------------------------------- #
# Optical flow: median-of-flow ego-motion estimate.
# --------------------------------------------------------------------- #

def _flow_with_background_pan_and_object(
    h: int, w: int,
    bg_dx: float, bg_dy: float,
    obj_dx: float, obj_dy: float,
    bx1: int, by1: int, bx2: int, by2: int,
) -> np.ndarray:
    """Build a synthetic dense flow array.

    Background pixels get ``(bg_dx, bg_dy)``; pixels inside the object
    bbox get ``(obj_dx, obj_dy)`` (these are absolute image-plane flows,
    just like Farneback would output).
    """
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = bg_dx
    flow[..., 1] = bg_dy
    flow[by1:by2, bx1:bx2, 0] = obj_dx
    flow[by1:by2, bx1:bx2, 1] = obj_dy
    return flow


def test_global_flow_recovers_background_pan():
    flow = _flow_with_background_pan_and_object(
        h=200, w=200,
        bg_dx=5.0, bg_dy=-2.0,
        obj_dx=15.0, obj_dy=8.0,
        bx1=80, by1=80, bx2=120, by2=120,  # small (20% area)
    )
    gdx, gdy = estimate_global_flow(flow)
    # Median is dominated by the static-but-panning background.
    assert abs(gdx - 5.0) < 0.5
    assert abs(gdy + 2.0) < 0.5


def test_per_bbox_flow_uncompensated_reports_absolute_motion():
    flow = _flow_with_background_pan_and_object(
        h=200, w=200,
        bg_dx=5.0, bg_dy=-2.0,
        obj_dx=15.0, obj_dy=8.0,
        bx1=80, by1=80, bx2=120, by2=120,
    )
    dx, dy, mag = per_bbox_flow_stats(flow, 80, 80, 120, 120)
    # No compensation -> we get the raw image-plane motion (15, 8).
    assert abs(dx - 15.0) < 0.1
    assert abs(dy - 8.0) < 0.1
    assert mag > 0


def test_per_bbox_flow_with_compensation_reports_motion_relative_to_background():
    flow = _flow_with_background_pan_and_object(
        h=200, w=200,
        bg_dx=5.0, bg_dy=-2.0,
        obj_dx=15.0, obj_dy=8.0,
        bx1=80, by1=80, bx2=120, by2=120,
    )
    gdx, gdy = estimate_global_flow(flow)
    dx, dy, mag = per_bbox_flow_stats(
        flow, 80, 80, 120, 120, global_dx=gdx, global_dy=gdy,
    )
    # After compensation, only the object's motion *relative to the
    # background* remains: (15-5, 8-(-2)) = (10, 10).
    assert abs(dx - 10.0) < 0.5
    assert abs(dy - 10.0) < 0.5


def test_static_object_under_pan_compensates_to_zero():
    """An object that's truly stationary in the world frame moves in
    image coordinates only because of the camera pan. After
    compensation, its per-bbox flow should be ~zero."""
    flow = _flow_with_background_pan_and_object(
        h=200, w=200,
        bg_dx=7.0, bg_dy=-3.0,
        obj_dx=7.0, obj_dy=-3.0,  # same as background -> truly static
        bx1=80, by1=80, bx2=120, by2=120,
    )
    gdx, gdy = estimate_global_flow(flow)
    dx, dy, mag = per_bbox_flow_stats(
        flow, 80, 80, 120, 120, global_dx=gdx, global_dy=gdy,
    )
    assert abs(dx) < 0.5
    assert abs(dy) < 0.5
    assert mag < 1.0


def test_global_flow_handles_empty_array():
    gdx, gdy = estimate_global_flow(np.zeros((0, 0, 2), dtype=np.float32))
    assert (gdx, gdy) == (0.0, 0.0)


def test_per_bbox_flow_handles_degenerate_bbox():
    flow = np.zeros((100, 100, 2), dtype=np.float32)
    dx, dy, mag = per_bbox_flow_stats(flow, 50, 50, 50, 50)
    assert (dx, dy, mag) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------- #
# Frame differencing: ECC translation alignment.
# --------------------------------------------------------------------- #

def _make_frame(h, w, square=(50, 50, 80, 80), bg=100, fg=200, shift=(0, 0)):
    """Build a grayscale frame with a bright square on a uniform background.

    The whole frame is shifted by ``shift = (dx, dy)`` so we can simulate
    a global pan.
    """
    img = np.full((h, w), bg, dtype=np.uint8)
    dx, dy = shift
    x1, y1, x2, y2 = square
    x1 += dx; x2 += dx
    y1 += dy; y2 += dy
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    img[y1:y2, x1:x2] = fg
    return img


def _textured_frame(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Build a uniquely-textured grayscale frame for ECC tests.

    A pure stripe / gradient pattern is too repetitive for ECC to lock
    on (translations modulo the stripe period are indistinguishable),
    so we add a layer of low-frequency noise that breaks symmetry.
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    base = (xs * 0.7 + ys * 0.4) % 200.0
    noise = rng.normal(0.0, 25.0, size=(h, w)).astype(np.float32)
    img = base + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def test_ecc_recovers_pure_translation():
    h, w = 200, 200
    prev = _textured_frame(h, w, seed=0)
    # Shift right by 4 pixels using array slicing (avoids wraparound).
    curr = np.zeros_like(prev)
    curr[:, 4:] = prev[:, :-4]
    curr[:, :4] = prev[:, :4]  # any plausible fill is fine; ECC ignores edges
    tx, ty = estimate_translation_ecc(prev, curr)
    # Should recover ~+4 in x and ~0 in y, within ECC tolerance.
    assert abs(tx - 4.0) < 1.5
    assert abs(ty) < 1.0


def test_affine_ecc_recovers_pure_rotation():
    """A small (~3deg) rotation around the image center should be
    recovered by the affine ECC model, where the translation model
    cannot represent it at all."""
    import cv2
    h, w = 200, 200
    prev = _textured_frame(h, w, seed=2)
    angle_deg = 3.0
    R = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    curr = cv2.warpAffine(prev, R, (w, h), borderMode=cv2.BORDER_REPLICATE)

    warp_aff = estimate_ego_motion_ecc(prev, curr, motion_model="affine")
    # Apply the recovered warp to prev and compare to curr.
    prev_aligned = cv2.warpAffine(
        prev, warp_aff, (w, h), borderMode=cv2.BORDER_REPLICATE,
    )
    aff_resid = float(np.mean(np.abs(
        prev_aligned.astype(np.int32) - curr.astype(np.int32),
    )))

    # Translation-only cannot represent rotation -> residual stays high.
    tx, ty = estimate_translation_ecc(prev, curr)
    W_t = np.float32([[1, 0, tx], [0, 1, ty]])
    prev_t = cv2.warpAffine(
        prev, W_t, (w, h), borderMode=cv2.BORDER_REPLICATE,
    )
    tr_resid = float(np.mean(np.abs(
        prev_t.astype(np.int32) - curr.astype(np.int32),
    )))

    # The affine residual should be substantially lower.
    assert aff_resid < tr_resid * 0.6, (
        f"affine should outperform translation on rotation: "
        f"aff={aff_resid:.2f} tr={tr_resid:.2f}"
    )


def test_frame_diff_with_ego_motion_compensation_reduces_false_motion(tmp_path):
    """Synthesize a static-textured scene panned by 4 px between the two
    frames. Without compensation the absdiff highlights the whole frame;
    with compensation, the mask should have far fewer nonzero pixels."""
    import cv2
    h, w = 200, 200
    prev = _textured_frame(h, w, seed=1)
    curr = np.zeros_like(prev)
    curr[:, 4:] = prev[:, :-4]
    curr[:, :4] = prev[:, :4]
    prev_path = tmp_path / "prev.png"
    curr_path = tmp_path / "curr.png"
    cv2.imwrite(str(prev_path), prev)
    cv2.imwrite(str(curr_path), curr)

    mask_off = compute_frame_difference(
        prev_path, curr_path, threshold=10, kernel_size=3,
        ego_motion_compensation=False,
    )
    mask_on = compute_frame_difference(
        prev_path, curr_path, threshold=10, kernel_size=3,
        ego_motion_compensation=True,
    )
    nz_off = int(np.count_nonzero(mask_off))
    nz_on = int(np.count_nonzero(mask_on))
    # Compensation should cut the false-motion area by at least half.
    # We don't aim for ~0 because the edge band (4 px) can't be aligned.
    assert nz_on < nz_off * 0.5, (
        f"compensation should suppress most pan-induced motion: "
        f"off={nz_off} on={nz_on}"
    )
