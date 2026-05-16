"""Unit tests for frame differencing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.frame_diff import compute_frame_difference


def _save_gray_frame(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def test_changed_rectangle_creates_nonzero_motion_mask(tmp_path):
    """A bright rectangle that appears in frame 2 should light up the mask."""
    h, w = 200, 300
    f1 = np.full((h, w), 30, dtype=np.uint8)
    f2 = f1.copy()
    cv2.rectangle(f2, (80, 60), (180, 140), 230, thickness=-1)

    p1 = tmp_path / "f1.jpg"
    p2 = tmp_path / "f2.jpg"
    _save_gray_frame(p1, f1)
    _save_gray_frame(p2, f2)

    mask = compute_frame_difference(p1, p2, threshold=25, kernel_size=5)
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    nonzero = int(np.count_nonzero(mask))
    assert nonzero > 0, "Expected the changed rectangle to produce motion pixels."

    # Most motion should be roughly where the rectangle was drawn.
    sub = mask[60:140, 80:180]
    assert np.count_nonzero(sub) > 0.5 * nonzero


def test_identical_frames_produce_empty_mask(tmp_path):
    h, w = 100, 100
    f = np.full((h, w), 100, dtype=np.uint8)
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    _save_gray_frame(p1, f)
    _save_gray_frame(p2, f)
    mask = compute_frame_difference(p1, p2, threshold=25, kernel_size=5)
    assert int(np.count_nonzero(mask)) == 0
