"""Tests for the codec-fallback video writer in src.utils.

The historical bug we're guarding against: cv2.VideoWriter with the
`mp4v` (MPEG-4 Part 2) fourcc produces files that play in VLC but
render as a solid green/pink color in macOS QuickTime, Finder Quick
Look, and Cursor's built-in video preview. open_video_writer now
prefers `avc1` (H.264) so the generated files play everywhere.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import (
    _VIDEO_FOURCC_FALLBACK,
    open_video_writer,
    write_video_from_frames,
)


def _make_test_frame(path: Path, h=64, w=64, color=(0, 200, 0)) -> None:
    img = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_fallback_chain_lists_h264_first():
    """Future-proof against accidentally re-ordering the chain. H.264
    MUST come before mp4v or we're back to the QuickTime green-screen
    bug."""
    assert _VIDEO_FOURCC_FALLBACK[0] in ("avc1", "H264")
    assert "mp4v" in _VIDEO_FOURCC_FALLBACK
    assert _VIDEO_FOURCC_FALLBACK.index("mp4v") > _VIDEO_FOURCC_FALLBACK.index(
        _VIDEO_FOURCC_FALLBACK[0]
    )


def test_open_video_writer_returns_opened_writer(tmp_path):
    out = tmp_path / "tiny.mp4"
    writer, fourcc = open_video_writer(out, fps=2, width=64, height=64)
    try:
        assert writer.isOpened()
        assert fourcc in _VIDEO_FOURCC_FALLBACK
    finally:
        writer.release()


def test_write_video_from_frames_produces_readable_mp4(tmp_path):
    """Round-trip: write frames -> read the resulting mp4 -> we should
    get exactly the same number of frames back at the same size."""
    n = 5
    frame_paths = []
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.jpg"
        _make_test_frame(p)
        frame_paths.append(p)
    out = tmp_path / "out.mp4"
    write_video_from_frames(frame_paths, out, fps=2)
    assert out.exists()
    assert out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened(), "the file we just wrote should be readable"
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (64, 64)
        count += 1
    cap.release()
    assert count == n, f"expected {n} frames back, read {count}"


def test_write_video_from_frames_returns_empty_on_no_frames(tmp_path):
    out = tmp_path / "empty.mp4"
    result = write_video_from_frames([], out, fps=2)
    assert result == ""
    assert not out.exists()
