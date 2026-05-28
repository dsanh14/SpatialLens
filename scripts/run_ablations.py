"""Isolated-component ablation runner for the final report.

For each contribution introduced over the baseline cascade, this script
disables ONLY that component and re-runs the relevant stages. The
result is a table of (full system accuracy) - (full minus X) per
contribution -- the isolated cost of removing X.

Run:
    python scripts/run_ablations.py

Writes outputs/evaluation/ablations.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.evaluation import evaluate_hazards, load_labels  # noqa: E402
from src.hazard_classifier import HazardClassifier  # noqa: E402
from src.motion_features import compute_track_motion_features  # noqa: E402
from src.track_reid import (  # noqa: E402
    _same_frame_nms,
    _stitch_fragments,
    clean_tracks,
)

DET = Path("outputs/detections")
TRACKS = Path("outputs/tracks")
FRAMES = Path("data/frames")
LABELS_RAW = "data/labels/hazard_labels.csv"
LABELS_DEDUP = "data/labels/hazard_labels_dedup.csv"


def _image_size(vid: str) -> tuple:
    fd = FRAMES / vid
    if not fd.exists():
        return (960, 1707)
    for f in sorted(fd.glob("*.jpg")):
        img = cv2.imread(str(f))
        if img is not None:
            return (img.shape[1], img.shape[0])
    return (960, 1707)


def _run_classifier_only(disabled_rules: set, cfg: dict) -> pd.DataFrame:
    """Re-classify existing track features with rules toggled."""
    clf = HazardClassifier(cfg, disabled_rules=disabled_rules)
    out = []
    for vid in sorted(p.stem.replace("_track_features", "")
                      for p in TRACKS.glob("*_track_features.csv")):
        feats = pd.read_csv(TRACKS / f"{vid}_track_features.csv")
        w, h = _image_size(vid)
        out.append(clf.classify_all(feats, image_width=w, image_height=h))
    return pd.concat(out, ignore_index=True)


def _run_tracker_variant(
    cfg: dict,
    use_nms: bool,
    use_stitching: bool,
) -> pd.DataFrame:
    """Re-run track cleanup + features + classification with toggles."""
    from src.tracking import assign_tracks
    from src.optical_flow import compute_object_flow_features
    from src.frame_diff import compute_object_motion_overlap

    # We need flow + frame_diff attached per detection. Read the current
    # tracks csv (which already has them) and recover detections by
    # dropping track_id/track_age. Then re-track from scratch so the
    # toggle is honest.
    out = []
    for vid in sorted(p.stem.replace("_detections", "")
                      for p in DET.glob("*_detections.csv")):
        det = pd.read_csv(DET / f"{vid}_detections.csv")
        w, h = _image_size(vid)
        tracks = assign_tracks(
            det, w, h, cfg,
            output_dir=None, video_id=vid,
        )
        # Attach motion features (flow + frame diff).
        frame_paths = sorted((FRAMES / vid).glob("*.jpg"))
        if frame_paths:
            tracks = compute_object_motion_overlap(
                detections_df=tracks,
                frame_paths=[str(p) for p in frame_paths],
                config=cfg,
                output_dir=None,
                video_id=vid,
            )
            tracks = compute_object_flow_features(
                tracks_df=tracks,
                frame_paths=[str(p) for p in frame_paths],
                config=cfg,
                output_dir=None,
                video_id=vid,
                save_visualizations=False,
            )
        # Apply selected cleanups.
        if use_nms and not tracks.empty:
            tracks = _same_frame_nms(tracks, iou_threshold=0.5)
        if use_stitching and not tracks.empty:
            tracks = _stitch_fragments(
                tracks, image_width=w, image_height=h,
                max_gap_frames=2, max_dist_frac=0.45,
                max_area_ratio=5.0, appearance_threshold=0.45,
                fragment_threshold=1,
            )
        feats = compute_track_motion_features(
            tracks, cfg, w, h, output_dir=None, video_id=vid,
        )
        clf = HazardClassifier(cfg)
        out.append(clf.classify_all(feats, image_width=w, image_height=h))
    return pd.concat(out, ignore_index=True)


def _accuracy(preds: pd.DataFrame, labels: pd.DataFrame) -> tuple:
    res = evaluate_hazards(preds, labels)
    return (
        res.get("overall_accuracy", 0.0),
        res.get("num_evaluated_tracks", 0),
        res.get("macro_f1", 0.0),
    )


def main() -> None:
    cfg = load_config("config.yaml")
    labels_raw = load_labels(LABELS_RAW)
    labels_dedup = load_labels(LABELS_DEDUP)

    rows = []

    # Full system (baseline for the table).
    full = _run_classifier_only(set(), cfg)
    acc_r, n_r, f1_r = _accuracy(full, labels_raw)
    acc_d, n_d, f1_d = _accuracy(full, labels_dedup)
    rows.append(["full system", acc_r, n_r, f1_r, acc_d, n_d, f1_d])

    # Disable classifier rules one at a time.
    for rule in [
        "2f_crossing_salvage",
        "2f_approach_salvage",
        "soft_moving_away",
        "slow_approach",
        "trajectory_reversal",
    ]:
        preds = _run_classifier_only({rule}, cfg)
        a_r, _, m_r = _accuracy(preds, labels_raw)
        a_d, _, m_d = _accuracy(preds, labels_dedup)
        rows.append([f"-{rule}", a_r, n_r, m_r, a_d, n_d, m_d])

    # Disable tracker improvements one at a time.
    for name, nms, st in [
        ("-appearance_reid", True, False),
        ("-same_frame_nms", False, True),
        ("-nms -reid (both off)", False, False),
    ]:
        preds = _run_tracker_variant(cfg, use_nms=nms, use_stitching=st)
        a_r, n_rv, m_r = _accuracy(preds, labels_raw)
        a_d, n_dv, m_d = _accuracy(preds, labels_dedup)
        rows.append([name, a_r, n_rv, m_r, a_d, n_dv, m_d])

    df = pd.DataFrame(rows, columns=[
        "variant", "raw_acc", "raw_n", "raw_macro_f1",
        "dedup_acc", "dedup_n", "dedup_macro_f1",
    ])
    out = Path("outputs/evaluation/ablations.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
