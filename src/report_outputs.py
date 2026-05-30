"""Collect the key images / videos / texts a single video produced into
``outputs/slide_assets/{video_id}/`` for easy drag-and-drop into Google
Slides. Also writes a small README that maps each asset to a slide.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .utils import ensure_dir

# All source paths are relative to the repo root.
FRAMES_DIR = Path("data/frames")
ANNOTATED_FRAMES_DIR = Path("outputs/annotated_frames")
ANNOTATED_VIDEOS_DIR = Path("outputs/annotated_videos")
PLOTS_DIR = Path("outputs/plots")
ALERTS_DIR = Path("outputs/alerts")
EVALUATION_DIR = Path("outputs/evaluation")
SLIDE_ASSETS_DIR = Path("outputs/slide_assets")


def _pick_middle(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return paths[len(paths) // 2]


def _copy(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def export_slide_assets(video_id: str, config: dict) -> Dict[str, str]:
    """Copy the key files for ``video_id`` into ``outputs/slide_assets/``.

    Returns a dict of ``{asset_kind: destination_path}`` (only entries
    that actually existed are included).
    """
    out_dir = ensure_dir(SLIDE_ASSETS_DIR / video_id)
    written: Dict[str, str] = {}

    sample_frames = sorted((FRAMES_DIR / video_id).glob("frame_*.jpg"))
    sample = _pick_middle(sample_frames)
    if sample:
        path = _copy(sample, out_dir / "01_sample_frame.jpg")
        if path:
            written["sample_frame"] = path

    det_frames = sorted((ANNOTATED_FRAMES_DIR / video_id).glob("frame_*.jpg"))
    det = _pick_middle(det_frames)
    if det:
        path = _copy(det, out_dir / "02_detection_frame.jpg")
        if path:
            written["detection_frame"] = path

    track_frames = sorted(
        (ANNOTATED_FRAMES_DIR / f"{video_id}_tracks").glob("frame_*.jpg")
    )
    trk = _pick_middle(track_frames)
    if trk:
        path = _copy(trk, out_dir / "03_tracking_frame.jpg")
        if path:
            written["tracking_frame"] = path

    hazard_frames = sorted(
        (ANNOTATED_FRAMES_DIR / f"{video_id}_hazards").glob("frame_*.jpg")
    )
    haz = _pick_middle(hazard_frames)
    if haz:
        path = _copy(haz, out_dir / "04_hazard_frame.jpg")
        if path:
            written["hazard_frame"] = path

    for src_name, dst_name, key in (
        (f"{video_id}_trajectories.png", "05_trajectories.png", "trajectories_plot"),
        (f"{video_id}_bbox_area.png", "06_bbox_area.png", "bbox_area_plot"),
        (f"{video_id}_approach_scores.png", "07_approach_scores.png",
         "approach_scores_plot"),
        (f"{video_id}_hazard_timeline.png", "08_hazard_timeline.png",
         "hazard_timeline_plot"),
        (f"{video_id}_motion_features.png", "09_motion_features.png",
         "motion_features_plot"),
        (f"{video_id}_confusion_matrix.png", "10_confusion_matrix.png",
         "confusion_matrix_plot"),
        (f"{video_id}_uncertain_reasons.png", "13_uncertain_reasons.png",
         "uncertain_reasons_plot"),
    ):
        path = _copy(PLOTS_DIR / src_name, out_dir / dst_name)
        if path:
            written[key] = path

    alerts_txt = ALERTS_DIR / f"{video_id}_alerts.txt"
    if alerts_txt.exists():
        path = _copy(alerts_txt, out_dir / "11_alerts.txt")
        if path:
            written["alerts_txt"] = path

    eval_json = EVALUATION_DIR / f"{video_id}_evaluation_summary.json"
    if eval_json.exists():
        path = _copy(eval_json, out_dir / "12_evaluation_summary.json")
        if path:
            written["evaluation_summary_json"] = path

    if bool(config.get("final_outputs", {}).get("export_demo_video", True)):
        for src_name, dst_name, key in (
            (f"{video_id}_hazards.mp4", "demo_hazards.mp4", "hazard_video"),
            (f"{video_id}_tracks.mp4", "demo_tracks.mp4", "tracking_video"),
            (f"{video_id}_detections.mp4", "demo_detections.mp4",
             "detection_video"),
        ):
            path = _copy(ANNOTATED_VIDEOS_DIR / src_name, out_dir / dst_name)
            if path:
                written[key] = path

    readme_path = out_dir / "README_slide_assets.txt"
    readme_path.write_text(_build_readme_text(video_id, written), encoding="utf-8")
    written["readme"] = str(readme_path)
    print(f"[report_outputs] wrote {len(written)} slide assets -> {out_dir}")
    return written


def _build_readme_text(video_id: str, written: Dict[str, str]) -> str:
    def present(key: str) -> str:
        return f"  -> {Path(written[key]).name}" if key in written else "  (not generated)"

    return (
        f"SpatialLens Assist — Slide assets for video '{video_id}'\n"
        + "=" * 60 + "\n\n"
        "Suggested mapping to Google Slides:\n\n"
        "Title / Problem slide:\n"
        f"  Use the sample frame.\n{present('sample_frame')}\n\n"
        "Method slide (perception pipeline):\n"
        "  Show 'detection frame' next to 'tracking frame' to convey the\n"
        "  detect -> track pipeline.\n"
        f"  Detection:{present('detection_frame')}\n"
        f"  Tracking: {present('tracking_frame')}\n\n"
        "Motion features slide:\n"
        f"  Trajectories:    {present('trajectories_plot')}\n"
        f"  Bbox area:       {present('bbox_area_plot')}\n"
        f"  Motion features: {present('motion_features_plot')}\n\n"
        "Results slide (hazards):\n"
        f"  Hazard frame (use as the hero image):\n{present('hazard_frame')}\n"
        f"  Hazard timeline: {present('hazard_timeline_plot')}\n"
        f"  Approach scores: {present('approach_scores_plot')}\n"
        f"  Confusion matrix (if labels exist): "
        f"{present('confusion_matrix_plot')}\n"
        f"  Uncertain reasons breakdown:        "
        f"{present('uncertain_reasons_plot')}\n\n"
        "Demo slide:\n"
        f"  Final hazard video: {present('hazard_video')}\n"
        f"  Or tracking video:  {present('tracking_video')}\n"
        f"  Or detection video: {present('detection_video')}\n\n"
        "Spoken-alert handout:\n"
        f"  {present('alerts_txt')}\n"
        f"\nFull evaluation JSON: {present('evaluation_summary_json')}\n"
    )


def _fmt_pct(x: Optional[float]) -> str:
    return f"{100.0 * float(x):.1f}%" if x is not None else "n/a"


def build_results_summary_text(agg: dict) -> str:
    """One-screen results summary for the Results slide, headlining the
    selective ("decidable") accuracy next to the overall number."""
    sel = agg.get("selective_accuracy") or {}
    lines = [
        "SpatialLens Assist — aggregate results",
        "=" * 44,
        "",
    ]
    if sel:
        lines.append(
            f"Decidable accuracy (tracks >= {sel.get('min_track_frames', 3)} "
            f"frames): {_fmt_pct(sel.get('decidable_accuracy'))} "
            f"({sel.get('num_decidable_tracks', 0)} tracks)"
        )
    lines.append(
        f"Overall accuracy (all evaluated): "
        f"{_fmt_pct(agg.get('overall_accuracy'))} "
        f"({agg.get('num_evaluated_tracks', 0)} tracks)"
    )
    if sel:
        lines.append(
            f"  (of which {sel.get('num_short_tracks', 0)} are 1-2 frame "
            f"tracks the system abstains on by design)"
        )
    lines.append(f"Macro F1: {agg.get('macro_f1', 0):.3f}")
    lines.append(
        f"'approaching' (safety-critical): "
        f"precision {agg.get('approaching_precision', 0):.2f}, "
        f"recall {agg.get('approaching_recall', 0):.2f}, "
        f"F1 {agg.get('approaching_f1', 0):.2f}"
    )
    lines.append("")
    lines.append("Per-class F1:")
    for lab, m in (agg.get("per_class_metrics") or {}).items():
        lines.append(f"  {lab:24s} F1={m.get('f1', 0):.2f}  (n={m.get('support', 0)})")
    return "\n".join(lines) + "\n"


def build_results_card_png(agg: dict, png_path: str | Path) -> Optional[str]:
    """Render a clean slide-ready metrics card (PNG) from an aggregate
    evaluation summary dict. Returns the path, or None if matplotlib is
    unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    sel = agg.get("selective_accuracy") or {}
    ensure_dir(Path(png_path).parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")

    headline = _fmt_pct(sel.get("decidable_accuracy")) if sel else _fmt_pct(
        agg.get("overall_accuracy"))
    ax.text(0.5, 0.93, "SpatialLens Assist — Results", ha="center",
            va="top", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.78, headline, ha="center", va="center", fontsize=40,
            fontweight="bold", color="#1a6f3c")
    sub = (f"decidable accuracy  (tracks \u2265 "
           f"{sel.get('min_track_frames', 3)} frames, "
           f"{sel.get('num_decidable_tracks', 0)} tracks)") if sel else \
          f"overall accuracy ({agg.get('num_evaluated_tracks', 0)} tracks)"
    ax.text(0.5, 0.62, sub, ha="center", va="center", fontsize=12,
            color="#444444")

    rows = [
        ("Overall accuracy (incl. abstentions)",
         f"{_fmt_pct(agg.get('overall_accuracy'))} "
         f"({agg.get('num_evaluated_tracks', 0)} tracks)"),
        ("Macro F1", f"{agg.get('macro_f1', 0):.3f}"),
        ("'approaching' precision / recall",
         f"{agg.get('approaching_precision', 0):.2f} / "
         f"{agg.get('approaching_recall', 0):.2f}"),
    ]
    if sel:
        rows.append(
            ("Single-frame tracks abstained (by design)",
             f"{sel.get('num_short_tracks', 0)}"))
    y = 0.46
    for k, v in rows:
        ax.text(0.06, y, k, ha="left", va="center", fontsize=11)
        ax.text(0.94, y, v, ha="right", va="center", fontsize=11,
                fontweight="bold")
        y -= 0.11
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return str(png_path)


def export_aggregate_results(
    eval_summary_path: str | Path,
    out_dir: str | Path,
) -> Dict[str, str]:
    """Write slide-ready aggregate results assets (text card + PNG card)
    from an ``all_videos_evaluation_summary.json``."""
    path = Path(eval_summary_path)
    written: Dict[str, str] = {}
    if not path.exists():
        return written
    with path.open(encoding="utf-8") as f:
        agg = json.load(f)
    if "overall_accuracy" not in agg:
        return written
    out = ensure_dir(out_dir)
    txt = out / "results_summary.txt"
    txt.write_text(build_results_summary_text(agg), encoding="utf-8")
    written["results_summary_txt"] = str(txt)
    png = build_results_card_png(agg, out / "results_card.png")
    if png:
        written["results_card_png"] = png
    return written
