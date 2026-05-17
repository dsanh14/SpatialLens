"""Week 3 pipeline: hazard classifier + alerts + plots + evaluation + slides.

Assumes the Week 1-2 pipeline has already produced::

    outputs/tracks/<video_id>_track_features.csv
    outputs/tracks/<video_id>_tracks.csv
    data/frames/<video_id>/frame_XXXX.jpg  (optional, for hazard overlay)
    outputs/detections/<video_id>_detections.csv  (optional, for summary)

Usage:

    python scripts/run_week3_pipeline.py --video-id bike_approaching_left
    python scripts/run_week3_pipeline.py --all

If no track-features CSVs exist, prints a hint and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts import generate_alerts, save_alerts  # noqa: E402
from src.config import load_config  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_hazards,
    load_labels,
    save_evaluation,
)
from src.hazard_classifier import run_hazard_classification  # noqa: E402
from src.report_outputs import export_slide_assets  # noqa: E402
from src.summarize import summarize_final  # noqa: E402
from src.utils import ensure_dir  # noqa: E402
from src.visualize import (  # noqa: E402
    draw_hazard_labels_on_frames,
    make_hazard_annotated_video,
    plot_approach_scores,
    plot_confusion_matrix,
    plot_hazard_timeline,
)

FRAMES_DIR = Path("data/frames")
DETECTIONS_DIR = Path("outputs/detections")
TRACKS_DIR = Path("outputs/tracks")
HAZARDS_DIR = Path("outputs/hazards")
ALERTS_DIR = Path("outputs/alerts")
EVALUATION_DIR = Path("outputs/evaluation")
ANNOTATED_FRAMES_DIR = Path("outputs/annotated_frames")
ANNOTATED_VIDEOS_DIR = Path("outputs/annotated_videos")
PLOTS_DIR = Path("outputs/plots")
SUMMARIES_DIR = Path("outputs/summaries")


def _discover_video_ids() -> List[str]:
    if not TRACKS_DIR.exists():
        return []
    return sorted(
        p.stem.replace("_track_features", "")
        for p in TRACKS_DIR.glob("*_track_features.csv")
    )


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _image_size_from_frames(video_id: str) -> Optional[tuple]:
    frame_dir = FRAMES_DIR / video_id
    if not frame_dir.exists():
        return None
    samples = sorted(frame_dir.glob("frame_*.jpg"))
    if not samples:
        return None
    img = cv2.imread(str(samples[0]))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def run_week3_for_video_id(video_id: str, cfg: dict) -> Dict[str, str]:
    print(f"\n=== Week 3 pipeline: {video_id} ===")
    features_csv = TRACKS_DIR / f"{video_id}_track_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Missing {features_csv}. Run Week 1-2 pipeline first.")

    features_df = pd.read_csv(features_csv)
    tracks_df = _read_csv_if_exists(TRACKS_DIR / f"{video_id}_tracks.csv")
    detections_df = _read_csv_if_exists(
        DETECTIONS_DIR / f"{video_id}_detections.csv"
    )

    image_size = _image_size_from_frames(video_id)
    image_w, image_h = (image_size if image_size else (None, None))

    hazards_df = run_hazard_classification(
        track_features_df=features_df,
        config=cfg,
        video_id=video_id,
        output_dir=HAZARDS_DIR,
        image_width=image_w,
        image_height=image_h,
    )

    alerts = generate_alerts(hazards_df, config=cfg)
    alert_paths = save_alerts(
        alerts=alerts,
        hazards_df=hazards_df,
        video_id=video_id,
        output_dir=ALERTS_DIR,
        config=cfg,
    )

    plot_paths: Dict[str, str] = {}
    if cfg["outputs"].get("save_plots", True):
        plot_paths["hazard_timeline_plot"] = plot_hazard_timeline(
            hazards_df, PLOTS_DIR / f"{video_id}_hazard_timeline.png"
        )
        plot_paths["approach_scores_plot"] = plot_approach_scores(
            hazards_df, PLOTS_DIR / f"{video_id}_approach_scores.png"
        )

    hazard_video_path = ""
    hazard_frames_dir = ANNOTATED_FRAMES_DIR / f"{video_id}_hazards"
    frame_paths = sorted((FRAMES_DIR / video_id).glob("frame_*.jpg"))
    if frame_paths:
        ensure_dir(hazard_frames_dir)
        hazard_frames = draw_hazard_labels_on_frames(
            frame_paths=[str(p) for p in frame_paths],
            tracks_df=tracks_df,
            hazards_df=hazards_df,
            output_dir=hazard_frames_dir,
        )
        if hazard_frames and cfg["outputs"].get("save_annotated_video", True):
            ensure_dir(ANNOTATED_VIDEOS_DIR)
            hazard_video_path = make_hazard_annotated_video(
                hazard_frames,
                ANNOTATED_VIDEOS_DIR / f"{video_id}_hazards.mp4",
                fps=int(cfg["video"].get("annotated_video_fps", 2)),
            )
    else:
        print(f"[week3] no extracted frames for {video_id}; "
              "skipping hazard overlay video.")

    evaluation: Optional[dict] = None
    eval_paths: Dict[str, str] = {}
    label_path = Path(cfg["evaluation"].get(
        "label_file", "data/labels/hazard_labels.csv"))
    eval_out_dir = Path(cfg["evaluation"].get(
        "output_dir", "outputs/evaluation"))
    labels_df = load_labels(label_path)
    if not labels_df.empty:
        evaluation = evaluate_hazards(hazards_df, labels_df, video_id=video_id)
        eval_paths = save_evaluation(evaluation, eval_out_dir, video_id=video_id)
        if (cfg["outputs"].get("save_plots", True)
                and "confusion_matrix" in evaluation):
            cm_df = pd.DataFrame.from_dict(
                evaluation["confusion_matrix"], orient="index"
            )
            if "labels" in evaluation:
                cols = [c for c in evaluation["labels"] if c in cm_df.columns]
                cm_df = (
                    cm_df.reindex(index=evaluation["labels"], columns=cols)
                         .fillna(0)
                         .astype(int)
                )
            plot_paths["confusion_matrix_plot"] = plot_confusion_matrix(
                cm_df, PLOTS_DIR / f"{video_id}_confusion_matrix.png",
                title=f"Hazard confusion matrix — {video_id}",
            )

    summary_text = summarize_final(
        video_id=video_id,
        detections_df=detections_df,
        tracks_df=tracks_df,
        track_features_df=features_df,
        hazards_df=hazards_df,
        alerts=alerts,
        video_summary=alert_paths.get("video_summary"),
        evaluation=evaluation,
        output_dir=SUMMARIES_DIR,
    )
    print("\n" + summary_text)

    if cfg["final_outputs"].get("export_slide_assets", True):
        export_slide_assets(video_id, cfg)

    outputs = {
        "video_id": video_id,
        "hazards_csv": str(HAZARDS_DIR / f"{video_id}_hazards.csv"),
        "hazards_json": str(HAZARDS_DIR / f"{video_id}_hazards.json"),
        "alerts_json": alert_paths.get("json", ""),
        "alerts_txt": alert_paths.get("txt", ""),
        "video_summary": alert_paths.get("video_summary", ""),
        "hazard_frames_dir": str(hazard_frames_dir),
        "hazard_video": hazard_video_path,
        "summary_txt": str(SUMMARIES_DIR / f"{video_id}_final_summary.txt"),
        "summary_json": str(SUMMARIES_DIR / f"{video_id}_final_summary.json"),
        "slide_assets_dir": str(Path("outputs/slide_assets") / video_id),
        **plot_paths,
        **{f"eval_{k}": v for k, v in eval_paths.items()},
    }
    return outputs


def main() -> None:
    p = argparse.ArgumentParser(description="Run SpatialLens Assist Week 3 pipeline.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="video_id to run.")
    g.add_argument("--all", action="store_true",
                   help="Run on every video_id with track features.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.video_id:
        targets = [args.video_id]
    else:
        targets = _discover_video_ids()
        if not targets:
            print("No track-features CSVs found in outputs/tracks/. "
                  "Run Week 1-2 pipeline first.")
            return

    all_outputs = []
    for vid in targets:
        try:
            all_outputs.append(run_week3_for_video_id(vid, cfg))
        except FileNotFoundError as e:
            print(f"[week3][skip] {e}")

    # Aggregate evaluation across every processed video, if labels exist.
    if args.all and all_outputs:
        label_path = Path(cfg["evaluation"].get(
            "label_file", "data/labels/hazard_labels.csv"))
        labels_df = load_labels(label_path)
        if not labels_df.empty:
            hazards = []
            for o in all_outputs:
                path = Path(o["hazards_csv"])
                if path.exists():
                    hazards.append(pd.read_csv(path))
            if hazards:
                preds = pd.concat(hazards, ignore_index=True)
                agg = evaluate_hazards(preds, labels_df, video_id=None)
                save_evaluation(
                    agg,
                    Path(cfg["evaluation"].get(
                        "output_dir", "outputs/evaluation")),
                )
                print(f"\n[week3] aggregate evaluation: "
                      f"accuracy={agg.get('overall_accuracy', 0):.2f}, "
                      f"approaching_F1={agg.get('approaching_f1', 0):.2f}, "
                      f"macro_F1={agg.get('macro_f1', 0):.2f}")

    print("\n=== Week 3 outputs ===")
    for out in all_outputs:
        print(f"\n[{out['video_id']}]")
        for k, v in out.items():
            if k == "video_id":
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
