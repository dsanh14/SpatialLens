"""End-to-end SpatialLens Assist pipeline: Week 1 + Week 2 + Week 3.

Reuses :mod:`scripts.run_week1_week2_pipeline.process_one` and
:func:`scripts.run_week3_pipeline.run_week3_for_video_id` so nothing is
duplicated.

Usage::

    python scripts/run_final_pipeline.py --video data/raw_videos/example.mp4
    python scripts/run_final_pipeline.py --all
    python scripts/run_final_pipeline.py --mock
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.evaluation import evaluate_hazards, load_labels, save_evaluation  # noqa: E402

# Import the Week 1-2 pipeline module by file name (it has dashes in spirit
# but a clean Python identifier, so a normal import works).
_w12 = importlib.import_module("scripts.run_week1_week2_pipeline")
_w3 = importlib.import_module("scripts.run_week3_pipeline")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the full SpatialLens Assist pipeline "
                    "(Week 1 + Week 2 + Week 3).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--video", help="Single video file to process.")
    g.add_argument("--all", action="store_true",
                   help="Process all videos in data/raw_videos/.")
    g.add_argument("--mock", action="store_true",
                   help="Generate + process synthetic mock videos.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)
    targets = _w12._resolve_targets(args, cfg)

    week12_outputs: List[Dict] = []
    week3_outputs: List[Dict] = []
    for vp, scenario in targets:
        out12 = _w12.process_one(vp, cfg, mock_scenario=scenario)
        week12_outputs.append(out12)
        out3 = _w3.run_week3_for_video_id(out12["video_id"], cfg)
        week3_outputs.append(out3)

    if len(week3_outputs) > 1:
        hazards = []
        for o in week3_outputs:
            hp = Path(o["hazards_csv"])
            if hp.exists():
                hazards.append(pd.read_csv(hp))
        if hazards:
            preds = pd.concat(hazards, ignore_index=True)

            if cfg["outputs"].get("save_plots", True):
                from src.visualize import plot_uncertain_reasons
                plots_dir = Path("outputs/plots")
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_uncertain_reasons(
                    preds, plots_dir / "all_videos_uncertain_reasons.png",
                    title="Why tracks were 'uncertain' — all videos",
                )

            label_path = Path(cfg["evaluation"].get(
                "label_file", "data/labels/hazard_labels.csv"))
            labels_df = load_labels(label_path)
            if not labels_df.empty:
                eval_dir = Path(cfg["evaluation"].get(
                    "output_dir", "outputs/evaluation"))
                agg = evaluate_hazards(preds, labels_df)
                save_evaluation(agg, eval_dir)
                sel = agg.get("selective_accuracy") or {}
                print(f"\n[final] decidable accuracy="
                      f"{sel.get('decidable_accuracy', float('nan')):.2f} "
                      f"(>= {sel.get('min_track_frames', 3)} frames), "
                      f"overall={agg.get('overall_accuracy', 0):.2f}, "
                      f"approaching_F1={agg.get('approaching_f1', 0):.2f}, "
                      f"macro_F1={agg.get('macro_f1', 0):.2f}")
                # Slide-ready aggregate results card (headlines decidable acc).
                from src.report_outputs import export_aggregate_results
                agg_dir = Path("outputs/slide_assets/_aggregate")
                export_aggregate_results(
                    eval_dir / "all_videos_evaluation_summary.json", agg_dir)

    print("\n=== Final pipeline outputs ===")
    for o12, o3 in zip(week12_outputs, week3_outputs):
        vid = o12["video_id"]
        print(f"\n[{vid}]")
        print(f"  frames={o12['num_frames']}, "
              f"detections={o12['num_detections']}, "
              f"tracks={o12['num_tracks']}")
        for k, v in o12.items():
            if k in {"video_id", "num_frames", "num_detections", "num_tracks"}:
                continue
            print(f"  W1-2 {k}: {v}")
        for k, v in o3.items():
            if k == "video_id":
                continue
            print(f"  W3   {k}: {v}")


if __name__ == "__main__":
    main()
