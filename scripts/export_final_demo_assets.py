"""Re-export Google-Slides-ready assets for already-processed videos.

Use this when you don't want to re-run the whole pipeline but you tweaked
plots / overlays and want to refresh ``outputs/slide_assets/``.

Usage::

    python scripts/export_final_demo_assets.py --video-id bike_approaching_left
    python scripts/export_final_demo_assets.py --all
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.report_outputs import export_slide_assets  # noqa: E402
from src.visualize import plot_uncertain_reasons  # noqa: E402

HAZARDS_DIR = Path("outputs/hazards")
TRACKS_DIR = Path("outputs/tracks")
PLOTS_DIR = Path("outputs/plots")
SLIDE_ASSETS_DIR = Path("outputs/slide_assets")


def _discover_video_ids() -> List[str]:
    ids: set = set()
    for p in HAZARDS_DIR.glob("*_hazards.csv"):
        ids.add(p.stem.replace("_hazards", ""))
    for p in TRACKS_DIR.glob("*_track_features.csv"):
        ids.add(p.stem.replace("_track_features", ""))
    return sorted(ids)


def main() -> None:
    p = argparse.ArgumentParser(description="Export Google-Slides assets.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="Single video_id to export.")
    g.add_argument("--all", action="store_true",
                   help="Export assets for every processed video_id.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.video_id:
        targets = [args.video_id]
    else:
        targets = _discover_video_ids()
        if not targets:
            print("No processed videos found. Run the final pipeline first.")
            return

    for vid in targets:
        written = export_slide_assets(vid, cfg)
        print(f"[{vid}] {len(written)} files written to "
              f"outputs/slide_assets/{vid}/")

    # If we're exporting more than one video, also generate (or refresh)
    # the cross-video uncertain-reasons summary plot and stash it in a
    # top-level slide_assets/_aggregate/ folder so it's easy to find for
    # the "limitations / failure modes" slide.
    if len(targets) > 1:
        hazards = []
        for vid in targets:
            hp = HAZARDS_DIR / f"{vid}_hazards.csv"
            if hp.exists():
                hazards.append(pd.read_csv(hp))
        if hazards:
            preds = pd.concat(hazards, ignore_index=True)
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            agg_src = PLOTS_DIR / "all_videos_uncertain_reasons.png"
            plot_uncertain_reasons(
                preds, agg_src,
                title="Why tracks were 'uncertain' — all videos",
            )
            agg_dir = SLIDE_ASSETS_DIR / "_aggregate"
            agg_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(agg_src, agg_dir / "all_videos_uncertain_reasons.png")
            print(f"[aggregate] uncertain-reasons summary -> "
                  f"{agg_dir / 'all_videos_uncertain_reasons.png'}")


if __name__ == "__main__":
    main()
