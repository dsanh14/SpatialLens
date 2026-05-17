"""Re-export Google-Slides-ready assets for already-processed videos.

Use this when you don't want to re-run the whole pipeline but you tweaked
plots / overlays and want to refresh ``outputs/slide_assets/``.

Usage::

    python scripts/export_final_demo_assets.py --video-id bike_approaching_left
    python scripts/export_final_demo_assets.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.report_outputs import export_slide_assets  # noqa: E402

HAZARDS_DIR = Path("outputs/hazards")
TRACKS_DIR = Path("outputs/tracks")


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


if __name__ == "__main__":
    main()
