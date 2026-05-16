"""Generate the four synthetic mock videos used to test the pipeline.

Usage:
    python scripts/generate_mock_videos.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.mock_data import generate_all_mock_videos  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Generate mock videos.")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (default: config.yaml).")
    p.add_argument("--out-dir", default="data/mock_videos",
                   help="Directory to write mock videos into.")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = generate_all_mock_videos(args.out_dir, cfg)
    print("\nGenerated mock videos:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
