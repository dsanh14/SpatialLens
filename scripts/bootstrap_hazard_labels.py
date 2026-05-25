"""Build `data/labels/hazard_labels.csv` with as much auto-filled ground
truth as is defensible, plus a prioritized review queue for the rest.

Why this exists
---------------

The Week 3 evaluation needs a ground-truth labels file to compute
accuracy / precision / recall / F1. Manually labeling every track is
tedious; this script does the easy ones for you and tells you exactly
which rows you still need to look at.

What gets auto-labeled (defensible)
-----------------------------------

* **Synthetic mock videos** whose video_id starts with a known mock
  scenario (e.g. ``bike_approaching_left_*``). The scenario name IS
  the ground truth, so labeling is just a lookup. These rows are
  marked ``label_source = "scenario"``.

What is NEVER auto-labeled (would be circular)
----------------------------------------------

* **Real-video tracks.** Using the model's own prediction as the
  ground truth would make every accuracy metric trivially perfect and
  meaningless. Instead, the script:

  1. Writes the predicted label into ``true_label`` *suggested* fields
     but leaves ``true_label`` itself blank (so it's clear nothing
     "counts" until you confirm it).
  2. Writes a prioritized review queue
     (``data/labels/REVIEW_QUEUE.md``) telling you which rows to
     verify, ranked by safety-criticality and amount of evidence.

Usage
-----

::

    # First: make sure templates exist (one per video) and all hazards
    # have been generated.
    python scripts/run_final_pipeline.py --all

    # Then: bootstrap the labels file + review queue.
    python scripts/bootstrap_hazard_labels.py

After that, open ``data/labels/REVIEW_QUEUE.md``, watch the listed
demo videos, and either accept the suggested label (paste it into the
``true_label`` column of ``data/labels/hazard_labels.csv``) or correct
it. Re-run ``scripts/run_week3_pipeline.py --all`` and the per-class
PRF + confusion matrix will appear in ``outputs/evaluation/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

HAZARDS_DIR = Path("outputs/hazards")
TRACKS_DIR = Path("outputs/tracks")
ANNOTATED_VIDEOS_DIR = Path("outputs/annotated_videos")
LABELS_DIR = Path("data/labels")

# Scenario name -> ground-truth hazard label. The mock-data generator
# encodes the scenario in the video_id, so we can label synthetic
# tracks deterministically without watching anything.
SCENARIO_TO_TRUE_LABEL: Dict[str, str] = {
    "bike_approaching_left": "approaching",
    "scooter_crossing_left_to_right": "crossing_left_to_right",
    "person_walking_away": "moving_away",
    "static_nonhazard": "static",
}

# Output columns in hazard_labels.csv. Matches what evaluate_hazards()
# reads (it uses `video_id` + `track_id` to join against predictions
# and `true_label` as the ground truth).
LABEL_COLUMNS = [
    "video_id",
    "track_id",
    "class_name",
    "true_label",
    "suggested_label",         # the model's prediction (NOT ground truth)
    "label_source",            # "scenario" | "manual_review_needed" | "manual"
    "first_frame",
    "last_frame",
    "num_frames",
    "evidence",
    "notes",
]


def _detect_scenario(video_id: str) -> str | None:
    """If `video_id` starts with a known mock scenario, return that
    scenario name; otherwise return None."""
    for scenario in SCENARIO_TO_TRUE_LABEL:
        if video_id == scenario or video_id.startswith(scenario + "_"):
            return scenario
    return None


def _load_hazards(video_id: str) -> pd.DataFrame:
    p = HAZARDS_DIR / f"{video_id}_hazards.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _discover_video_ids() -> List[str]:
    ids: set = set()
    for p in HAZARDS_DIR.glob("*_hazards.csv"):
        ids.add(p.stem.replace("_hazards", ""))
    for p in TRACKS_DIR.glob("*_track_features.csv"):
        ids.add(p.stem.replace("_track_features", ""))
    return sorted(ids)


def _row_to_label_record(
    row: pd.Series,
    video_id: str,
    scenario: str | None,
) -> Dict:
    predicted = str(row.get("hazard_label", ""))
    if scenario is not None:
        true = SCENARIO_TO_TRUE_LABEL[scenario]
        source = "scenario"
    else:
        true = ""
        source = "manual_review_needed"
    return {
        "video_id": video_id,
        "track_id": str(row.get("track_id", "")),
        "class_name": str(row.get("class_name", "")),
        "true_label": true,
        "suggested_label": predicted,
        "label_source": source,
        "first_frame": int(row.get("first_frame", -1)),
        "last_frame": int(row.get("last_frame", -1)),
        "num_frames": int(row.get("num_frames", 0)),
        "evidence": str(row.get("evidence", ""))[:240],
        "notes": "",
    }


def _build_review_queue_md(
    df: pd.DataFrame, out_path: Path,
) -> Path:
    """Write a per-video, priority-ordered review checklist for the
    rows that still need a human label."""
    pending = df[df["label_source"] == "manual_review_needed"].copy()
    auto = df[df["label_source"] == "scenario"]

    # Priority order: approaching first (safety-critical), then
    # crossings, then moving_away, then static, then uncertain.
    priority = {
        "approaching": 0,
        "crossing_left_to_right": 1,
        "crossing_right_to_left": 1,
        "moving_away": 2,
        "static": 3,
        "uncertain": 4,
    }
    pending["_prio"] = pending["suggested_label"].map(priority).fillna(99)
    # Within the same priority, longer tracks first (more evidence
    # makes verification faster).
    pending = pending.sort_values(
        ["video_id", "_prio", "num_frames"],
        ascending=[True, True, False],
    )

    lines: List[str] = []
    lines.append("# Hazard label review queue\n")
    lines.append(
        f"Auto-labeled from scenario name: **{len(auto)} rows**.\n"
        f"Needing manual review: **{len(pending)} rows** across "
        f"**{pending['video_id'].nunique()} videos**.\n\n"
        "Open each video below, then for each track confirm or correct "
        "the suggested label and paste the final value into the "
        "`true_label` column of `data/labels/hazard_labels.csv`.\n\n"
        "Priority order within a video: `approaching` (safety-critical) "
        ", then crossings, then everything else.\n"
    )

    for vid, grp in pending.groupby("video_id"):
        demo = ANNOTATED_VIDEOS_DIR / f"{vid}_hazards.mp4"
        demo_str = (
            f"[`{demo}`]({demo})" if demo.exists()
            else f"`{demo}` *(not generated)*"
        )
        lines.append(f"\n## `{vid}` ({len(grp)} tracks)\n")
        lines.append(f"Demo video: {demo_str}\n")
        lines.append(
            "| track_id | class | suggested | n_frames | frame range | evidence (truncated) |"
        )
        lines.append(
            "|---|---|---|---|---|---|"
        )
        for _, r in grp.iterrows():
            ev = str(r["evidence"]).replace("|", "\\|")
            if len(ev) > 90:
                ev = ev[:90] + "..."
            lines.append(
                f"| `{r['track_id']}` | {r['class_name']} | "
                f"**{r['suggested_label']}** | {int(r['num_frames'])} | "
                f"{int(r['first_frame'])}-{int(r['last_frame'])} | {ev} |"
            )

    lines.append("\n---\n")
    lines.append(
        "When done, re-run:\n\n"
        "```\n"
        "python scripts/run_week3_pipeline.py --all\n"
        "```\n\n"
        "Evaluation metrics will appear in `outputs/evaluation/`.\n"
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bootstrap hazard_labels.csv + a review queue.",
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--out",
        default=None,
        help="Override output path for the labels CSV. Defaults to "
             "`evaluation.label_file` from the config.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing labels CSV. Without this flag, an "
             "existing file is preserved and only the review queue is "
             "regenerated.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    label_path = Path(
        args.out or cfg["evaluation"].get(
            "label_file", "data/labels/hazard_labels.csv")
    )

    video_ids = _discover_video_ids()
    if not video_ids:
        print("No hazards or track features found in outputs/. "
              "Run scripts/run_final_pipeline.py --all first.")
        return

    records: List[Dict] = []
    for vid in video_ids:
        scenario = _detect_scenario(vid)
        haz = _load_hazards(vid)
        if haz.empty:
            print(f"[bootstrap] {vid}: no hazards CSV, skipping.")
            continue
        for _, row in haz.iterrows():
            records.append(_row_to_label_record(row, vid, scenario))

    if not records:
        print("Nothing to label. Run the Week 3 pipeline first.")
        return

    df = pd.DataFrame(records, columns=LABEL_COLUMNS)

    ensure_dir(LABELS_DIR)
    if label_path.exists() and not args.overwrite:
        # Be careful: someone may have already entered manual labels.
        # Merge: keep existing true_label values, only fill in scenario
        # ones for new (video_id, track_id) pairs.
        existing = pd.read_csv(label_path)
        # Normalize column set in case the existing CSV is older.
        for col in LABEL_COLUMNS:
            if col not in existing.columns:
                existing[col] = ""
        existing_keys = set(
            zip(existing["video_id"].astype(str),
                existing["track_id"].astype(str))
        )
        new_rows = df[~df.apply(
            lambda r: (str(r["video_id"]), str(r["track_id"]))
            in existing_keys,
            axis=1,
        )]
        merged = pd.concat([existing, new_rows], ignore_index=True)
        merged.to_csv(label_path, index=False)
        print(f"[bootstrap] merged {len(new_rows)} new rows into "
              f"existing {label_path} ({len(merged)} total).")
        out_df = merged
    else:
        df.to_csv(label_path, index=False)
        print(f"[bootstrap] wrote {len(df)} rows -> {label_path}")
        out_df = df

    n_scenario = int((out_df["label_source"] == "scenario").sum())
    n_pending = int((out_df["label_source"] == "manual_review_needed").sum())
    n_manual = int((out_df["label_source"] == "manual").sum())
    n_filled = int(((out_df["true_label"].fillna("") != "") &
                    (out_df["label_source"] != "scenario")).sum())
    print(
        f"[bootstrap] summary: scenario-labeled={n_scenario}, "
        f"manual-already-done={n_filled}, manual-still-todo={n_pending}, "
        f"explicitly-marked-manual={n_manual}"
    )

    queue_path = LABELS_DIR / "REVIEW_QUEUE.md"
    _build_review_queue_md(out_df, queue_path)
    print(f"[bootstrap] review queue -> {queue_path}")


if __name__ == "__main__":
    main()
