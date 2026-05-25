# SpatialLens Assist: Motion-Aware Hazard Detection for Campus Navigation

## Problem

Blind and low-vision pedestrians moving through a campus need awareness of
moving hazards — bikes, scooters, and other pedestrians. A static label like
"bike" is not enough; what matters is whether that bike is **approaching**,
**crossing**, **moving away**, or **static**. SpatialLens Assist is a
prototype computer vision pipeline that takes a short campus walkway video
and reasons about per-object motion to produce assistive alerts such as
"Bike approaching from the left."

This is a solo 3-week CS131 final project. It is **not** a deployable
navigation tool — see the *Limitations* section.

## What the system does

Given a short campus walkway video, the pipeline:

1. **Detects objects** per frame (Ultralytics YOLO `yolov8n`, COCO classes:
   `person`, `bicycle`, `motorcycle`, `skateboard`). A `mock` backend is
   also available so the full pipeline runs without any model download.
2. **Tracks objects** across frames with a simple IoU + centroid matcher
   (per class), producing stable `track_id`s like `bicycle_1`, `person_1`.
3. **Computes motion features per track** — centroid trajectory, bounding-box
   growth / shrinkage, Farneback optical flow inside the bbox, and frame
   differencing overlap.
4. **Classifies each track** as one of six hazard labels:
   `approaching`, `crossing_left_to_right`, `crossing_right_to_left`,
   `moving_away`, `static`, `uncertain`. When the label is `uncertain`,
   the classifier also picks a sub-reason
   (`short_track`, `near_approaching`, `near_crossing_left_to_right`,
   `near_crossing_right_to_left`, `near_moving_away`, `conflicting_cues`,
   `low_signal`) and writes it to the `uncertain_reason` column so the
   final report can quantify *why* the model was unsure.
5. **Generates short assistive alerts** ("Scooter crossing left-to-right.",
   or for uncertain tracks, the reason-specific phrasing like
   "Person detected briefly; not enough frames to judge motion.").
6. **Evaluates** against a manually labeled CSV when one is provided.
7. **Exports slide-ready assets** (annotated frames, plots, hazard video,
   alerts) into `outputs/slide_assets/{video_id}/`.

## Computer vision methods used

- **Detection** — Ultralytics YOLO (`yolov8n.pt`, COCO).
- **Tracking** — by default the from-scratch per-class IoU +
  centroid-distance matcher with weighted scoring
  (`0.7 * IoU + 0.3 * (1 - normalized centroid distance)`). For
  comparison, a ByteTrack baseline (via the `supervision` library,
  Kalman-filtered, two-stage matching, lost-track buffer) is wired in
  as an opt-in backend selectable via `tracking.backend: "bytetrack"`
  in `config.yaml`. The default backend is the from-scratch tracker
  for pedagogical clarity; ByteTrack is useful for ablation
  comparisons in the report ("our simple tracker drops 12% of tracks
  to short_track; ByteTrack drops 4%", etc.).
- **Centroid trajectory analysis** — `dx_total`, `dy_total`, normalized
  displacement.
- **Bounding-box growth/shrinkage** — `bbox_growth_ratio = (end_area − start_area) / start_area`.
- **Optical flow** — OpenCV Farneback, median dx/dy and mean magnitude
  inside each bbox.
- **Frame differencing** — thresholded absolute difference + morphological
  open/close, then fraction of bbox area covered by the motion mask.
- **Ego-motion compensation** — on hand-held / walking footage, the
  whole frame pans every step and that pan contaminates both optical
  flow (every static object reads "moving") and frame differencing
  (every bbox lights up). The pipeline compensates for this in two
  ways, both gated by `motion.ego_motion_compensation` (set `false` for
  ablation):
  1. **Optical flow** — for each frame pair we take the per-pixel
     median of the dense flow field as a robust estimate of camera
     translation (the median is dominated by the static-but-panning
     background, since moving foreground objects cover only a small
     fraction of the image), and subtract that global flow per pixel
     before computing the per-bbox median dx/dy and mean magnitude.
     The result is *object motion relative to the background*, not
     absolute pixel motion. The per-frame estimate is also exposed as
     `avg_global_flow_dx/dy/mag` on each track so the report can
     quantify how much camera motion was compensated for.
  2. **Frame differencing** — we use `cv2.findTransformECC` to align
     the previous frame to the current frame before the absdiff. By
     default this uses a translation-only motion model (2 parameters,
     fast and stable). Setting `motion.ego_motion_model: "affine"`
     switches to a 6-parameter affine model that also handles
     rotation, scale (zoom), and shear — useful for hand-held footage
     with sharp turns. Either way, this removes most of the
     false-positive motion-mask area caused by the camera moving and
     `frame_diff_overlap` measures genuine object motion.
- **Rule-based hazard classification** — the main CV contribution of the
  project. Combines an `approach_score` (weighted blend of bbox growth,
  center motion, frame-diff overlap, flow magnitude) with horizontal
  displacement and shrink/growth thresholds. Every decision carries an
  `evidence` string explaining why.
- **Explicit `uncertain` sub-classification** — instead of a generic
  "uncertain" bucket, the classifier picks the most specific near-miss
  reason and exposes it as the `uncertain_reason` column:

  | reason | meaning |
  |---|---|
  | `short_track` | fewer than `hazard.min_track_frames` detections — trajectory cues unreliable |
  | `near_crossing_left_to_right` / `near_crossing_right_to_left` | horizontal displacement was in the band just below the crossing threshold |
  | `near_moving_away` | bbox shrank but not past the shrink threshold |
  | `near_approaching` | bbox grew and `approach_score` was inside the near-miss window of the threshold |
  | `conflicting_cues` | track both grew and crossed; diagonal motion, neither cue dominates |
  | `low_signal` | motion was detected but every directional cue was near zero |

  This makes the "uncertain" rate honest and actionable in the report —
  you can quantify e.g. "X% of uncertain tracks were caused by short
  tracks; X% by diagonal motion."

- **Evaluation** — overall accuracy, per-class accuracy, confusion matrix,
  per-class precision/recall/F1, macro F1, and a focus on
  `approaching` precision/recall/F1.

The detector is the least important piece. **The contribution is the
motion-reasoning layer** (tracking → motion features → hazard rules →
alerts).

## Quickstart

```bash
pip install -r requirements.txt
```

### Run on synthetic mock data (works today, no real videos needed)

```bash
python scripts/run_final_pipeline.py --mock --config config.yaml
```

This generates four synthetic videos in `data/mock_videos/` and runs the
full Week 1 + 2 + 3 pipeline on each.

### Run on real videos

1. Drop your `.mp4` / `.mov` / `.avi` / `.mkv` / `.m4v` files into
   `data/raw_videos/`. The filename (without extension) becomes the
   `video_id` used throughout `outputs/`.
2. Optionally inspect a clip first:
   ```bash
   python scripts/inspect_video.py --video data/raw_videos/example.mp4
   ```
3. Run on a single video:
   ```bash
   python scripts/run_final_pipeline.py \
       --video data/raw_videos/example.mp4 \
       --config config.yaml
   ```
4. Or run on everything in `data/raw_videos/`:
   ```bash
   python scripts/run_final_pipeline.py --all --config config.yaml
   ```

### Run just one stage

```bash
# Weeks 1-2 only (extract, detect, track, motion features)
python scripts/run_week1_week2_pipeline.py --all --config config.yaml

# Week 3 only (hazards + alerts + plots + evaluation + slides)
python scripts/run_week3_pipeline.py --all --config config.yaml
```

## Labeling and evaluation

Manual evaluation is optional but recommended for the final report.

The fastest path:

```bash
python scripts/bootstrap_hazard_labels.py
```

This walks every processed video and:

- Auto-fills `true_label` for any synthetic mock-scenario video
  (`bike_approaching_left_*`, `scooter_crossing_left_to_right_*`,
  `person_walking_away_*`, `static_nonhazard_*`) — the scenario name
  encodes the ground truth, so no manual work is required for those.
- For every real-video track, writes the model's prediction into the
  `suggested_label` column **but leaves `true_label` blank** (auto-
  filling real videos with the model's own prediction would make
  every metric trivially perfect).
- Generates `data/labels/REVIEW_QUEUE.md`: a per-video checklist of
  every track that still needs a `true_label`, ordered by safety
  priority (`approaching` first), with a deep link to the right demo
  video and a one-line evidence summary.

Re-running the script preserves any `true_label` values you've already
entered (it merges by `(video_id, track_id)`); pass `--overwrite` to
start over from scratch.

Allowed values for `true_label`:

```
approaching | crossing_left_to_right | crossing_right_to_left
moving_away | static | uncertain
```

Manual path (if you'd rather start from blank templates):

```bash
python scripts/create_label_template.py --all --config config.yaml
# … fill in true_label, save as data/labels/hazard_labels.csv …
```

Then re-run Week 3 to get evaluation metrics:

```bash
python scripts/run_week3_pipeline.py --all --config config.yaml
```

Metrics land in `outputs/evaluation/`:
- `<video_id>_evaluation_summary.json`
- `<video_id>_confusion_matrix.csv`
- `all_videos_evaluation_summary.json` (aggregate)

## Slide-ready assets

After a successful pipeline run, `outputs/slide_assets/{video_id}/`
contains a numbered set of files (`01_sample_frame.jpg`,
`02_detection_frame.jpg`, …) plus `README_slide_assets.txt` which tells
you which asset to use for which slide (Method, Results, Demo, …).

You can also re-export the slide-assets folder without rerunning the
pipeline:

```bash
python scripts/export_final_demo_assets.py --all --config config.yaml
```

## Expected outputs

Per `video_id`, after `run_final_pipeline.py`:

| Path | Contents |
|------|----------|
| `data/frames/<vid>/` | sampled JPEG frames |
| `outputs/detections/<vid>_detections.{csv,json}` | per-frame detections |
| `outputs/tracks/<vid>_tracks.csv` | per-frame track rows + flow + frame-diff overlap |
| `outputs/tracks/<vid>_track_features.csv` | per-track aggregated motion features |
| `outputs/motion/<vid>/` | optical flow + frame-diff visualizations |
| `outputs/hazards/<vid>_hazards.{csv,json}` | per-track hazard labels + evidence |
| `outputs/alerts/<vid>_alerts.{txt,json}` | assistive alerts |
| `outputs/annotated_frames/<vid>{,_tracks,_hazards}/` | overlay JPEGs |
| `outputs/annotated_videos/<vid>_{detections,tracks,hazards}.mp4` | demo videos |
| `outputs/plots/<vid>_*.png` | trajectories, bbox area, approach scores, hazard timeline, confusion matrix |
| `outputs/summaries/<vid>_final_summary.{txt,json}` | one-pager per video |
| `outputs/evaluation/<vid>_evaluation_summary.json` | metrics (when labels exist) |
| `outputs/slide_assets/<vid>/` | curated set of images / videos / README |

## Repo layout

```
SpatialLens/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw_videos/        # drop your real videos here
│   ├── mock_videos/       # generated synthetic videos
│   ├── frames/            # extracted frames per video_id
│   └── labels/            # hazard label templates + final labels CSV
├── outputs/
│   ├── detections/  tracks/  motion/
│   ├── hazards/     alerts/  evaluation/
│   ├── annotated_frames/  annotated_videos/
│   ├── plots/       summaries/  slide_assets/
├── src/
│   ├── config.py            # YAML loader + Week 3 defaults backfill
│   ├── utils.py             # shared path / color / video helpers
│   ├── mock_data.py         # synthetic videos + deterministic mock detections
│   ├── extract_frames.py    # video -> JPEGs
│   ├── detect_objects.py    # YOLO or mock detection backend
│   ├── tracking.py          # IoU + centroid tracker
│   ├── frame_diff.py        # motion masks + per-bbox overlap
│   ├── optical_flow.py      # Farneback flow + per-bbox dx/dy/magnitude
│   ├── motion_features.py   # per-track aggregated features (Week 2)
│   ├── annotate.py          # detection + track OpenCV overlays
│   ├── visualize.py         # matplotlib plots + hazard frame overlay
│   ├── hazard_classifier.py # Week 3: rule-based 6-class hazard classifier
│   ├── alerts.py            # Week 3: assistive alert text generation
│   ├── evaluation.py        # Week 3: accuracy / PRF / confusion matrix
│   ├── report_outputs.py    # Week 3: curate slide_assets/ per video
│   └── summarize.py         # Week 1-2 + final per-video summaries
├── scripts/
│   ├── inspect_video.py
│   ├── generate_mock_videos.py
│   ├── run_week1_pipeline.py
│   ├── run_week2_pipeline.py
│   ├── run_week1_week2_pipeline.py
│   ├── run_week3_pipeline.py
│   ├── run_final_pipeline.py
│   ├── create_label_template.py
│   ├── export_final_demo_assets.py
│   └── week1_week2_check.py
├── tests/    (pytest, all CPU-fast)
└── notebooks/01_week1_week2_check.ipynb
```

## Limitations

- **No dedicated `scooter` class** in COCO YOLO — scooters often appear as
  some combination of `person` + `skateboard` / `motorcycle` / `bicycle`.
  The Week 3 classifier reasons on motion, not class, so this is partially
  mitigated.
- **Camera motion** is handled by ego-motion compensation (see the
  *Computer vision methods used* section). Translation is on by
  default; rotation/zoom can be enabled via
  `motion.ego_motion_model: "affine"` for hand-held footage with
  sharp turns. A fast pan can still briefly outrun ECC's convergence
  on low-texture frames, so for best results film at roughly walking
  pace and aim the camera at textured surfaces (buildings, foliage)
  rather than blank sky.
- **Bbox-growth as a proxy for "approaching"** is an approximation. Without
  a depth backend (intentionally out of scope), a track far from the
  camera that grows because of perspective changes can be misread.
- **Simple IoU + centroid tracker** can swap IDs when same-class
  objects cross or overlap heavily. A ByteTrack baseline is wired in
  as an opt-in `tracking.backend: "bytetrack"` config — it survives
  brief detection misses much better but, since it's a library call,
  the default keeps the from-scratch tracker for the report's
  pedagogical story.
- **No safety deployment**. This is a 3-week prototype on controlled,
  low-traffic clips — not a production assistive technology. Real-world
  use would require careful HCI work, latency budgets, redundancy, and
  user testing with the target community.
- **No model training**. The classifier is rule-based, which is easier to
  explain in the report but doesn't generalize the way a learned model
  could.

## Final report framing

> *SpatialLens Assist is a prototype for motion-aware hazard detection from
> short campus walkway videos. The main computer vision contribution is
> the motion-reasoning layer: combining tracking, centroid trajectory,
> bounding-box scale, optical flow, and frame differencing into a
> rule-based, explainable per-track hazard classifier that produces
> short assistive alerts. The system is evaluated on a small set of
> manually labeled tracks across mock and real videos. It is not a
> deployable navigation tool.*
