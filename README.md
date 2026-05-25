# SpatialLens Assist

CS131 final project — motion-aware hazard detection from short campus walkway videos.

**Repo:** https://github.com/dsanh14/SpatialLens

## What this is

If you're blind or low-vision and walking on campus, knowing there's a "bike" nearby isn't enough. You need to know whether it's coming toward you, cutting across your path, moving away, or just sitting there. This repo runs a CV pipeline on phone footage: detect people/bikes/scooters, track them across frames, classify motion, and print short alerts like *"Bike approaching from the left."*

It's a prototype for a class project, not something you'd ship as a navigation app.

## Methodology

The detector is off-the-shelf YOLO. The project work is everything after that: tracking, motion features, a rule-based hazard classifier with explainable outputs, and evaluation against hand-labeled tracks.

### 1. Frame extraction

Videos in `data/raw_videos/` are sampled at `video.sample_fps` (default 2 fps), resized to `video.resize_width` (960 px), and saved as JPEGs under `data/frames/<video_id>/`.

### 2. Detection

- **Backend:** Ultralytics YOLOv8n (`detection.backend: yolo`) or a deterministic `mock` backend for tests.
- **Classes:** COCO `person`, `bicycle`, `motorcycle`, `skateboard` (no dedicated scooter class).
- **Threshold:** `confidence_threshold: 0.35` on real footage — lower values caused a lot of 1–2 frame flicker detections.
- **Outputs:** `outputs/detections/<video_id>_detections.csv` (+ optional JSON).

### 3. Tracking

Tracks are built **per class** so a `person_1` never merges with `bicycle_1`.

**Default — `iou_centroid` (`src/tracking.py`):**  
Match detections frame-to-frame with weighted score  
`0.7 × IoU + 0.3 × (1 − normalized_centroid_distance)`.  
A track can skip up to `max_frame_gap` frames (default 4) before it dies. IDs look like `bicycle_1`, `person_2`.

**Optional — `bytetrack` (`src/tracking_bytetrack.py`):**  
`supervision.ByteTrack` for ablation. On our low-fps offline clips it dropped a large fraction of detections, so the simple tracker stays default.

**Outputs:** `outputs/tracks/<video_id>_tracks.csv` (per-frame rows with bbox, flow stats, frame-diff overlap).

### 4. Motion features (per track)

Aggregated in `src/motion_features.py` → `outputs/tracks/<video_id>_track_features.csv`:

| Feature | What it measures |
|--------|-------------------|
| Centroid trajectory | `dx_total`, `dy_total`, normalized displacement over the track |
| Bbox growth | `(end_area − start_area) / start_area` — rough “getting closer / farther” |
| Optical flow | Farneback dense flow; median dx/dy and mean magnitude **inside the bbox** |
| Frame differencing | Thresholded abs-diff mask, morphological cleanup, fraction of bbox covered |

**Ego-motion compensation** (`motion.ego_motion_compensation` in `config.yaml`):  
Hand-held walking footage pans the whole image, which makes static objects look like they're moving.

- **Flow:** subtract the per-frame **median** flow vector (background-dominated) before bbox stats → object motion relative to the scene.
- **Frame diff:** align previous frame to current with `cv2.findTransformECC` before differencing. `motion.ego_motion_model: translation` (default) or `affine` for rotation/zoom on sharp turns.

### 5. Hazard classification (rule cascade)

`src/hazard_classifier.py` assigns one label per track:

`approaching` · `crossing_left_to_right` · `crossing_right_to_left` · `moving_away` · `static` · `uncertain`

Rules combine `approach_score` (bbox growth, motion toward image center/lower half, flow, frame-diff), horizontal displacement (`dx_norm` as fraction of image width), and shrink/growth thresholds. Order matters — e.g. strong lateral motion can override a weak “approaching” cue (`strong_crossing_threshold_frac_width`), and bbox shrinkage is checked before crossing rules.

Every row gets an **`evidence`** string (quoted in alerts and useful for labeling).  
If the label is `uncertain`, **`uncertain_reason`** explains why:

| `uncertain_reason` | Typical cause |
|--------------------|---------------|
| `short_track` | Fewer than `hazard.min_track_frames` detections |
| `near_approaching` / `near_crossing_*` / `near_moving_away` | Just below a threshold |
| `conflicting_cues` | Diagonal motion — growth and lateral signals disagree |
| `low_signal` | Motion detected but no directional rule fired |

### 6. Alerts and evaluation

- **`src/alerts.py`** — short text per track; uncertain tracks use the reason in the wording.
- **`src/evaluation.py`** — when `data/labels/hazard_labels.csv` exists: accuracy, per-class precision/recall/F1, confusion matrix → `outputs/evaluation/`.

### 7. Visualization and report assets

`src/visualize.py`, `src/annotate.py`, `src/report_outputs.py` write plots, annotated frames/videos, and a curated `outputs/slide_assets/<video_id>/` folder for slides.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`yolov8n.pt` lives in the repo root; Ultralytics can also download it on first run.

## Run it

**Mock data (no real videos):**

```bash
python scripts/run_final_pipeline.py --mock --config config.yaml
```

**Your videos:** `.mp4` / `.mov` / `.avi` / `.mkv` / `.m4v` in `data/raw_videos/`. Filename (no extension) = `video_id`.

```bash
python scripts/run_final_pipeline.py --video data/raw_videos/IMG_4972.MOV --config config.yaml
python scripts/run_final_pipeline.py --all --config config.yaml
```

```bash
python scripts/inspect_video.py --video data/raw_videos/foo.mp4
```

**By week:**

```bash
python scripts/run_week1_week2_pipeline.py --all --config config.yaml   # extract, detect, track, motion
python scripts/run_week3_pipeline.py --all --config config.yaml         # hazards, alerts, eval, plots, slides
python scripts/run_final_pipeline.py --all --config config.yaml         # everything
```

## Labeling (for evaluation)

Ground truth: `data/labels/hazard_labels.csv` (`video_id`, `track_id`, `true_label`, …).

```bash
python scripts/bootstrap_hazard_labels.py
```

- Mock videos: `true_label` can be inferred from scenario filenames.
- Real videos: model prediction goes in `suggested_label`; you fill `true_label`. Checklist: `data/labels/REVIEW_QUEUE.md`.

Blank templates: `python scripts/create_label_template.py --all --config config.yaml`

Labels: `approaching` | `crossing_left_to_right` | `crossing_right_to_left` | `moving_away` | `static` | `uncertain`

Re-run week 3 or the full pipeline after editing labels to refresh `outputs/evaluation/`.

## Outputs (per `video_id`)

| Path | Contents |
|------|----------|
| `data/frames/<vid>/` | sampled JPEGs |
| `outputs/detections/<vid>_detections.{csv,json}` | per-frame boxes |
| `outputs/tracks/<vid>_tracks.csv` | per-frame track rows |
| `outputs/tracks/<vid>_track_features.csv` | per-track motion aggregates |
| `outputs/motion/<vid>/` | flow / frame-diff debug images |
| `outputs/hazards/<vid>_hazards.{csv,json}` | labels, evidence, `uncertain_reason` |
| `outputs/alerts/<vid>_alerts.{txt,json}` | assistive messages |
| `outputs/annotated_frames/<vid>{,_tracks,_hazards}/` | overlay JPEGs |
| `outputs/annotated_videos/<vid>_{detections,tracks,hazards}.mp4` | demo videos (H.264-friendly on macOS) |
| `outputs/plots/<vid>_*.png` | trajectories, bbox area, approach scores, timeline, confusion matrix, uncertain reasons |
| `outputs/summaries/<vid>_final_summary.{txt,json}` | one-page text summary |
| `outputs/evaluation/<vid>_*.json` | metrics when labels exist |
| `outputs/slide_assets/<vid>/` | numbered frames + `README_slide_assets.txt` |

Re-export slides only:

```bash
python scripts/export_final_demo_assets.py --all --config config.yaml
```

## Config

`config.yaml` — detection threshold, `tracking.backend`, `max_frame_gap`, hazard thresholds, ego-motion flags, ByteTrack params. Defaults are merged with `src/config.py` if keys are missing.

## Tests

```bash
pytest -q
```

CPU-only unit tests for classifier rules, ego-motion, tracking backends, evaluation, video writer, etc.

## Data

Most current clips are **controlled** captures (set routes, waited for bikes/pedestrians) so tracks are labelable. The plan also includes one **uncontrolled** normal campus-walk video for a qualitative demo — not part of the numbered labeled eval set.

## Limitations

- No COCO scooter class; scooters often map to skateboard/motorcycle/person.
- 2 fps sampling → many 1–2 frame track fragments → `uncertain` / `short_track` even when the object is clearly crossing in the video.
- Bbox growth is a depth proxy, not real depth.
- IoU tracker can swap IDs on overlap; ByteTrack didn't help on our low-fps offline setup.
- Not a deployable assistive product — no latency budget, audio UI, or user study.

## Repo layout

```
SpatialLens/
├── README.md
├── requirements.txt
├── config.yaml
├── yolov8n.pt
│
├── data/
│   ├── raw_videos/          # your .mov / .mp4 inputs
│   ├── mock_videos/         # generated by --mock
│   ├── frames/<video_id>/   # extracted JPEGs
│   └── labels/
│       ├── hazard_labels.csv
│       └── REVIEW_QUEUE.md  # labeling checklist (from bootstrap script)
│
├── outputs/
│   ├── detections/
│   ├── tracks/
│   ├── motion/
│   ├── hazards/
│   ├── alerts/
│   ├── evaluation/
│   ├── annotated_frames/
│   ├── annotated_videos/
│   ├── plots/
│   ├── summaries/
│   └── slide_assets/<video_id>/
│
├── src/
│   ├── config.py              # load YAML + defaults
│   ├── extract_frames.py
│   ├── detect_objects.py      # YOLO + mock
│   ├── tracking.py            # IoU + centroid (default)
│   ├── tracking_bytetrack.py  # optional ByteTrack backend
│   ├── optical_flow.py
│   ├── frame_diff.py
│   ├── motion_features.py     # per-track aggregates
│   ├── hazard_classifier.py   # rule cascade + uncertain_reason
│   ├── alerts.py
│   ├── evaluation.py
│   ├── annotate.py            # bbox overlays
│   ├── visualize.py           # matplotlib plots
│   ├── report_outputs.py      # slide_assets curation
│   ├── summarize.py
│   ├── mock_data.py
│   └── utils.py               # paths, colors, video writer (codec fallback)
│
├── scripts/
│   ├── run_final_pipeline.py
│   ├── run_week1_pipeline.py
│   ├── run_week2_pipeline.py
│   ├── run_week1_week2_pipeline.py
│   ├── run_week3_pipeline.py
│   ├── generate_mock_videos.py
│   ├── inspect_video.py
│   ├── bootstrap_hazard_labels.py
│   ├── create_label_template.py
│   ├── export_final_demo_assets.py
│   └── week1_week2_check.py
│
├── tests/                     # pytest
├── notebooks/                 # optional checks
└── reports/progress_report/   # LaTeX progress report (main.tex + figures/)
```
