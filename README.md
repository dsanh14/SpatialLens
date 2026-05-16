# SpatialLens Assist: Motion-Aware Hazard Detection for Campus Navigation

SpatialLens Assist aims to detect moving campus hazards for blind/low-vision
pedestrians. It analyzes short walkway videos of bikes, scooters, and
pedestrians. **Weeks 1–2** implement video processing, object detection,
tracking, frame differencing, optical flow, and motion feature extraction.
**Week 3** will implement the final hazard classifier and evaluation.

This repo is a solo CS131 final project.

## Current status

- **Week 1**: perception pipeline — video loading, frame extraction, object
  detection (Ultralytics YOLO or mock), annotated frames/video.
- **Week 2**: tracking + motion features — simple IoU/centroid tracker, frame
  differencing, Farneback optical flow, bbox-area growth, per-track motion
  feature tables, detection/track summaries, plots.
- **Week 3 (not implemented yet)**: final hazard classifier
  (`approaching`, `crossing left-to-right`, `crossing right-to-left`,
  `moving away`, `static`, `uncertain`), evaluation, slides + report.

`TODO(Week 3)` markers in the code show exactly where the hazard classifier
will plug in.

## Dataset plan

Real campus videos will be collected and dropped into `data/raw_videos/` in
2–3 days. Until then, **mock mode** generates synthetic videos so the full
Week 1–2 pipeline can be tested without any real videos and without a GPU.

### Video scenarios to collect

1. Bike approaching from front-left, 10–12 seconds.
2. Scooter crossing left-to-right, 8–10 seconds.
3. Person walking away, 10–12 seconds.
4. Static bike/person non-hazard, 8 seconds.
5. Person crossing right-to-left, 8–10 seconds.
6. Mixed scene (multiple agents), 12–15 seconds.

### Safety note

Film controlled, slow, low-traffic scenes. Do not stage dangerous
near-collisions. Treat all subjects respectfully and avoid filming people who
have not consented to be on camera.

### Class / model note

The default YOLO model (`yolov8n.pt`, COCO classes) recognizes
`person`, `bicycle`, `motorcycle`, and `skateboard`, but **does not have a
dedicated `scooter` class**. In practice, scooters are usually detected as
some combination of `person` + `skateboard` / `motorcycle` / `bicycle`
depending on the model and angle. This is documented again in the final
report as a known limitation; the Week 3 hazard classifier will operate on
motion features, not on the raw class name alone, so this limitation is
partially mitigated.

## Quickstart

```bash
pip install -r requirements.txt
```

### Run the full pipeline on mock data (works today, no real videos needed)

```bash
python scripts/run_week1_week2_pipeline.py --mock --config config.yaml
```

This will:

1. Generate four synthetic videos in `data/mock_videos/`.
2. Extract frames into `data/frames/<video_id>/`.
3. Run mock detections (deterministic, no model download required).
4. Track objects across frames.
5. Compute frame differencing and Farneback optical flow per detection.
6. Aggregate per-track motion features.
7. Save annotated detection + tracking videos, plots, and a text/JSON summary.

### Run on a single real video (once available)

```bash
python scripts/run_week1_week2_pipeline.py \
    --video data/raw_videos/example.mp4 \
    --config config.yaml
```

### Run on all real videos in `data/raw_videos/`

```bash
python scripts/run_week1_week2_pipeline.py --all --config config.yaml
```

### Inspect a video's metadata

```bash
python scripts/inspect_video.py --video data/raw_videos/example.mp4
```

### Regenerate just the mock videos

```bash
python scripts/generate_mock_videos.py --config config.yaml
```

### Run unit tests

```bash
pytest tests/
```

## Repo layout

```
spatiallens-assist/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw_videos/        # drop your collected videos here
│   ├── mock_videos/       # generated synthetic videos
│   ├── frames/            # extracted frames per video_id
│   └── processed/
├── outputs/
│   ├── detections/        # per-video detection CSV + JSON
│   ├── tracks/            # per-video tracks CSV + track features CSV
│   ├── motion/            # frame-diff / optical-flow visualizations
│   ├── annotated_frames/  # frames with bbox / track overlays
│   ├── annotated_videos/  # mp4 of annotated frames
│   ├── plots/             # matplotlib plots for the deck
│   └── summaries/         # human-readable summary per video
├── src/                   # library code (see modules below)
├── scripts/               # CLI entry points
├── notebooks/
│   └── 01_week1_week2_check.ipynb
└── tests/
```

### `src/` modules

- `config.py` — YAML loader + validation.
- `utils.py` — small shared helpers (paths, color maps, video I/O).
- `mock_data.py` — synthetic videos + deterministic mock detections.
- `extract_frames.py` — sample + resize frames from a video.
- `detect_objects.py` — YOLO or mock detection backend.
- `annotate.py` — draw bboxes / tracks, write annotated videos.
- `tracking.py` — IoU + centroid tracker, per-class.
- `frame_diff.py` — frame differencing motion masks + per-bbox overlap.
- `optical_flow.py` — Farneback flow + per-bbox dx/dy/magnitude.
- `motion_features.py` — per-track Week 2 motion feature table.
- `summarize.py` — Week 1–2 text/JSON summary per video.
- `visualize.py` — matplotlib plots used in the slide deck.

## Expected outputs (per `video_id`)

After a successful pipeline run, you should see:

- `outputs/detections/<video_id>_detections.csv` and `.json`
- `outputs/tracks/<video_id>_tracks.csv`
- `outputs/tracks/<video_id>_track_features.csv`
- `outputs/motion/<video_id>/frame_diff_XXXX.jpg`
- `outputs/motion/<video_id>/flow_XXXX.jpg`
- `outputs/annotated_frames/<video_id>/frame_XXXX.jpg`
- `outputs/annotated_frames/<video_id>_tracks/frame_XXXX.jpg`
- `outputs/annotated_videos/<video_id>_detections.mp4`
- `outputs/annotated_videos/<video_id>_tracks.mp4`
- `outputs/plots/<video_id>_detections_per_frame.png`
- `outputs/plots/<video_id>_trajectories.png`
- `outputs/plots/<video_id>_bbox_area.png`
- `outputs/plots/<video_id>_motion_features.png`
- `outputs/summaries/<video_id>_week1_week2_summary.txt` and `.json`

## Limitations (Weeks 1–2)

- **No dedicated `scooter` class** in COCO YOLO — see note above.
- YOLO detections can be noisy on low-light or low-resolution video.
- The Week 2 motion labels (`growing`, `shrinking`, `static`, `moving`,
  `left_to_right`, `right_to_left`) are **preliminary features**, not the
  final hazard alerts. They are inputs to the Week 3 classifier.
- Farneback optical flow assumes a roughly stable camera. If the camera
  shakes a lot, per-bbox flow is unreliable.
- The simple IoU + centroid tracker can swap IDs when two objects of the
  same class cross or strongly overlap. A more robust tracker
  (ByteTrack / SORT) is intentionally out of scope for Weeks 1–2.

## Roadmap

- **Week 3**: combine per-track motion features (displacement direction,
  bbox growth, flow magnitude, frame-diff overlap, track length) into a
  rule-based or lightweight learned hazard classifier and evaluate on real
  campus videos. Polish the slide deck and final report.
