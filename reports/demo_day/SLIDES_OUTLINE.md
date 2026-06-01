# SpatialLens Assist — Demo Day Slides (4 min)

**CS131 Demo Day** · 4-minute oral presentation · submit slides as **PDF on Gradescope**.
This file is the build sheet: 7 slides, each with its title, on-slide content, the exact
image to drop in, and a word-for-word speaker script with a running clock (target **4:00**).

---

## How to assemble (≈30 min)

1. New Google Slides deck → **File ▸ Page setup ▸ Widescreen 16:9**.
2. Pick a clean theme (e.g. *Simple Light*). Keep one accent color; use it for the
   headline numbers and the "money line" callouts.
3. For each slide below: paste the **Title** and **bullets**, then **Insert ▸ Image ▸
   Upload from computer** and pick the matching file from
   `reports/demo_day/assets/` (filenames are pre-numbered per slide).
4. Paste the **Speaker script** into the per-slide **Speaker notes** pane (View ▸ Show
   speaker notes) so you can rehearse to time.
5. Export: **File ▸ Download ▸ PDF document (.pdf)** → upload to Gradescope.
   Sanity-check the PDF is legible (figures large, ≤6 bullets/slide).

**Design rules for the 6% presentation grade:** one idea per slide, ≤6 short bullets,
figures big enough to read from the back row, bold the 3–4 numbers that matter, consistent
fonts/colors throughout.

> All image files live in `reports/demo_day/assets/` (copies — originals untouched).
> Original source paths are listed under each slide in case you want full-res versions.

---

## Slide 1 — Title  ·  0:00–0:15  (15s)

**Title:** SpatialLens Assist
**Subtitle:** Motion-Aware Hazard Detection for Campus Navigation from Low-FPS Egocentric Video
**Footer:** Diego Sanchez · Stanford CS131

- Image: `assets/slide1_title_thumb.jpg` (hazard-overlay frame, right half of slide)
  · source: `outputs/slide_assets/IMG_4972/04_hazard_frame.jpg`

**Speaker script (→0:15):**
> "Hi, I'm Diego. SpatialLens Assist turns a cheap phone video of a campus walkway into
> spoken hazard alerts for blind and low-vision pedestrians — telling them not just *what*
> is nearby, but whether it's *coming at them*."

---

## Slide 2 — The Problem  ·  0:15–0:55  (40s)   [Problem clarity — 7%]

**Title:** Detection tells you *what*, not *what it's doing*

- A **parked bicycle** and a bicycle **bearing down on you** are both just `bicycle, 0.86`.
- A sighted person reads intent from context; a blind/low-vision pedestrian can't — the
  safety signal is **motion**, not the label.
- **Our task:** for each tracked object in a ~10s phone clip @ **2 fps**, classify the
  motion into **6 classes**:
  `approaching` · `crossing_LTR` · `crossing_RTL` · `moving_away` · `static` · `uncertain`.

- Image: `assets/slide2_hazard_overlay.jpg` (annotated frame)
  · source: `reports/progress_report/figures/fig_hazard_overlay.jpg`

**Speaker script (→0:55):**
> "Here's the problem. A parked bike and a bike speeding toward you are detected
> identically — `bicycle, point eight-six`. A sighted person fills in the intent from
> context; the user we care about can't. So the safety-critical signal isn't the label,
> it's the *motion*. We frame it as per-track classification: given a ten-second phone clip
> at just two frames a second, decide whether each object is approaching, crossing left or
> right, moving away, static — or explicitly *uncertain*."

---

## Slide 3 — Methodology  ·  0:55–1:40  (45s)   [Methodology clarity — 7%]

**Title:** A transparent, CPU-only pipeline

- Image (full width, top): `assets/slide3_pipeline.png`
  · source: `reports/progress_report/figures/fig_pipeline.png`
  6 stages: **2 fps frames → YOLOv8n → IoU+centroid tracker + appearance re-ID →
  ego-motion-compensated motion features → interpretable rule cascade → alert + evidence string.**

- Three design pillars:
  - **Explainable** — every prediction carries a natural-language *evidence string*.
  - **Calibrated abstention** — `uncertain` is a first-class output; short tracks don't bluff.
  - **CPU-only** — full pipeline runs in **~4 min** on a GPU-less Mac.
- Key trick: **ego-motion compensation** (median-subtracted Farnebäck flow + ECC frame-diff)
  so camera pan doesn't make every *static* object look like it's moving.

**Speaker script (→1:40):**
> "The pipeline has six stages. Off-the-shelf YOLOv8n detects; a lightweight IoU-plus-
> centroid tracker links objects across frames, with an appearance-based re-ID pass to
> stitch fragments. Then we compute motion features — and the key trick is ego-motion
> compensation: we subtract the camera's own panning, using median optical flow and
> frame alignment, so a parked bench doesn't look like it's charging at you. Finally an
> *interpretable rule cascade* — no black box — emits the label plus a plain-English
> evidence string. Three principles drive it: everything is explainable, the system
> *abstains* rather than guesses on thin evidence, and it all runs on CPU in about four
> minutes."

---

## Slide 4 — It works: one clip, four stages  ·  1:40–2:25  (45s)

**Title:** From pixels to a spoken alert

- 4-up horizontal strip (equal sizes, left→right), label each underneath:
  1. `assets/slide4a_raw.jpg` — **Raw frame** (2 fps input)
  2. `assets/slide4b_detection.jpg` — **YOLOv8n detections**
  3. `assets/slide4c_tracking.jpg` — **Tracking + re-ID** (centroid trails)
  4. `assets/slide4d_hazard.jpg` — **Hazard overlay**
  · source: `outputs/slide_assets/IMG_4972/0{1,2,3,4}_*.jpg`
- Big alert callout (accent color):
  **🔊 "Bicycle approaching directly ahead." [high]**
- Sub-caption: evidence string — *bbox grew 21.98×, centroid moved toward image center.*

**Speaker script (→2:25):**
> "Here's a real run, left to right. Raw frame. YOLO's detections. The tracker links the
> cyclist across frames — that's the yellow trail. And the motion stage fires: the
> bounding box grew almost twenty-two-fold and the centroid moved toward the center, so
> the system says, out loud, *'bicycle approaching directly ahead,'* high confidence —
> and it shows you exactly *why* it said that."

---

## Slide 5 — Results  ·  2:25–3:05  (40s)   [Results clarity — 7%]

**Title:** Accurate where it counts

- Headline numbers (big, accent color):
  - **82.5%** overall · **93.3%** on decidable tracks (≥3 frames)
  - **`approaching`: precision 0.95 / recall 0.88**  (the safety-critical class)
- Image (right side): `assets/slide5_confusion_matrix.png`
  · source: `reports/final_report/figures/final/confusion_matrix.png`
- **Money line:** **zero `approaching` ↔ `moving_away` confusion.**
  Off-diagonal mass is abstention into `uncertain`, *not* dangerous flips.

**Speaker script (→3:05):**
> "On 11 controlled walkway videos and 61 hand-labeled tracks: 82.5% overall, and 93.3%
> on tracks long enough to decide — three frames or more. The safety-critical
> *approaching* class hits 0.95 precision, 0.88 recall. And the number I care most about:
> *zero* confusion between approaching and moving-away. When the system is unsure it
> abstains into *uncertain* — it never told a user something was leaving when it was
> actually coming at them. For an alert system, that's the right kind of asymmetry."

---

## Slide 6 — Analysis & Learnings  ·  3:05–3:45  (40s)   [Analysis/Discussion — 7%]

**Title:** Continuity beats classifier complexity

- Image (left): `assets/slide6_ablation_study.png`
  · source: `reports/final_report/figures/final/ablation_study.png`
- Lead finding: the biggest gains come from **keeping tracks intact**, not fancier rules:
  - appearance-gated re-ID: **+6.2 pts** (largest single contributor)
  - two short-track salvage rules: **+3.5 pts** each
- **Negative result:** **ByteTrack → 32.8%** on 2 fps offline footage — a strong *online*
  tracker's motion-continuity assumption breaks under sparse sampling.
- Abstention is the **"fragmentation tax made auditable"**: 23 of 24 abstentions are
  short tracks, not classifier confusion.

**Speaker script (→3:45):**
> "The ablations taught me the real lesson. The biggest accuracy gains didn't come from
> smarter classification rules — they came from *keeping tracks from fragmenting*.
> Appearance re-ID alone is worth six points. And a useful negative result: ByteTrack, a
> state-of-the-art online tracker, drops to 33% here, because at two frames a second its
> motion-continuity assumption just doesn't hold. The system's uncertainty is almost
> entirely short broken tracks — so the abstentions are an honest, auditable tax, not the
> classifier being confused."

---

## Slide 7 — Conclusion  ·  3:45–4:00  (15s)

**Title:** Takeaways & what's next

- **Takeaway:** simple, *interpretable* motion cues recover useful hazard labels from
  cheap low-fps video — and continuity matters more than classifier sophistication.
- **Next:** monocular depth (MiDaS) · learned re-ID embedding · denser sampling (5–10 fps)
  · user study with blind/low-vision walkers.
- *Thank you!*

**Speaker script (→4:00):**
> "So: cheap, low-frame-rate video plus interpretable motion cues is enough to give useful,
> explainable hazard alerts. Next steps are adding depth, a learned re-ID embedding, and a
> study with actual blind and low-vision walkers. Thank you."

---

## Timing summary

| Slide | Topic | Budget | Cumulative |
|------:|-------|-------:|-----------:|
| 1 | Title / hook         | 0:15 | 0:15 |
| 2 | Problem              | 0:40 | 0:55 |
| 3 | Methodology          | 0:45 | 1:40 |
| 4 | Demo stage-strip     | 0:45 | 2:25 |
| 5 | Results              | 0:40 | 3:05 |
| 6 | Analysis & learnings | 0:40 | 3:45 |
| 7 | Conclusion           | 0:15 | 4:00 |

**Total: 4:00.** Scripts above run slightly under each budget — that's intentional buffer
for transitions. If you're long, trim the evidence-string aside on Slide 4 and the
ByteTrack sentence on Slide 6 first.

## Rubric coverage
- **Problem & Methodology clarity (7%):** Slides 2–3 (Slide 4 grounds it in a real run).
- **Results / Analysis / Discussion clarity (7%):** Slides 5–6.
- **Presentation quality (6%):** consistent theme, big legible figures, ≤6 bullets/slide,
  timed to 4:00.

## Backup asset
`assets/slideX_results_card_backup.png` — a pre-rendered summary card with all headline
numbers. Use it as an alternative to the confusion matrix on Slide 5, or as a fallback
"results at a glance" slide if you add time.
