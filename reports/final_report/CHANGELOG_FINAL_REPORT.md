# CHANGELOG — Final Report Rewrite

This document summarizes the differences between the original 4-page
checkpoint-style report and the rewritten 8-page research-style final
report.

## High-level changes

1. **Added an explicit research question to the Introduction.** The
   paper now opens by asking *"can a small, interpretable set of
   motion cues recover useful hazard labels from low-fps handheld
   egocentric video, without training a learned trajectory model?"*
   and frames the system as an answer to that question rather than a
   pipeline tour.
2. **Reframed `uncertain` as conservative abstention** with sub-reason
   tagging, and emphasized that 23/24 of the abstentions are
   `short_track` (i.e. tracker fragmentation, not classifier
   ambiguity).
3. **Split out a dedicated Dataset and Evaluation Protocol section**
   covering controlled vs. uncontrolled splits, track--label
   matching, the raw vs. consolidated denominators, decidable-subset
   reasoning, and the choice of metrics.
4. **Added a Failure Analysis and Limitations section** with an
   uncertainty-reason figure, a 5-row error taxonomy table, and an
   explicit listing of dataset, fps, depth, and user-study
   limitations.
5. **Rewrote the conclusion as a research-style takeaway** rather than
   a progress update: tracker continuity matters more than classifier
   sophistication at this scale.
6. **Added five contribution bullets** at the end of the Introduction
   explicitly enumerating what the project contributes.
7. **Removed all checkpoint / progress-update language** ("with one
   week remaining", "remaining work", etc.). The conclusion is now
   final.

## New figures (in `figures/final/`)

| File | Source | Purpose |
| --- | --- | --- |
| `temporal_strip.pdf` | `data/frames/IMG_4972` + `outputs/tracks/IMG_4972_tracks.csv` | 4-frame strip of cyclist approaching, with centroid trail, bounding-box growth, and the verbatim evidence string the alert layer quotes. Demonstrates motion *over time* rather than a single static frame. |
| `feature_space_scatter.pdf` | Joined per-track features + labels (all 11 videos) | Every labelled track plotted in the (`dx_norm`, `growth`) feature space the cascade actually uses. Filled = correct, hollow = misclassified. Dashed lines mark the classifier thresholds. Visually confirms the rule cascade matches class clusters. |
| `confusion_matrix.pdf` | `outputs/evaluation/all_videos_evaluation_summary.json` | Polished row-normalized confusion matrix with both counts and recall percent in each cell. |
| `ablation_study.pdf` | `outputs/evaluation/ablations.csv` | Dual-denominator (raw + consolidated) ablation bars showing accuracy drop per removed component. |
| `class_distribution.pdf` | `data/labels/hazard_labels.csv` | Ground-truth class distribution bar chart highlighting `static` and `crossing_RTL` rarity. |
| `qualitative_panel.pdf` | Frames + tracks for IMG_4972/4981/4982 | 2×2 success/failure panel: approaching, crossing R→L, moving away, conservative abstention (static). |
| `uncertainty_breakdown.pdf` | Joined predictions | Shows 23 of 24 `uncertain` predictions are `short_track` abstentions; tracker fragmentation dominates uncertainty. |

The original `figures/confusion_matrix.png` and `ablation_bars.png`
are still in the repo but the rewritten paper now reads from
`figures/final/*.pdf` for higher-resolution publication-quality
versions.

## New tables (LaTeX sources in `tables/`)

| File | What it shows |
| --- | --- |
| `dataset_summary.tex` | Controlled / matched / decidable / uncontrolled splits with track counts and quantitative-use flag. |
| `class_distribution.tex` | Per-class label counts and percentages with notes on rarity. |
| `per_class_metrics.tex` | Per-class precision / recall / F1 / support; overall, decidable, and macro-F1 summarized below. |
| `ablation_study.tex` | Configuration / change / raw accuracy / consolidated accuracy / macro F1. |
| `error_taxonomy.tex` | Five residual error types with cause / example / proposed fix. |

All tables use `booktabs` rules. They are emitted by
`scripts/make_final_figures.py` so they regenerate automatically when
new evaluation results land.

## Code changes

* **Added `scripts/make_final_figures.py`** — single source of truth
  for all final-report figures and tables. Reads from
  `data/labels/hazard_labels.csv`, the per-track features in
  `outputs/tracks/`, the predictions in `outputs/hazards/`, and the
  evaluation summary in `outputs/evaluation/`. Functions:
  `load_tracks`, `plot_feature_space`, `plot_confusion_matrix`,
  `plot_ablation`, `plot_class_distribution`, `plot_temporal_strip`,
  `plot_qualitative_panel`, `plot_uncertainty_breakdown`, and one
  table-export function per table.
* The original `scripts/make_paper_figures.py` and
  `scripts/make_failure_figure.py` are left in place for reproducing
  the older figures.

## LaTeX style changes

* Added `booktabs`, `siunitx`, `cleveref` (with `capitalize` and
  `noabbrev`), `microtype`, `colortbl`, `array`, `makecell`,
  `amssymb` packages.
* Use `\Cref{...}` everywhere instead of mixing `\ref{...}` /
  `Sec.~\ref{}`.
* Switched the wide tables (dataset summary, ablation, error
  taxonomy) to `table*` so they span both columns cleanly; the
  narrower per-class and class-distribution tables use `\resizebox`
  to stay within a column without manual reflow.
* Bumped target length to 8 pages (was 4) to accommodate the new
  Dataset & Evaluation Protocol and Failure Analysis sections.

## Numbers preserved from existing outputs

All numerical claims in the rewritten paper are sourced directly from
files committed in the repository, not invented:

* 82.5% raw accuracy / 92.2% consolidated accuracy / 93.3% decidable
  accuracy: from `outputs/evaluation/all_videos_evaluation_summary.json`.
* Per-class precision/recall/F1: same JSON.
* Ablation deltas (−6.2, −3.5, −7.0, etc.): from
  `outputs/evaluation/ablations.csv`.
* 23/24 short-track uncertainty breakdown: derived live from
  `outputs/hazards/*.csv` by `make_final_figures.py`.
* 11 videos, 61 / 57 / 51 / 30 track counts: derived from
  `data/labels/hazard_labels.csv`, `..._dedup.csv`, and the eval
  JSON's `selective_accuracy.num_decidable_tracks`.

No citations, numbers, or qualitative claims were fabricated. The
`static` F1 of 0.00 is reported honestly with an explanation that it
reflects $n=2$ short fragments, not a general failure of the
`static` branch.

## How to regenerate the report from scratch

```bash
# from repo root
python scripts/make_final_figures.py
cd reports/final_report
pdflatex main.tex && pdflatex main.tex   # two passes for cross-refs
```

The two-pass build is required for `\Cref` references and the table
of contents counters; there is no bibliography database (refs are
inline in a `thebibliography` block).
