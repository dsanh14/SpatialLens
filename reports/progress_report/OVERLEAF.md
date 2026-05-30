# Overleaf setup

## Upload these files

Paste `main.tex` and create folder `figures/` with **one file per figure**:

| File | What it shows |
|------|----------------|
| `fig_pipeline.png` | End-to-end pipeline diagram |
| `fig_hazard_overlay.jpg` | Annotated demo frame (IMG\_4976) |
| `fig_accuracy.png` | Decidable vs overall accuracy bars |
| `fig_ablations.png` | Ablation bar chart |
| `fig_confusion_matrix.png` | Confusion matrix heatmap |
| `fig_uncertain_reasons.png` | Uncertain-reason breakdown |
| `fig_trajectories.png` | Track trajectories (IMG\_4974) |

Generate locally:

```bash
python scripts/generate_progress_report_figures.py
cd reports/progress_report && pdflatex main.tex
```

Zip upload:

```bash
cd reports/progress_report
zip -r progress_report_overleaf.zip main.tex figures/fig_*.png figures/fig_*.jpg
```
