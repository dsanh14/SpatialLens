# Overleaf setup

## Quick steps

1. Open [Overleaf](https://www.overleaf.com) → **New Project** → **Blank Project**
2. Paste `main.tex` from this folder
3. Create folder `figures/` and upload:
   - `figure1_overview.png`
   - `figure2_evaluation.png`
   - `figure3_qualitative.png`
4. **Recompile** (pdfLaTeX) → **Download PDF**

## Regenerate figures locally

```bash
python scripts/generate_progress_report_figures.py
cd reports/progress_report && pdflatex main.tex && pdflatex main.tex
```

## Zip upload

```bash
cd reports/progress_report
zip -r progress_report_overleaf.zip main.tex figures/figure1_overview.png figures/figure2_evaluation.png figures/figure3_qualitative.png
```

Expected layout: **up to 5 pages**, CS231N-style two-column report with abstract, numbered sections/subsections, tables, and three full-width figures.
