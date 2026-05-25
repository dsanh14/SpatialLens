# Overleaf setup

## Quick steps

1. Open [Overleaf](https://www.overleaf.com) → **New Project** → **Blank Project**
2. Select all text in the default `main.tex`, delete it, and **paste** the contents of `main.tex` from this folder
3. In the left file tree: **New Folder** → name it `figures`
4. Upload these three files into `figures/`:
   - `figures/confusion_matrix.png`
   - `figures/uncertain_reasons.png`
   - `figures/annotated_example.jpg`
5. **Recompile** (pdfLaTeX, default) → **Download PDF**

## Or upload as a zip

From the repo root:

```bash
cd reports/progress_report
zip -r progress_report_overleaf.zip main.tex figures/
```

In Overleaf: **New Project** → **Upload Project** → choose `progress_report_overleaf.zip`

## Project layout Overleaf expects

```
main.tex          ← paste or upload
figures/
  confusion_matrix.png
  uncertain_reasons.png
  annotated_example.jpg
```
