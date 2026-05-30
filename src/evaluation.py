"""Evaluation against manually filled hazard labels (Week 3).

Labels live in a single CSV (default ``data/labels/hazard_labels.csv``) with
columns ``video_id``, ``track_id``, ``class_name``, ``true_label`` (one of
the six hazard labels), and optional ``notes``. Predictions come from
:mod:`src.hazard_classifier`.

Falls back to a pure-Python precision/recall/F1 implementation if
scikit-learn is not installed. Macro F1 is reported when sklearn is
available, otherwise we approximate it as the unweighted mean of per-class
F1 scores.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .hazard_classifier import HAZARD_LABELS
from .utils import ensure_dir

POSITIVE_LABEL = "approaching"

LABEL_FILE_COLUMNS = ("video_id", "track_id", "class_name", "true_label")


def load_labels(label_path: str | Path) -> pd.DataFrame:
    """Load the manual label CSV, or return an empty frame if missing.

    Drops rows where ``true_label`` is empty, and warns about unknown labels
    (anything outside :data:`src.hazard_classifier.HAZARD_LABELS`).
    """
    path = Path(label_path)
    if not path.exists():
        print(f"[evaluation] no label file at {path}; evaluation will be skipped.")
        return pd.DataFrame(columns=list(LABEL_FILE_COLUMNS))
    df = pd.read_csv(path)
    missing = [c for c in LABEL_FILE_COLUMNS if c not in df.columns]
    if missing:
        print(f"[evaluation][warning] label file missing columns {missing}; "
              "ignoring it.")
        return pd.DataFrame(columns=list(LABEL_FILE_COLUMNS))
    df = df.dropna(subset=["true_label"])
    df["true_label"] = df["true_label"].astype(str).str.strip()
    df = df[df["true_label"] != ""]
    unknown = sorted(set(df["true_label"]) - set(HAZARD_LABELS))
    if unknown:
        print(f"[evaluation][warning] unknown true_label values found "
              f"(kept, but may inflate uncertain): {unknown}")
    return df.reset_index(drop=True)


def _confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> pd.DataFrame:
    """Pure-Python confusion matrix as a labeled DataFrame (rows=true)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    return pd.DataFrame(mat, index=labels, columns=labels)


def _per_class_prf(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / F1 in pure Python."""
    out: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        out[lab] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(tp + fn),
        }
    return out


def _per_class_accuracy(
    y_true: List[str], y_pred: List[str], labels: List[str],
) -> Dict[str, float]:
    counts = Counter(y_true)
    correct = Counter(
        t for t, p in zip(y_true, y_pred) if t == p
    )
    return {
        lab: (correct.get(lab, 0) / counts[lab]) if counts.get(lab, 0) else 0.0
        for lab in labels
    }


def evaluate_hazards(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    video_id: Optional[str] = None,
    min_track_frames: int = 3,
) -> dict:
    """Compute accuracy / per-class metrics / confusion matrix.

    Returns a dict with all metrics. If no matched rows exist (no labels
    for the predicted tracks), the result is still a dict but with a
    ``message`` explaining why metrics are empty.

    When the predictions carry a ``num_frames`` column, a *selective*
    ("decidable") accuracy is also reported: accuracy restricted to
    tracks with at least ``min_track_frames`` detections. Below that
    threshold the classifier deliberately abstains as
    ``uncertain/short_track`` (a single frame carries no motion signal),
    so this number isolates classifier quality from tracker
    fragmentation. It is reported *alongside* — never instead of — the
    overall accuracy and the abstention count.
    """
    if labels_df is None or labels_df.empty:
        return {
            "video_id": video_id,
            "message": "no labels available; evaluation skipped",
            "num_labeled_tracks": 0,
            "num_predicted_tracks": int(len(pred_df)) if pred_df is not None else 0,
        }
    if pred_df is None or pred_df.empty:
        return {
            "video_id": video_id,
            "message": "no predictions to evaluate",
            "num_labeled_tracks": int(len(labels_df)),
            "num_predicted_tracks": 0,
        }

    if video_id is not None:
        labels_df = labels_df[labels_df["video_id"].astype(str) == str(video_id)]
        pred_df = pred_df[pred_df["video_id"].astype(str) == str(video_id)]
        if labels_df.empty:
            return {
                "video_id": video_id,
                "message": f"no labels found for video_id={video_id}",
                "num_labeled_tracks": 0,
                "num_predicted_tracks": int(len(pred_df)),
            }

    merged = pred_df.merge(
        labels_df[list(LABEL_FILE_COLUMNS)],
        on=["video_id", "track_id"],
        how="inner",
        suffixes=("", "_label"),
    )
    if merged.empty:
        return {
            "video_id": video_id,
            "message": "labels exist but no track_id matched predictions",
            "num_labeled_tracks": int(len(labels_df)),
            "num_predicted_tracks": int(len(pred_df)),
        }

    y_true = merged["true_label"].astype(str).tolist()
    y_pred = merged["hazard_label"].astype(str).tolist()
    labels = list(HAZARD_LABELS)

    overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    per_class_acc = _per_class_accuracy(y_true, y_pred, labels)
    per_class_prf = _per_class_prf(y_true, y_pred, labels)
    cm = _confusion_matrix(y_true, y_pred, labels)

    # Selective ("decidable") accuracy on tracks with enough temporal
    # support to classify. Requires a num_frames column on the predictions.
    selective: Optional[dict] = None
    if "num_frames" in merged.columns:
        nf = pd.to_numeric(merged["num_frames"], errors="coerce")
        dec_mask = nf >= int(min_track_frames)
        dec = merged[dec_mask]
        if len(dec) > 0:
            dt = dec["true_label"].astype(str).tolist()
            dp = dec["hazard_label"].astype(str).tolist()
            dec_acc = sum(1 for t, p in zip(dt, dp) if t == p) / len(dt)
            selective = {
                "min_track_frames": int(min_track_frames),
                "num_decidable_tracks": int(len(dec)),
                "decidable_accuracy": float(dec_acc),
                "num_short_tracks": int(len(merged) - len(dec)),
            }

    approach_metrics = per_class_prf.get(POSITIVE_LABEL, {})

    macro_f1 = float(
        sum(m["f1"] for m in per_class_prf.values()) / max(len(per_class_prf), 1)
    )
    sklearn_macro_f1: Optional[float] = None
    try:
        from sklearn.metrics import f1_score  # type: ignore
        sklearn_macro_f1 = float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )
    except Exception:
        sklearn_macro_f1 = None

    return {
        "video_id": video_id,
        "num_labeled_tracks": int(len(labels_df)),
        "num_predicted_tracks": int(len(pred_df)),
        "num_evaluated_tracks": int(len(merged)),
        "overall_accuracy": float(overall_acc),
        "selective_accuracy": selective,
        "per_class_accuracy": per_class_acc,
        "per_class_metrics": per_class_prf,
        "approaching_precision": float(approach_metrics.get("precision", 0.0)),
        "approaching_recall": float(approach_metrics.get("recall", 0.0)),
        "approaching_f1": float(approach_metrics.get("f1", 0.0)),
        "macro_f1": macro_f1,
        "macro_f1_sklearn": sklearn_macro_f1,
        # Nested as {true_label: {predicted_label: count}} so JSON, CSV
        # and the heatmap all share the same orientation (rows=true).
        "confusion_matrix": cm.to_dict(orient="index"),
        "confusion_matrix_orientation": "rows=true_label, cols=predicted_label",
        "labels": labels,
    }


def save_evaluation(
    metrics: dict,
    output_dir: str | Path,
    video_id: Optional[str] = None,
) -> Dict[str, str]:
    """Persist the evaluation summary + confusion matrix to disk."""
    out_dir = ensure_dir(output_dir)
    suffix = video_id or "all_videos"
    summary_path = out_dir / f"{suffix}_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    written: Dict[str, str] = {"summary_json": str(summary_path)}

    if "confusion_matrix" in metrics and isinstance(metrics["confusion_matrix"], dict):
        cm_df = pd.DataFrame.from_dict(
            metrics["confusion_matrix"], orient="index"
        )
        if "labels" in metrics:
            cols = [c for c in metrics["labels"] if c in cm_df.columns]
            cm_df = cm_df.reindex(index=metrics["labels"], columns=cols).fillna(0).astype(int)
        cm_df.index.name = "true_label"
        cm_path = out_dir / f"{suffix}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        written["confusion_matrix_csv"] = str(cm_path)
    return written


def evaluate_all_and_save(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: str | Path,
) -> Dict[str, dict]:
    """Run evaluation per ``video_id`` AND on the aggregate, save both."""
    results: Dict[str, dict] = {}
    if pred_df is None or pred_df.empty or labels_df is None or labels_df.empty:
        agg = evaluate_hazards(pred_df, labels_df)
        save_evaluation(agg, output_dir)
        results["all_videos"] = agg
        return results

    for vid in sorted(set(labels_df["video_id"].astype(str))):
        m = evaluate_hazards(pred_df, labels_df, video_id=vid)
        save_evaluation(m, output_dir, video_id=vid)
        results[vid] = m

    agg = evaluate_hazards(pred_df, labels_df)
    save_evaluation(agg, output_dir)
    results["all_videos"] = agg
    return results
