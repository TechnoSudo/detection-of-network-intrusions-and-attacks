from __future__ import annotations

import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _save(fig: plt.Figure, outpath: str) -> None:
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"[plot] saved -> {outpath}")
    plt.close(fig)


def _load_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "y_true" not in out:
        raise ValueError(f"NPZ must contain y_true. Found: {sorted(out.keys())}")
    return out


def _maybe_binarize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        y_str = y.astype(str)
        if np.all(np.isin(y_str, ["0", "1"])):
            return y_str.astype(int)
    return y.astype(int)


def _phase_bounds(n_pre: int, n_drift: int, n_post: int) -> Dict[str, int]:
    pre_end = n_pre - 1
    drift_end = n_pre + n_drift - 1
    post_end = n_pre + n_drift + n_post - 1
    return {"pre_end": pre_end, "drift_end": drift_end, "post_end": post_end}


# ============================================================================
# Scenario 2 and 3 plots 
#   - model metrics bar chart
#   - confusion matrix (best model / ensemble)
#   - precision-recall curve (best model / ensemble)
# ============================================================================

def plot_model_metrics(results_csv: str, outdir: str = "plots", title: Optional[str] = None) -> None:
    """
    Bar chart comparing models on key metrics.
    Expects CSV columns: model, f1, precision, recall, bac
    """
    _ensure_outdir(outdir)

    df = pd.read_csv(results_csv)
    required_cols = {"model", "f1", "precision", "recall", "bac"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{results_csv} must contain columns: {sorted(required_cols)}. Found: {df.columns.tolist()}"
        )

    df = df.copy()
    df["model"] = df["model"].astype(str)

    metrics = ["f1", "precision", "recall", "bac"]
    x = np.arange(len(df["model"]))
    width = 0.2

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[m].values, width=width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].values, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title or f"Model performance ({os.path.basename(results_csv)})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    outpath = os.path.join(outdir, f"metrics_{os.path.splitext(os.path.basename(results_csv))[0]}.png")
    _save(fig, outpath)


def plot_confusion(npz_path: str, outdir: str = "plots", title: str = "Confusion Matrix") -> None:
    """
    Confusion matrix for best model / ensemble.
    Expects NPZ keys: y_true, y_pred
    """
    _ensure_outdir(outdir)
    data = _load_npz(npz_path)

    if "y_pred" not in data:
        raise ValueError("NPZ must contain y_pred for confusion matrix.")

    y_true = _maybe_binarize(data["y_true"])
    y_pred = _maybe_binarize(data["y_pred"])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal(0)", "attack(1)"])
    ax.set_yticklabels(["normal(0)", "attack(1)"])

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    outpath = os.path.join(outdir, f"confusion_{os.path.splitext(os.path.basename(npz_path))[0]}.png")
    _save(fig, outpath)


def plot_prec_recall(npz_path: str, outdir: str = "plots", title: str = "Precision-Recall Curve") -> None:
    """
    PR curve for best model / ensemble (recommended for imbalanced data).
    Expects NPZ keys: y_true, y_proba
    """
    _ensure_outdir(outdir)
    data = _load_npz(npz_path)

    if "y_proba" not in data:
        raise ValueError("NPZ must contain y_proba for PR curve.")

    y_true = _maybe_binarize(data["y_true"])
    y_proba = np.asarray(data["y_proba"]).astype(float)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.plot(recall, precision, label=f"AP={ap:.4f}")

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")

    outpath = os.path.join(outdir, f"pr_{os.path.splitext(os.path.basename(npz_path))[0]}.png")
    _save(fig, outpath)


# ============================================================================
# Scenario 3 plots 
#   - batch-wise F1 over time (multi-model)
#   - phase shading (pre/drift/post)
#   - drift markers (no vertical label spam)
# ============================================================================

def plot_batch_f1_multi(
    df_metrics: pd.DataFrame,
    outpath: str,
    title: str,
    n_pre: int,
    n_drift: int,
    n_post: int,
    use_ema: bool = True,
    y_col: Optional[str] = None,
    show_drift_markers: bool = True,
    df_events: Optional[pd.DataFrame] = None,
) -> None:

    # Choose metric column
    if y_col is not None:
        metric_col = y_col
    else:
        metric_col = "f1_ema" if (use_ema and "f1_ema" in df_metrics.columns) else "f1"

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    # -----------------------------
    # Phase shading + readable labels
    # -----------------------------
    pre_end = n_pre - 1
    drift_end = n_pre + n_drift - 1
    post_end = n_pre + n_drift + n_post - 1

    ax.axvspan(-0.5, pre_end + 0.5, alpha=0.08)
    ax.text(
        (0 + pre_end) / 2,
        0.97,
        "Pre-drift",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.axvspan(pre_end + 0.5, drift_end + 0.5, alpha=0.12)
    ax.text(
        (pre_end + 1 + drift_end) / 2,
        0.97,
        "Drift",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.axvspan(drift_end + 0.5, post_end + 0.5, alpha=0.08)
    ax.text(
        (drift_end + 1 + post_end) / 2,
        0.97,
        "Post-drift",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    # -----------------------------
    # Model performance curves
    # -----------------------------
    for model_name in df_metrics["model"].unique():
        sub = df_metrics[df_metrics["model"] == model_name].sort_values("batch_idx")
        ax.plot(
            sub["batch_idx"],
            sub[metric_col],
            marker="o",
            linewidth=2,
            markersize=4,
            label=model_name,
        )

    # -----------------------------
    # Drift markers (distinct styles)
    # -----------------------------
    if show_drift_markers and df_events is not None and len(df_events) > 0:
        drift_styles = {
            "ADWIN": dict(marker="^", color="tab:blue", label="Drift: ADWIN"),
            "PageHinkley": dict(marker="s", color="tab:orange", label="Drift: Page-Hinkley"),
        }

        for det, style in drift_styles.items():
            sub = df_events[df_events["detector"] == det]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["batch_idx"],
                np.full(len(sub), 0.985),
                s=70,
                marker=style["marker"],
                color=style["color"],
                edgecolors="black",
                linewidths=0.8,
                label=style["label"],
                zorder=5,
            )

    # -----------------------------
    # Axis + styling
    # -----------------------------
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Batch index")
    ax.set_ylabel(metric_col.upper())

    # 🔑 IMPORTANT: zoomed Y-axis
    ax.set_ylim(0.5, 1.02)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", ncol=2)

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] saved -> {outpath}")


def plot_anomaly_score_hist(anomaly_csv: str, outdir: str, title: str = "Anomaly score distribution") -> None:
    _ensure_outdir(outdir)
    df = pd.read_csv(anomaly_csv)
    if not {"y_true", "anomaly_score"}.issubset(df.columns):
        raise ValueError("anomaly_csv must contain y_true, anomaly_score")

    y = df["y_true"].astype(int).to_numpy()
    s = df["anomaly_score"].astype(float).to_numpy()

    s0 = s[y == 0]
    s1 = s[y == 1]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(s0, bins=60, alpha=0.6, label="Normal (0)")
    ax.hist(s1, bins=60, alpha=0.6, label="Attack (1)")
    ax.set_title(title)
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    outpath = os.path.join(outdir, "anomaly_score_hist.png")
    _save(fig, outpath)