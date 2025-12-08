from typing import Dict
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    bac = balanced_accuracy_score(y_true, y_pred)

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "bac": float(bac),
    }


def compute_metrics_from_proba(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    return compute_metrics(y_true, y_pred)
