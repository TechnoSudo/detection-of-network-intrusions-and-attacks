from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score, matthews_corrcoef



def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "bac": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def compute_metrics_from_proba(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba_pos = np.asarray(y_proba_pos)

    y_pred = (y_proba_pos >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, average="binary")