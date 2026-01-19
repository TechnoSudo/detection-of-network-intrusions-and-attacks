from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    contamination: float = 0.01     # expected anomaly rate; tune for NetFlow
    random_state: int = 42


class IsolationForestAD:
    """
    Simple anomaly detector wrapper.

    - anomaly_score: higher => more anomalous
    - is_anomaly: boolean flag using percentile threshold or fixed threshold
    """
    def __init__(self, cfg: IsolationForestConfig = IsolationForestConfig()):
        self.cfg = cfg
        self.model = IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self._fitted = True

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsolationForestAD not fitted.")
        # sklearn: score_samples higher => more normal
        normality = self.model.score_samples(X)
        return -normality  # flip: higher => more anomalous

    def is_anomaly(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None,
        percentile: float = 99.0,
    ) -> np.ndarray:
        scores = self.anomaly_score(X)
        if threshold is None:
            threshold = float(np.percentile(scores, percentile))
        return scores >= threshold

    def info(self) -> Dict[str, Any]:
        return {
            "type": "IsolationForest",
            "n_estimators": self.cfg.n_estimators,
            "contamination": self.cfg.contamination,
            "random_state": self.cfg.random_state,
        }