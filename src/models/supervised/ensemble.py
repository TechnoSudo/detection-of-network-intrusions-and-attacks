# https://medium.com/@awanurrahman.cse/understanding-soft-voting-and-hard-voting-a-comparative-analysis-of-ensemble-learning-methods-db0663d2c008 

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np


@dataclass
class EnsembleConfig:
    # If weights are None => equal weights
    weights: Optional[Dict[str, float]] = None


class WeightedSoftVotingEnsemble:
    def __init__(self, models: Dict[str, object], config: EnsembleConfig = EnsembleConfig()):
        if not models:
            raise ValueError("models dict cannot be empty")
        self.models = models
        self.cfg = config
        self.classes_: Optional[np.ndarray] = None
        self._weights: Optional[Dict[str, float]] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeightedSoftVotingEnsemble":
        for m in self.models.values():
            m.fit(X, y)

        # Ensure consistent class ordering across models
        classes_list = [getattr(m, "classes_", None) for m in self.models.values()]
        if any(c is None for c in classes_list):
            raise ValueError("All base models must expose classes_ after fit()")

        # Use first model's class order as canonical
        self.classes_ = classes_list[0]
        for c in classes_list[1:]:
            if not np.array_equal(self.classes_, c):
                raise ValueError("Base models have different classes_ ordering/sets. Align labels first.")

        self._weights = self._compute_weights()
        return self

    def _compute_weights(self) -> Dict[str, float]:
        if self.cfg.weights is None:
            # Equal weights
            w = {name: 1.0 for name in self.models.keys()}
        else:
            # Use provided weights (e.g., from validation F1)
            w = dict(self.cfg.weights)

        # Guard: ensure all models have a weight
        for name in self.models.keys():
            w.setdefault(name, 1.0)

        # Normalize weights
        s = sum(float(v) for v in w.values())
        if s <= 0:
            raise ValueError("Sum of weights must be > 0")
        return {k: float(v) / s for k, v in w.items()}

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.cfg.weights = weights
        self._weights = self._compute_weights()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self._weights is None:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        probs = None
        for name, model in self.models.items():
            p = model.predict_proba(X)
            w = self._weights.get(name, 0.0)
            probs = p * w if probs is None else probs + (p * w)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Confidence = max class probability per sample.
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)