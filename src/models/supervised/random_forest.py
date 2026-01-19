# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: Optional[str] = "balanced"
    n_jobs: int = -1
    random_state: int = 42


class RFModel:
    def __init__(self, config: RandomForestConfig = RandomForestConfig()):
        self.cfg = config
        self.model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_split=self.cfg.min_samples_split,
            min_samples_leaf=self.cfg.min_samples_leaf,
            class_weight=self.cfg.class_weight,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFModel":
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

    def set_params(self, **kwargs) -> "RFModel":
        self.model.set_params(**kwargs)
        return self
