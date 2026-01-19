# https://xgboost.readthedocs.io/en/stable/get_started.html

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


@dataclass
class XGBConfig:
    # Core boosting params
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6

    # Regularization / robustness
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0

    # Practical
    n_jobs: int = -1
    random_state: int = 42

    # CPU-friendly, fast histogram algorithm
    tree_method: str = "hist"


class XGBModel:
    def __init__(self, config: XGBConfig = XGBConfig()):
        self.cfg = config
        self.model: Optional[XGBClassifier] = None
        self.le = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBModel":
        y = np.asarray(y)

        # Encode labels -> required for XGBoost multiclass, safe for binary too
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        num_class = len(self.classes_)

        # Choose objective based on number of classes
        if num_class <= 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        params = dict(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_lambda=self.cfg.reg_lambda,
            reg_alpha=self.cfg.reg_alpha,
            min_child_weight=self.cfg.min_child_weight,
            gamma=self.cfg.gamma,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            tree_method=self.cfg.tree_method,
            objective=objective,
            eval_metric=eval_metric,
        )

        # Only set num_class for multi-class
        if num_class > 2:
            params["num_class"] = num_class

        self.model = XGBClassifier(**params)
        self.model.fit(X, y_enc)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("XGBModel not fitted. Call fit() first.")

        proba = self.model.predict_proba(X)

        # Ensure shape is [N, 2] for binary edge cases
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = np.vstack([1.0 - proba, proba]).T

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.model is None or self.classes_ is None:
            raise RuntimeError("XGBModel not fitted. Call fit() first.")

        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def get_params(self) -> Dict[str, Any]:
        out = dict(vars(self.cfg))
        if self.model is not None:
            out.update(self.model.get_params())
        return out

    def set_params(self, **kwargs) -> "XGBModel":
        # Update config fields
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        return self
