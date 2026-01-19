from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier


class AdaptiveSGDModel:

    def __init__(
        self,
        seed: int = 42,
        alpha: float = 1e-4,
        max_iter: int = 1000,
        tol: float = 1e-3,
        class_weight: str | dict | None = "balanced",
    ):
        self._requested_class_weight = class_weight
        internal_class_weight = None if class_weight == "balanced" else class_weight

        self.clf = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
            class_weight=internal_class_weight,
        )

        self.classes_: np.ndarray | None = None
        self._pf_initialized: bool = False

    def _balanced_sample_weight(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).astype(int)
        n = len(y)
        classes = np.array([0, 1], dtype=int)
        counts = np.array([(y == c).sum() for c in classes], dtype=float)
        counts = np.maximum(counts, 1.0) 
        weights = n / (len(classes) * counts)
        sw = np.ones(n, dtype=float)
        for c, w in zip(classes, weights):
            sw[y == c] = w
        return sw

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveSGDModel":
        y = np.asarray(y).astype(int)

        if self._requested_class_weight == "balanced":
            sw = self._balanced_sample_weight(y)
            self.clf.fit(X, y, sample_weight=sw)
        else:
            self.clf.fit(X, y)

        self.classes_ = np.asarray(self.clf.classes_)
        self._pf_initialized = True
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveSGDModel":
        y = np.asarray(y).astype(int)

        sw = None
        if self._requested_class_weight == "balanced":
            sw = self._balanced_sample_weight(y)

        if not self._pf_initialized:
            self.clf.partial_fit(X, y, classes=np.array([0, 1], dtype=int), sample_weight=sw)
            self._pf_initialized = True
        else:
            self.clf.partial_fit(X, y, sample_weight=sw)

        self.classes_ = np.asarray(self.clf.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.clf, "predict_proba"):
            raise RuntimeError("SGDClassifier has no predict_proba. Ensure loss='log_loss'.")

        p = np.asarray(self.clf.predict_proba(X))

        if p.ndim == 1:
            p = p.reshape(-1, 1)

        if p.shape[1] == 1:
            cls = int(self.classes_[0])
            if cls == 0:
                p = np.hstack([np.ones((len(p), 1)), np.zeros((len(p), 1))])
            else:
                p = np.hstack([np.zeros((len(p), 1)), np.ones((len(p), 1))])

        return p