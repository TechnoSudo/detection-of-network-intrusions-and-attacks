import copy
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier

ModelBundle = Dict[str, BaseEstimator]

def build_base_models(random_state: int = 42) -> ModelBundle:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
    )

    sgd = SGDClassifier(
        loss="log_loss",          # enables predict_proba
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=random_state,
    )

    return {
        "rf": rf,
        "xgb": xgb,
        "sgd": sgd,
    }


def train_offline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> ModelBundle:
    models = build_base_models(random_state=random_state)
    models["rf"].fit(X_train, y_train)
    models["xgb"].fit(X_train, y_train)
    models["sgd"].fit(X_train, y_train)
    return models


def predict_all_models(
    models: ModelBundle,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        preds[name] = model.predict(X)

    return preds


def predict_proba_all_models(
    models: ModelBundle,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    proba: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[:, 1]
        else:
            # Fallback: if no predict_proba, approximate via decision_function if available
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                # Map scores to [0,1] via logistic
                p = 1.0 / (1.0 + np.exp(-scores))
            else:
                # As a last resort, use hard predictions as pseudo-probabilities
                preds = model.predict(X)
                p = preds.astype(float)

        proba[name] = p

    return proba


def make_sgd_static_and_adaptive(
    models: ModelBundle,
) -> Tuple[SGDClassifier, SGDClassifier]:
    base_sgd = models["sgd"]
    sgd_static = copy.deepcopy(base_sgd)
    sgd_adaptive = copy.deepcopy(base_sgd)
    return sgd_static, sgd_adaptive


def ensemble_predict_proba(
    models: ModelBundle,
    X: np.ndarray,
) -> np.ndarray:
    proba_dict = predict_proba_all_models(models, X)
    stacked = np.vstack(list(proba_dict.values()))  # shape: (n_models, n_samples)
    return stacked.mean(axis=0)


def ensemble_predict_labels(
    models: ModelBundle,
    X: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    p_hat = ensemble_predict_proba(models, X)
    return (p_hat >= threshold).astype(int)