from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.data_loader import load_kdd, load_nsl_kdd, load_netflow, load_cores_iot
from src.data.preprocessing import preprocess_with_split
from src.evaluation.metrics import compute_metrics

from src.models.supervised.adaptive_sgd import AdaptiveSGDModel
from src.models.supervised.random_forest import RFModel
from src.models.supervised.xgboost import XGBModel
from src.models.supervised.ensemble import WeightedSoftVotingEnsemble


def run_scenario1(
    dataset: str,
    outdir: str,
    sample_frac: float = 1.0, 
    seed: int = 42,
) -> Dict[str, Any]:

    print("\n" + "=" * 80)
    print(f"RUNNING SCENARIO 1 — KNOWN ATTACKS | Dataset: {dataset.upper()}")
    print("=" * 80)

    os.makedirs(outdir, exist_ok=True)

    if dataset == "kdd":
        X, y, _ = load_kdd("data/dataset-1/kddcup.data")
        bundle = preprocess_with_split(
            X=X,
            y=y,
            cat_cols=["protocol_type", "service", "flag"],
            name="SCENARIO1_KDD",
        )

    elif dataset == "nsl":
        X, y, _ = load_nsl_kdd("data/dataset-4/NSL-KDD-train.txt")
        bundle = preprocess_with_split(
            X=X,
            y=y,
            cat_cols=["protocol_type", "service", "flag"],
            name="SCENARIO1_NSL",
        )

    elif dataset == "netflow":
        X, y, _ = load_netflow("data/dataset-2/train_net.csv")
        bundle = preprocess_with_split(
            X=X,
            y=y,
            cat_cols=["PROTOCOL_MAP"],
            name="SCENARIO1_NETFLOW",
        )

    elif dataset == "iot":
        X, y = load_cores_iot("data/dataset-3/cores_iot.csv")
        bundle = preprocess_with_split(
            X=X,
            y=y,
            cat_cols=[],
            name="SCENARIO1_IOT",
        )

    else:
        raise ValueError("scenario1 datasets: kdd, nsl, netflow, iot")

    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_val, y_val = bundle["X_val"], bundle["y_val"]

    results = []

    print("\nTraining Random Forest...")
    rf = RFModel()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    results.append({"model": "RF", **compute_metrics(y_val, y_pred)})

    print("\nTraining Adaptive SGD...")
    sgd = AdaptiveSGDModel(seed=seed) if "seed" in AdaptiveSGDModel.__init__.__code__.co_varnames else AdaptiveSGDModel()
    if hasattr(sgd, "fit"):
        sgd.fit(X_train, y_train)
    else:
        sgd.partial_fit(X_train, y_train)
    y_pred = sgd.predict(X_val)
    results.append({"model": "SGD", **compute_metrics(y_val, y_pred)})

    print("\nTraining XGBoost...")
    xgb = XGBModel()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_val)
    results.append({"model": "XGB", **compute_metrics(y_val, y_pred)})

    print("\nTraining Ensemble...")
    ensemble = WeightedSoftVotingEnsemble(
        models={"rf": rf, "sgd": sgd, "xgb": xgb}
    )
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_val)

    # Save ensemble arrays (for confusion / PR plots)
    y_proba = None
    if hasattr(ensemble, "predict_proba"):
        try:
            p = ensemble.predict_proba(X_val)
            if p.ndim == 2 and p.shape[1] >= 2:
                y_proba = p[:, 1]
        except Exception:
            pass

    results.append({"model": "Ensemble", **compute_metrics(y_val, y_pred)})

    df = pd.DataFrame(results)
    out_csv = os.path.join(outdir, f"results_scenario1_{dataset}.csv")
    df.to_csv(out_csv, index=False)

    out_npz = os.path.join(outdir, "arrays_ensemble.npz")
    if y_proba is None:
        np.savez(out_npz, y_true=y_val, y_pred=y_pred)
    else:
        np.savez(out_npz, y_true=y_val, y_pred=y_pred, y_proba=y_proba)

    print("\nRESULTS:")
    print(df)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_npz}")

    return {
        "results_csv": out_csv,
        "npz": {"ensemble": out_npz},
    }