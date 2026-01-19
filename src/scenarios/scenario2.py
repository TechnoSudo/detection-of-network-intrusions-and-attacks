import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.data.data_loader import load_kdd, load_nsl_kdd, load_netflow
from src.data.preprocessing import preprocess_given_train_val

from src.models.supervised.random_forest import RFModel
from src.models.supervised.adaptive_sgd import AdaptiveSGDModel
from src.models.supervised.xgboost import XGBModel
from src.models.supervised.ensemble import WeightedSoftVotingEnsemble


ATTACK_CATEGORIES_KDD: Dict[str, List[str]] = {
    "dos": ["back", "land", "neptune", "pod", "smurf", "teardrop"],
    "probe": ["ipsweep", "nmap", "portsweep", "satan"],
    "r2l": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"],
    "u2r": ["buffer_overflow", "loadmodule", "perl", "rootkit"],
}

SCENARIO2_UNKNOWN_KDD_NSL: List[str] = [
    "teardrop",        # dos
    "nmap",            # probe
    "warezclient",     # r2l
    "rootkit",         # u2r
]
SCENARIO2_NETFLOW_UNKNOWN: List[str] = ["Malware"]
SCENARIO2_NETFLOW_KNOWN: List[str] = ["Port Scanning", "Denial of Service"]


def sub_sample(
    X: pd.DataFrame,
    y: pd.Series,
    y_attack: pd.Series,
    sample_frac: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Randomly sample a fraction of rows (default 50%)."""
    if sample_frac >= 1.0:
        return X.reset_index(drop=True), y.reset_index(drop=True), y_attack.reset_index(drop=True)

    n = len(X)
    n_sample = int(n * sample_frac)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_sample, replace=False)
    idx = np.sort(idx)

    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].reset_index(drop=True)
    yas = y_attack.iloc[idx].reset_index(drop=True)

    print(f"[subset] original={n}, subset={len(Xs)} (frac={sample_frac})")
    return Xs, ys, yas


def preprocess_train_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str,
    cat_cols: List[str],
):
    """Fit preprocess on train only; transform test using same fitted objects."""
    bundle = preprocess_given_train_val(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        cat_cols=cat_cols,
        name=f"SCENARIO2_{dataset_name}",
    )
    return bundle["X_train"], bundle["y_train"], bundle["X_val"], bundle["y_val"]


def _normalize_attack_labels(y_attack: pd.Series, dataset: str) -> pd.Series:
    ya = y_attack.astype(str).str.strip()

    if dataset == "netflow":
        ya = ya.replace({"None": "normal", "<null>": "normal", "nan": "normal", "NaN": "normal"})

    ya = ya.str.replace(".", "", regex=False)
    return ya


def _kdd_nsl_known_unknown_from_present_attacks(y_attack: pd.Series) -> Tuple[List[str], List[str]]:
    present = set(y_attack.unique().tolist())
    present.discard("normal")

    all_tax_attacks = set()
    for v in ATTACK_CATEGORIES_KDD.values():
        all_tax_attacks.update(v)

    present_tax_attacks = sorted(list(all_tax_attacks & present))

    unknown = [a for a in SCENARIO2_UNKNOWN_KDD_NSL if a in present]
    known = [a for a in present_tax_attacks if a not in unknown]

    if len(unknown) == 0:
        raise ValueError(
            f"[scenario2] None of the hardcoded unknown KDD/NSL attacks are present in this file. "
            f"Present attacks: {sorted(list(present))[:50]}"
        )
    if len(known) == 0:
        raise ValueError("[scenario2] Known attacks ended up empty after filtering. Adjust unknown list.")

    print(f"[attack-types:KDD/NSL] present_attacks={len(present)} known={len(known)} unknown={len(unknown)}")
    print(f"[attack-types:KDD/NSL] unknown={unknown}")
    return known, unknown


def _netflow_known_unknown_from_present_attacks(y_attack: pd.Series) -> Tuple[List[str], List[str]]:
    present = set(y_attack.unique().tolist())
    present.discard("normal")

    unknown = [a for a in SCENARIO2_NETFLOW_UNKNOWN if a in present]
    known = [a for a in SCENARIO2_NETFLOW_KNOWN if a in present]

    if len(unknown) == 0:
        raise ValueError(
            f"[scenario2] Hardcoded NetFlow unknown attack not found in this file. "
            f"Present attacks: {sorted(list(present))}"
        )
    if len(known) == 0:
        raise ValueError(
            f"[scenario2] Hardcoded NetFlow known attacks not found in this file. "
            f"Present attacks: {sorted(list(present))}"
        )

    print(f"[attack-types:NETFLOW] present_attacks={len(present)} known={len(known)} unknown={len(unknown)}")
    print(f"[attack-types:NETFLOW] unknown={unknown}")
    return known, unknown


def build_train_test_from_types(
    X: pd.DataFrame,
    y: pd.Series,
    y_attack: pd.Series,
    known_types: List[str],
    unknown_types: List[str],
    train_frac_within_subset: float = 0.6,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    normal_mask = (y_attack == "normal")
    normal_idx = np.where(normal_mask.to_numpy())[0]
    rng.shuffle(normal_idx)

    n_train_target = int(len(X) * train_frac_within_subset)
    n_test_target = len(X) - n_train_target

    n_train_normal = int(len(normal_idx) * train_frac_within_subset)
    train_normal_idx = normal_idx[:n_train_normal]
    test_normal_idx = normal_idx[n_train_normal:]

    known_mask = y_attack.isin(known_types)
    known_idx = np.where(known_mask.to_numpy())[0]
    rng.shuffle(known_idx)

    unknown_mask = y_attack.isin(unknown_types)
    unknown_idx = np.where(unknown_mask.to_numpy())[0]
    rng.shuffle(unknown_idx)

    # Train: normals + known
    train_idx = list(train_normal_idx)
    remaining_train = n_train_target - len(train_idx)
    if remaining_train > 0:
        take = min(remaining_train, len(known_idx))
        train_idx += list(known_idx[:take])

    # Test: normals + unknown
    test_idx = list(test_normal_idx)
    remaining_test = n_test_target - len(test_idx)
    if remaining_test > 0:
        take = min(remaining_test, len(unknown_idx))
        test_idx += list(unknown_idx[:take])

    if len(train_idx) < n_train_target:
        print(f"[warn] train target={n_train_target}, got={len(train_idx)} (not enough known attacks?)")
    if len(test_idx) < n_test_target:
        print(f"[warn] test  target={n_test_target}, got={len(test_idx)} (not enough unknown attacks?)")

    train_idx = np.array(sorted(set(train_idx)))
    test_idx = np.array(sorted(set(test_idx)))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"[split] train={len(X_train)} test={len(X_test)} (targets were {n_train_target}/{n_test_target})")
    print(f"[split] train positives={int(y_train.sum())} negatives={len(y_train)-int(y_train.sum())}")
    print(f"[split] test  positives={int(y_test.sum())} negatives={len(y_test)-int(y_test.sum())}")

    ya_train = y_attack.iloc[train_idx].reset_index(drop=True)
    ya_test = y_attack.iloc[test_idx].reset_index(drop=True)
    print(f"[split] train attack-types (top): {ya_train[ya_train!='normal'].value_counts().head(8).to_dict()}")
    print(f"[split] test  attack-types (top): {ya_test[ya_test!='normal'].value_counts().head(8).to_dict()}")

    return X_train, y_train, X_test, y_test


def run_scenario2(
    dataset: str,
    outdir: str,
    sample_frac: float = 1.0,
    seed: int = 42,
):
    print("\n" + "=" * 80)
    print(f"RUNNING SCENARIO 2 — UNSEEN ATTACK TYPES | Dataset: {dataset.upper()}")
    print("=" * 80)

    os.makedirs(outdir, exist_ok=True)

    # Load dataset
    if dataset == "kdd":
        X, y, y_attack = load_kdd("data/dataset-1/kddcup.data")
        cat_cols = ["protocol_type", "service", "flag"]

    elif dataset == "nsl":
        X, y, y_attack = load_nsl_kdd("data/dataset-4/NSL-KDD-train.txt")
        cat_cols = ["protocol_type", "service", "flag"]

    elif dataset == "netflow":
        X, y, y_attack = load_netflow("data/dataset-2/train_net.csv")
        cat_cols = ["PROTOCOL_MAP"]

    else:
        raise ValueError("scenario2 datasets: kdd, nsl, netflow")

    # Normalize labels
    y_attack = _normalize_attack_labels(y_attack, dataset=dataset)
    if "normal" not in set(y_attack.unique().tolist()):
        raise ValueError(f"[scenario2] Expected 'normal' after normalization. Got: {y_attack.unique()}")

    # Subsample
    Xs, ys, yas = sub_sample(X, y, y_attack, sample_frac=sample_frac, seed=seed)

    # Known/unknown selection
    if dataset in {"kdd", "nsl"}:
        known_types, unknown_types = _kdd_nsl_known_unknown_from_present_attacks(yas)
    else:
        known_types, unknown_types = _netflow_known_unknown_from_present_attacks(yas)

    # Build train/test split
    X_train, y_train, X_test, y_test = build_train_test_from_types(
        Xs, ys, yas,
        known_types=known_types,
        unknown_types=unknown_types,
        train_frac_within_subset=0.6,
        seed=seed,
    )

    # Preprocess (fit on train only)
    Xtr, ytr, Xte, yte = preprocess_train_test(
        X_train, y_train, X_test, y_test,
        dataset_name=dataset.upper(),
        cat_cols=cat_cols,
    )

    # Train supervised models
    results = []

    print("\nTraining Random Forest...")
    rf = RFModel()
    rf.fit(Xtr, ytr)
    y_pred_rf = rf.predict(Xte)
    results.append({"model": "RF", **compute_metrics(yte, y_pred_rf)})

    print("\nTraining SGD...")
    sgd = AdaptiveSGDModel()
    sgd.fit(Xtr, ytr)
    y_pred_sgd = sgd.predict(Xte)
    results.append({"model": "SGD", **compute_metrics(yte, y_pred_sgd)})

    print("\nTraining XGBoost...")
    xgb = XGBModel()
    xgb.fit(Xtr, ytr)
    y_pred_xgb = xgb.predict(Xte)
    results.append({"model": "XGB", **compute_metrics(yte, y_pred_xgb)})

    print("\nTraining Ensemble...")
    ensemble = WeightedSoftVotingEnsemble(models={"rf": rf, "sgd": sgd, "xgb": xgb})
    ensemble.fit(Xtr, ytr)
    y_pred_ens = ensemble.predict(Xte)

    # Try to get ensemble proba (for PR curve)
    y_proba_ens: Optional[np.ndarray] = None
    if hasattr(ensemble, "predict_proba"):
        try:
            p = ensemble.predict_proba(Xte)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                y_proba_ens = p[:, 1]
            elif p.ndim == 1:
                y_proba_ens = p
        except Exception:
            y_proba_ens = None

    results.append({"model": "Ensemble", **compute_metrics(yte, y_pred_ens)})

    # Save results table
    df = pd.DataFrame(results)
    out_csv = os.path.join(outdir, f"results_scenario2_{dataset}.csv")
    df.to_csv(out_csv, index=False)

    # Save split info
    out_txt = os.path.join(outdir, f"scenario2_{dataset}_splits.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Subset fraction used: {sample_frac}\n")
        f.write("Within subset: train=60%, test=40%\n\n")
        f.write(f"Known attack types ({len(known_types)}):\n")
        f.write(", ".join(known_types) + "\n\n")
        f.write(f"Unknown attack types ({len(unknown_types)}):\n")
        f.write(", ".join(unknown_types) + "\n\n")
        f.write(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")
        f.write(f"Train positives: {int(np.sum(y_train))} | Train negatives: {len(y_train)-int(np.sum(y_train))}\n")
        f.write(f"Test positives: {int(np.sum(y_test))} | Test negatives: {len(y_test)-int(np.sum(y_test))}\n")

    # Save arrays for plots (ensemble only)
    out_npz = os.path.join(outdir, "arrays_ensemble.npz")
    if y_proba_ens is None:
        np.savez(out_npz, y_true=np.asarray(yte), y_pred=np.asarray(y_pred_ens))
    else:
        np.savez(out_npz, y_true=np.asarray(yte), y_pred=np.asarray(y_pred_ens), y_proba=np.asarray(y_proba_ens))

    print("\n" + "-" * 60)
    print("KNOWN attack types:", known_types)
    print("UNKNOWN attack types:", unknown_types)
    print("\nRESULTS:")
    print(df)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_npz}")

    out = {
        "results_csv": out_csv,
        "npz": {"ensemble": out_npz},
        "splits_txt": out_txt,
        "known_types": known_types,
        "unknown_types": unknown_types,
    }
    return out