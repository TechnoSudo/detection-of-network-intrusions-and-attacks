from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def build_attack_pools(
    X: pd.DataFrame,
    y: pd.Series,
    y_attack_type: pd.Series,
    known_attacks: List[str],
    new_attacks: List[str],
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:

    normal_like = y_attack_type.astype(str).str.lower().isin(["normal", "none"])
    is_normal = (y == 0) | normal_like

    # Known/new attacks based on the attack-type labels
    is_known = y_attack_type.isin(known_attacks)
    is_new = y_attack_type.isin(new_attacks)

    X_normal = X.loc[is_normal]
    y_normal = y.loc[is_normal]

    X_known = X.loc[is_known]
    y_known = y.loc[is_known]

    X_new = X.loc[is_new]
    y_new = y.loc[is_new]

    return {
        "normal": (X_normal, y_normal),
        "known": (X_known, y_known),
        "new": (X_new, y_new),
    }



def _sample_pool(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    n_samples: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    idx = np.random.choice(len(X_pool), size=n_samples, replace=True)
    return X_pool.iloc[idx], y_pool.iloc[idx]


def create_streaming_batches(
    pools: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    batch_size: int,
    n_pre_drift: int,
    n_drift_onset: int,
    n_post_drift: int,
    ratios_pre_drift: Tuple[float, float] = (0.80, 0.20),
    ratios_onset: Tuple[float, float, float] = (0.75, 0.20, 0.05),
    ratios_post: Tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> List[Tuple[pd.DataFrame, pd.Series]]:

    X_norm, y_norm = pools["normal"]
    X_known, y_known = pools["known"]
    X_new, y_new = pools["new"]

    batches = []
    
    # Pre-drift batches
    normal_r, known_r = ratios_pre_drift
    for _ in range(n_pre_drift):
        n_normal = int(batch_size * normal_r)
        n_known = int(batch_size * known_r)

        Xn, yn = _sample_pool(X_norm, y_norm, n_normal)
        Xa, ya = _sample_pool(X_known, y_known, n_known)

        X_batch = pd.concat([Xn, Xa], axis=0)
        y_batch = pd.concat([yn, ya], axis=0)

        batches.append((X_batch, y_batch))

    # Drift onset batches
    normal_r, known_r, new_r = ratios_onset
    for _ in range(n_drift_onset):
        n_normal = int(batch_size * normal_r)
        n_known = int(batch_size * known_r)
        n_new = int(batch_size * new_r)

        Xn, yn = _sample_pool(X_norm, y_norm, n_normal)
        Xa, ya = _sample_pool(X_known, y_known, n_known)
        Xnew, ynew = _sample_pool(X_new, y_new, n_new)

        X_batch = pd.concat([Xn, Xa, Xnew], axis=0)
        y_batch = pd.concat([yn, ya, ynew], axis=0)

        batches.append((X_batch, y_batch))

    # Post-drift batches
    normal_r, known_r, new_r = ratios_post
    for _ in range(n_post_drift):
        n_normal = int(batch_size * normal_r)
        n_known = int(batch_size * known_r)
        n_new = int(batch_size * new_r)

        Xn, yn = _sample_pool(X_norm, y_norm, n_normal)
        Xa, ya = _sample_pool(X_known, y_known, n_known)
        Xnew, ynew = _sample_pool(X_new, y_new, n_new)

        X_batch = pd.concat([Xn, Xa, Xnew], axis=0)
        y_batch = pd.concat([yn, ya, ynew], axis=0)

        batches.append((X_batch, y_batch))

    return batches
