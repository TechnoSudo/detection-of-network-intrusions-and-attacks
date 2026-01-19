import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_df(X: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, then fill NaN with 0."""
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def get_cat_cols(X: pd.DataFrame, forced_cat_cols: Optional[List[str]] = None) -> List[str]:
    """Detect categorical columns (object/string), or use forced list."""
    if forced_cat_cols is not None:
        return list(forced_cat_cols)
    return [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("string")]


def fit_preprocess(
    X_train: pd.DataFrame,
    cat_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fit scaler (numeric) + onehot encoder (categorical) on TRAIN only.
    Returns a dict containing fitted objects + column lists.
    """
    X_train = clean_df(X_train)
    cat_cols = get_cat_cols(X_train, cat_cols)
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    print(f"[fit_preprocess] numeric cols: {len(num_cols)}")
    print(f"[fit_preprocess] categorical cols: {len(cat_cols)} -> {cat_cols}")

    scaler = StandardScaler()
    scaler.fit(X_train[num_cols].astype(float))

    encoder = None
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(X_train[cat_cols].astype(str))

    return {
        "scaler": scaler,
        "encoder": encoder,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def transform_preprocess(
    X: pd.DataFrame,
    scaler: StandardScaler,
    num_cols: List[str],
    encoder: Optional[OneHotEncoder] = None,
    cat_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Transform any split using fitted scaler/encoder.
    Returns (X_np, feature_names).
    """
    X = clean_df(X)

    X_num = scaler.transform(X[num_cols].astype(float))
    feature_names = list(num_cols)

    if encoder is not None and cat_cols is not None and len(cat_cols) > 0:
        X_cat = encoder.transform(X[cat_cols].astype(str))
        cat_names = encoder.get_feature_names_out(cat_cols).tolist()
        X_out = np.hstack([X_num, X_cat])
        feature_names = feature_names + cat_names
    else:
        X_out = X_num

    return X_out, feature_names


def preprocess_with_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    cat_cols: Optional[List[str]] = None,
    val_size: float = 0.3,
    seed: int = 42,
    stratify: bool = True,
    name: str = "dataset",
) -> Dict[str, Any]:
    """
    Use ONE dataset file and create a train/val split.
    Fit preprocessing on train, transform train+val.
    """
    if y is None:
        raise ValueError("y must be provided for splitting.")

    strat = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=strat
    )

    print("\n" + "-" * 80)
    print(f"[preprocess:{name}] split train: {X_train.shape}, val: {X_val.shape}")

    fitted = fit_preprocess(X_train, cat_cols=cat_cols)

    X_train_np, feat_names = transform_preprocess(
        X_train, fitted["scaler"], fitted["num_cols"], fitted["encoder"], fitted["cat_cols"]
    )
    X_val_np, _ = transform_preprocess(
        X_val, fitted["scaler"], fitted["num_cols"], fitted["encoder"], fitted["cat_cols"]
    )

    print(f"[preprocess:{name}] output train: {X_train_np.shape}, val: {X_val_np.shape}")
    print(f"[preprocess:{name}] total features: {len(feat_names)}")
    print(f"[preprocess:{name}] sample features: {feat_names[:15]}")
    print(f"[preprocess:{name}] NaNs train: {np.isnan(X_train_np).sum()}  val: {np.isnan(X_val_np).sum()}")

    return {
        "X_train": X_train_np,
        "y_train": np.asarray(y_train),
        "X_val": X_val_np,
        "y_val": np.asarray(y_val),
        "feature_names": feat_names,
        "preprocess": fitted,  # contains scaler/encoder/col lists
    }


def preprocess_given_train_val(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series],
    X_val: pd.DataFrame,
    y_val: Optional[pd.Series],
    cat_cols: Optional[List[str]] = None,
    name: str = "dataset",
) -> Dict[str, Any]:
    """
    If you already have separate train/val (or train/test), use this.
    Fits on train only, transforms both.
    """
    print("\n" + "-" * 80)
    print(f"[preprocess:{name}] given train: {X_train.shape}, val: {X_val.shape}")

    fitted = fit_preprocess(X_train, cat_cols=cat_cols)

    X_train_np, feat_names = transform_preprocess(
        X_train, fitted["scaler"], fitted["num_cols"], fitted["encoder"], fitted["cat_cols"]
    )
    X_val_np, _ = transform_preprocess(
        X_val, fitted["scaler"], fitted["num_cols"], fitted["encoder"], fitted["cat_cols"]
    )

    print(f"[preprocess:{name}] output train: {X_train_np.shape}, val: {X_val_np.shape}")
    print(f"[preprocess:{name}] total features: {len(feat_names)}")
    print(f"[preprocess:{name}] sample features: {feat_names[:15]}")
    print(f"[preprocess:{name}] NaNs train: {np.isnan(X_train_np).sum()}  val: {np.isnan(X_val_np).sum()}")

    out = {
        "X_train": X_train_np,
        "X_val": X_val_np,
        "feature_names": feat_names,
        "preprocess": fitted,
    }
    if y_train is not None:
        out["y_train"] = np.asarray(y_train)
    if y_val is not None:
        out["y_val"] = np.asarray(y_val)
    return out
