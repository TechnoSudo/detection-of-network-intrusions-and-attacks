from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, scaler


def load_and_prepare_kdd(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    col_names = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
    ]

    df = pd.read_csv(path, header=None, names=col_names)

    df["label"] = df["label"].str.strip().str.replace(".", "", regex=False)

    y_attack_type = df["label"].copy()
    y = (df["label"] != "normal").astype(int)

    # One-hot encode symbolic features
    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["label"]]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)

    X_num = df[num_cols].astype(float).values

    X_all = np.hstack([X_num, X_cat])
    feature_names = num_cols + list(cat_feature_names)

    X = pd.DataFrame(X_all, columns=feature_names)

    return X, y, y_attack_type


def load_and_prepare_netflow(path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)

    y_attack_type = df["ANOMALY"].astype(str).str.strip()

    # Normalize all "no attack" cases
    y_attack_type = y_attack_type.replace(
        "<null>",
        "None"
    )

    # Binary label: 0 = normal, 1 = attack
    y = (y_attack_type != "None").astype(int)

    # Drop identifiers
    drop_cols = [
        "FLOW_ID",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "ANALYSIS_TIMESTAMP",
        "ID",
        "ANOMALY",
    ]

    drop_cols = [c for c in drop_cols if c in df.columns]
    df_features = df.drop(columns=drop_cols)
    df_features = standardize(df_features)

    # Encode PROTOCOL_MAP if present
    if "PROTOCOL_MAP" in df_features.columns:
        proto_map = df_features["PROTOCOL_MAP"].astype(str)
        proto_dummies = pd.get_dummies(proto_map, prefix="proto")
        df_features = pd.concat(
            [df_features.drop(columns=["PROTOCOL_MAP"]), proto_dummies],
            axis=1,
        )

    X = df_features.astype(float)

    return X, y, y_attack_type


def load_and_prepare_cores_iot(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    y = df.iloc[:, -1].astype(int)
    X = df.iloc[:, :-1].astype(float)

    return X, y


def prep_column_ip_to_int(ip):
    try:
        parts = ip.split(".")
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except:
        return 0


def standardize(df):
    df_numeric = df.select_dtypes(include=["number"])
    df_non_numeric = df.select_dtypes(exclude=["number"])

    df_numeric = (df_numeric - df_numeric.mean()) / df_numeric.std()

    return pd.concat([df_numeric, df_non_numeric], axis=1)[df.columns]