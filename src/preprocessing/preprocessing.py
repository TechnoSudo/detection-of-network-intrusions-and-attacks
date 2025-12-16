from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def preprocess_features(X_train, X_test):

    print("[preprocess] Before cleaning: ",
          X_train.isna().sum().sum(),
          X_test.isna().sum().sum())

    X_train = X_train.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print("[preprocess] After cleaning: ",
          X_train.isna().sum().sum(),
          X_test.isna().sum().sum())

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    print("[preprocess] After scaling: ",
          np.isnan(X_train_std).sum(),
          np.isnan(X_test_std).sum())

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

    # Strip trailing dots in labels like "normal."
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
    print(f"[NetFlow] Loading file: {path}")
    df = pd.read_csv(path)
    print(f"[NetFlow] Raw shape: {df.shape}")

    # 1) Attack TYPE from ALERT (string)
    if "ALERT" not in df.columns:
        raise ValueError("NetFlow file must contain an 'ALERT' column.")

    y_attack_type = df["ALERT"].astype(str).str.strip()

    # Normalize "no attack" labels
    y_attack_type = y_attack_type.replace(
        {
            "<null>": "None",
            "nan": "None",
            "NaN": "None",
        }
    )

    print("[NetFlow] Unique ALERT / attack types:", y_attack_type.unique())

    # 2) Binary label: 0 = normal, 1 = attack
    #   Anything that is NOT "None" is considered an attack
    y = (y_attack_type != "None").astype(int)

    # 3) Drop identifiers + label columns from features
    drop_cols = [
        "FLOW_ID",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "ANALYSIS_TIMESTAMP",
        "ID",
        "ANOMALY",   # numeric flag – we don't want to leak it
        "ALERT",     # attack type – label, not feature
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df_features = df.drop(columns=drop_cols)

    # 4) Split numeric vs categorical
    num_cols = df_features.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df_features.columns if c not in num_cols]

    X_num = df_features[num_cols]

    if cat_cols:
        X_cat = pd.get_dummies(df_features[cat_cols].astype(str))
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num.copy()

    print(f"[NetFlow] Feature matrix shape before cleaning: {X.shape}")
    print(f"[NetFlow] NaNs before cleaning: {X.isna().sum().sum()}")

    # 5) Clean NaNs / inf
    X = X.replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0.0)

    print(f"[NetFlow] NaNs after cleaning: {X.isna().sum().sum()}")

    X = X.astype(float)

    return X, y, y_attack_type



def load_and_prepare_cores_iot(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    y = df.iloc[:, -1].astype(int)
    X = df.iloc[:, :-1].astype(float)

    return X, y