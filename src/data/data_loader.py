import pandas as pd
import numpy as np


KDD_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]


def clean_df(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def load_kdd(path: str):
    print(f"[KDD] Loading: {path}")
    df = pd.read_csv(path, header=None, names=KDD_COLS)

    df["label"] = df["label"].astype(str).str.strip().str.replace(".", "", regex=False)

    y_attack = df["label"]
    y_binary = (df["label"] != "normal").astype(int)

    X = df.drop(columns=["label"])
    X = clean_df(X)

    print("[KDD] Shape:", X.shape)
    print("[KDD] Labels:", y_attack.unique())

    return X, y_binary, y_attack


def load_nsl_kdd(train_path: str):
    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
        "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
        "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"
    ]

    print(f"[NSL-KDD] Train: {train_path}")
    df = pd.read_csv(train_path, header=None, names=col_names)
    print(f"[NSL-KDD] Train shape: {df.shape}")

    # Attack type labels (string)
    y_attack_type = df["label"].astype(str).str.strip().str.replace(".", "", regex=False)

    # Binary label: 0 normal, 1 attack
    y = (y_attack_type != "normal").astype(int)

    # Features: drop label + difficulty
    X = df.drop(columns=["label", "difficulty"])

    return X, y, y_attack_type


def load_netflow(path: str):
    print(f"[NetFlow] Loading: {path}")
    df = pd.read_csv(path)

    y_attack = df["ALERT"].fillna("None").astype(str)
    y_binary = (y_attack != "None").astype(int)

    drop_cols = [
        "FLOW_ID",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "ANALYSIS_TIMESTAMP",
        "ID",
        "ANOMALY",
        "ALERT"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    X = clean_df(X)

    print("[NetFlow] Shape:", X.shape)
    print("[NetFlow] Attack labels:", y_attack.unique())

    return X, y_binary, y_attack


def load_cores_iot(path: str):
    print(f"[Cores-IoT] Loading: {path}")

    df = pd.read_csv(path, header=None)
    y = df.iloc[:, -1].astype(int)
    X = df.iloc[:, :-1].astype(float)

    print("[Cores-IoT] Shape:", X.shape)
    print("[Cores-IoT] Label distribution:", y.value_counts().to_dict())

    return X, y