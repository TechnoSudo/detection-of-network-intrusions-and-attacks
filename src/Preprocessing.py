import numpy as np
import pandas as pd


def fill_missing(df, strategy="median"):
    df_numeric = df.select_dtypes(include=["number"])
    df_non_numeric = df.select_dtypes(exclude=["number"])

    if strategy == "mean":
        df_numeric = df_numeric.fillna(df_numeric.mean())
    elif strategy == "median":
        df_numeric = df_numeric.fillna(df_numeric.median())
    elif strategy == "constant":
        df_numeric = df_numeric.fillna(0)
    else:
        raise ValueError("Unknown strategy.")

    return pd.concat([df_numeric, df_non_numeric], axis=1)[df.columns]

def ip_to_int(ip):
    try:
        parts = ip.split(".")
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except:
        return 0

def encode_protocol(df):
    df["PROTOCOL_MAP"] = df["PROTOCOL_MAP"].astype("category").cat.codes
    return df

def split_train_test(df, test_columns):
    """
    df: pandas DataFrame
    test_columns: can be:
        - list of column names
        - list of column indices
        - mixed list of indices + names
        - single name or single index
    """
    # Convert single element
    if not isinstance(test_columns, (list, tuple, set)):
        test_columns = [test_columns]

    # Separate names and indices
    col_names = [c for c in test_columns if isinstance(c, str)]
    col_indices = [c for c in test_columns if isinstance(c, int)]

    # Convert indices → names
    col_names_from_indices = [df.columns[i] for i in col_indices]

    # Final combined list of names
    final_names = col_names + col_names_from_indices

    # Build test + train
    test_df = df[final_names].copy()
    train_df = df.drop(columns=final_names).copy()

    return train_df, test_df


def encode_attack_labels(df, attack_column="ANOMALY"):
    df[attack_column] = df[attack_column].astype("category").cat.codes
    return df


def convert_ip_columns(df):
    ip_cols = [col for col in df.columns if "IPV4" in col or "IP" in col]

    for col in ip_cols:
        df[col] = df[col].apply(ip_to_int)

    return df

def describe(df):
    return df.describe(include='all')

def drop_irrelevant_identifiers(df):
    cols_to_drop = [col for col in df.columns if col in ["ID", "_ID"]]
    return df.drop(columns=cols_to_drop, errors="ignore")


def normalize(df):
    df_numeric = df.select_dtypes(include=["number"])
    df_non_numeric = df.select_dtypes(exclude=["number"])

    df_numeric = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

    return pd.concat([df_numeric, df_non_numeric], axis=1)[df.columns]


def standardize(df):
    df_numeric = df.select_dtypes(include=["number"])
    df_non_numeric = df.select_dtypes(exclude=["number"])

    df_numeric = (df_numeric - df_numeric.mean()) / df_numeric.std()

    return pd.concat([df_numeric, df_non_numeric], axis=1)[df.columns]

def create_equal_batches(df, batch_size):
    num_batches = len(df) // batch_size
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append(df.iloc[start:end])
    return batches


def create_progressive_attack_batches(df, label_column, batch_size):
    attack_df = df[df[label_column] == 1]
    normal_df = df[df[label_column] == 0]

    total_records = len(df)
    num_batches = total_records // batch_size

    batches = []
    attack_ratio_step = 1 / num_batches

    for i in range(num_batches):
        ratio = attack_ratio_step * i
        normal_needed = int(batch_size * (1 - ratio))
        attack_needed = batch_size - normal_needed

        normal_sample = normal_df.sample(n=normal_needed, replace=True)
        attack_sample = attack_df.sample(n=attack_needed, replace=True)

        batch = pd.concat([normal_sample, attack_sample]).sample(frac=1)
        batches.append(batch)

    return batches


def create_equal_batchesa(df, batch_size):
    num_batches = len(df) // batch_size
    batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append(df.iloc[start:end])

    if len(df) % batch_size != 0:
        batches.append(df.iloc[num_batches * batch_size:])

    return batches


def create_progressive_attack_batchesa(df, batch_size, attack_column="ANOMALY"):
    normal_df = df[df[attack_column] == 0]
    attack_df = df[df[attack_column] == 1]

    normal_size = len(normal_df)
    attack_size = len(attack_df)

    progressive_batches = []
    pointer_normal = 0
    pointer_attack = 0

    grow_ratio = attack_size / (len(df) / batch_size)

    total_batches = (len(df) // batch_size) + 1

    for i in range(total_batches):
        allowed_attacks = int(i * grow_ratio)

        start_attack = pointer_attack
        end_attack = min(pointer_attack + allowed_attacks, attack_size)
        attack_part = attack_df.iloc[start_attack:end_attack]
        pointer_attack = end_attack

        needed_normals = batch_size - len(attack_part)
        start_normal = pointer_normal
        end_normal = min(pointer_normal + needed_normals, normal_size)
        normal_part = normal_df.iloc[start_normal:end_normal]
        pointer_normal = end_normal

        batch = pd.concat([normal_part, attack_part], ignore_index=True)
        progressive_batches.append(batch)

        if pointer_normal >= normal_size and pointer_attack >= attack_size:
            break

    return progressive_batches