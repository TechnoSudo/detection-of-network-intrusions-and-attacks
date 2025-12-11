from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import (
    ModelBundle,
    train_offline_models,
    predict_all_models,
    make_sgd_static_and_adaptive,
    ensemble_predict_labels, retrain_offline_models,
)
from metrics import compute_metrics
from preprocessing import preprocess_features


RANDOM_STATE = 42

def run_scenario_1(
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Dict[str, float]]:
    print("\n Scenario 1: All attacks known ")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train_std, X_test_std, _ = preprocess_features(X_train, X_test)

    models: ModelBundle = train_offline_models(
        X_train_std,
        y_train.values,
    )

    preds = predict_all_models(models, X_test_std)

    metrics_per_model: Dict[str, Dict[str, float]] = {}

    for name, y_pred in preds.items():
        metrics_per_model[name] = compute_metrics(y_test.values, y_pred)
        print(f"Scenario 1 metrics ({name}): {metrics_per_model[name]}")

    y_ens = ensemble_predict_labels(models, X_test_std, threshold=0.5)
    metrics_per_model["ensemble"] = compute_metrics(y_test.values, y_ens)
    print(f"Scenario 1 metrics (ensemble): {metrics_per_model['ensemble']}")

    return metrics_per_model



def run_scenario_2_kdd(
    X: pd.DataFrame,
    y: pd.Series,
    y_attack_type: pd.Series,
    known_attacks: List[str],
    unseen_attacks: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:

    print("\n Scenario 2: Some attacks appear only during testing (KDD'99)")

    is_normal = (y_attack_type == "normal")
    is_known_attack = y_attack_type.isin(known_attacks)
    train_mask = is_normal | is_known_attack

    test_mask = np.ones(len(y_attack_type), dtype=bool)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    y_type_test = y_attack_type.loc[test_mask]

    X_train_std, X_test_std, _ = preprocess_features(X_train, X_test)

    models: ModelBundle = train_offline_models(
        X_train_std,
        y_train.values,
    )
    preds = predict_all_models(models, X_test_std)

    unseen_mask_test = y_type_test.isin(unseen_attacks).values

    metrics_per_model: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name, y_pred_all in preds.items():
        metrics_all = compute_metrics(y_test.values, y_pred_all)

        if unseen_mask_test.sum() > 0:
            y_true_u = y_test.values[unseen_mask_test]
            y_pred_u = y_pred_all[unseen_mask_test]
            metrics_unseen = compute_metrics(y_true_u, y_pred_u)
        else:
            metrics_unseen = {
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "bac": np.nan,
            }

        metrics_per_model[name] = {
            "all": metrics_all,
            "unseen": metrics_unseen,
        }

        print(f"\n{name} – ALL:    {metrics_all}")
        print(f"{name} – UNSEEN: {metrics_unseen}")

    # X_train_partial, X_test_partial, _ = preprocess_features(X_train, X_test)
    # models = retrain_offline_models(
    #     X_train_partial[unseen_mask_test],
    #     y_train.values[unseen_mask_test],
    #     models=models,
    # )
    # preds_partial = predict_all_models(models, X_test_partial)
    #
    # for name, y_pred_all in preds_partial.items():
    #     metrics_all = compute_metrics(y_test.values[unseen_mask_test], y_pred_all[unseen_mask_test])
    #
    #     if unseen_mask_test.sum() > 0:
    #         y_true_u = y_test.values[unseen_mask_test]
    #         y_pred_u = y_pred_all[unseen_mask_test]
    #         metrics_unseen = compute_metrics(y_true_u, y_pred_u)
    #     else:
    #         metrics_unseen = {
    #             "f1": np.nan,
    #             "precision": np.nan,
    #             "recall": np.nan,
    #             "bac": np.nan,
    #         }
    #
    #     metrics_per_model[name + "_partial"] = {
    #         "all": metrics_all,
    #         "unseen": metrics_unseen,
    #     }
    #
    #     print(f"\n{name} – ALL:    {metrics_all}")
    #     print(f"{name} – UNSEEN: {metrics_unseen}")


    return metrics_per_model



def run_scenario_2_netflow(
    X: pd.DataFrame,
    y: pd.Series,
    y_attack_type: pd.Series,
    known_attacks: List[str],
    unseen_attacks: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:

    print("\n Scenario 2: Some attacks appear only during testing (NetFlow v9)")

    is_normal = (y_attack_type == "None")
    is_known_attack = y_attack_type.isin(known_attacks)
    train_mask = is_normal | is_known_attack

    test_mask = np.ones(len(y_attack_type), dtype=bool)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    y_type_test = y_attack_type.loc[test_mask]

    X_train_std, X_test_std, _ = preprocess_features(X_train, X_test)

    models: ModelBundle = train_offline_models(
        X_train_std,
        y_train.values,
    )

    preds = predict_all_models(models, X_test_std)

    unseen_mask_test = y_type_test.isin(unseen_attacks).values

    metrics_per_model: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name, y_pred_all in preds.items():
        metrics_all = compute_metrics(y_test.values, y_pred_all)

        if unseen_mask_test.sum() > 0:
            y_true_u = y_test.values[unseen_mask_test]
            y_pred_u = y_pred_all[unseen_mask_test]
            metrics_unseen = compute_metrics(y_true_u, y_pred_u)
        else:
            metrics_unseen = {
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "bac": np.nan,
            }

        metrics_per_model[name] = {
            "all": metrics_all,
            "unseen": metrics_unseen,
        }

        print(f"\n{name} – ALL:    {metrics_all}")
        print(f"{name} – UNSEEN: {metrics_unseen}")

    return metrics_per_model



def run_scenario_3_streaming(
    batches: List[Tuple[pd.DataFrame, pd.Series]],
    n_pre_drift: int,
    random_state: int = 42,
) -> Dict[str, List[Dict[str, float]]]:

    print("\nScenario 3: Evolving attacks")

    if n_pre_drift > len(batches):
        raise ValueError("n_pre_drift cannot exceed number of batches.")

    X_list, y_list = [], []

    for i in range(n_pre_drift):
        X_b, y_b = batches[i]
        X_list.append(X_b)
        y_list.append(y_b)

    X_offline = pd.concat(X_list, axis=0)
    y_offline = pd.concat(y_list, axis=0).values

    scaler = StandardScaler()
    X_offline_std = scaler.fit_transform(X_offline)

    models: ModelBundle = train_offline_models(
        X_offline_std,
        y_offline,
        random_state=random_state,
    )

    sgd_static, sgd_adaptive = make_sgd_static_and_adaptive(models)

    batch_metrics: Dict[str, List[Dict[str, float]]] = {
        "rf": [],
        "xgb": [],
        "sgd_static": [],
        "sgd_adaptive": [],
        "ensemble": [],
    }

    for X_b, y_b in batches:

        X_std = scaler.transform(X_b)
        y_true = y_b.values

        y_rf = models["rf"].predict(X_std)
        batch_metrics["rf"].append(compute_metrics(y_true, y_rf))

        y_xgb = models["xgb"].predict(X_std)
        batch_metrics["xgb"].append(compute_metrics(y_true, y_xgb))

        y_sgd_s = sgd_static.predict(X_std)
        batch_metrics["sgd_static"].append(compute_metrics(y_true, y_sgd_s))

        y_sgd_a = sgd_adaptive.predict(X_std)
        batch_metrics["sgd_adaptive"].append(compute_metrics(y_true, y_sgd_a))

        p_rf = models["rf"].predict_proba(X_std)[:, 1]
        p_xgb = models["xgb"].predict_proba(X_std)[:, 1]
        p_sgd_a = sgd_adaptive.predict_proba(X_std)[:, 1]

        p_ens = (p_rf + p_xgb + p_sgd_a) / 3
        y_ens = (p_ens >= 0.5).astype(int)

        batch_metrics["ensemble"].append(compute_metrics(y_true, y_ens))

        sgd_adaptive.partial_fit(X_std, y_true)

    print("\nScenario 3:")
    for k in batch_metrics:
        print(f"  {k}: {batch_metrics[k][-1]}")

    return batch_metrics
