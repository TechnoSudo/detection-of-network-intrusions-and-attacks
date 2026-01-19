from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.data_loader import load_kdd, load_nsl_kdd, load_netflow
from src.data.preprocessing import fit_preprocess, transform_preprocess
from src.evaluation.metrics import compute_metrics

from src.models.supervised.adaptive_sgd import AdaptiveSGDModel
from src.models.supervised.random_forest import RFModel
from src.models.supervised.xgboost import XGBModel
from src.models.supervised.ensemble import WeightedSoftVotingEnsemble, EnsembleConfig

from src.models.drift_detection.adwin import ADWINDriftDetector
from src.models.drift_detection.page_hinkley import PageHinkleyDriftDetector

from src.scenarios.feature_evolution import FeatureEvolutionConfig, DriftScheduleConfig, detect_numeric_cols, pick_target_cols, fit_feature_stats, apply_feature_evolution

KDD_KNOWN: List[str] = [ "neptune", "smurf", "back", "ipsweep", "portsweep", "satan", "guess_passwd", "warezmaster", "buffer_overflow"]
KDD_UNKNOWN: List[str] = ["teardrop", "nmap", "warezclient", "rootkit"]

NSL_KNOWN: List[str] = KDD_KNOWN
NSL_UNKNOWN: List[str] = KDD_UNKNOWN

NETFLOW_KNOWN: List[str] = ["Port Scanning", "Denial of Service"]
NETFLOW_UNKNOWN: List[str] = ["Malware"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cat_cols(dataset: str) -> List[str]:
    return ["protocol_type", "service", "flag"] if dataset in {"kdd", "nsl"} else ["PROTOCOL_MAP"]


def _normalize_attack_labels(dataset: str, y_attack: pd.Series) -> pd.Series:
    ya = y_attack.astype(str).str.strip()
    if dataset == "netflow":
        ya = ya.replace({"None": "normal", "<null>": "normal", "nan": "normal", "NaN": "normal"})
    return ya.str.replace(".", "", regex=False)


def _select_known_unknown(dataset: str) -> Tuple[List[str], List[str]]:
    if dataset == "kdd":
        return KDD_KNOWN, KDD_UNKNOWN
    if dataset == "nsl":
        return NSL_KNOWN, NSL_UNKNOWN
    if dataset == "netflow":
        return NETFLOW_KNOWN, NETFLOW_UNKNOWN
    raise ValueError("Unsupported dataset")


def _plot_stream_multi(
    df_metrics: pd.DataFrame,
    df_events: pd.DataFrame,
    out_png: str,
    title: str,
    n_pre: int,
    n_drift: int,
    n_post: int,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    pre_end = n_pre - 1
    drift_end = n_pre + n_drift - 1
    post_end = n_pre + n_drift + n_post - 1

    ax.axvspan(-0.5, pre_end + 0.5, alpha=0.06)
    ax.axvspan(pre_end + 0.5, drift_end + 0.5, alpha=0.10)
    ax.axvspan(drift_end + 0.5, post_end + 0.5, alpha=0.06)

    y_phase = 0.505
    ax.text(pre_end / 2, y_phase, "pre-drift", ha="center", va="bottom", fontsize=10)
    ax.text((pre_end + drift_end) / 2, y_phase, "drift", ha="center", va="bottom", fontsize=10)
    ax.text((drift_end + post_end) / 2, y_phase, "post-drift", ha="center", va="bottom", fontsize=10)

    for model in df_metrics["model"].unique():
        sub = df_metrics[df_metrics["model"] == model].sort_values("batch_idx")
        ax.plot(sub["batch_idx"], sub["f1_ema"], linewidth=2.2, label=model)

    if not df_events.empty:
        for det, marker, color in [
            ("ADWIN", "x", "black"),
            ("PageHinkley", "^", "dimgray"),
        ]:
            sub = df_events[df_events["detector"] == det]
            if len(sub) > 0:
                ax.scatter(
                    sub["batch_idx"],
                    np.full(len(sub), 0.98),
                    marker=marker,
                    color=color,
                    s=70,
                    linewidths=2,
                    label=det,
                    zorder=5,
                )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Batch index")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_initial_batch_indices(
    y: pd.Series,
    y_attack: pd.Series,
    known_types: List[str],
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """
    Build X0 so it contains BOTH classes (0 and 1) to avoid XGBoost logistic base_score crash.
    We sample ~50% normal and ~50% known attacks (whatever is available).
    """
    rng = np.random.default_rng(seed + 12345)

    y_int = np.asarray(y).astype(int)
    normal_idx = np.where(y_int == 0)[0]
    attack_idx = np.where(y_int == 1)[0]

    # Prefer known attacks when available (scenario definition)
    known_mask = y_attack.isin(known_types).to_numpy()
    known_attack_idx = np.where((y_int == 1) & known_mask)[0]
    if len(known_attack_idx) > 0:
        attack_idx = known_attack_idx

    rng.shuffle(normal_idx)
    rng.shuffle(attack_idx)

    n_norm = batch_size // 2
    n_att = batch_size - n_norm

    take_norm = normal_idx[: min(n_norm, len(normal_idx))]
    take_att = attack_idx[: min(n_att, len(attack_idx))]

    idx = np.concatenate([take_norm, take_att])

    # If one side is short, fill from the other side (still keeps at least 1 of each if possible)
    if len(idx) < batch_size:
        need = batch_size - len(idx)
        pool = np.setdiff1d(np.arange(len(y_int)), idx, assume_unique=False)
        extra = rng.choice(pool, size=need, replace=False) if len(pool) >= need else rng.choice(pool, size=need, replace=True)
        idx = np.concatenate([idx, extra])

    rng.shuffle(idx)

    # Hard safety check
    uniq = np.unique(y_int[idx])
    if len(uniq) < 2:
        raise ValueError(
            f"[scenario3] Initial batch has only one class {uniq.tolist()}. "
            "Increase batch_size or change sampling (or ensure y has both classes)."
        )

    return idx

def _make_batch_indices_balanced(
    y: pd.Series,
    y_attack: pd.Series,
    known_types: List[str],
    unknown_types: List[str],
    b: int,
    n_pre: int,
    n_drift: int,
    n_post: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 10000 + b)

    y_int = np.asarray(y).astype(int)
    normal_idx = np.where(y_int == 0)[0]

    known_attack_idx = np.where((y_int == 1) & y_attack.isin(known_types).to_numpy())[0]
    unknown_attack_idx = np.where((y_int == 1) & y_attack.isin(unknown_types).to_numpy())[0]

    # phase controls which attack pool we sample from
    if b < n_pre:
        attack_pool = known_attack_idx
    elif b < n_pre + n_drift:
        # mix known -> unknown gradually across drift window
        t = (b - n_pre + 1) / max(1, n_drift)
        use_unknown = rng.random() < t
        attack_pool = unknown_attack_idx if (use_unknown and len(unknown_attack_idx) > 0) else known_attack_idx
    else:
        attack_pool = unknown_attack_idx if len(unknown_attack_idx) > 0 else known_attack_idx

    # sample half normals + half attacks
    n_norm = batch_size // 2
    n_att = batch_size - n_norm

    take_norm = rng.choice(normal_idx, size=n_norm, replace=(len(normal_idx) < n_norm))
    take_att = rng.choice(attack_pool, size=n_att, replace=(len(attack_pool) < n_att))

    idx = np.concatenate([take_norm, take_att])
    rng.shuffle(idx)
    return idx



def run_scenario3(
    dataset: str,
    outdir: str,
    seed: int = 42,
    sample_frac: float = 1.0,
    n_pre: int = 10,
    n_drift: int = 10,
    n_post: int = 10,
    batch_size: int = 2000,
    drift_type: str = "incremental",
    change_point: Optional[int] = None,
    start_t: Optional[int] = None,
    end_t: Optional[int] = None,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
    min_mix: float = 0.0,
    max_mix: float = 1.0,
) -> Dict[str, Any]:
    _ensure_dir(outdir)

    if dataset == "kdd":
        X, y, y_attack = load_kdd("data/dataset-1/kddcup.data")
    elif dataset == "nsl":
        X, y, y_attack = load_nsl_kdd("data/dataset-4/NSL-KDD-train.txt")
    elif dataset == "netflow":
        X, y, y_attack = load_netflow("data/dataset-2/train_net.csv")
    else:
        raise ValueError("Unsupported dataset")

    if sample_frac < 1.0:
        n = len(X)
        n_sample = int(n * sample_frac)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=n_sample, replace=False))
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        y_attack = y_attack.iloc[idx].reset_index(drop=True)

    y_attack = _normalize_attack_labels(dataset, y_attack)
    known_types, unknown_types = _select_known_unknown(dataset)

    total_batches = n_pre + n_drift + n_post
    total_needed = total_batches * batch_size
    if len(X) < total_needed:
        raise ValueError(
            f"[scenario3] Not enough rows for streaming: need {total_needed}, have {len(X)}. "
            f"Reduce batch_size or n_pre/n_drift/n_post or increase sample_frac."
        )

    dt = drift_type.lower().strip()
    if dt not in {"sudden", "incremental", "gradual"}:
        raise ValueError("drift_type must be one of: sudden, incremental, gradual")

    if dt == "sudden":
        if change_point is None:
            change_point = n_pre
        sched = DriftScheduleConfig(
            drift_type="sudden",
            change_point=int(change_point),
            start_t=0,
            end_t=total_batches - 1,
            min_intensity=float(min_intensity),
            max_intensity=float(max_intensity),
            min_mix=float(min_mix),
            max_mix=float(max_mix),
        )
    else:
        if start_t is None:
            start_t = n_pre
        if end_t is None:
            end_t = n_pre + n_drift - 1
        sched = DriftScheduleConfig(
            drift_type=dt,
            change_point=0,
            start_t=int(start_t),
            end_t=int(end_t),
            min_intensity=float(min_intensity),
            max_intensity=float(max_intensity),
            min_mix=float(min_mix),
            max_mix=float(max_mix),
        )

    init_idx = _make_initial_batch_indices(
        y=y,
        y_attack=y_attack,
        known_types=known_types,
        batch_size=batch_size,
        seed=seed,
    )
    X0 = X.iloc[init_idx].reset_index(drop=True)
    y0 = y.iloc[init_idx].reset_index(drop=True)

    fitted = fit_preprocess(X0, _cat_cols(dataset))

    def transform(df: pd.DataFrame) -> np.ndarray:
        Xn, _ = transform_preprocess(df, **fitted)
        return Xn

    evo_cfg = FeatureEvolutionConfig(seed=seed)
    evo_cfg.numeric_cols = detect_numeric_cols(X0)
    evo_cfg.target_cols = pick_target_cols(X0, evo_cfg.numeric_cols, evo_cfg.topk)
    evo_stats = fit_feature_stats(X0, evo_cfg.target_cols)

    X0_np = transform(X0)
    y0_np = np.asarray(y0).astype(int)

    rf = RFModel().fit(X0_np, y0_np)
    xgb = XGBModel().fit(X0_np, y0_np)

    sgd_static = AdaptiveSGDModel(seed=seed)
    sgd_adaptive = AdaptiveSGDModel(seed=seed)
    sgd_static.partial_fit(X0_np, y0_np)
    sgd_adaptive.partial_fit(X0_np, y0_np)

    ensemble = WeightedSoftVotingEnsemble(
        models={"rf": rf, "xgb": xgb, "sgd_static": sgd_static, "sgd_adaptive": sgd_adaptive},
        config=EnsembleConfig(),
    ).fit(X0_np, y0_np)

    adwin = ADWINDriftDetector()
    ph = PageHinkleyDriftDetector()

    metrics: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    for b in range(total_batches):
        idx_batch = _make_batch_indices_balanced(
            y=y,
            y_attack=y_attack,
            known_types=known_types,
            unknown_types=unknown_types,
            b=b,
            n_pre=n_pre,
            n_drift=n_drift,
            n_post=n_post,
            batch_size=batch_size,
            seed=seed,
        )
        Xb = X.iloc[idx_batch].reset_index(drop=True)
        yb = y.iloc[idx_batch].reset_index(drop=True)
        yab = y_attack.iloc[idx_batch].reset_index(drop=True)


        Xb = apply_feature_evolution(
            Xb_df=Xb,
            y_attack_b=yab,
            unknown_types=unknown_types,
            t=b,
            cfg=evo_cfg,
            stats=evo_stats,
            sched=sched,
        )

        Xb_np = transform(Xb)
        yb_np = np.asarray(yb).astype(int)

        preds = {
            "rf": np.asarray(rf.predict(Xb_np)).astype(int),
            "xgb": np.asarray(xgb.predict(Xb_np)).astype(int),
            "sgd_static": np.asarray(sgd_static.predict(Xb_np)).astype(int),
            "sgd_adaptive": np.asarray(sgd_adaptive.predict(Xb_np)).astype(int),
            "ensemble": np.asarray(ensemble.predict(Xb_np)).astype(int),
        }

        for name, yp in preds.items():
            r = compute_metrics(yb_np, yp)
            metrics.append({"batch_idx": b, "model": name, **r})

        err = float(np.mean((preds["ensemble"] != yb_np).astype(float)))
        if adwin.update(err):
            events.append({"batch_idx": b, "detector": "ADWIN"})
        if ph.update(err):
            events.append({"batch_idx": b, "detector": "PageHinkley"})

        sgd_adaptive.partial_fit(Xb_np, yb_np)

    df_metrics = pd.DataFrame(metrics)
    df_metrics["f1_ema"] = df_metrics.groupby("model")["f1"].transform(
        lambda s: s.ewm(alpha=0.35, adjust=False).mean()
    )

    df_events = pd.DataFrame(events)

    out_png = os.path.join(outdir, "stream_plot.png")
    out_metrics = os.path.join(outdir, "batch_metrics.csv")
    out_events = os.path.join(outdir, "drift_events.csv")

    df_metrics.to_csv(out_metrics, index=False)
    df_events.to_csv(out_events, index=False)

    title = f"Scenario 3 — {dt.capitalize()} Concept Drift ({dataset.upper()})"
    _plot_stream_multi(
        df_metrics=df_metrics,
        df_events=df_events,
        out_png=out_png,
        title=title,
        n_pre=n_pre,
        n_drift=n_drift,
        n_post=n_post,
    )

    return {
        "plot_png": out_png,
        "metrics_csv": out_metrics,
        "events_csv": out_events,
    }