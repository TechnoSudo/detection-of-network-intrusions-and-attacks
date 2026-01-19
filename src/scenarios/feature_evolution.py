from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureEvolutionConfig:
    numeric_cols: Optional[List[str]] = None
    target_cols: Optional[List[str]] = None
    topk: int = 8

    mode: str = "shift_scale"   # "shift_scale" | "noise"
    max_shift_std: float = 1.5
    max_scale: float = 0.35
    noise_std: float = 0.15

    only_unknown: bool = True

    clip_sigma: float = 6.0
    seed: int = 42
    consistent_direction: bool = True


@dataclass
class DriftScheduleConfig:
    drift_type: str = "incremental"  # "sudden" | "incremental" | "gradual"

    change_point: int = 50
    start_t: int = 0
    end_t: int = 200

    min_intensity: float = 0.0
    max_intensity: float = 1.0

    min_mix: float = 0.0
    max_mix: float = 1.0


def detect_numeric_cols(df: pd.DataFrame, exclude: Optional[Sequence[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def pick_target_cols(df: pd.DataFrame, numeric_cols: List[str], topk: int) -> List[str]:
    variances: List[Tuple[float, str]] = []
    for c in numeric_cols:
        v = float(pd.to_numeric(df[c], errors="coerce").var())
        if np.isfinite(v):
            variances.append((v, c))
    variances.sort(reverse=True)
    return [c for _, c in variances[: max(1, min(topk, len(variances)) )]]


def fit_feature_stats(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        stats[c] = (mu, max(sd, 1e-6))
    return stats


def _linear_ramp(t: int, t0: int, t1: int, y0: float, y1: float) -> float:
    if t <= t0:
        return y0
    if t >= t1:
        return y1
    return y0 + (t - t0) / (t1 - t0) * (y1 - y0)


def drift_intensity_and_mix(t: int, sched: DriftScheduleConfig) -> Tuple[float, float]:
    dt = sched.drift_type.lower()

    if dt == "sudden":
        intensity = sched.max_intensity if t >= sched.change_point else sched.min_intensity
        return float(intensity), 1.0

    if dt in ("incremental", "gradual"):
        intensity = _linear_ramp(
            t, sched.start_t, sched.end_t,
            sched.min_intensity, sched.max_intensity
        )
        mix = 1.0 if dt == "incremental" else _linear_ramp(
            t, sched.start_t, sched.end_t,
            sched.min_mix, sched.max_mix
        )
        return float(intensity), float(mix)

    raise ValueError(f"Unknown drift_type: {sched.drift_type}")


def _column_signs(cols: List[str], seed: int) -> Dict[str, float]:
    signs: Dict[str, float] = {}
    for c in cols:
        s = seed ^ (abs(hash(c)) % (2**31 - 1))
        rng = np.random.default_rng(s)
        signs[c] = float(rng.choice([-1.0, 1.0]))
    return signs


def apply_feature_evolution(
    Xb_df: pd.DataFrame,
    y_attack_b: pd.Series,
    unknown_types: List[str],
    t: int,
    cfg: FeatureEvolutionConfig,
    stats: Dict[str, Tuple[float, float]],
    sched: DriftScheduleConfig,
) -> pd.DataFrame:

    intensity, mix_prob = drift_intensity_and_mix(t, sched)
    if intensity <= 0.0:
        return Xb_df

    X = Xb_df.copy()

    base_mask = (
        y_attack_b.isin(unknown_types).to_numpy()
        if cfg.only_unknown
        else np.ones(len(X), dtype=bool)
    )

    if sched.drift_type == "gradual":
        rng_mix = np.random.default_rng(cfg.seed + 7777 + t)
        mask = base_mask & (rng_mix.random(len(X)) < mix_prob)
    else:
        mask = base_mask

    if not mask.any():
        return X

    num_cols = cfg.numeric_cols or detect_numeric_cols(X)
    target_cols = cfg.target_cols or pick_target_cols(X, num_cols, cfg.topk)

    if not target_cols:
        return X

    if cfg.consistent_direction:
        shift_signs = _column_signs(target_cols, cfg.seed)
        scale_signs = _column_signs([c + "_scale" for c in target_cols], cfg.seed + 123)
    else:
        shift_signs = {}
        scale_signs = {}

    rng = np.random.default_rng(cfg.seed + 1000 + t)

    for c in target_cols:
        if c not in stats:
            continue

        mu, sd = stats[c]
        x = pd.to_numeric(X.loc[mask, c], errors="coerce").astype(float).to_numpy()

        if cfg.mode == "shift_scale":
            s1 = shift_signs.get(c, rng.choice([-1.0, 1.0]))
            s2 = scale_signs.get(c + "_scale", rng.choice([-1.0, 1.0]))
            shift = s1 * cfg.max_shift_std * sd * intensity
            scale = 1.0 + s2 * cfg.max_scale * intensity
            x_new = (x * scale) + shift

        elif cfg.mode == "noise":
            noise = rng.normal(0.0, cfg.noise_std * sd * intensity, size=x.shape)
            x_new = x + noise

        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

        lo = mu - cfg.clip_sigma * sd
        hi = mu + cfg.clip_sigma * sd
        X.loc[mask, c] = np.clip(x_new, lo, hi)

    return X