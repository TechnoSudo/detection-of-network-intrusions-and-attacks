from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


@dataclass(frozen=True)
class StatConfig:
    results_root: str = os.path.join("results", "scenario3")
    metric_col: str = "f1"  # use raw f1 for statistical testing (recommended)
    alpha: float = 0.05
    zero_method: str = "wilcox"  # "wilcox" | "pratt" | "zsplit"
    alternative: str = "two-sided"  # "two-sided" | "greater" | "less"


def _list_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def _read_batch_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"batch_idx", "model"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns {required} in {csv_path}")
    return df


def _pivot_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found. Available: {df.columns.tolist()}")
    wide = df.pivot_table(index="batch_idx", columns="model", values=metric_col, aggfunc="mean")
    wide = wide.sort_index()
    return wide


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta in [-1, 1]. Positive => x tends to be larger than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    # O(n*m) but your n ~ 30 batches, totally fine.
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    denom = len(x) * len(y)
    return (gt - lt) / denom


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for k, idx in enumerate(order):
        mult = (m - k)
        val = p[idx] * mult
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def _wilcoxon_paired(
    a: np.ndarray,
    b: np.ndarray,
    cfg: StatConfig,
) -> Tuple[float, float, int]:
    """
    Returns: (p_value, median_diff, n_used)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = int(len(a))
    if n < 3:
        return float("nan"), float("nan"), n

    # if arrays are identical -> wilcoxon may complain; handle gracefully
    if np.allclose(a, b, atol=1e-12):
        return 1.0, float(np.median(a - b)), n

    stat = wilcoxon(
        a,
        b,
        zero_method=cfg.zero_method,
        alternative=cfg.alternative,
        correction=False,
        mode="auto",
    )
    p = float(stat.pvalue)
    med = float(np.median(a - b))
    return p, med, n


def run_stats(
    cfg: StatConfig = StatConfig(),
    comparisons: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    comparisons: list of (A, B) where test is A vs B (paired by batch_idx).
    If None, uses a sensible default set.
    """
    if comparisons is None:
        comparisons = [
            ("ensemble", "sgd_static"),
            ("sgd_adaptive", "sgd_static"),
            ("ensemble", "rf"),
            ("ensemble", "xgb"),
        ]

    rows: List[Dict[str, object]] = []
    datasets = _list_dirs(cfg.results_root)

    for dataset in datasets:
        ds_dir = os.path.join(cfg.results_root, dataset)
        drifts = _list_dirs(ds_dir)

        for drift in drifts:
            csv_path = os.path.join(ds_dir, drift, "batch_metrics.csv")
            if not os.path.exists(csv_path):
                continue

            df = _read_batch_metrics(csv_path)
            wide = _pivot_metric(df, cfg.metric_col)

            for a_name, b_name in comparisons:
                if a_name not in wide.columns or b_name not in wide.columns:
                    rows.append(
                        {
                            "dataset": dataset,
                            "drift": drift,
                            "metric": cfg.metric_col,
                            "A": a_name,
                            "B": b_name,
                            "n_batches_used": 0,
                            "p_value": float("nan"),
                            "p_holm": float("nan"),
                            "significant": False,
                            "median_diff(A-B)": float("nan"),
                            "mean_diff(A-B)": float("nan"),
                            "cliffs_delta": float("nan"),
                        }
                    )
                    continue

                a = wide[a_name].to_numpy()
                b = wide[b_name].to_numpy()

                p, med, n_used = _wilcoxon_paired(a, b, cfg)
                mean_diff = float(np.nanmean(a - b)) if n_used > 0 else float("nan")
                cd = cliffs_delta(a, b) if n_used > 0 else float("nan")

                rows.append(
                    {
                        "dataset": dataset,
                        "drift": drift,
                        "metric": cfg.metric_col,
                        "A": a_name,
                        "B": b_name,
                        "n_batches_used": n_used,
                        "p_value": p,
                        "p_holm": float("nan"),  # filled after
                        "significant": False,    # filled after
                        "median_diff(A-B)": med,
                        "mean_diff(A-B)": mean_diff,
                        "cliffs_delta": cd,
                    }
                )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError(f"No results found under: {cfg.results_root}")

    # Holm correction across ALL tests we actually computed (non-nan)
    mask = np.isfinite(out["p_value"].to_numpy())
    pvals = out.loc[mask, "p_value"].tolist()
    if len(pvals) > 0:
        adj = holm_bonferroni(pvals)
        out.loc[mask, "p_holm"] = adj
        out["significant"] = (out["p_holm"] <= cfg.alpha) & np.isfinite(out["p_holm"])
    else:
        out["p_holm"] = np.nan
        out["significant"] = False

    # Sort nicely
    out = out.sort_values(["dataset", "drift", "A", "B"]).reset_index(drop=True)
    return out


def save_outputs(df: pd.DataFrame, results_root: str) -> None:
    out_csv = os.path.join(results_root, "stat_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"[stats] saved -> {out_csv}")

    # LaTeX table (compact)
    df_latex = df.copy()
    df_latex["p_value"] = df_latex["p_value"].map(lambda x: f"{x:.4g}" if np.isfinite(x) else "NA")
    df_latex["p_holm"] = df_latex["p_holm"].map(lambda x: f"{x:.4g}" if np.isfinite(x) else "NA")
    df_latex["median_diff(A-B)"] = df_latex["median_diff(A-B)"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "NA")
    df_latex["cliffs_delta"] = df_latex["cliffs_delta"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")

    cols = ["dataset", "drift", "A", "B", "n_batches_used", "p_value", "p_holm", "significant", "median_diff(A-B)", "cliffs_delta"]
    tex = df_latex[cols].to_latex(index=False, escape=True)

    out_tex = os.path.join(results_root, "stat_summary.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[stats] saved -> {out_tex}")


def main() -> None:
    cfg = StatConfig(
        results_root=os.path.join("results", "scenario2"),
        metric_col="f1",
        alpha=0.05,
        zero_method="wilcox",
        alternative="two-sided",
    )

    df = run_stats(cfg=cfg)
    save_outputs(df, cfg.results_root)

    # quick console summary
    sig = df[df["significant"] == True]
    print("\n[stats] Significant results (Holm-corrected):")
    if len(sig) == 0:
        print("  none")
    else:
        for _, r in sig.iterrows():
            print(
                f"  {r['dataset']}/{r['drift']}: {r['A']} > {r['B']}? "
                f"median_diff={r['median_diff(A-B)']:.4f} p_holm={r['p_holm']:.4g}"
            )


if __name__ == "__main__":
    main()