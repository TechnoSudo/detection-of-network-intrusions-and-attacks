from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =========================
# Configuration
# =========================

@dataclass(frozen=True)
class StatConfig:
    results_root: str = os.path.join("results", "scenario2")
    metric_col: str = "f1"
    alpha: float = 0.05
    zero_method: str = "wilcox"       # recommended
    alternative: str = "two-sided"    # standard

def _list_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def _find_results_csv(run_dir: str) -> str | None:
    for f in os.listdir(run_dir):
        if f.startswith("results_scenario2") and f.endswith(".csv"):
            return os.path.join(run_dir, f)
    return None


def _read_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"model", "f1"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns {required}: {csv_path}")
    return df

def run_stats(cfg: StatConfig) -> pd.DataFrame:
    """
    Performs paired Wilcoxon tests for Scenario 2.
    Pairing is done across models within the same dataset.
    """

    comparisons = [
        ("ensemble", "rf"),
        ("ensemble", "xgb"),
        ("ensemble", "sgd_static"),
        ("sgd_adaptive", "sgd_static"),
    ]

    rows: List[Dict[str, object]] = []

    for dataset in _list_dirs(cfg.results_root):
        ds_dir = os.path.join(cfg.results_root, dataset)
        runs = _list_dirs(ds_dir)

        for run in runs:
            run_dir = os.path.join(ds_dir, run)
            csv_path = _find_results_csv(run_dir)
            if csv_path is None:
                continue

            df = _read_results(csv_path)

            wide = df.pivot_table(
                index=None,
                columns="model",
                values=cfg.metric_col,
                aggfunc="mean",
            )

            for A, B in comparisons:
                if A not in wide.columns or B not in wide.columns:
                    continue

                a = np.asarray([wide[A].values[0]])
                b = np.asarray([wide[B].values[0]])

                if np.allclose(a, b):
                    p = 1.0
                else:
                    p = wilcoxon(
                        a,
                        b,
                        zero_method=cfg.zero_method,
                        alternative=cfg.alternative,
                    ).pvalue

                rows.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "metric": cfg.metric_col,
                        "A": A,
                        "B": B,
                        "A_mean": float(a.mean()),
                        "B_mean": float(b.mean()),
                        "diff(A-B)": float(a.mean() - b.mean()),
                        "p_value": float(p),
                        "significant": bool(p <= cfg.alpha),
                    }
                )

    if not rows:
        raise RuntimeError("No Scenario 2 results found.")

    return pd.DataFrame(rows)


# =========================
# Save outputs
# =========================

def save_outputs(df: pd.DataFrame, cfg: StatConfig) -> None:
    out_csv = os.path.join(cfg.results_root, "stat_summary_scenario2.csv")
    df.to_csv(out_csv, index=False)
    print(f"[stats] saved -> {out_csv}")

    print("\n[stats] Significant results (p ≤ alpha):")
    sig = df[df["significant"]]
    if sig.empty:
        print("  none")
    else:
        for _, r in sig.iterrows():
            print(
                f"  {r['dataset']} | {r['A']} vs {r['B']} "
                f"(diff={r['diff(A-B)']:.4f}, p={r['p_value']:.4g})"
            )


# =========================
# Main
# =========================

def main() -> None:
    cfg = StatConfig()
    df = run_stats(cfg)
    save_outputs(df, cfg)


if __name__ == "__main__":
    main()