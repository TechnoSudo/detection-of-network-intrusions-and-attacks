from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Any, Optional

from src.scenarios.scenario1 import run_scenario1
from src.scenarios.scenario2 import run_scenario2
from src.scenarios.scenario3 import run_scenario3

from src.evaluation.plots import (
    plot_model_metrics,
    plot_confusion,
    plot_prec_recall,
)


def _make_run_dir(scenario: int, dataset: str) -> Dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"scenario{scenario}", dataset, ts)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return {"run_dir": run_dir, "plots_dir": plots_dir}


def _run_plots_from_npz(run_info: Dict[str, Any], plots_dir: str) -> None:
    npz_map = run_info.get("npz", {})
    npz_path: Optional[str] = None

    if isinstance(npz_map, dict) and len(npz_map) > 0:
        npz_path = npz_map.get("ensemble") or next(iter(npz_map.values()), None)

    if npz_path and os.path.exists(npz_path):
        plot_confusion(npz_path, outdir=plots_dir, title="Confusion Matrix")
        try:
            plot_prec_recall(npz_path, outdir=plots_dir, title="Precision-Recall Curve")
        except Exception as e:
            print(f"[main] PR curve skipped: {e}")
    else:
        print("[main] No .npz arrays found; skipping confusion/PR plots.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenarios + auto-generate plots into results/")

    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--dataset", type=str, required=True, choices=["kdd", "nsl", "netflow", "iot"])

    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-pre", type=int, default=10)
    parser.add_argument("--n-drift", type=int, default=10)
    parser.add_argument("--n-post", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2000)

    parser.add_argument(
        "--drift-type",
        type=str,
        default="incremental",
        choices=["sudden", "incremental", "gradual"],
        help="Scenario3 drift schedule type",
    )
    parser.add_argument("--change-point", type=int, default=10, help="Sudden drift: batch where drift starts")
    parser.add_argument("--start-t", type=int, default=10, help="Incremental/gradual: ramp start batch")
    parser.add_argument("--end-t", type=int, default=19, help="Incremental/gradual: ramp end batch")
    parser.add_argument("--min-intensity", type=float, default=0.0)
    parser.add_argument("--max-intensity", type=float, default=1.0)
    parser.add_argument("--min-mix", type=float, default=0.0, help="Gradual: affected fraction at start")
    parser.add_argument("--max-mix", type=float, default=1.0, help="Gradual: affected fraction at end")

    args = parser.parse_args()

    dirs = _make_run_dir(args.scenario, args.dataset)
    run_dir, plots_dir = dirs["run_dir"], dirs["plots_dir"]

    print("\n" + "=" * 90)
    print(f"RUN: scenario{args.scenario} | dataset={args.dataset}")
    print(f"Run dir: {run_dir}")
    print("=" * 90)

    run_info: Dict[str, Any] = {}

    if args.scenario == 1:
        run_info = run_scenario1(
            dataset=args.dataset,
            outdir=run_dir,
            sample_frac=args.sample_frac,
            seed=args.seed,
        )

        print("\n[main] Scenario 1 finished. Generating plots...")
        results_csv = run_info.get("results_csv")
        if results_csv and os.path.exists(results_csv):
            plot_model_metrics(results_csv, outdir=plots_dir)
        _run_plots_from_npz(run_info, plots_dir)

    elif args.scenario == 2:
        run_info = run_scenario2(
            dataset=args.dataset,
            outdir=run_dir,
            sample_frac=args.sample_frac,
            seed=args.seed,
        )

        print("\n[main] Scenario 2 finished. Generating plots...")
        results_csv = run_info.get("results_csv")
        if results_csv and os.path.exists(results_csv):
            plot_model_metrics(results_csv, outdir=plots_dir)
        _run_plots_from_npz(run_info, plots_dir)

    elif args.scenario == 3:
        run_info = run_scenario3(
            dataset=args.dataset,
            outdir=run_dir,
            seed=args.seed,
            sample_frac=args.sample_frac,
            n_pre=args.n_pre,
            n_drift=args.n_drift,
            n_post=args.n_post,
            batch_size=args.batch_size,
            drift_type=args.drift_type,
            change_point=args.change_point,
            start_t=args.start_t,
            end_t=args.end_t,
            min_intensity=args.min_intensity,
            max_intensity=args.max_intensity,
            min_mix=args.min_mix,
            max_mix=args.max_mix,
        )
        print("\n[main] Scenario 3 finished. (Scenario 3 saves its own drift plot + CSVs.)")

    print("\n" + "-" * 90)
    print("DONE.")
    print(f"Results folder: {run_dir}")
    print(f"Plots folder  : {plots_dir}")

    if isinstance(run_info, dict):
        if "plot_png" in run_info:
            print(f"Main plot     : {run_info['plot_png']}")
        if "results_csv" in run_info:
            print(f"Results CSV   : {run_info['results_csv']}")
        if "metrics_csv" in run_info:
            print(f"Metrics CSV   : {run_info['metrics_csv']}")
        if "events_csv" in run_info:
            print(f"Events CSV    : {run_info['events_csv']}")


if __name__ == "__main__":
    main()