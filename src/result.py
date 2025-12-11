from typing import Dict, List, Optional, Any
import os
import json

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_json(obj: Any, save_path: str) -> None:
    _ensure_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


## Printing json results

def print_scenario1_results(dataset_name: str,
                            metrics_per_model: Dict[str, Dict[str, float]]) -> None:
    print(f"\n[Scenario 1] Dataset: {dataset_name}")

    for model_name, model_metrics in metrics_per_model.items():
        print(f"\n  Model: {model_name}")
        for metric_name, value in model_metrics.items():
            print(f"    {metric_name:>10}: {value:.4f}")


def save_scenario1_results(
    dataset_name: str,
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    payload = {
        "dataset": dataset_name,
        "scenario": "scenario_1_all_attacks_known",
        "metrics": metrics,
    }
    save_json(payload, save_path)


def print_scenario2_results(
    dataset_name: str,
    metrics_per_model: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1",
) -> None:
    print(f"\n[Scenario 2] Dataset: {dataset_name}")
    print(f"Metric: {metric.upper()} (all vs unseen)")
    for model_name, subsets in metrics_per_model.items():
        m_all = subsets.get("all", {})
        m_unseen = subsets.get("unseen", {})
        v_all = m_all.get(metric, float("nan"))
        v_unseen = m_unseen.get(metric, float("nan"))
        print(f"  {model_name:>12}: all = {v_all:.4f} | unseen = {v_unseen:.4f}")


def save_scenario2_results(
    dataset_name: str,
    metrics_per_model: Dict[str, Dict[str, Dict[str, float]]],
    save_path: str,
) -> None:
    payload = {
        "dataset": dataset_name,
        "scenario": "scenario_2_unseen_attacks",
        "metrics_per_model": metrics_per_model,
    }
    save_json(payload, save_path)


## Print plots

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def plot_scenario1_radar(
    metrics_per_model: Dict[str, Dict[str, float]],
    dataset_name: str,
    save_path: str = None,
) -> None:

    # Order of metrics around the circle
    metric_names = ["f1", "precision", "recall", "bac"]
    num_metrics = len(metric_names)

    # Angles for each metric axis
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # close the loop

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Plot one polygon per model
    for model_name, m in metrics_per_model.items():
        values = [m[metric] for metric in metric_names]
        values.append(values[0])  # close the polygon

        ax.plot(angles, values, linewidth=1.5, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name.upper() for name in metric_names])

    # Limit from 0 to 1 since all metrics are in [0, 1]
    ax.set_ylim(0.0, 1.0)

    ax.set_title(f"Scenario 1 – {dataset_name}", pad=20)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_scenario2_unseen_vs_all(
    metrics_per_model: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "f1",
    title: str = "Scenario 2: Performance on all vs unseen attacks",
    dataset_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:

    model_names = list(metrics_per_model.keys())
    n_models = len(model_names)

    all_vals = [metrics_per_model[m]["all"][metric] for m in model_names]
    unseen_vals = [metrics_per_model[m]["unseen"][metric] for m in model_names]

    x = np.arange(n_models)
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width / 2, all_vals, width, label="All attacks")
    plt.bar(x + width / 2, unseen_vals, width, label="Unseen attacks")

    plt.xticks(x, model_names)
    plt.ylabel(metric.upper())

    full_title = title
    if dataset_name:
        full_title += f" ({dataset_name})"
    plt.title(full_title)

    plt.legend()
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_scenario3_batch_curves(
    batch_metrics: Dict[str, List[Dict[str, float]]],
    metric: str = "f1",
    n_pre_drift: Optional[int] = None,
    n_drift_onset: Optional[int] = None,
    title: str = "Scenario 3: Batch-wise performance",
    dataset_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:

    any_model = next(iter(batch_metrics.values()))
    n_batches = len(any_model)
    batch_indices = np.arange(1, n_batches + 1)
    plt.figure(figsize=(10, 6))

    for model_name, metrics_list in batch_metrics.items():
        values = [m[metric] for m in metrics_list]
        plt.plot(batch_indices, values, marker="o", linewidth=1.5, label=model_name)

    if n_pre_drift is not None and n_drift_onset is not None:
        start_pre = 1
        end_pre = n_pre_drift

        start_onset = n_pre_drift + 1
        end_onset = n_pre_drift + n_drift_onset

        start_post = end_onset + 1
        end_post = n_batches

        plt.axvspan(start_pre - 0.5, end_pre + 0.5, alpha=0.1)
        plt.axvspan(start_onset - 0.5, end_onset + 0.5, alpha=0.15)
        plt.axvspan(start_post - 0.5, end_post + 0.5, alpha=0.08)

        ymin, ymax = plt.ylim()
        ytext = ymax - 0.05 * (ymax - ymin)

        plt.text((start_pre + end_pre) / 2, ytext, "pre-drift",
                 ha="center", va="top", fontsize=9)
        plt.text((start_onset + end_onset) / 2, ytext, "drift onset",
                 ha="center", va="top", fontsize=9)
        plt.text((start_post + end_post) / 2, ytext, "post-drift",
                 ha="center", va="top", fontsize=9)

    plt.xlabel("Batch index")
    plt.ylabel(metric.upper())

    full_title = title
    if dataset_name:
        full_title += f" ({dataset_name})"
    plt.title(full_title)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()



def summarize_scenario3_adaptation(
    batch_metrics: Dict[str, List[Dict[str, float]]],
    metric: str = "f1",
    n_pre_drift: Optional[int] = None,
    n_drift_onset: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:

    any_model = next(iter(batch_metrics.values()))
    n_batches = len(any_model)

    if n_pre_drift is None or n_drift_onset is None:
        # If phase information is not provided, just compute overall means
        summary: Dict[str, Dict[str, float]] = {}
        for model_name, metrics_list in batch_metrics.items():
            values = np.array([m[metric] for m in metrics_list], dtype=float)
            summary[model_name] = {
                "mean_overall": float(np.mean(values)),
            }
        return summary

    start_pre = 0
    end_pre = n_pre_drift
    start_onset = end_pre
    end_onset = end_pre + n_drift_onset
    start_post = end_onset
    end_post = n_batches
    summary: Dict[str, Dict[str, float]] = {}

    for model_name, metrics_list in batch_metrics.items():
        values = np.array([m[metric] for m in metrics_list], dtype=float)

        mean_pre = float(np.mean(values[start_pre:end_pre])) if end_pre > start_pre else float("nan")
        mean_onset = float(np.mean(values[start_onset:end_onset])) if end_onset > start_onset else float("nan")
        mean_post = float(np.mean(values[start_post:end_post])) if end_post > start_post else float("nan")
        mean_overall = float(np.mean(values))

        summary[model_name] = {
            "mean_pre": mean_pre,
            "mean_onset": mean_onset,
            "mean_post": mean_post,
            "mean_overall": mean_overall,
        }

    return summary


def print_scenario3_summary(
    dataset_name: str,
    summary: Dict[str, Dict[str, float]],
    metric: str = "f1",
) -> None:
    print(f"\n[Scenario 3] Dataset: {dataset_name}")
    print(f"Summary metric: {metric.upper()}")
    for model_name, vals in summary.items():
        mean_pre = vals.get("mean_pre", float("nan"))
        mean_onset = vals.get("mean_onset", float("nan"))
        mean_post = vals.get("mean_post", float("nan"))
        mean_overall = vals.get("mean_overall", float("nan"))
        print(
            f"  {model_name:>12}: "
            f"pre={mean_pre:.4f}, onset={mean_onset:.4f}, "
            f"post={mean_post:.4f}, overall={mean_overall:.4f}"
        )


def save_scenario3_summary(
    dataset_name: str,
    summary: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    payload = {
        "dataset": dataset_name,
        "scenario": "scenario_3_evolving_attacks",
        "summary": summary,
    }
    save_json(payload, save_path)