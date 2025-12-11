import numpy as np
import pandas as pd

from .preprocessing import (
    load_and_prepare_kdd,
    load_and_prepare_netflow,
    load_and_prepare_cores_iot,
)

from .streaming import build_attack_pools, create_streaming_batches

from .result import (
    print_scenario1_results,
    save_scenario1_results,
    print_scenario2_results,
    save_scenario2_results,
    plot_scenario1_radar,
    plot_scenario2_unseen_vs_all,
    plot_scenario3_batch_curves,
    summarize_scenario3_adaptation,
    print_scenario3_summary,
    save_scenario3_summary,
)
from .scenarios import (
    run_scenario_1,
    run_scenario_2_kdd,
    run_scenario_2_netflow,
    run_scenario_3_streaming,
)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths to datasets
KDD_PATH = "data/dataset-1/kddcup.data"
NETFLOW_PATH_TRAIN = "data/dataset-2/train_net.csv"
NETFLOW_PATH_TEST = "data/dataset-2/test_net.csv"
CORES_IOT_PATH = "data/dataset-3/cores_iot.csv"

# datasets to run
RUN_KDD = False
RUN_NETFLOW = True
RUN_CORES_IOT = False

# Scenario 3 streaming config
BATCH_SIZE = 300
N_PRE_DRIFT = 10
N_DRIFT_ONSET = 10
N_POST_DRIFT = 10


def main() -> None:
    # Load datasets
    if RUN_KDD:
        X_kdd, y_kdd, y_type_kdd = load_and_prepare_kdd(KDD_PATH)

    if RUN_NETFLOW:
        X_nf, y_nf, y_type_nf = load_and_prepare_netflow(NETFLOW_PATH_TRAIN)

    if RUN_CORES_IOT:
        X_iot, y_iot = load_and_prepare_cores_iot(CORES_IOT_PATH)



    # ## Scenario 1 – All attacks known
    # if RUN_KDD:
    #     s1_kdd_metrics = run_scenario_1(X_kdd, y_kdd)
    #     print_scenario1_results("KDD'99", s1_kdd_metrics)
    #     save_scenario1_results("KDD'99", s1_kdd_metrics, "results/kdd_s1.json")
    #     plot_scenario1_radar(
    #         metrics_per_model=s1_kdd_metrics,
    #         dataset_name="KDD'99",
    #         save_path="plots/kdd_s1_radar.png",
    #     )

    # if RUN_NETFLOW:
    #     s1_nf_metrics = run_scenario_1(X_nf, y_nf)
    #     print_scenario1_results("NetFlow v9", s1_nf_metrics)
    #     save_scenario1_results("NetFlow v9", s1_nf_metrics, "results/netflow_s1.json")
    #     plot_scenario1_radar(
    #         metrics_per_model=s1_nf_metrics,
    #         dataset_name="NetFlow v9",
    #         save_path="plots/netflow_s1_radar.png",
    #     )

    # if RUN_CORES_IOT:
    #     s1_iot_metrics = run_scenario_1(X_iot, y_iot)
    #     print_scenario1_results("CORES-IoT", s1_iot_metrics)
    #     save_scenario1_results("CORES-IoT", s1_iot_metrics, "results/cores_iot_s1.json")
    #     plot_scenario1_radar(
    #         metrics_per_model=s1_iot_metrics,
    #         dataset_name="CORES-IoT",
    #         save_path="plots/cores_iot_s1_radar.png",
    #     )



    ## Scenario 2 – Some attacks appear only during testing

    # # KDD'99: known vs unseen attacks
    # if RUN_KDD:
    #     known_kdd = [
    #         "back", "land", "neptune", "pod", "smurf", "teardrop",      # DoS
    #         "ipsweep", "nmap", "portsweep", "satan"                     # Probe
    #     ]
    #     unseen_kdd = [
    #         "buffer_overflow", "loadmodule", "perl", "rootkit",        # U2R
    #         "ftp_write", "guess_passwd", "imap", "multihop", "phf",
    #         "spy", "warezclient", "warezmaster"                        # R2L
    #     ]
    #     s2_kdd_metrics = run_scenario_2_kdd(
    #         X_kdd,
    #         y_kdd,
    #         y_type_kdd,
    #         known_attacks=known_kdd,
    #         unseen_attacks=unseen_kdd,
    #     )
    #     print_scenario2_results("KDD'99", s2_kdd_metrics, metric="f1")
    #     save_scenario2_results("KDD'99", s2_kdd_metrics, "results/kdd_s2.json")
    #     plot_scenario2_unseen_vs_all(
    #         s2_kdd_metrics,
    #         metric="f1",
    #         dataset_name="KDD'99",
    #         save_path="plots/kdd_s2_f1.png",
    #     )
    #


    # NetFlow v9: known vs unseen categories
    if RUN_NETFLOW:
        
        known_nf = ["Port Scanning", "Denial of Service"]
        unseen_nf = ["Malware"]
    
        print("\n[NetFlow v9] Scenario 2")
        s2_nf_metrics = run_scenario_2_netflow(
            X_nf,
            y_nf,
            y_type_nf,
            known_attacks=known_nf,
            unseen_attacks=unseen_nf,
        )
        print_scenario2_results("NetFlow v9", s2_nf_metrics, metric="f1")
        save_scenario2_results("NetFlow v9", s2_nf_metrics, "results/netflow_s2.json")
        plot_scenario2_unseen_vs_all(
            s2_nf_metrics,
            metric="f1",
            dataset_name="NetFlow v9",
            save_path="plots/netflow_s2_f1.png",
        )
    

    ## Scenario 3 – Evolving attacks

    # KDD'99 streaming
    # if RUN_KDD:
    #     known_kdd = [
    #         "back", "land", "neptune", "pod", "smurf", "teardrop",
    #         "ipsweep", "nmap", "portsweep", "satan"
    #     ]
    #     new_kdd = [
    #         "ftp_write", "guess_passwd", "imap", "multihop", "phf",
    #         "spy", "warezclient", "warezmaster", "buffer_overflow",
    #         "loadmodule", "perl", "rootkit"
    #     ]
    #
    #     kdd_pools = build_attack_pools(
    #         X_kdd,
    #         y_kdd,
    #         y_type_kdd,
    #         known_attacks=known_kdd,
    #         new_attacks=new_kdd,
    #     )
    #
    #     kdd_batches = create_streaming_batches(
    #         kdd_pools,
    #         batch_size=BATCH_SIZE,
    #         n_pre_drift=N_PRE_DRIFT,
    #         n_drift_onset=N_DRIFT_ONSET,
    #         n_post_drift=N_POST_DRIFT,
    #     )
    #
    #     s3_kdd_batch_metrics = run_scenario_3_streaming(
    #         kdd_batches,
    #         n_pre_drift=N_PRE_DRIFT,
    #     )
    #     plot_scenario3_batch_curves(
    #         s3_kdd_batch_metrics,
    #         metric="f1",
    #         n_pre_drift=N_PRE_DRIFT,
    #         n_drift_onset=N_DRIFT_ONSET,
    #         dataset_name="KDD'99",
    #         save_path="plots/kdd_s3_f1.png",
    #     )
    #     kdd_summary = summarize_scenario3_adaptation(
    #         s3_kdd_batch_metrics,
    #         metric="f1",
    #         n_pre_drift=N_PRE_DRIFT,
    #         n_drift_onset=N_DRIFT_ONSET,
    #     )
    #     print_scenario3_summary("KDD'99", kdd_summary, metric="f1")
    #     save_scenario3_summary("KDD'99", kdd_summary, "results/kdd_s3.json")

    # # NetFlow v9 streaming
    # if RUN_NETFLOW:
    #     known_nf = ["Port Scanning", "Denial of Service"]
    #     new_nf = ["Malware"]
    
    #     nf_pools = build_attack_pools(
    #         X_nf,
    #         y_nf,
    #         y_type_nf,
    #         known_attacks=known_nf,
    #         new_attacks=new_nf,
    #     )
    
    #     nf_batches = create_streaming_batches(
    #         nf_pools,
    #         batch_size=BATCH_SIZE,
    #         n_pre_drift=N_PRE_DRIFT,
    #         n_drift_onset=N_DRIFT_ONSET,
    #         n_post_drift=N_POST_DRIFT,
    #     )
    
    #     s3_nf_batch_metrics = run_scenario_3_streaming(
    #         nf_batches,
    #         n_pre_drift=N_PRE_DRIFT,
    #     )
    
    #     plot_scenario3_batch_curves(
    #         s3_nf_batch_metrics,
    #         metric="f1",
    #         n_pre_drift=N_PRE_DRIFT,
    #         n_drift_onset=N_DRIFT_ONSET,
    #         dataset_name="NetFlow v9",
    #         save_path="plots/netflow_s3_f1.png",
    #     )
    
    #     nf_summary = summarize_scenario3_adaptation(
    #         s3_nf_batch_metrics,
    #         metric="f1",
    #         n_pre_drift=N_PRE_DRIFT,
    #         n_drift_onset=N_DRIFT_ONSET,
    #     )
    #     print_scenario3_summary("NetFlow v9", nf_summary, metric="f1")
    #     save_scenario3_summary("NetFlow v9", nf_summary, "results/netflow_s3.json")

    # # CORES-IoT Scenario 3 streaming with drift


if __name__ == "__main__":
    main()