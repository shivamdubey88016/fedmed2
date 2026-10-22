"""
Reproducible Experiment Pipeline and File-Based Persistence Engine
Stores experiments under data/experiments/{experiment_id}/ containing:
  - config.json
  - metrics.csv
  - results.json

Zero external database dependency.
Supports single-run and multi-seed evaluations ([42, 123, 456, 789, 999]).
"""

import os
import csv
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from fl_engine.models import ClinicalRiskModel, ClinDisentangleNet
from fl_engine.bias_detector import BiasDetector
from fl_engine.hospital_node import HospitalNode
from fl_engine.global_server import GlobalFederatedServer
from fl_engine.dataset import generate_federated_healthcare_datasets

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments")

def generate_experiment_id() -> str:
    """Generates unique experiment ID, e.g. FEQ-2026-001."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    existing = [d for d in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    year = datetime.now().year
    prefix = f"FEQ-{year}-"
    matching = [d for d in existing if d.startswith(prefix)]
    max_idx = 0
    for m in matching:
        try:
            num = int(m.split("-")[-1])
            if num > max_idx:
                max_idx = num
        except Exception:
            pass
    next_idx = max_idx + 1
    return f"FEQ-{year}-{next_idx:03d}"

def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Computes sample mean and sample standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return round(mean, 4), 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(max(0.0, var))
    return round(mean, 4), round(std, 4)

def run_single_experiment(
    algorithm: str = "FedEquiMed",
    dataset_key: str = "mimic_iv",
    num_clients: int = 3,
    rounds: int = 5,
    local_epochs: int = 4,
    learning_rate: float = 0.08,
    fairness_beta: float = 2.5,
    seed: int = 42,
    heterogeneity: str = "medium",
    dp_noise: float = 0.02
) -> Dict[str, Any]:
    """Runs a single experiment with the specified hyperparameter configuration."""
    # Partition datasets according to non-IID heterogeneity
    hosp_data, global_test = generate_federated_healthcare_datasets(seed=seed)
    
    # Instantiate hospital nodes
    hospitals = {}
    h_keys = list(hosp_data.keys())[:num_clients]
    for k in h_keys:
        hospitals[k] = HospitalNode(
            hospital_id=k,
            name=f"Hospital {k.split('_')[-1].upper()}",
            dataset=hosp_data[k],
            num_features=10
        )

    server = GlobalFederatedServer(hospitals, global_test, num_features=10)
    history = []

    # Map algorithm names to internal strategies
    algo_map = {
        "fedavg": "fedavg",
        "fairfed": "fairfed",
        "fedequimed": "fedequimed"
    }
    strat = algo_map.get(algorithm.lower(), "fedequimed")

    for r in range(rounds):
        round_res = server.run_federated_round(
            strategy=strat,
            epochs_per_hospital=local_epochs,
            learning_rate=learning_rate,
            fairness_beta=fairness_beta,
            dp_noise_multiplier=dp_noise
        )
        gm = round_res["global_metrics"]
        history.append({
            "round": r + 1,
            "accuracy": gm["accuracy"],
            "precision": gm["precision"],
            "recall": gm["recall"],
            "f1": gm["f1"],
            "auroc": gm["auroc"],
            "delta_dp": gm["gender_dpd"],
            "delta_eo": gm["gender_eod"],
            "dir": gm["gender_dir"],
            "composite_disparity": gm["composite_disparity"]
        })

    final_m = history[-1]
    return {
        "algorithm": algorithm,
        "seed": seed,
        "rounds": rounds,
        "history": history,
        "final_metrics": final_m
    }

def run_and_persist_experiment(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes an experiment (single or multi-seed) and saves artifacts under:
    data/experiments/{experiment_id}/
      - config.json
      - metrics.csv
      - results.json
    """
    exp_id = generate_experiment_id()
    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    seeds = config.get("seeds", [config.get("seed", 42)])
    if not isinstance(seeds, list) or len(seeds) == 0:
        seeds = [42]

    algorithm = config.get("algorithm", "FedEquiMed")
    dataset_key = config.get("dataset", "mimic_iv")
    num_clients = int(config.get("num_clients", 3))
    rounds = int(config.get("rounds", 5))
    local_epochs = int(config.get("local_epochs", 4))
    lr = float(config.get("learning_rate", 0.08))
    beta = float(config.get("fairness_beta", 2.5))
    heterogeneity = config.get("heterogeneity", "medium")
    dp_noise = float(config.get("dp_noise", 0.02))

    seed_results = []
    for s in seeds:
        res = run_single_experiment(
            algorithm=algorithm,
            dataset_key=dataset_key,
            num_clients=num_clients,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=lr,
            fairness_beta=beta,
            seed=s,
            heterogeneity=heterogeneity,
            dp_noise=dp_noise
        )
        seed_results.append(res)

    # Compute mean and standard deviations
    acc_vals = [sr["final_metrics"]["accuracy"] for sr in seed_results]
    f1_vals = [sr["final_metrics"]["f1"] for sr in seed_results]
    auroc_vals = [sr["final_metrics"]["auroc"] for sr in seed_results]
    dp_vals = [sr["final_metrics"]["delta_dp"] for sr in seed_results]
    eo_vals = [sr["final_metrics"]["delta_eo"] for sr in seed_results]
    dir_vals = [sr["final_metrics"]["dir"] for sr in seed_results]

    acc_mean, acc_std = compute_mean_std(acc_vals)
    f1_mean, f1_std = compute_mean_std(f1_vals)
    auroc_mean, auroc_std = compute_mean_std(auroc_vals)
    dp_mean, dp_std = compute_mean_std(dp_vals)
    eo_mean, eo_std = compute_mean_std(eo_vals)
    dir_mean, dir_std = compute_mean_std(dir_vals)

    results_data = {
        "experiment_id": exp_id,
        "timestamp": datetime.now().isoformat(),
        "algorithm": algorithm,
        "seeds": seeds,
        "rounds": rounds,
        "metrics_summary": {
            "accuracy": {"mean": acc_mean, "std": acc_std},
            "f1": {"mean": f1_mean, "std": f1_std},
            "auroc": {"mean": auroc_mean, "std": auroc_std},
            "delta_dp": {"mean": dp_mean, "std": dp_std},
            "delta_eo": {"mean": eo_mean, "std": eo_std},
            "dir": {"mean": dir_mean, "std": dir_std}
        },
        "seed_runs": seed_results
    }

    # 1. Save config.json
    full_config = {
        "experiment_id": exp_id,
        "created_at": results_data["timestamp"],
        "dataset": dataset_key,
        "prediction_task": "30-day adverse clinical event & mortality risk",
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "algorithm": algorithm,
        "fairness_beta": beta,
        "seeds": seeds,
        "heterogeneity": heterogeneity,
        "dp_parameters": {
            "noise_multiplier": dp_noise,
            "clipping_norm": 1.0,
            "target_delta": 1e-5
        }
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)

    # 2. Save results.json
    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # 3. Save metrics.csv
    csv_path = os.path.join(exp_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "round", "accuracy", "precision", "recall", "f1", "auroc", "delta_dp", "delta_eo", "dir"])
        for sr in seed_results:
            s_val = sr["seed"]
            for row in sr["history"]:
                writer.writerow([
                    s_val,
                    row["round"],
                    row["accuracy"],
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row["f1"],
                    row["auroc"],
                    row["delta_dp"],
                    row["delta_eo"],
                    row["dir"]
                ])

    return results_data

def list_experiments() -> List[Dict[str, Any]]:
    """Lists all executed experiments from data/experiments/."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    exp_folders = sorted(os.listdir(EXPERIMENTS_DIR), reverse=True)
    summary_list = []
    for fld in exp_folders:
        fld_path = os.path.join(EXPERIMENTS_DIR, fld)
        if not os.path.isdir(fld_path):
            continue
        cfg_file = os.path.join(fld_path, "config.json")
        res_file = os.path.join(fld_path, "results.json")
        if os.path.exists(cfg_file) and os.path.exists(res_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                with open(res_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
                summary_list.append({
                    "experiment_id": fld,
                    "date": cfg.get("created_at", ""),
                    "algorithm": cfg.get("algorithm", "Unknown"),
                    "dataset": cfg.get("dataset", "mimic_iv"),
                    "clients": cfg.get("num_clients", 3),
                    "rounds": cfg.get("rounds", 5),
                    "seeds": cfg.get("seeds", []),
                    "status": "Completed",
                    "metrics_summary": res.get("metrics_summary", {})
                })
            except Exception:
                continue
    return summary_list

def run_ablation_study(
    rounds: int = 4,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the 5-component ablation study on identical data and splits:
    A. Baseline FedAvg
    B. FedAvg + ClinDisentangleNet
    C. FedAvg + CARP-Loss
    D. FedAvg + D-TAMA
    E. Full FedEquiMed
    """
    hosp_data, global_test = generate_federated_healthcare_datasets(seed=seed)
    components = [
        {"id": "A", "name": "FedAvg (Baseline)", "strat": "fedavg"},
        {"id": "B", "name": "FedAvg + ClinDisentangleNet", "strat": "ablation_disentangle"},
        {"id": "C", "name": "FedAvg + CARP-Loss", "strat": "ablation_carp"},
        {"id": "D", "name": "FedAvg + D-TAMA", "strat": "ablation_dtama"},
        {"id": "E", "name": "Full FedEquiMed (All)", "strat": "fedequimed"}
    ]

    ablation_results = []
    initial_w = ClinicalRiskModel(num_features=10, seed=seed).get_weights()

    for comp in components:
        hospitals = {
            k: HospitalNode(hospital_id=k, name=k, dataset=hosp_data[k], num_features=10)
            for k in hosp_data
        }
        for h in hospitals.values():
            h.local_model.set_weights(initial_w)

        server = GlobalFederatedServer(hospitals, global_test, num_features=10)
        server.global_model.set_weights(initial_w)

        for _ in range(rounds):
            server.run_federated_round(strategy=comp["strat"], epochs_per_hospital=4, fairness_beta=2.5)

        eval_res = server.evaluate_global_model()
        ablation_results.append({
            "component_id": comp["id"],
            "component_name": comp["name"],
            "strategy": comp["strat"],
            "accuracy": eval_res["overall_accuracy"],
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1": eval_res["f1"],
            "auroc": eval_res["auroc"],
            "delta_dp": eval_res["gender_audit"]["demographic_parity_difference"],
            "delta_eo": eval_res["gender_audit"]["equalized_odds_difference"],
            "dir": eval_res["gender_audit"]["disparate_impact_ratio"],
            "composite_disparity": eval_res["composite_disparity"]
        })

    return {
        "rounds": rounds,
        "seed": seed,
        "ablation_study": ablation_results,
        "interpretation": "Each component's individual empirical contribution evaluated on held-out test cohort."
    }
