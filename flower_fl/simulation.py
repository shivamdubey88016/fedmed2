"""
Flower Simulation Engine for Multi-Hospital Federated Learning
Orchestrates simulated hospital nodes, executes communication rounds,
and logs fairness trajectories on the global holdout cohort.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import flwr as fl
from flwr.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from flower_fl.model import PyTorchLogisticRegression
from flower_fl.dataset_loader import load_all_hospital_datasets
from flower_fl.metrics import compute_clinical_fairness_metrics
from flower_fl.client import HospitalFlowerClient, LocalClientProxy
from flower_fl.strategy import FairnessAwareFedAvg

def create_global_eval_fn(global_test_data: Dict[str, Any]):
    """Creates a centralized evaluation function for the Flower strategy."""
    X_test = global_test_data["X"]
    y_test = global_test_data["y"]
    gender_test = global_test_data["gender"]
    
    eval_model = PyTorchLogisticRegression(num_features=X_test.shape[1])
    last_weights_holder = {"weights": eval_model.get_weights()}

    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, Any]):
        eval_model.set_weights(parameters)
        eval_model.eval()
        last_weights_holder["weights"] = [w.copy() for w in parameters]

        with torch.no_grad():
            preds = eval_model.predict(X_test)
            probs = eval_model.predict_proba(X_test)
            metrics = compute_clinical_fairness_metrics(
                y_true=y_test,
                y_pred=preds,
                y_prob=probs,
                sensitive_attr=gender_test,
                attribute_name="gender"
            )
            # BCE Loss
            logits = eval_model(X_test)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_test.unsqueeze(1)
            ).item()

        return float(loss), metrics

    return evaluate, last_weights_holder

def evaluate_model_node_wise(
    weights: List[np.ndarray],
    hospitals_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates global model weights individually on each hospital node's local population.
    Returns: Dict[hospital_id -> {accuracy, f1, dpd, eod, auroc, dir, loss, sample_size, subgroups}]
    """
    node_results = {}
    for hid, data in hospitals_data.items():
        X_h = data["X"]
        y_h = data["y"]
        g_h = data["gender"]
        
        eval_model = PyTorchLogisticRegression(num_features=X_h.shape[1])
        eval_model.set_weights(weights)
        eval_model.eval()

        with torch.no_grad():
            preds = eval_model.predict(X_h)
            probs = eval_model.predict_proba(X_h)
            metrics = compute_clinical_fairness_metrics(
                y_true=y_h,
                y_pred=preds,
                y_prob=probs,
                sensitive_attr=g_h,
                attribute_name="gender"
            )
            logits = eval_model(X_h)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_h.unsqueeze(1)
            ).item()

        metrics["loss"] = round(float(loss), 4)
        metrics["sample_size"] = data["sample_size"]
        node_results[hid] = metrics

    return node_results

def run_federated_simulation(
    strategy_type: str = "fairness_aware",  # "fedavg" or "fairness_aware"
    num_rounds: int = 5,
    epochs_per_hospital: int = 3,
    lr: float = 0.05,
    fairness_beta: float = 2.5,
    momentum_gamma: float = 0.8,
    use_local_reweighting: Optional[bool] = None,
    data_dir: str = "data",
    hospitals_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes an end-to-end Flower federated learning simulation across participating hospital clients.
    Supports default or custom user-provided hospital datasets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load institutional cohorts (default or custom)
    if hospitals_dict is not None and len(hospitals_dict) > 0:
        hospitals_data = hospitals_dict
        _, global_test = load_all_hospital_datasets(data_dir=data_dir)
    else:
        hospitals_data, global_test = load_all_hospital_datasets(data_dir=data_dir)

    num_clients = len(hospitals_data)
    if num_clients == 0:
        raise ValueError("No hospital clients available for federated training.")

    # If use_local_reweighting is None:
    # FedAvg defaults to False (standard ERM)
    # FairnessAware defaults to True (FedIDA local balance)
    if use_local_reweighting is None:
        use_local_reweighting = (strategy_type.lower() == "fairness_aware")

    # 2. Initialize hospital clients
    hospital_names = {
        "hospital_a": "Hospital A (Urban Medical Center)",
        "hospital_b": "Hospital B (Rural Community Clinic)",
        "hospital_c": "Hospital C (Heart & Cardiovascular Institute)"
    }
    
    clients = {}
    for hid, data in hospitals_data.items():
        h_label = hospital_names.get(hid, hid.replace("_", " ").title())
        clients[hid] = HospitalFlowerClient(
            client_id=hid,
            hospital_name=h_label,
            data=data,
            use_adaptive_reweighting=use_local_reweighting
        )

    # 3. Create initial global model weights (zeros)
    num_features = global_test["X"].shape[1]
    init_model = PyTorchLogisticRegression(num_features=num_features)
    init_params = ndarrays_to_parameters(init_model.get_weights())

    # 4. Centralized evaluation function
    eval_fn, last_weights_holder = create_global_eval_fn(global_test)

    # Fit config callback
    def on_fit_config(server_round: int) -> Dict[str, Any]:
        return {
            "epochs": epochs_per_hospital,
            "lr": lr,
            "use_reweighting": use_local_reweighting,
            "round": server_round
        }

    # 5. Instantiate Flower Strategy
    if strategy_type.lower() == "fairness_aware":
        strategy = FairnessAwareFedAvg(
            fairness_beta=fairness_beta,
            momentum_gamma=momentum_gamma,
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=eval_fn,
            on_fit_config_fn=on_fit_config,
            initial_parameters=init_params,
        )
    else:
        strategy = FedAvg(
            fraction_fit=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=eval_fn,
            on_fit_config_fn=on_fit_config,
            initial_parameters=init_params,
        )

    # 6. Register clients with ClientManager
    client_manager = SimpleClientManager()
    for hid, cl in clients.items():
        client_manager.register(LocalClientProxy(hid, cl))

    # 7. Create Flower Server and run training rounds
    server = Server(client_manager=client_manager, strategy=strategy)
    history, elapsed_sec = server.fit(num_rounds=num_rounds, timeout=None)

    # 8. Extract metrics trajectory from history
    metrics_cent = history.metrics_centralized
    trajectory = []
    
    # metrics_cent maps metric_name -> [(round_num, val), ...]
    num_eval_rounds = len(metrics_cent.get("accuracy", []))
    for r_idx in range(num_eval_rounds):
        r_num = metrics_cent["accuracy"][r_idx][0]
        round_data = {"round": r_num}
        for m_name, entries in metrics_cent.items():
            if r_idx < len(entries):
                round_data[m_name] = entries[r_idx][1]
        trajectory.append(round_data)

    # Weights history (if FairnessAware)
    weights_history = getattr(strategy, "weights_history", [])

    final_metrics = trajectory[-1] if trajectory else {}

    # 9. Evaluate final global model node-wise across all participating hospitals
    node_wise_metrics = evaluate_model_node_wise(last_weights_holder["weights"], hospitals_data)

    return {
        "strategy": strategy_type,
        "num_rounds": num_rounds,
        "elapsed_seconds": round(elapsed_sec, 3),
        "trajectory": trajectory,
        "weights_history": weights_history,
        "final_metrics": final_metrics,
        "final_weights": last_weights_holder["weights"],
        "node_wise_metrics": node_wise_metrics
    }

def run_comparative_benchmark(
    num_rounds: int = 5,
    epochs_per_hospital: int = 3,
    lr: float = 0.05,
    fairness_beta: float = 2.5,
    momentum_gamma: float = 0.8,
    data_dir: str = "data",
    hospitals_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes both Standard FedAvg and Custom Fairness-Aware Strategy
    under identical data partitions and seeds, returning direct side-by-side comparisons.
    """
    fedavg_res = run_federated_simulation(
        strategy_type="fedavg",
        num_rounds=num_rounds,
        epochs_per_hospital=epochs_per_hospital,
        lr=lr,
        data_dir=data_dir,
        hospitals_dict=hospitals_dict,
        seed=seed
    )

    fair_res = run_federated_simulation(
        strategy_type="fairness_aware",
        num_rounds=num_rounds,
        epochs_per_hospital=epochs_per_hospital,
        lr=lr,
        fairness_beta=fairness_beta,
        momentum_gamma=momentum_gamma,
        data_dir=data_dir,
        hospitals_dict=hospitals_dict,
        seed=seed
    )

    # Node-wise comparisons across all participating hospitals
    node_comparison = {}
    fed_nodes = fedavg_res.get("node_wise_metrics", {})
    fair_nodes = fair_res.get("node_wise_metrics", {})
    for hid in fed_nodes.keys():
        f_m = fed_nodes.get(hid, {})
        fa_m = fair_nodes.get(hid, {})
        node_comparison[hid] = {
            "sample_size": f_m.get("sample_size"),
            "fedavg_accuracy": f_m.get("accuracy"),
            "fairness_accuracy": fa_m.get("accuracy"),
            "fedavg_f1": f_m.get("f1"),
            "fairness_f1": fa_m.get("f1"),
            "fedavg_dpd": f_m.get("dpd"),
            "fairness_dpd": fa_m.get("dpd"),
            "fedavg_eod": f_m.get("eod"),
            "fairness_eod": fa_m.get("eod"),
            "fedavg_auroc": f_m.get("auroc"),
            "fairness_auroc": fa_m.get("auroc"),
            "fedavg_loss": f_m.get("loss"),
            "fairness_loss": fa_m.get("loss"),
        }

    return {
        "standard_fedavg": fedavg_res,
        "fairness_aware": fair_res,
        "node_comparison": node_comparison,
        "comparison_table": {
            "Accuracy": {
                "FedAvg": fedavg_res["final_metrics"].get("accuracy"),
                "Fairness-Aware": fair_res["final_metrics"].get("accuracy")
            },
            "F1-Score": {
                "FedAvg": fedavg_res["final_metrics"].get("f1"),
                "Fairness-Aware": fair_res["final_metrics"].get("f1")
            },
            "DPD (Demographic Parity)": {
                "FedAvg": fedavg_res["final_metrics"].get("dpd"),
                "Fairness-Aware": fair_res["final_metrics"].get("dpd")
            },
            "EOD (Equalized Odds)": {
                "FedAvg": fedavg_res["final_metrics"].get("eod"),
                "Fairness-Aware": fair_res["final_metrics"].get("eod")
            },
            "DIR (Disparate Impact)": {
                "FedAvg": fedavg_res["final_metrics"].get("dir"),
                "Fairness-Aware": fair_res["final_metrics"].get("dir")
            },
            "AUROC": {
                "FedAvg": fedavg_res["final_metrics"].get("auroc"),
                "Fairness-Aware": fair_res["final_metrics"].get("auroc")
            }
        }
    }
