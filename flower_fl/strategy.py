"""
Federated Aggregation Strategies for Flower:
1. Standard FedAvg (Sample-Volume Proportional Baseline)
2. FairnessAwareFedAvg (Dual-Tier Adaptive Momentum Aggregation - FedEquiMed D-TAMA)
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes, Parameters, Scalar,
    parameters_to_ndarrays, ndarrays_to_parameters
)

class FairnessAwareFedAvg(FedAvg):
    """
    Fairness-Aware Adaptive Momentum Aggregation Strategy for Flower.
    
    Subclasses flwr.server.strategy.FedAvg to mitigate non-IID institutional disparities:
    1. Tracks institutional bias momentum m_k across rounds:
       m_k^(t) = gamma * m_k^(t-1) + (1 - gamma) * Delta_k^(t)
    2. Calculates adaptive dual-tier aggregation weights:
       alpha_k = (n_k / N) * exp(-beta * (m_k / (m_bar + eps)))
       w_k = alpha_k / sum(alpha_i)
    
    Nodes contributing models with persistent demographic bias are smoothly down-weighted,
    preventing large biased hospitals from monopolizing global model updates.
    """
    def __init__(
        self,
        fairness_beta: float = 2.5,
        momentum_gamma: float = 0.80,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        evaluate_fn: Optional[Callable[[int, List[np.ndarray], Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
        self.fairness_beta = fairness_beta
        self.momentum_gamma = momentum_gamma
        
        # State tracking
        self.institutional_momentum: Dict[str, float] = {}
        self.weights_history: List[Dict[str, float]] = []
        self.disparity_history: List[Dict[str, float]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates client parameters using fairness-adaptive dual-tier momentum weighting.
        """
        if not results:
            return None, {}

        # If failures occurred and we have no successful results
        if not self.accept_failures and failures:
            return None, {}

        # 1. Parse client updates and metrics
        client_updates = []
        total_samples = 0
        eps = 1e-6

        for client_proxy, fit_res in results:
            cid = getattr(client_proxy, "cid", fit_res.metrics.get("client_id", "unknown"))
            params = parameters_to_ndarrays(fit_res.parameters)
            n_k = fit_res.num_examples
            dpd_k = abs(float(fit_res.metrics.get("dpd", 0.0)))

            total_samples += n_k
            client_updates.append({
                "cid": str(cid),
                "params": params,
                "num_examples": n_k,
                "dpd": dpd_k
            })

        if total_samples <= 0:
            total_samples = len(client_updates)

        # 2. Update D-TAMA institutional momentum m_k
        current_disparities = {}
        for c in client_updates:
            cid = c["cid"]
            curr_dpd = c["dpd"]
            current_disparities[cid] = curr_dpd
            prev_m = self.institutional_momentum.get(cid, curr_dpd)
            updated_m = self.momentum_gamma * prev_m + (1.0 - self.momentum_gamma) * curr_dpd
            self.institutional_momentum[cid] = updated_m

        self.disparity_history.append(current_disparities)

        # Average momentum across participating nodes
        mean_momentum = (
            sum(self.institutional_momentum.values()) / max(1, len(self.institutional_momentum))
        )

        # 3. Compute adaptive dual-tier weights
        raw_alphas = []
        for c in client_updates:
            cid = c["cid"]
            s_size = c["num_examples"]
            m_k = self.institutional_momentum.get(cid, 0.0)
            
            # Tier 1: Volume tier
            vol_tier = s_size / total_samples
            
            # Tier 2: Disparity resistance tier
            norm_disparity = m_k / (mean_momentum + eps)
            exponent = max(-30.0, min(0.0, -abs(self.fairness_beta) * norm_disparity))
            momentum_tier = math.exp(exponent)
            
            alpha = vol_tier * momentum_tier
            raw_alphas.append(alpha)

        sum_alpha = sum(raw_alphas)
        if sum_alpha <= 0 or math.isnan(sum_alpha):
            # Fallback to standard volume weighting
            client_weights = [c["num_examples"] / total_samples for c in client_updates]
        else:
            client_weights = [a / sum_alpha for a in raw_alphas]

        # Record client weights for visualization
        round_weights_dict = {
            c["cid"]: round(float(w), 4) for c, w in zip(client_updates, client_weights)
        }
        self.weights_history.append(round_weights_dict)

        # 4. Compute weighted average of model parameters
        num_layers = len(client_updates[0]["params"])
        aggregated_ndarrays = []

        for layer_idx in range(num_layers):
            layer_acc = np.zeros_like(client_updates[0]["params"][layer_idx], dtype=np.float32)
            for c, w in zip(client_updates, client_weights):
                layer_acc += float(w) * c["params"][layer_idx].astype(np.float32)
            aggregated_ndarrays.append(layer_acc)

        # 5. Package into Flower Parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        metrics_aggregated: Dict[str, Scalar] = {
            "mean_disparity": float(mean_momentum),
            "max_weight": float(max(client_weights)),
            "min_weight": float(min(client_weights))
        }

        return parameters_aggregated, metrics_aggregated
