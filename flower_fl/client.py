"""
Flower Hospital Client Implementation
Implements flwr.client.NumPyClient with PyTorch training and fairness disparity auditing.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.client import NumPyClient
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
)

from flower_fl.model import PyTorchLogisticRegression
from flower_fl.metrics import compute_clinical_fairness_metrics

class HospitalFlowerClient(NumPyClient):
    """
    Flower Client representing a hospital institution (e.g. Hospital A, B, or C).
    Trains local PyTorch logistic regression model without exposing patient data.
    """
    def __init__(
        self,
        client_id: str,
        hospital_name: str,
        data: Dict[str, Any],
        use_adaptive_reweighting: bool = False
    ):
        self.client_id = client_id
        self.hospital_name = hospital_name
        self.data = data
        self.use_adaptive_reweighting = use_adaptive_reweighting
        
        self.X = data["X"]
        self.y = data["y"]
        self.gender = data["gender"]
        self.sample_size = data["sample_size"]
        
        # Initialize local PyTorch model
        self.model = PyTorchLogisticRegression(num_features=self.X.shape[1])

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns local model parameters to Flower."""
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Receives global parameters from Flower server, runs local gradient descent,
        evaluates local demographic disparity, and returns updated parameters + metrics.
        """
        # 1. Update local model with global parameters
        self.model.set_weights(parameters)

        # 2. Extract hyperparameters from config
        epochs = int(config.get("epochs", 5))
        lr = float(config.get("lr", 0.05))
        batch_size = int(config.get("batch_size", 32))
        use_reweight = bool(config.get("use_reweighting", self.use_adaptive_reweighting))

        # 3. Local PyTorch training
        loss = self.model.train_local(
            x_train=self.X,
            y_train=self.y,
            sensitive_attr=self.gender,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            use_reweighting=use_reweight
        )

        # 4. Audit local fairness metrics on institutional data
        self.model.eval()
        with torch.no_grad():
            preds = self.model.predict(self.X)
            probs = self.model.predict_proba(self.X)
            metrics = compute_clinical_fairness_metrics(
                y_true=self.y,
                y_pred=preds,
                y_prob=probs,
                sensitive_attr=self.gender,
                attribute_name="gender"
            )

        # Return updated parameters, dataset size, and local audit metrics
        client_metrics = {
            "loss": float(loss),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "dpd": float(metrics.get("dpd", 0.0)),
            "eod": float(metrics.get("eod", 0.0)),
            "client_id": self.client_id
        }

        return self.model.get_weights(), self.sample_size, client_metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluates global parameters on local hospital test data."""
        self.model.set_weights(parameters)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, self.y.unsqueeze(1)
            ).item()
            preds = self.model.predict(self.X)
            acc = float((preds == self.y).float().mean().item())

        return float(bce), self.sample_size, {"accuracy": acc}


class LocalClientProxy(ClientProxy):
    """
    In-memory Flower ClientProxy that bridges HospitalFlowerClient to Flower Server.
    Enables instant simulation without requiring Ray or external daemons.
    """
    def __init__(self, cid: str, client: NumPyClient):
        super().__init__(cid)
        self.client = client

    def get_parameters(self, ins, timeout=None, group_id=None):
        params = self.client.get_parameters(ins.config)
        return ndarrays_to_parameters(params)

    def fit(self, ins: FitIns, timeout: float | None = None, group_id: int | None = None) -> FitRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        out_ndarrays, num_examples, metrics = self.client.fit(ndarrays, ins.config)
        return FitRes(
            status=Status(Code.OK, "Success"),
            parameters=ndarrays_to_parameters(out_ndarrays),
            num_examples=num_examples,
            metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns, timeout: float | None = None, group_id: int | None = None) -> EvaluateRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        loss, num_examples, metrics = self.client.evaluate(ndarrays, ins.config)
        return EvaluateRes(
            status=Status(Code.OK, "Success"),
            loss=float(loss),
            num_examples=num_examples,
            metrics=metrics
        )

    def get_properties(self, ins, timeout=None, group_id=None):
        pass

    def reconnect(self, ins, timeout=None, group_id=None):
        pass
