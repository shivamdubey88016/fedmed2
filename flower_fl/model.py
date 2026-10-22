"""
PyTorch Logistic Regression Model for Clinical Risk Prediction
Supports local training with standard BCE loss or fairness-aware subgroup adaptive reweighting (FedIDA).
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

class PyTorchLogisticRegression(nn.Module):
    """
    Logistic Regression implemented in PyTorch for federated clinical risk prediction.
    Models probability of 30-day adverse clinical outcome: P(Y=1 | X) = sigmoid(w^T x + b).
    """
    def __init__(self, num_features: int = 10):
        super().__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 1)
        # Initialize with zeros for consistent FL initial state
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw linear logits."""
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes calibrated probability P(Y=1|X) via Sigmoid."""
        return torch.sigmoid(self.linear(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary classification predictions {0, 1}."""
        probs = self.predict_proba(x)
        return (probs >= threshold).long().squeeze(-1)

    def get_weights(self) -> List[np.ndarray]:
        """Returns model parameters as a list of NumPy ndarrays for Flower serialization."""
        return [
            self.linear.weight.detach().cpu().numpy().copy(),
            self.linear.bias.detach().cpu().numpy().copy()
        ]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Sets model parameters from NumPy ndarrays sent by Flower server."""
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weights[0]).float())
            self.linear.bias.copy_(torch.from_numpy(weights[1]).float())

    def train_local(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        sensitive_attr: Optional[torch.Tensor] = None,
        epochs: int = 5,
        lr: float = 0.05,
        batch_size: int = 32,
        use_reweighting: bool = False
    ) -> float:
        """
        Executes local gradient descent on hospital data.
        
        If use_reweighting is True (FedIDA):
          Computes sample-wise weights inversely proportional to demographic subgroup frequency:
          w_i = N / (2 * N_group)
        """
        self.train()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        n_samples = x_train.shape[0]

        # Compute sample weights if fairness reweighting is active
        sample_weights = torch.ones(n_samples, 1)
        if use_reweighting and sensitive_attr is not None:
            unique_groups = torch.unique(sensitive_attr)
            for g in unique_groups:
                mask = (sensitive_attr == g)
                n_g = mask.sum().item()
                if n_g > 0:
                    w_g = n_samples / (len(unique_groups) * n_g)
                    sample_weights[mask] = w_g

        last_epoch_loss = 0.0
        for _ in range(epochs):
            # Shuffle indices
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            num_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = perm[start_idx:start_idx + batch_size]
                bx = x_train[batch_idx]
                by = y_train[batch_idx].unsqueeze(1)
                bw = sample_weights[batch_idx]

                optimizer.zero_grad()
                logits = self(bx)
                
                # BCE with Logits loss
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, by, reduction='none'
                )
                loss = (bce * bw).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            last_epoch_loss = epoch_loss / max(1, num_batches)

        return float(last_epoch_loss)
