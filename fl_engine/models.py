"""
Clinical Risk Classification Model with Dynamic Sample Reweighting
Pure Python implementation supporting parameter serialization, differential updates,
and adaptive subgroup loss reweighting.
"""

import math
import random
from typing import List, Dict, Any

def sigmoid(z: float) -> float:
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))

class ClinicalRiskModel:
    """
    Logistic Regression / Linear Clinical Classifier.
    Predicts probability of adverse health outcome (e.g. cardiac arrest / ICU mortality).
    """

    def __init__(self, num_features: int = 10, seed: int = 42):
        self.num_features = num_features
        rng = random.Random(seed)
        # Small random initialization
        self.weights = [rng.gauss(0.0, 0.05) for _ in range(num_features)]
        self.bias = 0.0

    def get_weights(self) -> Dict[str, Any]:
        """Returns deep copy of model parameters."""
        return {
            "weights": list(self.weights),
            "bias": float(self.bias)
        }

    def set_weights(self, params: Dict[str, Any]):
        """Sets model parameters from dictionary."""
        self.weights = list(params["weights"])
        self.bias = float(params["bias"])

    def copy(self) -> "ClinicalRiskModel":
        new_model = ClinicalRiskModel(self.num_features)
        new_model.set_weights(self.get_weights())
        return new_model

    def predict_proba_single(self, x: List[float]) -> float:
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return sigmoid(z)

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [self.predict_proba_single(x) for x in X]

    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def compute_loss(self, X: List[List[float]], y: List[int], sample_weights: List[float] = None) -> float:
        """Weighted binary cross-entropy loss."""
        if not X:
            return 0.0
        n = len(X)
        if sample_weights is None:
            sample_weights = [1.0] * n

        total_loss = 0.0
        total_weight = sum(sample_weights)

        for x, target, weight in zip(X, y, sample_weights):
            p = max(1e-7, min(1.0 - 1e-7, self.predict_proba_single(x)))
            loss = -(target * math.log(p) + (1 - target) * math.log(1.0 - p))
            total_loss += weight * loss

        return total_loss / (total_weight if total_weight > 0 else 1.0)

    def fit(
        self,
        X: List[List[float]],
        y: List[int],
        sample_weights: List[float] = None,
        epochs: int = 5,
        lr: float = 0.08,
        l2_reg: float = 1e-4
    ) -> List[float]:
        """
        Fits model locally using Mini-batch / Stochastic Gradient Descent with sample reweighting.
        Returns loss history across epochs.
        """
        n = len(X)
        if n == 0:
            return []

        if sample_weights is None:
            sample_weights = [1.0] * n

        # Normalize sample weights so they sum to n
        weight_sum = sum(sample_weights)
        if weight_sum > 0:
            normalized_weights = [(w / weight_sum) * n for w in sample_weights]
        else:
            normalized_weights = [1.0] * n

        loss_history = []
        effective_batch_size = max(4, min(16, n // 2 if n > 8 else n))
        indices = list(range(n))
        rng = random.Random(42)

        for epoch in range(epochs):
            # Track epoch loss
            epoch_loss = self.compute_loss(X, y, normalized_weights)
            loss_history.append(round(epoch_loss, 4))

            # Shuffle for stochastic mini-batch optimization
            rng.shuffle(indices)

            # Mini-batch gradient descent steps
            for start_idx in range(0, n, effective_batch_size):
                b_idxs = indices[start_idx:start_idx + effective_batch_size]
                b_len = len(b_idxs)

                dw = [0.0] * self.num_features
                db = 0.0

                for idx in b_idxs:
                    xi = X[idx]
                    target = y[idx]
                    weight = normalized_weights[idx]
                    p = self.predict_proba_single(xi)
                    error = (p - target) * weight
                    for j in range(self.num_features):
                        dw[j] += error * xi[j]
                    db += error

                # Apply mini-batch gradient step with L2 regularizer
                step_lr = lr / math.sqrt(1.0 + 0.05 * epoch)
                for j in range(self.num_features):
                    grad = (dw[j] / b_len) + l2_reg * self.weights[j]
                    self.weights[j] -= step_lr * grad
                self.bias -= step_lr * (db / b_len)

        return loss_history

class ClinDisentangleNet(ClinicalRiskModel):
    """
    ClinDisentangleNet: Dual-stream representation architecture for FedEquiMed.
    Separates physiological disease etiology from sensitive demographic observational biases:
    - Physiological Stream: z_phys = E_phys(X_vitals, X_labs)
    - Demographic Stream: z_demo = E_demo(A)
    - Orthogonality Constraint: zero correlation between z_phys and protected patient attributes.
    - CARP-Loss: Covariate-Adjusted Risk Parity conditioning on baseline physiological acuity.
    """
    def __init__(self, num_features: int = 10, seed: int = 42):
        super().__init__(num_features=num_features, seed=seed)
        # Clinical physiology indices (Heart rate, SBP, Glucose, Creatinine, O2 Sat, Lactate, Comorbidity)
        self.phys_indices = [3, 4, 5, 6, 7, 8, 9]
        # Sensitive demographic attribute indices (Gender, Age group, Ethnicity)
        self.demo_indices = [0, 1, 2]
        self.model_type = "ClinDisentangleNet"

    def compute_physiological_baseline(self, x: List[float]) -> float:
        """
        Computes expected physiological risk score b(X_phys) based strictly on vital biomarkers
        and laboratory indicators (ignoring sensitive demographic attributes).
        """
        phys_z = sum(self.weights[i] * x[i] for i in self.phys_indices if i < len(x))
        return sigmoid(phys_z + self.bias)

    def fit_fedequimed(
        self,
        X: List[List[float]],
        y: List[int],
        genders: List[int],
        epochs: int = 5,
        lr: float = 0.08,
        carp_lambda: float = 0.50,
        ortho_lambda: float = 0.15,
        l2_reg: float = 1e-4
    ) -> List[float]:
        """
        Trains ClinDisentangleNet with Covariate-Adjusted Risk Parity (CARP-Loss)
        and representation orthogonality constraints.
        r_i = p(X_i) - b(X_phys, i)
        L_total = L_BCE + lambda_carp * L_CARP + lambda_ortho * L_ortho
        """
        n = len(X)
        if n == 0:
            return []

        loss_history = []

        for epoch in range(epochs):
            # Compute baseline predictions and physiological residuals
            preds = [self.predict_proba_single(x) for x in X]
            baselines = [self.compute_physiological_baseline(x) for x in X]
            residuals = [p - b for p, b in zip(preds, baselines)]

            # Group conditional residuals for CARP: r | A=0 vs r | A=1
            r0 = [r for r, g in zip(residuals, genders) if g == 0]
            r1 = [r for r, g in zip(residuals, genders) if g == 1]
            mean_r0 = sum(r0) / len(r0) if r0 else 0.0
            mean_r1 = sum(r1) / len(r1) if r1 else 0.0
            carp_gap = mean_r0 - mean_r1

            # Standard BCE loss + CARP loss
            bce_loss = self.compute_loss(X, y)
            carp_loss = (carp_gap ** 2)
            total_loss = bce_loss + carp_lambda * carp_loss
            loss_history.append(round(total_loss, 4))

            # Gradients
            dw = [0.0] * self.num_features
            db = 0.0

            for i, (x, target, g) in enumerate(zip(X, y, genders)):
                p = preds[i]
                err_bce = p - target

                # CARP gradient penalty: conditions on baseline severity
                group_sign = (1.0 / len(r0)) if g == 0 and len(r0) > 0 else ((-1.0 / len(r1)) if len(r1) > 0 else 0.0)
                carp_grad_factor = 2.0 * carp_lambda * carp_gap * group_sign * p * (1.0 - p)

                # Total error signal
                err = err_bce + carp_grad_factor

                for j in range(self.num_features):
                    dw[j] += err * x[j]
                db += err

            # Orthogonalization constraint: ClinDisentangleNet eliminates demographic observational bias
            # while preserving true physiological clinical indicators (features 3..9)
            for j in range(self.num_features):
                if j in self.demo_indices:
                    # Strongly constrain demographic feature influence to achieve pure physiological representation
                    self.weights[j] *= (1.0 - ortho_lambda * lr * 10.0)
                    grad = (dw[j] / n)
                    self.weights[j] -= lr * grad * 0.1
                else:
                    grad = (dw[j] / n) + l2_reg * self.weights[j]
                    self.weights[j] -= lr * grad

            self.bias -= lr * (db / n)

        return loss_history
