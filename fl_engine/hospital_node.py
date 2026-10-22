"""
Hospital Node Implementation for Federated Healthcare System
Maintains private local patient records, executes local model training with
adaptive subgroup reweighting, conducts secure bias auditing, and exports sanitized payloads.
"""

from typing import List, Dict, Any, Optional
from fl_engine.models import ClinicalRiskModel
from fl_engine.bias_detector import BiasDetector
from fl_engine.privacy_engine import PrivacyEngine

class HospitalNode:
    def __init__(
        self,
        hospital_id: str,
        name: str,
        dataset: List[Dict[str, Any]],
        num_features: int = 10,
        description: str = "",
        privacy_engine: Optional[PrivacyEngine] = None
    ):
        self.hospital_id = hospital_id
        self.name = name
        self.description = description
        # Private local dataset partitioned for this hospital node
        self._private_dataset = dataset or []
        self._sample_size = len(self._private_dataset)
        self.num_features = num_features
        self.local_model = ClinicalRiskModel(num_features=num_features)
        self.latest_exported_payload: Optional[Dict[str, Any]] = None
        self.latest_test_result: Optional[Dict[str, Any]] = None
        self.training_history: List[Dict[str, Any]] = []
        self._cached_summary = self._compute_summary(self._private_dataset)
        self.privacy_engine = privacy_engine or PrivacyEngine(clipping_norm=1.0, noise_multiplier=0.02)
        if len(self._private_dataset) > 0:
            self._cached_audit = BiasDetector.audit_model(self.local_model, self._private_dataset)
        else:
            self._cached_audit = {
                "status": "ready",
                "overall_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auroc": 0.50,
                "sample_size": 0,
                "gender_audit": {
                    "demographic_parity_difference": 0.0,
                    "equalized_odds_difference": 0.0,
                    "disparate_impact_ratio": 1.0,
                    "group_stats": {}
                },
                "age_audit": {
                    "demographic_parity_difference": 0.0,
                    "equalized_odds_difference": 0.0,
                    "disparate_impact_ratio": 1.0,
                    "group_stats": {}
                },
                "composite_disparity": 0.0
            }

    @property
    def sample_size(self) -> int:
        return self._sample_size

    def _compute_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(data)
        if n == 0:
            return {
                "hospital_id": self.hospital_id,
                "name": self.name,
                "sample_size": self._sample_size,
                "demographics": {
                    "gender": {"female": 0, "male": 0, "female_pct": 50.0},
                    "age_group": {"young_under_60": 0, "senior_60_plus": 0, "senior_pct": 50.0},
                    "ethnicity": {0: 0},
                    "adverse_outcome_rate": 0.0
                }
            }
        female_count = sum(1 for d in data if d["gender"] == 0)
        male_count = n - female_count
        young_count = sum(1 for d in data if d["age_group"] == 0)
        senior_count = n - young_count
        pos_outcomes = sum(1 for d in data if d["label"] == 1)

        eth_counts = {}
        for d in data:
            eth = d["ethnicity"]
            eth_counts[eth] = eth_counts.get(eth, 0) + 1

        return {
            "hospital_id": self.hospital_id,
            "name": self.name,
            "description": self.description,
            "sample_size": n,
            "demographics": {
                "gender": {"female": female_count, "male": male_count, "female_pct": round(female_count / n * 100, 1)},
                "age_group": {"young_under_60": young_count, "senior_60_plus": senior_count, "senior_pct": round(senior_count / n * 100, 1)},
                "ethnicity": eth_counts,
                "adverse_outcome_rate": round(pos_outcomes / n * 100, 1)
            }
        }

    def purge_raw_records(self):
        """Strict Privacy Enforcement: Discards local microdata, retaining summary stats."""
        self._private_dataset = []

    def get_summary(self) -> Dict[str, Any]:
        """Provides summary metrics without storing or leaking raw microdata."""
        summary = dict(self._cached_summary)
        summary["hospital_id"] = self.hospital_id
        summary["name"] = self.name
        summary["description"] = self.description
        summary["sample_size"] = self._sample_size
        summary["latest_test_result"] = self.latest_test_result
        return summary

    def get_sample_records(self, count: int = 5) -> List[Dict[str, Any]]:
        """Returns preview records for local hospital staff inspection."""
        samples = []
        for d in self._private_dataset[:count]:
            samples.append({
                "gender": "Female" if d["gender"] == 0 else "Male",
                "age_group": "< 60 yrs" if d["age_group"] == 0 else ">= 60 yrs",
                "ethnicity": f"Cohort {chr(65 + d['ethnicity'])}",
                "label": "Adverse Event (High Risk)" if d["label"] == 1 else "Normal (Low Risk)",
                "raw_vitals": d["raw_data"]
            })
        return samples

    def audit_bias(self) -> Dict[str, Any]:
        """Runs privacy-preserving local bias detector on current local model."""
        if self._private_dataset and len(self._private_dataset) > 0:
            self._cached_audit = BiasDetector.audit_model(self.local_model, self._private_dataset)
        return self._cached_audit

    def calculate_adaptive_sample_weights(self, mitigation_intensity: float = 1.0) -> List[float]:
        """
        Adaptive In-Processing Mitigation: Subgroup Balancing (Kamiran-Calders / FedIDA).
        Assigns weights to balance representation across subgroups:
        w(y=c, a=g) = [P(Y=c) * P(A=g)] / P(Y=c, A=g)
        This reduces representation imbalance during local gradient optimization.
        """
        n = len(self._private_dataset)
        if n == 0:
            return []

        y_counts = {0: 0, 1: 0}
        group_counts = {}
        cell_counts = {}

        for d in self._private_dataset:
            y = d["label"]
            g = d["gender"]
            y_counts[y] = y_counts.get(y, 0) + 1
            group_counts[g] = group_counts.get(g, 0) + 1
            cell = (y, g)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1

        weights = []
        for d in self._private_dataset:
            y = d["label"]
            g = d["gender"]
            cell = (y, g)

            p_y = y_counts[y] / n
            p_g = group_counts[g] / n
            p_cell = cell_counts[cell] / n if cell_counts[cell] > 0 else 1.0 / n

            base_w = (p_y * p_g) / p_cell
            final_w = 1.0 + mitigation_intensity * (base_w - 1.0)
            final_w = max(0.2, min(5.0, final_w))
            weights.append(final_w)

        return weights

    def train_local_model(
        self,
        global_weights: Optional[Dict[str, Any]] = None,
        epochs: int = 5,
        lr: float = 0.08,
        use_adaptive_mitigation: bool = True,
        mitigation_intensity: float = 1.2,
        method: str = "fairfed",
        dp_noise_multiplier: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes local model training on private records.
        Supports:
        - 'fedavg': Standard baseline SGD
        - 'fairfed': Dynamic subgroup reweighting (FedIDA)
        - 'fedequimed': ClinDisentangleNet with Covariate-Adjusted Risk Parity (CARP-Loss)
        """
        if global_weights is not None:
            self.local_model.set_weights(global_weights)

        pre_audit = self.audit_bias()

        if len(self._private_dataset) == 0:
            self.latest_exported_payload = {
                "hospital_id": self.hospital_id,
                "hospital_name": self.name,
                "sample_size": 0,
                "weights": self.local_model.get_weights(),
                "delta": {"weights": [0.0] * self.num_features, "bias": 0.0},
                "loss_history": [],
                "pre_audit": pre_audit,
                "post_audit": pre_audit,
                "adaptive_mitigation_active": use_adaptive_mitigation,
                "privacy_verification": {
                    "raw_records_shared_to_server": 0,
                    "differential_privacy_enabled": True,
                    "protection_mechanism": "Differential privacy mechanism prototype",
                    "note": "Raw patient microdata is not transmitted to central aggregator."
                }
            }
            return self.latest_exported_payload

        # Feature and label extraction from private data
        X = [d["features"] for d in self._private_dataset]
        y = [d["label"] for d in self._private_dataset]
        genders = [d["gender"] for d in self._private_dataset]

        if method.lower() == "fedequimed":
            from fl_engine.models import ClinDisentangleNet
            if not isinstance(self.local_model, ClinDisentangleNet):
                dis_model = ClinDisentangleNet(num_features=self.num_features)
                dis_model.set_weights(self.local_model.get_weights())
                self.local_model = dis_model

            loss_history = self.local_model.fit_fedequimed(
                X=X,
                y=y,
                genders=genders,
                epochs=epochs,
                lr=lr,
                carp_lambda=0.60,
                ortho_lambda=0.20
            )
        elif method.lower() == "ablation_disentangle":
            from fl_engine.models import ClinDisentangleNet
            if not isinstance(self.local_model, ClinDisentangleNet):
                dis_model = ClinDisentangleNet(num_features=self.num_features)
                dis_model.set_weights(self.local_model.get_weights())
                self.local_model = dis_model
            # Disentangle only (no CARP loss)
            loss_history = self.local_model.fit_fedequimed(
                X=X, y=y, genders=genders, epochs=epochs, lr=lr, carp_lambda=0.0, ortho_lambda=0.20
            )
        elif method.lower() == "ablation_carp":
            from fl_engine.models import ClinDisentangleNet
            if not isinstance(self.local_model, ClinDisentangleNet):
                dis_model = ClinDisentangleNet(num_features=self.num_features)
                dis_model.set_weights(self.local_model.get_weights())
                self.local_model = dis_model
            # CARP only (no ortho constraint)
            loss_history = self.local_model.fit_fedequimed(
                X=X, y=y, genders=genders, epochs=epochs, lr=lr, carp_lambda=0.60, ortho_lambda=0.0
            )
        else:
            if use_adaptive_mitigation:
                sample_weights = self.calculate_adaptive_sample_weights(mitigation_intensity)
            else:
                sample_weights = [1.0] * len(X)

            loss_history = self.local_model.fit(
                X=X,
                y=y,
                sample_weights=sample_weights,
                epochs=epochs,
                lr=lr
            )

        post_audit = self.audit_bias()

        # Compute parameter update delta
        current_weights = self.local_model.get_weights()
        if global_weights is not None:
            delta_w = [cw - gw for cw, gw in zip(current_weights["weights"], global_weights["weights"])]
            delta_b = current_weights["bias"] - global_weights["bias"]
        else:
            delta_w = list(current_weights["weights"])
            delta_b = current_weights["bias"]

        # Apply Differential Privacy (L2 clipping + Gaussian noise)
        if dp_noise_multiplier is not None:
            self.privacy_engine.noise_multiplier = float(dp_noise_multiplier)

        private_delta_w, private_delta_b, original_norm = self.privacy_engine.clip_and_perturb(
            delta_w, delta_b
        )

        # Apply private delta back to weights for export
        if global_weights is not None:
            sanitized_weights = {
                "weights": [gw + pdw for gw, pdw in zip(global_weights["weights"], private_delta_w)],
                "bias": global_weights["bias"] + private_delta_b
            }
        else:
            sanitized_weights = {
                "weights": private_delta_w,
                "bias": private_delta_b
            }

        # Privacy summary
        dp_summary = self.privacy_engine.get_summary(
            sample_size=self.sample_size,
            batch_size=min(16, self.sample_size),
            rounds=len(self.training_history) + 1,
            local_epochs=epochs
        )

        self.latest_exported_payload = {
            "hospital_id": self.hospital_id,
            "hospital_name": self.name,
            "sample_size": self.sample_size,
            "weights": sanitized_weights,
            "delta": {"weights": private_delta_w, "bias": private_delta_b},
            "loss_history": loss_history,
            "pre_audit": {
                "accuracy": pre_audit["overall_accuracy"],
                "gender_dpd": pre_audit["gender_audit"]["demographic_parity_difference"],
                "age_dpd": pre_audit["age_audit"]["demographic_parity_difference"],
                "equalized_odds_diff": pre_audit["gender_audit"]["equalized_odds_difference"],
                "composite_disparity": pre_audit["composite_disparity"]
            },
            "post_audit": {
                "accuracy": post_audit["overall_accuracy"],
                "gender_dpd": post_audit["gender_audit"]["demographic_parity_difference"],
                "age_dpd": post_audit["age_audit"]["demographic_parity_difference"],
                "equalized_odds_diff": post_audit["gender_audit"]["equalized_odds_difference"],
                "composite_disparity": post_audit["composite_disparity"],
                "gender_stats": post_audit["gender_audit"]["group_stats"],
                "age_stats": post_audit["age_audit"]["group_stats"]
            },
            "adaptive_mitigation_active": use_adaptive_mitigation,
            "privacy_verification": {
                "raw_records_shared": 0,
                "raw_records_shared_to_server": 0,
                "differential_privacy_enabled": self.privacy_engine.noise_multiplier > 0,
                "gradient_clipping_l2": self.privacy_engine.clipping_norm,
                "noise_std_dev_sigma": self.privacy_engine.noise_multiplier,
                "original_delta_l2_norm": original_norm,
                "dp_summary": dp_summary,
                "data_leakage_risk": "Mitigated (Raw records never transmitted offsite; DP noise applied)",
                "regulatory_design": "Designed to align with HIPAA Safe Harbor and GDPR data minimization requirements."
            }
        }

        self.training_history.append({
            "round": len(self.training_history) + 1,
            "loss": loss_history[-1] if loss_history else 0.0,
            "post_accuracy": post_audit["overall_accuracy"],
            "gender_dpd": post_audit["gender_audit"]["demographic_parity_difference"],
            "age_dpd": post_audit["age_audit"]["demographic_parity_difference"],
            "adaptive_mitigation": use_adaptive_mitigation
        })

        self._cached_audit = post_audit
        return self.latest_exported_payload

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node metadata and dataset for state persistence."""
        return {
            "hospital_id": self.hospital_id,
            "name": self.name,
            "description": self.description,
            "num_features": self.num_features,
            "sample_size": self._sample_size,
            "dataset": self._private_dataset,
            "training_history": self.training_history,
            "latest_test_result": self.latest_test_result,
            "weights": self.local_model.get_weights(),
            "cached_audit": self._cached_audit
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HospitalNode":
        """Reconstructs hospital node from serialized dictionary."""
        node = cls(
            hospital_id=data["hospital_id"],
            name=data["name"],
            dataset=data.get("dataset", []),
            num_features=data.get("num_features", 10),
            description=data.get("description", "")
        )
        if "weights" in data and data["weights"]:
            node.local_model.set_weights(data["weights"])
        node.training_history = data.get("training_history", [])
        node.latest_test_result = data.get("latest_test_result")
        if "cached_audit" in data and data["cached_audit"]:
            node._cached_audit = data["cached_audit"]
        else:
            node.audit_bias()
        return node
