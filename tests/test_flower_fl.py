"""
Unit Test Suite for Flower-based Federated Learning & Fairness Strategy
"""

import unittest
import numpy as np
import torch
import flwr as fl

from flower_fl.model import PyTorchLogisticRegression
from flower_fl.dataset_loader import load_all_hospital_datasets
from flower_fl.metrics import compute_clinical_fairness_metrics
from flower_fl.client import HospitalFlowerClient, LocalClientProxy
from flower_fl.strategy import FairnessAwareFedAvg
from flower_fl.simulation import run_federated_simulation, run_comparative_benchmark

class TestFlowerFederatedLearning(unittest.TestCase):

    def test_pytorch_model(self):
        """Tests PyTorch Logistic Regression forward pass and weight get/set."""
        model = PyTorchLogisticRegression(num_features=10)
        x = torch.randn(8, 10)
        probs = model.predict_proba(x)
        self.assertEqual(probs.shape, (8, 1))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        weights = model.get_weights()
        self.assertEqual(len(weights), 2)  # [weight, bias]
        self.assertEqual(weights[0].shape, (1, 10))
        self.assertEqual(weights[1].shape, (1,))

        # Update weights and verify
        new_w = [np.ones((1, 10)), np.array([2.5])]
        model.set_weights(new_w)
        updated_w = model.get_weights()
        np.testing.assert_allclose(updated_w[0], new_w[0])
        np.testing.assert_allclose(updated_w[1], new_w[1])

    def test_dataset_loader(self):
        """Tests loading MIMIC-IV hospital cohorts and global evaluation cohort."""
        hospitals, global_test = load_all_hospital_datasets(data_dir="data")
        self.assertIn("hospital_a", hospitals)
        self.assertIn("hospital_b", hospitals)
        self.assertIn("hospital_c", hospitals)
        
        # Verify sizes
        self.assertEqual(hospitals["hospital_a"]["sample_size"], 1200)
        self.assertEqual(hospitals["hospital_b"]["sample_size"], 800)
        self.assertEqual(hospitals["hospital_c"]["sample_size"], 1000)
        self.assertEqual(global_test["sample_size"], 1000)

        # Verify feature dimensions
        self.assertEqual(hospitals["hospital_a"]["X"].shape[1], 10)
        self.assertEqual(global_test["X"].shape[1], 10)

    def test_fairness_metrics(self):
        """Tests calculation of DPD, EOD, Accuracy, and F1."""
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
        y_prob = [0.9, 0.4, 0.2, 0.1, 0.8, 0.6, 0.7, 0.3]
        gender = [0, 0, 0, 0, 1, 1, 1, 1]  # 4 Female, 4 Male

        metrics = compute_clinical_fairness_metrics(y_true, y_pred, y_prob, gender)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("dpd", metrics)
        self.assertIn("eod", metrics)
        self.assertIn("subgroups", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_flower_client_proxy(self):
        """Tests Flower NumPyClient and LocalClientProxy bridge."""
        hospitals, _ = load_all_hospital_datasets(data_dir="data")
        client = HospitalFlowerClient("hospital_a", "Hospital A", hospitals["hospital_a"])
        proxy = LocalClientProxy("hospital_a", client)

        model = PyTorchLogisticRegression(num_features=10)
        params = fl.common.ndarrays_to_parameters(model.get_weights())
        fit_ins = fl.common.FitIns(parameters=params, config={"epochs": 1, "lr": 0.05})

        fit_res = proxy.fit(fit_ins)
        self.assertEqual(fit_res.status.code, fl.common.Code.OK)
        self.assertEqual(fit_res.num_examples, 1200)
        self.assertIn("dpd", fit_res.metrics)
        self.assertIn("loss", fit_res.metrics)

    def test_end_to_end_simulation(self):
        """Tests full 2-round simulation with both FedAvg and FairnessAware strategies."""
        res_fedavg = run_federated_simulation(
            strategy_type="fedavg",
            num_rounds=2,
            epochs_per_hospital=1,
            data_dir="data",
            seed=42
        )
        self.assertEqual(len(res_fedavg["trajectory"]), 3)  # Round 0 (init), Round 1, Round 2
        self.assertIn("accuracy", res_fedavg["final_metrics"])
        self.assertIn("f1", res_fedavg["final_metrics"])
        self.assertIn("dpd", res_fedavg["final_metrics"])
        self.assertIn("eod", res_fedavg["final_metrics"])

        res_fair = run_federated_simulation(
            strategy_type="fairness_aware",
            num_rounds=2,
            epochs_per_hospital=1,
            fairness_beta=2.5,
            momentum_gamma=0.8,
            data_dir="data",
            seed=42
        )
        self.assertEqual(len(res_fair["trajectory"]), 3)
        self.assertIn("accuracy", res_fair["final_metrics"])
        self.assertIn("dpd", res_fair["final_metrics"])

    def test_custom_hospital_integration(self):
        """Tests synthesizing custom hospital data and running 4-hospital Flower simulation."""
        from flower_fl.dataset_loader import (
            generate_synthetic_custom_hospital,
            parse_patient_dataframe
        )
        # Generate custom hospital D
        df_custom = generate_synthetic_custom_hospital(
            hospital_name="Hospital D (Maternity Clinic)",
            sample_size=300,
            female_pct=0.85,
            geriatric_pct=0.20,
            seed=999
        )
        self.assertEqual(len(df_custom), 300)
        parsed_custom = parse_patient_dataframe(df_custom)
        self.assertEqual(parsed_custom["sample_size"], 300)

        # Merge with existing hospitals
        base_hospitals, _ = load_all_hospital_datasets(data_dir="data")
        orig_count = len(base_hospitals)
        base_hospitals["hospital_d"] = parsed_custom
        self.assertEqual(len(base_hospitals), orig_count + 1)

        # Run FL simulation with all active hospital clients
        res_4h = run_federated_simulation(
            strategy_type="fairness_aware",
            num_rounds=2,
            epochs_per_hospital=1,
            hospitals_dict=base_hospitals,
            seed=42
        )
        self.assertEqual(res_4h["num_rounds"], 2)
        self.assertIn("accuracy", res_4h["final_metrics"])
        self.assertIn("dpd", res_4h["final_metrics"])
        self.assertIn("eod", res_4h["final_metrics"])
        
        # Verify custom client participated in aggregation
        weights_hist = res_4h["weights_history"]
        self.assertEqual(len(weights_hist[0]), len(base_hospitals))
        self.assertIn("hospital_d", weights_hist[0])

if __name__ == "__main__":
    unittest.main()
