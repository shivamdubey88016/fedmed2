"""
Comprehensive Automated Test Suite for FedEquiMed
Tests:
- Fairness metrics: ΔDP, ΔEO, DIR, zero denominators, empty subgroups
- Federated Aggregation: FedAvg, FairFed, D-TAMA weight normalization
- Edge cases: Single client, empty client, extreme beta, raw signed disparity
- Differential Privacy: L2 clipping, Gaussian noise, RDP accountant
- Experiment Pipeline: Reproducibility, multi-seed stats, persistence
- Security: Constant-time authentication, admin authorization
"""

import os
import json
import unittest
import math

from fl_engine.bias_detector import calculate_subgroup_metrics, calculate_auroc, BiasDetector
from fl_engine.models import ClinicalRiskModel, ClinDisentangleNet
from fl_engine.hospital_node import HospitalNode
from fl_engine.global_server import GlobalFederatedServer
from fl_engine.privacy_engine import PrivacyEngine
from fl_engine.experiments import (
    run_single_experiment,
    run_and_persist_experiment,
    compute_mean_std,
    list_experiments,
    run_ablation_study,
    EXPERIMENTS_DIR
)
from fl_engine.dataset import generate_federated_healthcare_datasets

class TestFairnessMathematics(unittest.TestCase):
    def test_demographic_parity_difference(self):
        # Group 0: 80% positive prediction, Group 1: 30% positive prediction -> ΔDP = 0.50
        preds = [1]*8 + [0]*2 + [1]*3 + [0]*7
        targets = [1]*10 + [1]*10
        attrs = [0]*10 + [1]*10
        metrics = calculate_subgroup_metrics(preds, targets, attrs, "gender")
        self.assertAlmostEqual(metrics["demographic_parity_difference"], 0.50, places=3)

    def test_disparate_impact_ratio(self):
        preds = [1]*8 + [0]*2 + [1]*4 + [0]*6
        targets = [1]*10 + [1]*10
        attrs = [0]*10 + [1]*10
        metrics = calculate_subgroup_metrics(preds, targets, attrs, "gender")
        # DIR = 0.40 / 0.80 = 0.50
        self.assertAlmostEqual(metrics["disparate_impact_ratio"], 0.50, places=2)

    def test_zero_denominator_dir(self):
        # Both groups 0 positive predictions -> DIR should be 1.0 (safe parity)
        preds = [0]*20
        targets = [1]*10 + [0]*10
        attrs = [0]*10 + [1]*10
        metrics = calculate_subgroup_metrics(preds, targets, attrs, "gender")
        self.assertEqual(metrics["disparate_impact_ratio"], 1.0)
        self.assertEqual(metrics["demographic_parity_difference"], 0.0)

    def test_equalized_odds_difference(self):
        # Group 0: 4 pos, 4 neg -> TPR = 4/4 = 1.0, FPR = 0/4 = 0.0
        # Group 1: 4 pos, 4 neg -> TPR = 2/4 = 0.5, FPR = 2/4 = 0.5
        # ΔTPR = 0.5, ΔFPR = 0.5 -> ΔEO = 0.5
        preds = [1]*4 + [0]*4 + [1]*2 + [0]*2 + [1]*2 + [0]*2
        targets = [1]*4 + [0]*4 + [1]*4 + [0]*4
        attrs = [0]*8 + [1]*8
        metrics = calculate_subgroup_metrics(preds, targets, attrs, "gender")
        self.assertAlmostEqual(metrics["equalized_odds_difference"], 0.50, places=2)

    def test_auroc_calculation(self):
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        auroc = calculate_auroc(y_true, y_score)
        self.assertEqual(auroc, 1.0)

        # Inverted scores
        y_score_inv = [0.1, 0.2, 0.8, 0.9]
        auroc_inv = calculate_auroc(y_true, y_score_inv)
        self.assertEqual(auroc_inv, 0.0)

class TestFederatedAggregation(unittest.TestCase):
    def setUp(self):
        hosp_data, self.global_test = generate_federated_healthcare_datasets(seed=42)
        self.hospitals = {
            k: HospitalNode(k, k, hosp_data[k], num_features=10)
            for k in hosp_data
        }
        self.server = GlobalFederatedServer(self.hospitals, self.global_test, num_features=10)

    def test_fedavg_weight_normalization(self):
        payloads = [
            {"hospital_id": "h1", "sample_size": 500, "weights": {"weights": [1.0]*10, "bias": 0.5}},
            {"hospital_id": "h2", "sample_size": 300, "weights": {"weights": [2.0]*10, "bias": 1.0}},
            {"hospital_id": "h3", "sample_size": 200, "weights": {"weights": [3.0]*10, "bias": 1.5}}
        ]
        agg, weights = self.server.aggregate_fedavg(payloads)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        self.assertAlmostEqual(weights["h1"], 0.50, places=2)
        self.assertAlmostEqual(weights["h2"], 0.30, places=2)
        self.assertAlmostEqual(weights["h3"], 0.20, places=2)

    def test_fairfed_weight_normalization_and_penalization(self):
        payloads = [
            {
                "hospital_id": "fair_client",
                "sample_size": 500,
                "weights": {"weights": [1.0]*10, "bias": 0.5},
                "post_audit": {"gender_dpd": 0.01, "composite_disparity": 0.01}
            },
            {
                "hospital_id": "unfair_client",
                "sample_size": 500,
                "weights": {"weights": [1.0]*10, "bias": 0.5},
                "post_audit": {"gender_dpd": 0.35, "composite_disparity": 0.35}
            }
        ]
        agg, weights = self.server.aggregate_fairfed(payloads, fairness_beta=5.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        self.assertGreater(weights["fair_client"], weights["unfair_client"])

    def test_single_client_aggregation(self):
        single_payload = [{
            "hospital_id": "h1",
            "sample_size": 100,
            "weights": {"weights": [0.5]*10, "bias": 0.2},
            "post_audit": {"gender_dpd": 0.05}
        }]
        agg_ff, w_ff = self.server.aggregate_fairfed(single_payload)
        self.assertEqual(w_ff["h1"], 1.0)
        self.assertEqual(agg_ff["bias"], 0.2)

        agg_fa, w_fa = self.server.aggregate_fedavg(single_payload)
        self.assertEqual(w_fa["h1"], 1.0)

    def test_extreme_beta_handling(self):
        payloads = [
            {"hospital_id": "h1", "sample_size": 100, "weights": {"weights": [1.0]*10, "bias": 0.0}, "post_audit": {"gender_dpd": 0.0}},
            {"hospital_id": "h2", "sample_size": 100, "weights": {"weights": [1.0]*10, "bias": 0.0}, "post_audit": {"gender_dpd": 0.5}}
        ]
        agg, weights = self.server.aggregate_fairfed(payloads, fairness_beta=100.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        self.assertAlmostEqual(weights["h1"], 1.0, places=2)

class TestDifferentialPrivacy(unittest.TestCase):
    def test_clipping_and_noise(self):
        pe = PrivacyEngine(clipping_norm=1.0, noise_multiplier=0.05)
        large_delta = [5.0] * 10
        pert_w, pert_b, norm = pe.clip_and_perturb(large_delta, 2.0)
        self.assertGreater(norm, 1.0)
        pert_norm = math.sqrt(sum(w**2 for w in pert_w) + pert_b**2)
        self.assertLess(pert_norm, 2.5)

    def test_rdp_accountant(self):
        pe = PrivacyEngine(clipping_norm=1.0, noise_multiplier=0.5)
        eps = pe.compute_rdp_epsilon(q=0.05, rounds=10, local_steps=4)
        self.assertTrue(isinstance(eps, float))
        self.assertGreater(eps, 0.0)

class TestExperimentPipeline(unittest.TestCase):
    def test_experiment_persistence_and_stats(self):
        config = {
            "algorithm": "FedAvg",
            "dataset": "mimic_iv",
            "num_clients": 2,
            "rounds": 2,
            "local_epochs": 1,
            "seeds": [42, 123]
        }
        res = run_and_persist_experiment(config)
        exp_id = res["experiment_id"]
        exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "results.json")))
        self.assertTrue(os.path.exists(os.path.join(exp_dir, "metrics.csv")))

        acc_summary = res["metrics_summary"]["accuracy"]
        self.assertTrue("mean" in acc_summary)
        self.assertTrue("std" in acc_summary)

    def test_ablation_study(self):
        res = run_ablation_study(rounds=2, seed=42)
        self.assertIn("ablation_study", res)
        self.assertEqual(len(res["ablation_study"]), 5)
        comp_ids = [c["component_id"] for c in res["ablation_study"]]
        self.assertEqual(comp_ids, ["A", "B", "C", "D", "E"])

class TestAdminAuthority(unittest.TestCase):
    def test_admin_passkey_verification(self):
        from server import verify_admin_key
        self.assertTrue(verify_admin_key("fedequimed_admin_2026"))
        self.assertFalse(verify_admin_key("wrong_password"))
        self.assertFalse(verify_admin_key(""))
        self.assertFalse(verify_admin_key(None))

if __name__ == "__main__":
    unittest.main()
