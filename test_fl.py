"""
Test suite for Fairness-Aware Federated Learning Engine
"""

import sys
from fl_engine.dataset import generate_federated_healthcare_datasets
from fl_engine.hospital_node import HospitalNode
from fl_engine.global_server import GlobalFederatedServer

def run_tests():
    print("1. Generating Federated Datasets...")
    hospitals_data, global_test = generate_federated_healthcare_datasets(seed=42)
    assert len(hospitals_data) == 3, f"Expected 3 hospitals, got {len(hospitals_data)}"
    assert len(global_test) == 1000, f"Expected 1000 test samples, got {len(global_test)}"
    print("   Dataset generated successfully!")

    print("2. Initializing Hospital Nodes...")
    hospitals = {
        "hospital_a": HospitalNode("hospital_a", "Hospital A (Urban Center)", hospitals_data["hospital_a"], description="Urban tertiary medical center with broad demographic catchment."),
        "hospital_b": HospitalNode("hospital_b", "Hospital B (Rural Clinic)", hospitals_data["hospital_b"], description="Rural clinic with high proportion of geriatric patients."),
        "hospital_c": HospitalNode("hospital_c", "Hospital C (Heart Institute)", hospitals_data["hospital_c"], description="Cardiovascular specialty center with higher male cardiac cohort.")
    }
    for hid, h in hospitals.items():
        summary = h.get_summary()
        print(f"   {h.name}: {summary['sample_size']} records, {summary['demographics']['gender']['female_pct']}% female")

    print("3. Testing Local Hospital Training & Secure Payload Export...")
    payload_a = hospitals["hospital_a"].train_local_model(epochs=3, lr=0.05, use_adaptive_mitigation=True)
    assert "weights" in payload_a, "Payload missing weights"
    assert "pre_audit" in payload_a, "Payload missing pre_audit"
    assert "post_audit" in payload_a, "Payload missing post_audit"
    assert payload_a["privacy_verification"]["raw_records_shared"] == 0, "Raw records leaked!"
    print(f"   Hospital A Local Accuracy: {payload_a['post_audit']['accuracy']}, Gender DPD: {payload_a['post_audit']['gender_dpd']}")
    print(f"   Privacy verification: {payload_a['privacy_verification']['data_leakage_risk']}")

    print("4. Testing Global Server & FairFed Aggregation...")
    server = GlobalFederatedServer(hospitals, global_test)
    round_result = server.run_federated_round(strategy="fairfed", epochs_per_hospital=3, use_adaptive_mitigation=True)
    print(f"   Round 1 Result: Strategy={round_result['strategy']}")
    print(f"   Aggregation Weights: {round_result['aggregation_weights']}")
    print(f"   Global Accuracy: {round_result['global_metrics']['accuracy']}")
    print(f"   Global Gender DPD: {round_result['global_metrics']['gender_dpd']}")
    print(f"   Global Equalized Odds Diff: {round_result['global_metrics']['gender_eod']}")

    print("5. Testing Multi-Round Comparative Benchmark (FedAvg vs FairFed vs FedEquiMed)...")
    benchmark = server.run_benchmark_comparison(rounds=3)
    sum_res = benchmark["comparison_summary"]
    print("   Benchmark Results:")
    print(f"     Standard FedAvg - Acc: {sum_res['standard_fedavg']['final_accuracy']}, Gender DPD: {sum_res['standard_fedavg']['final_gender_dpd']}")
    print(f"     Fairness-Aware FL - Acc: {sum_res['fairness_aware_fl']['final_accuracy']}, Gender DPD: {sum_res['fairness_aware_fl']['final_gender_dpd']}")
    print(f"     FedEquiMed (Proposed) - Acc: {sum_res['fedequimed']['final_accuracy']}, Gender DPD: {sum_res['fedequimed']['final_gender_dpd']}, DIR: {sum_res['fedequimed']['gender_dir']}")
    print(f"     Disparity Reduction (FedEquiMed): {sum_res['fedequimed']['disparity_reduction_pct']}%")

    print("6. Testing FedEquiMed End-to-End Federated Round with D-TAMA Aggregator...")
    fe_round = server.run_federated_round(strategy="fedequimed", epochs_per_hospital=3, use_adaptive_mitigation=True)
    assert fe_round["strategy"] == "fedequimed", "Strategy should be fedequimed"
    assert "institutional_momentum" in fe_round, "Missing institutional_momentum"
    print(f"   FedEquiMed Round Accuracy: {fe_round['global_metrics']['accuracy']}, Gender DPD: {fe_round['global_metrics']['gender_dpd']}")
    print(f"   D-TAMA Institutional Momentum: {fe_round['institutional_momentum']}")

    print("\nALL FEDERATED LEARNING & FEDEQUIMED ENGINE TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
