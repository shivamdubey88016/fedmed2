#!/usr/bin/env python3
"""
CLI Runner for Flower Federated Healthcare Simulation:
Compares Standard FedAvg vs Fairness-Aware Adaptive Aggregation (FedEquiMed)
using PyTorch Logistic Regression on MIMIC-IV hospital cohorts.
"""

import sys
import argparse
from flower_fl.simulation import run_comparative_benchmark

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Learning Fairness Benchmark")
    parser.add_argument("--mode", type=str, choices=["compare", "fairness_only", "fedavg_only"], default="compare",
                        help="Evaluation mode: compare (both FedAvg & Fairness-Aware), fairness_only, or fedavg_only (default: compare)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL communication rounds (default: 5)")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per hospital (default: 3)")
    parser.add_argument("--lr", type=float, default=0.05, help="Local learning rate (default: 0.05)")
    parser.add_argument("--beta", type=float, default=2.5, help="Fairness disparity penalty beta (default: 2.5)")
    parser.add_argument("--gamma", type=float, default=0.8, help="Disparity momentum gamma (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--custom-hospital", type=str, default=None, help="Path to custom hospital CSV file")
    parser.add_argument("--custom-id", type=str, default="hospital_custom", help="Identifier for custom hospital (default: hospital_custom)")
    parser.add_argument("--export-json", type=str, default=None, help="Path to save experiment results JSON bundle")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to save summary comparison CSV")
    parser.add_argument("--benchmark-preset", type=str, choices=["mimic_iv", "eicu", "medmnist", "covid19_imaging", "tcga"], default=None,
                        help="Enroll a preset clinical benchmark from the original repository (mimic_iv, eicu, medmnist, covid19_imaging, tcga)")
    parser.add_argument("--benchmark-count", type=int, default=300, help="Record count for enrolled benchmark (default: 300)")
    args = parser.parse_args()

    # Load custom hospital or benchmark preset if specified
    custom_dict = None
    if args.benchmark_preset:
        from flower_fl.dataset_loader import load_all_hospital_datasets, load_preset_benchmark_cohort
        base_hospitals, _ = load_all_hospital_datasets()
        preset_id = f"hospital_{args.benchmark_preset}"
        print(f"Enrolling clinical benchmark preset: {args.benchmark_preset} as {preset_id} ({args.benchmark_count} records)")
        parsed_b, _ = load_preset_benchmark_cohort(args.benchmark_preset, count=args.benchmark_count, seed=args.seed)
        base_hospitals[preset_id] = parsed_b
        custom_dict = base_hospitals
    elif args.custom_hospital:
        from flower_fl.dataset_loader import load_all_hospital_datasets, parse_patient_csv
        base_hospitals, _ = load_all_hospital_datasets()
        print(f"Loading custom hospital from: {args.custom_hospital}")
        base_hospitals[args.custom_id] = parse_patient_csv(args.custom_hospital)
        custom_dict = base_hospitals

    print("=" * 72)
    print(" FLOWER FEDERATED HEALTHCARE BENCHMARK (MIMIC-IV COHORTS)")
    print("=" * 72)
    print(f" • Architecture: PyTorch Logistic Regression")
    active_clients_str = ", ".join(custom_dict.keys()) if custom_dict else "hospital_a, hospital_b, hospital_c"
    print(f" • Hospital Clients ({len(custom_dict) if custom_dict else 3}): {active_clients_str}")
    print(f" • Rounds: {args.rounds} | Local Epochs: {args.epochs} | Learning Rate: {args.lr}")
    from flower_fl.simulation import run_comparative_benchmark, run_federated_simulation

    print(f" • Mode: {args.mode.upper()}")
    print(f" • Fairness Parameters: beta = {args.beta}, gamma = {args.gamma}")
    print("=" * 72)

    if args.mode == "compare":
        print("\n[1/2] Running Comparative FL Benchmark (FedAvg vs Fairness-Aware)...")
        benchmark = run_comparative_benchmark(
            num_rounds=args.rounds,
            epochs_per_hospital=args.epochs,
            lr=args.lr,
            fairness_beta=args.beta,
            momentum_gamma=args.gamma,
            hospitals_dict=custom_dict,
            seed=args.seed
        )
        fedavg = benchmark["standard_fedavg"]
        fair = benchmark["fairness_aware"]
        comp = benchmark["comparison_table"]

        print("\n[2/2] Benchmark Completed Successfully!")
        print(f"  FedAvg Runtime: {fedavg['elapsed_seconds']}s")
        print(f"  Fairness-Aware Runtime: {fair['elapsed_seconds']}s")

        print("\n" + "=" * 72)
        print(f"{'METRIC':<30} | {'STANDARD FEDAVG':<18} | {'FAIRNESS-AWARE':<18}")
        print("-" * 72)
        for metric_name, vals in comp.items():
            v_fed = vals["FedAvg"]
            v_fair = vals["Fairness-Aware"]
            s_fed = f"{v_fed:.4f}" if v_fed is not None else "N/A"
            s_fair = f"{v_fair:.4f}" if v_fair is not None else "N/A"
            
            star = ""
            if "DPD" in metric_name or "EOD" in metric_name:
                if v_fed is not None and v_fair is not None and v_fair < v_fed:
                    pct = ((v_fed - v_fair) / max(1e-6, v_fed)) * 100
                    star = f" (-{pct:.1f}% disparity)"
            elif "DIR" in metric_name:
                if v_fed is not None and v_fair is not None and v_fair > v_fed:
                    star = " (+fairer)"
            
            print(f"{metric_name:<30} | {s_fed:<18} | {s_fair:<18}{star}")
        print("=" * 72)

        # Subgroup breakdown
        subgroups = fair["final_metrics"].get("subgroups", {})
        if subgroups:
            print("\nSubgroup Demographic Performance (Fairness-Aware Global Model):")
            for g_name, g_vals in subgroups.items():
                print(f"  • {g_name} (N={g_vals['count']}): Accuracy = {g_vals['accuracy']:.4f}, TPR = {g_vals['tpr']:.4f}, FPR = {g_vals['fpr']:.4f}, F1 = {g_vals['f1']:.4f}")

        # Aggregation weights history
        weights_hist = fair.get("weights_history", [])
        if weights_hist:
            print("\nAdaptive Client Aggregation Weights across Rounds:")
            for r_idx, w_dict in enumerate(weights_hist, 1):
                weights_str = ", ".join(f"{k}: {v:.3f}" for k, v in w_dict.items())
                print(f"  Round {r_idx}: {weights_str}")

        # Node-wise Evaluation Table
        node_comp = benchmark.get("node_comparison", {})
        if node_comp:
            print("\n" + "=" * 72)
            print(" NODE-WISE INSTITUTIONAL PERFORMANCE")
            print("=" * 72)
            print(f"{'HOSPITAL NODE':<22} | {'SAMPLE':<7} | {'FEDAVG ACC':<11} | {'FAIR ACC':<10} | {'FED DPD':<9} | {'FAIR DPD':<9}")
            print("-" * 72)
            for hid, nv in node_comp.items():
                h_lbl = hid.replace("_", " ").title()[:21]
                n_sz = nv.get("sample_size", 0)
                f_acc = f"{nv.get('fedavg_accuracy', 0):.4f}"
                fa_acc = f"{nv.get('fairness_accuracy', 0):.4f}"
                f_dpd = f"{nv.get('fedavg_dpd', 0):.4f}"
                fa_dpd = f"{nv.get('fairness_dpd', 0):.4f}"
                print(f"{h_lbl:<22} | {n_sz:<7} | {f_acc:<11} | {fa_acc:<10} | {f_dpd:<9} | {fa_dpd:<9}")
            print("=" * 72)

    elif args.mode == "fairness_only":
        print("\nRunning Proposed Fairness-Aware FL Simulation (D-TAMA)...")
        fair = run_federated_simulation(
            strategy_type="fairness_aware",
            num_rounds=args.rounds,
            epochs_per_hospital=args.epochs,
            lr=args.lr,
            fairness_beta=args.beta,
            momentum_gamma=args.gamma,
            hospitals_dict=custom_dict,
            seed=args.seed
        )
        fedavg = None
        comp = {
            "Accuracy": {"Fairness-Aware": fair["final_metrics"].get("accuracy")},
            "F1-Score": {"Fairness-Aware": fair["final_metrics"].get("f1")},
            "DPD (Demographic Parity)": {"Fairness-Aware": fair["final_metrics"].get("dpd")},
            "EOD (Equalized Odds)": {"Fairness-Aware": fair["final_metrics"].get("eod")},
            "DIR (Disparate Impact)": {"Fairness-Aware": fair["final_metrics"].get("dir")},
            "AUROC": {"Fairness-Aware": fair["final_metrics"].get("auroc")}
        }
        print(f"\nSimulation Complete in {fair['elapsed_seconds']}s")
        print("\n" + "=" * 52)
        print(f"{'METRIC':<32} | {'FAIRNESS-AWARE':<16}")
        print("-" * 52)
        for k, v in comp.items():
            print(f"{k:<32} | {v['Fairness-Aware']:.4f}")
        print("=" * 52)

        node_comp = fair.get("node_wise_metrics", {})
        if node_comp:
            print("\n" + "=" * 60)
            print(" NODE-WISE PERFORMANCE (FAIRNESS-AWARE)")
            print("=" * 60)
            print(f"{'HOSPITAL NODE':<22} | {'SAMPLE':<7} | {'ACCURACY':<10} | {'DPD (ΔDP)':<10}")
            print("-" * 60)
            for hid, nv in node_comp.items():
                print(f"{hid.replace('_', ' ').title():<22} | {nv.get('sample_size', 0):<7} | {nv.get('accuracy', 0):<10.4f} | {nv.get('dpd', 0):<10.4f}")
            print("=" * 60)

    else: # fedavg_only
        print("\nRunning Standard FedAvg FL Simulation...")
        fedavg = run_federated_simulation(
            strategy_type="fedavg",
            num_rounds=args.rounds,
            epochs_per_hospital=args.epochs,
            lr=args.lr,
            hospitals_dict=custom_dict,
            seed=args.seed
        )
        fair = None
        comp = {
            "Accuracy": {"FedAvg": fedavg["final_metrics"].get("accuracy")},
            "F1-Score": {"FedAvg": fedavg["final_metrics"].get("f1")},
            "DPD (Demographic Parity)": {"FedAvg": fedavg["final_metrics"].get("dpd")},
            "EOD (Equalized Odds)": {"FedAvg": fedavg["final_metrics"].get("eod")},
            "DIR (Disparate Impact)": {"FedAvg": fedavg["final_metrics"].get("dir")},
            "AUROC": {"FedAvg": fedavg["final_metrics"].get("auroc")}
        }
        print(f"\nSimulation Complete in {fedavg['elapsed_seconds']}s")
        print("\n" + "=" * 52)
        print(f"{'METRIC':<32} | {'STANDARD FEDAVG':<16}")
        print("-" * 52)
        for k, v in comp.items():
            print(f"{k:<32} | {v['FedAvg']:.4f}")
        print("=" * 52)

        node_comp = fedavg.get("node_wise_metrics", {})
        if node_comp:
            print("\n" + "=" * 60)
            print(" NODE-WISE PERFORMANCE (STANDARD FEDAVG)")
            print("=" * 60)
            print(f"{'HOSPITAL NODE':<22} | {'SAMPLE':<7} | {'ACCURACY':<10} | {'DPD (ΔDP)':<10}")
            print("-" * 60)
            for hid, nv in node_comp.items():
                print(f"{hid.replace('_', ' ').title():<22} | {nv.get('sample_size', 0):<7} | {nv.get('accuracy', 0):<10.4f} | {nv.get('dpd', 0):<10.4f}")
            print("=" * 60)

    # Export results if requested
    if args.export_json:
        import json
        export_payload = {
            "metadata": {
                "rounds": args.rounds,
                "epochs": args.epochs,
                "lr": args.lr,
                "fairness_beta": args.beta,
                "momentum_gamma": args.gamma,
                "seed": args.seed,
                "clients": list(custom_dict.keys()) if custom_dict else ["hospital_a", "hospital_b", "hospital_c"]
            },
            "comparison_table": comp,
            "node_comparison": node_comp,
            "standard_fedavg": fedavg,
            "fairness_aware": fair
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2, default=str)
        print(f"\n[Artifact Saved] JSON Results Bundle: {args.export_json}")

    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Standard_FedAvg", "Fairness_Aware"])
            for m_k, m_v in comp.items():
                v_fed = f"{m_v['FedAvg']:.4f}" if m_v.get("FedAvg") is not None else "N/A"
                v_fair = f"{m_v['Fairness-Aware']:.4f}" if m_v.get("Fairness-Aware") is not None else "N/A"
                writer.writerow([m_k, v_fed, v_fair])
        print(f"[Artifact Saved] Summary Comparison CSV: {args.export_csv}")

if __name__ == "__main__":
    main()
