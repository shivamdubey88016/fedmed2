"""
FedEquiMed Streamlit Dashboard
Interactive Medical AI Interface for Fairness-Aware Federated Learning
Using PyTorch Logistic Regression and Flower Simulation across MIMIC-IV Hospital Cohorts.
Includes dynamic custom hospital upload and synthetic hospital generation.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from flower_fl.dataset_loader import (
    load_all_hospital_datasets,
    parse_patient_dataframe,
    generate_synthetic_custom_hospital,
    save_custom_hospital_to_disk,
    load_preset_benchmark_cohort,
    BENCHMARK_DATASETS_META
)
from flower_fl.simulation import run_comparative_benchmark, run_federated_simulation
from flower_fl.model import PyTorchLogisticRegression
from flower_fl.metrics import compute_clinical_fairness_metrics

# Page Configuration
st.set_page_config(
    page_title="FedEquiMed: Fairness FL in Healthcare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-custom {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title & Clinical Header
st.markdown('<div class="main-header">🏥 FedEquiMed: Fairness-Aware Federated Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<b>RO1 Objective</b>: Non-IID Multi-Hospital Simulation with <b>Flower (flwr)</b>, '
    '<b>PyTorch Logistic Regression</b>, and <b>Fairness-Aware Adaptive Aggregation (D-TAMA)</b> on MIMIC-IV Clinical Cohorts.'
    '</div>',
    unsafe_allow_html=True
)

# Initialize Session State
if "custom_hospitals" not in st.session_state:
    st.session_state["custom_hospitals"] = {}
if "benchmark_results" not in st.session_state:
    st.session_state["benchmark_results"] = None

# Load Base Datasets + custom additions
base_hospitals, global_test = load_all_hospital_datasets(
    data_dir="data",
    custom_hospitals=st.session_state["custom_hospitals"]
)

# Merge all available hospitals
all_available_hospitals = {**base_hospitals, **st.session_state["custom_hospitals"]}

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Controls")

sim_mode = st.sidebar.radio(
    "Evaluation Mode",
    ["Compare FedAvg vs Fairness-Aware", "Fairness-Aware Only", "Standard FedAvg Only"]
)

# Hospital Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 Participating Hospital Nodes")
hospital_options = list(all_available_hospitals.keys())
selected_hospitals_keys = st.sidebar.multiselect(
    "Select Hospital Clients for FL",
    options=hospital_options,
    default=hospital_options,
    help="Select which hospital clients will participate in this federated round."
)

if not selected_hospitals_keys:
    st.sidebar.warning("Please select at least 1 hospital client.")
    selected_hospitals_keys = hospital_options[:1]

active_hospitals = {k: all_available_hospitals[k] for k in selected_hospitals_keys}
st.sidebar.caption(f"**{len(active_hospitals)} Hospital Client(s)** active in federated network.")

st.sidebar.markdown("---")
rounds = st.sidebar.slider("Communication Rounds", min_value=1, max_value=10, value=5, step=1)
epochs = st.sidebar.slider("Local Epochs per Hospital", min_value=1, max_value=8, value=3, step=1)
lr = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.02, 0.05, 0.08, 0.1, 0.15], value=0.05)

if sim_mode != "Standard FedAvg Only":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Fairness-Aware Parameters")
    fairness_beta = st.sidebar.slider("Disparity Penalty (β)", min_value=0.5, max_value=5.0, value=2.5, step=0.5,
                                      help="Higher values penalize hospitals with persistent demographic bias more strongly.")
    momentum_gamma = st.sidebar.slider("Momentum Memory (γ)", min_value=0.1, max_value=0.95, value=0.80, step=0.05,
                                       help="Tracks institutional disparity momentum across communication rounds.")
else:
    st.sidebar.markdown("---")
    st.sidebar.info("ℹ️ **Standard FedAvg Mode**: Global parameters are aggregated strictly by sample size ($n_k/N$) without fairness penalty.")
    fairness_beta = 0.0
    momentum_gamma = 0.0

seed = st.sidebar.number_input("Random Seed", value=42, step=1)

run_button = st.sidebar.button("🚀 Run Federated Simulation", type="primary", use_container_width=True)

# Direct Sidebar Download for PhD Research Report PDF
pdf_report_path = "FedEquiMed_PhD_Research_Report.pdf"
if os.path.exists(pdf_report_path):
    with open(pdf_report_path, "rb") as f_pdf:
        pdf_bytes = f_pdf.read()
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📑 Download PhD Report (PDF)",
        data=pdf_bytes,
        file_name="FedEquiMed_PhD_Research_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

# Run or load simulation state respecting sim_mode
if "last_sim_mode" not in st.session_state:
    st.session_state["last_sim_mode"] = sim_mode

need_sim = run_button or (st.session_state["benchmark_results"] is None) or (st.session_state.get("last_sim_mode") != sim_mode)

if need_sim:
    st.session_state["last_sim_mode"] = sim_mode
    with st.spinner(f"Running simulation in [{sim_mode}] mode across {len(active_hospitals)} hospitals with Flower & PyTorch..."):
        if sim_mode == "Compare FedAvg vs Fairness-Aware":
            benchmark = run_comparative_benchmark(
                num_rounds=rounds,
                epochs_per_hospital=epochs,
                lr=lr,
                fairness_beta=fairness_beta,
                momentum_gamma=momentum_gamma,
                data_dir="data",
                hospitals_dict=active_hospitals,
                seed=int(seed)
            )
            benchmark["mode"] = "compare"
        elif sim_mode == "Fairness-Aware Only":
            fair_run = run_federated_simulation(
                strategy_type="fairness_aware",
                num_rounds=rounds,
                epochs_per_hospital=epochs,
                lr=lr,
                fairness_beta=fairness_beta,
                momentum_gamma=momentum_gamma,
                data_dir="data",
                hospitals_dict=active_hospitals,
                seed=int(seed)
            )
            node_comp = {
                hid: {
                    "sample_size": m.get("sample_size"),
                    "fairness_accuracy": m.get("accuracy"),
                    "fairness_f1": m.get("f1"),
                    "fairness_dpd": m.get("dpd"),
                    "fairness_eod": m.get("eod"),
                    "fairness_auroc": m.get("auroc"),
                    "fairness_loss": m.get("loss"),
                } for hid, m in fair_run.get("node_wise_metrics", {}).items()
            }
            benchmark = {
                "mode": "fairness_only",
                "fairness_aware": fair_run,
                "standard_fedavg": None,
                "node_comparison": node_comp,
                "comparison_table": {
                    "Accuracy": {"Fairness-Aware": fair_run["final_metrics"].get("accuracy")},
                    "F1-Score": {"Fairness-Aware": fair_run["final_metrics"].get("f1")},
                    "DPD (Demographic Parity)": {"Fairness-Aware": fair_run["final_metrics"].get("dpd")},
                    "EOD (Equalized Odds)": {"Fairness-Aware": fair_run["final_metrics"].get("eod")},
                    "DIR (Disparate Impact)": {"Fairness-Aware": fair_run["final_metrics"].get("dir")},
                    "AUROC": {"Fairness-Aware": fair_run["final_metrics"].get("auroc")}
                }
            }
        else: # Standard FedAvg Only
            fed_run = run_federated_simulation(
                strategy_type="fedavg",
                num_rounds=rounds,
                epochs_per_hospital=epochs,
                lr=lr,
                data_dir="data",
                hospitals_dict=active_hospitals,
                seed=int(seed)
            )
            node_comp = {
                hid: {
                    "sample_size": m.get("sample_size"),
                    "fedavg_accuracy": m.get("accuracy"),
                    "fedavg_f1": m.get("f1"),
                    "fedavg_dpd": m.get("dpd"),
                    "fedavg_eod": m.get("eod"),
                    "fedavg_auroc": m.get("auroc"),
                    "fedavg_loss": m.get("loss"),
                } for hid, m in fed_run.get("node_wise_metrics", {}).items()
            }
            benchmark = {
                "mode": "fedavg_only",
                "standard_fedavg": fed_run,
                "fairness_aware": None,
                "node_comparison": node_comp,
                "comparison_table": {
                    "Accuracy": {"FedAvg": fed_run["final_metrics"].get("accuracy")},
                    "F1-Score": {"FedAvg": fed_run["final_metrics"].get("f1")},
                    "DPD (Demographic Parity)": {"FedAvg": fed_run["final_metrics"].get("dpd")},
                    "EOD (Equalized Odds)": {"FedAvg": fed_run["final_metrics"].get("eod")},
                    "DIR (Disparate Impact)": {"FedAvg": fed_run["final_metrics"].get("dir")},
                    "AUROC": {"FedAvg": fed_run["final_metrics"].get("auroc")}
                }
            }
        st.session_state["benchmark_results"] = benchmark

results = st.session_state["benchmark_results"]
mode = results.get("mode", "compare")
fedavg_res = results.get("standard_fedavg")
fair_res = results.get("fairness_aware")
comp_table = results.get("comparison_table", {})

# Top Federated Training Status & Privacy Guarantee Banner (from old website)
st.markdown("""
<div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.65rem 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
    <div style="display: flex; align-items: center; gap: 0.6rem;">
        <span style="font-size: 1.15rem;">🔒</span>
        <span style="font-size: 0.88rem; font-weight: 600; color: #166534;">
            <b>Zero-PHI Transfer Guarantee</b>: Raw patient records remain 100% on-premise within hospital perimeters. Only encrypted parameter updates are communicated.
        </span>
    </div>
    <span style="background: #DCFCE7; color: #15803D; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid #86EFAC;">
        HIPAA / GDPR Aligned
    </span>
</div>
""", unsafe_allow_html=True)

fl_stat_col1, fl_stat_col2, fl_stat_col3, fl_stat_col4, fl_stat_col5 = st.columns(5)
with fl_stat_col1:
    st.metric("Clients Active", f"{len(active_hospitals)} Hospital Silos", help="Decentralized on-premise hospital nodes participating in FL.")
with fl_stat_col2:
    st.metric("FL Rounds", f"{rounds} Rounds", help="Communication cycles between hospital nodes and global server.")
with fl_stat_col3:
    st.metric("Local Epochs", f"{epochs} Epochs", help="Local gradient descent epochs computed on-premise per round.")
with fl_stat_col4:
    strat_short = "D-TAMA vs FedAvg" if sim_mode == "Compare FedAvg vs Fairness-Aware" else ("D-TAMA Adaptive" if sim_mode == "Fairness-Aware Only" else "Standard FedAvg")
    st.metric("Aggregation", strat_short, help="Consensus parameter aggregation strategy.")
with fl_stat_col5:
    st.metric("Data Transferred", "0 Bytes PHI", delta="Weights Only", delta_color="normal")

st.markdown("---")

# Navigation Tabs (Focused 5-Tab Clinical Healthcare Suite)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Simulation & Benchmarks",
    "🏥 Hospital Cohorts",
    "➕ Add Custom Hospital & Benchmarks",
    "⚖️ Demographic Fairness Audit",
    "🩺 Patient Risk Predictor"
])

# ----------------------------------------------------
# TAB 1: Simulation & Benchmarks
# ----------------------------------------------------
with tab1:
    st.subheader(f"Federated Convergence & Fairness Performance ({len(active_hospitals)} Active Clients)")

    with st.expander("🔬 Decentralized Federated Architecture & Training Workflow (Zero-PHI)", expanded=False):
        st.markdown("""
        ```
        🏥 Participating Nodes (Hosp A, B, C & Custom)
              │
              ▼ [Local Epochs On-Premise Optimization]
        ⚙️ Local Model Training (PyTorch BCE Loss + Subgroup Reweighting)
              │
              ▼ [Weights Only — 0 Bytes Patient PHI Transmitted]
        📡 Encrypted Parameter Vectors (Δw, Δb)
              │
              ▼ [D-TAMA Disparity-Penalized Aggregation]
        🌐 Central Federated Aggregator (Flower flwr)
              │
              ▼ [Consensus Parameters Dispatched]
        🔄 Global Clinical Risk Model
              │
              ▼ [Holdout Multi-Center Cohort Evaluation]
        ⚖️ Comprehensive Fairness & Utility Audit (Accuracy, F1, AUROC, ΔDP, ΔEO, DIR)
        ```
        """)

    # Metrics Summary Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    if mode == "compare":
        fed_acc = fedavg_res["final_metrics"].get("accuracy", 0.0)
        fair_acc = fair_res["final_metrics"].get("accuracy", 0.0)
        acc_diff = (fair_acc - fed_acc) * 100

        fed_f1 = fedavg_res["final_metrics"].get("f1", 0.0)
        fair_f1 = fair_res["final_metrics"].get("f1", 0.0)
        f1_diff = (fair_f1 - fed_f1) * 100

        fed_dpd = fedavg_res["final_metrics"].get("dpd", 0.0)
        fair_dpd = fair_res["final_metrics"].get("dpd", 0.0)
        dpd_diff = ((fed_dpd - fair_dpd) / max(1e-6, fed_dpd)) * 100

        fed_eod = fedavg_res["final_metrics"].get("eod", 0.0)
        fair_eod = fair_res["final_metrics"].get("eod", 0.0)
        eod_diff = ((fed_eod - fair_eod) / max(1e-6, fed_eod)) * 100

        with m_col1:
            st.metric(
                label="Accuracy (Global Cohort)",
                value=f"{fair_acc * 100:.2f}%",
                delta=f"{acc_diff:+.2f}% vs FedAvg ({fed_acc * 100:.2f}%)",
                delta_color="normal"
            )
        with m_col2:
            st.metric(
                label="F1-Score (ICU Mortality)",
                value=f"{fair_f1 * 100:.2f}%",
                delta=f"{f1_diff:+.2f}% vs FedAvg ({fed_f1 * 100:.2f}%)",
                delta_color="normal"
            )
        with m_col3:
            st.metric(
                label="Demographic Parity (ΔDP)",
                value=f"{fair_dpd * 100:.2f}%",
                delta=f"-{dpd_diff:.1f}% disparity" if dpd_diff >= 0 else f"+{abs(dpd_diff):.1f}% disparity",
                delta_color="normal" if fair_dpd <= fed_dpd else "inverse"
            )
        with m_col4:
            st.metric(
                label="Equalized Odds (ΔEO)",
                value=f"{fair_eod * 100:.2f}%",
                delta=f"-{eod_diff:.1f}% disparity" if eod_diff >= 0 else f"+{abs(eod_diff):.1f}% disparity",
                delta_color="normal" if fair_eod <= fed_eod else "inverse"
            )
    elif mode == "fairness_only":
        fair_acc = fair_res["final_metrics"].get("accuracy", 0.0)
        fair_f1 = fair_res["final_metrics"].get("f1", 0.0)
        fair_dpd = fair_res["final_metrics"].get("dpd", 0.0)
        fair_eod = fair_res["final_metrics"].get("eod", 0.0)
        with m_col1:
            st.metric(label="Accuracy (Global Cohort)", value=f"{fair_acc * 100:.2f}%", delta="Fairness-Aware FL", delta_color="off")
        with m_col2:
            st.metric(label="F1-Score (ICU Mortality)", value=f"{fair_f1 * 100:.2f}%", delta="Fairness-Aware FL", delta_color="off")
        with m_col3:
            st.metric(label="Demographic Parity (ΔDP)", value=f"{fair_dpd * 100:.2f}%", delta="Fairness Metric", delta_color="off")
        with m_col4:
            st.metric(label="Equalized Odds (ΔEO)", value=f"{fair_eod * 100:.2f}%", delta="Fairness Metric", delta_color="off")
    else: # fedavg_only
        fed_acc = fedavg_res["final_metrics"].get("accuracy", 0.0)
        fed_f1 = fedavg_res["final_metrics"].get("f1", 0.0)
        fed_dpd = fedavg_res["final_metrics"].get("dpd", 0.0)
        fed_eod = fedavg_res["final_metrics"].get("eod", 0.0)
        with m_col1:
            st.metric(label="Accuracy (Global Cohort)", value=f"{fed_acc * 100:.2f}%", delta="Standard FedAvg", delta_color="off")
        with m_col2:
            st.metric(label="F1-Score (ICU Mortality)", value=f"{fed_f1 * 100:.2f}%", delta="Standard FedAvg", delta_color="off")
        with m_col3:
            st.metric(label="Demographic Parity (ΔDP)", value=f"{fed_dpd * 100:.2f}%", delta="Standard FedAvg", delta_color="off")
        with m_col4:
            st.metric(label="Equalized Odds (ΔEO)", value=f"{fed_eod * 100:.2f}%", delta="Standard FedAvg", delta_color="off")

    st.markdown("---")

    # Round Trajectories Plot
    c_col1, c_col2 = st.columns(2)

    active_traj = fair_res["trajectory"] if fair_res else fedavg_res["trajectory"]
    rounds_axis = [t["round"] for t in active_traj]

    with c_col1:
        st.markdown("#### Accuracy & F1-Score over Communication Rounds")
        fig_acc, ax_acc = plt.subplots(figsize=(6, 3.8))
        if mode == "compare":
            fed_traj = fedavg_res["trajectory"]
            fair_traj = fair_res["trajectory"]
            ax_acc.plot(rounds_axis, [t["accuracy"] * 100 for t in fed_traj], "o--", color="#6B7280", label="FedAvg Accuracy", linewidth=1.8)
            ax_acc.plot(rounds_axis, [t["accuracy"] * 100 for t in fair_traj], "s-", color="#10B981", label="Fairness-Aware Accuracy", linewidth=2.2)
            ax_acc.plot(rounds_axis, [t["f1"] * 100 for t in fed_traj], "^--", color="#9CA3AF", label="FedAvg F1", linewidth=1.5)
            ax_acc.plot(rounds_axis, [t["f1"] * 100 for t in fair_traj], "d-", color="#059669", label="Fairness-Aware F1", linewidth=2.0)
        elif mode == "fairness_only":
            fair_traj = fair_res["trajectory"]
            ax_acc.plot(rounds_axis, [t["accuracy"] * 100 for t in fair_traj], "s-", color="#10B981", label="Fairness-Aware Accuracy", linewidth=2.2)
            ax_acc.plot(rounds_axis, [t["f1"] * 100 for t in fair_traj], "d-", color="#059669", label="Fairness-Aware F1", linewidth=2.0)
        else: # fedavg_only
            fed_traj = fedavg_res["trajectory"]
            ax_acc.plot(rounds_axis, [t["accuracy"] * 100 for t in fed_traj], "o--", color="#6B7280", label="FedAvg Accuracy", linewidth=1.8)
            ax_acc.plot(rounds_axis, [t["f1"] * 100 for t in fed_traj], "^--", color="#9CA3AF", label="FedAvg F1", linewidth=1.5)

        ax_acc.set_xlabel("Federated Round", fontsize=10)
        ax_acc.set_ylabel("Score (%)", fontsize=10)
        ax_acc.set_ylim(40.0, 90.0)
        ax_acc.grid(True, linestyle=":", alpha=0.6)
        ax_acc.legend(fontsize=8, loc="lower right")
        st.pyplot(fig_acc)

    with c_col2:
        st.markdown("#### Demographic Disparity Trajectory (ΔDP & ΔEO)")
        fig_fair, ax_fair = plt.subplots(figsize=(6, 3.8))
        if mode == "compare":
            fed_traj = fedavg_res["trajectory"]
            fair_traj = fair_res["trajectory"]
            ax_fair.plot(rounds_axis, [t["dpd"] * 100 for t in fed_traj], "o--", color="#EF4444", label="FedAvg ΔDP (Gender)", linewidth=1.8)
            ax_fair.plot(rounds_axis, [t["dpd"] * 100 for t in fair_traj], "s-", color="#3B82F6", label="Fairness-Aware ΔDP", linewidth=2.2)
            ax_fair.plot(rounds_axis, [t["eod"] * 100 for t in fed_traj], "^--", color="#F87171", label="FedAvg ΔEO (Odds)", linewidth=1.5)
            ax_fair.plot(rounds_axis, [t["eod"] * 100 for t in fair_traj], "d-", color="#6366F1", label="Fairness-Aware ΔEO", linewidth=2.0)
            all_disps = [t["dpd"] * 100 for t in fed_traj] + [t["eod"] * 100 for t in fed_traj] + [t["dpd"] * 100 for t in fair_traj] + [t["eod"] * 100 for t in fair_traj]
        elif mode == "fairness_only":
            fair_traj = fair_res["trajectory"]
            ax_fair.plot(rounds_axis, [t["dpd"] * 100 for t in fair_traj], "s-", color="#3B82F6", label="Fairness-Aware ΔDP", linewidth=2.2)
            ax_fair.plot(rounds_axis, [t["eod"] * 100 for t in fair_traj], "d-", color="#6366F1", label="Fairness-Aware ΔEO", linewidth=2.0)
            all_disps = [t["dpd"] * 100 for t in fair_traj] + [t["eod"] * 100 for t in fair_traj]
        else: # fedavg_only
            fed_traj = fedavg_res["trajectory"]
            ax_fair.plot(rounds_axis, [t["dpd"] * 100 for t in fed_traj], "o--", color="#EF4444", label="FedAvg ΔDP (Gender)", linewidth=1.8)
            ax_fair.plot(rounds_axis, [t["eod"] * 100 for t in fed_traj], "^--", color="#F87171", label="FedAvg ΔEO (Odds)", linewidth=1.5)
            all_disps = [t["dpd"] * 100 for t in fed_traj] + [t["eod"] * 100 for t in fed_traj]

        ax_fair.set_xlabel("Federated Round", fontsize=10)
        ax_fair.set_ylabel("Disparity (%) [Lower = Fairer]", fontsize=10)
        max_d = max(all_disps + [10.0])
        ax_fair.set_ylim(0.0, max_d * 1.25)
        ax_fair.grid(True, linestyle=":", alpha=0.6)
        ax_fair.legend(fontsize=8, loc="upper right")
        st.pyplot(fig_fair)

    # Adaptive Client Aggregation Weights
    st.markdown("#### Adaptive Client Aggregation Weights (D-TAMA)")
    if fair_res and fair_res.get("weights_history"):
        weights_hist = fair_res.get("weights_history", [])
        w_df = pd.DataFrame(weights_hist)
        w_df.index = [f"Round {i+1}" for i in range(len(w_df))]
        
        col_w1, col_w2 = st.columns([1.5, 1])
        with col_w1:
            st.bar_chart(w_df)
        with col_w2:
            st.dataframe(w_df, use_container_width=True)
            st.info(
                "💡 **D-TAMA Observation**: Notice how the proposed strategy dynamically allocates aggregation weights: "
                "facilities with persistent demographic bias are proportionally down-weighted, while balanced nodes "
                "anchor the global model update."
            )
    else:
        st.info("ℹ️ **Standard FedAvg Mode**: Aggregation weights are strictly proportional to client dataset sizes ($w_k = n_k/N$) without disparity feedback penalty.")

    # Comprehensive Comparison Table
    st.markdown("#### Global Performance & Fairness Matrix")
    formatted_comp = {}
    for metric_name, m_dict in comp_table.items():
        formatted_comp[metric_name] = {}
        for algo_name, val in m_dict.items():
            if val is None:
                formatted_comp[metric_name][algo_name] = "N/A"
            elif metric_name == "DIR (Disparate Impact)":
                formatted_comp[metric_name][algo_name] = f"{val:.4f}"
            else:
                formatted_comp[metric_name][algo_name] = f"{val * 100:.2f}% ({val:.4f})"

    df_comp = pd.DataFrame(formatted_comp).T
    if mode == "compare":
        df_comp.columns = ["Standard FedAvg (Flower)", "Fairness-Aware Adaptive (Proposed)"]
        gain_col = []
        for idx in df_comp.index:
            raw_fed = comp_table[idx].get("FedAvg", 0.0)
            raw_fair = comp_table[idx].get("Fairness-Aware", 0.0)
            if idx in ["Accuracy", "F1-Score", "AUROC"]:
                diff = (raw_fair - raw_fed) * 100
                gain_col.append(f"{diff:+.2f}%")
            elif idx in ["DPD (Demographic Parity)", "EOD (Equalized Odds)"]:
                diff = ((raw_fed - raw_fair) / max(1e-6, raw_fed)) * 100
                gain_col.append(f"-{diff:.1f}% disparity" if diff >= 0 else f"+{abs(diff):.1f}% disparity")
            else: # DIR
                diff = raw_fair - raw_fed
                gain_col.append(f"{diff:+.4f} (closer to 1.0)")
        df_comp["Disparity Reduction / Gain"] = gain_col
    elif mode == "fairness_only":
        df_comp.columns = ["Fairness-Aware Adaptive (Proposed)"]
    else: # fedavg_only
        df_comp.columns = ["Standard FedAvg (Flower)"]
    st.table(df_comp)

    # Node-Wise Evaluation Across Participating Hospitals
    st.markdown("---")
    st.markdown("#### 🏥 Node-Wise Institutional Performance")
    st.markdown("Evaluation of global models directly on each participating hospital client's local patient population:")

    node_comp = results.get("node_comparison", {})
    if node_comp:
        node_rows = []
        for hid, vals in node_comp.items():
            h_label = hid.replace("_", " ").title()
            row = {
                "Hospital Node": h_label,
                "Patients": vals.get("sample_size"),
            }
            if mode == "compare":
                row.update({
                    "FedAvg Accuracy": f"{vals.get('fedavg_accuracy', 0)*100:.2f}%",
                    "Fairness Accuracy": f"{vals.get('fairness_accuracy', 0)*100:.2f}%",
                    "FedAvg F1": f"{vals.get('fedavg_f1', 0)*100:.2f}%",
                    "Fairness F1": f"{vals.get('fairness_f1', 0)*100:.2f}%",
                    "FedAvg DPD": f"{vals.get('fedavg_dpd', 0)*100:.2f}%",
                    "Fairness DPD": f"{vals.get('fairness_dpd', 0)*100:.2f}%",
                    "FedAvg EOD": f"{vals.get('fedavg_eod', 0)*100:.2f}%",
                    "Fairness EOD": f"{vals.get('fairness_eod', 0)*100:.2f}%",
                })
            elif mode == "fairness_only":
                row.update({
                    "Fairness Accuracy": f"{vals.get('fairness_accuracy', 0)*100:.2f}%",
                    "Fairness F1": f"{vals.get('fairness_f1', 0)*100:.2f}%",
                    "Fairness DPD (ΔDP)": f"{vals.get('fairness_dpd', 0)*100:.2f}%",
                    "Fairness EOD (ΔEO)": f"{vals.get('fairness_eod', 0)*100:.2f}%",
                })
            else: # fedavg_only
                row.update({
                    "FedAvg Accuracy": f"{vals.get('fedavg_accuracy', 0)*100:.2f}%",
                    "FedAvg F1": f"{vals.get('fedavg_f1', 0)*100:.2f}%",
                    "FedAvg DPD (ΔDP)": f"{vals.get('fedavg_dpd', 0)*100:.2f}%",
                    "FedAvg EOD (ΔEO)": f"{vals.get('fedavg_eod', 0)*100:.2f}%",
                })
            node_rows.append(row)
        df_node_display = pd.DataFrame(node_rows)
        st.dataframe(df_node_display, use_container_width=True)

        # Node-Wise Comparison Bar Chart
        fig_node, (ax_n1, ax_n2) = plt.subplots(1, 2, figsize=(11, 3.4))
        node_labels = [hid.replace("_", " ").title() for hid in node_comp.keys()]
        x_n = np.arange(len(node_labels))
        b_width = 0.35

        if mode == "compare":
            fed_node_acc = [node_comp[h].get("fedavg_accuracy", 0) * 100 for h in node_comp.keys()]
            fair_node_acc = [node_comp[h].get("fairness_accuracy", 0) * 100 for h in node_comp.keys()]
            ax_n1.bar(x_n - b_width/2, fed_node_acc, b_width, label="Standard FedAvg", color="#9CA3AF")
            ax_n1.bar(x_n + b_width/2, fair_node_acc, b_width, label="Fairness-Aware FL", color="#10B981")

            fed_node_dpd = [node_comp[h].get("fedavg_dpd", 0) * 100 for h in node_comp.keys()]
            fair_node_dpd = [node_comp[h].get("fairness_dpd", 0) * 100 for h in node_comp.keys()]
            ax_n2.bar(x_n - b_width/2, fed_node_dpd, b_width, label="Standard FedAvg", color="#EF4444")
            ax_n2.bar(x_n + b_width/2, fair_node_dpd, b_width, label="Fairness-Aware FL", color="#3B82F6")
            max_dpd_val = max(fed_node_dpd + fair_node_dpd + [10.0])
        elif mode == "fairness_only":
            fair_node_acc = [node_comp[h].get("fairness_accuracy", 0) * 100 for h in node_comp.keys()]
            ax_n1.bar(x_n, fair_node_acc, b_width, label="Fairness-Aware FL", color="#10B981")

            fair_node_dpd = [node_comp[h].get("fairness_dpd", 0) * 100 for h in node_comp.keys()]
            ax_n2.bar(x_n, fair_node_dpd, b_width, label="Fairness-Aware FL", color="#3B82F6")
            max_dpd_val = max(fair_node_dpd + [10.0])
        else: # fedavg_only
            fed_node_acc = [node_comp[h].get("fedavg_accuracy", 0) * 100 for h in node_comp.keys()]
            ax_n1.bar(x_n, fed_node_acc, b_width, label="Standard FedAvg", color="#9CA3AF")

            fed_node_dpd = [node_comp[h].get("fedavg_dpd", 0) * 100 for h in node_comp.keys()]
            ax_n2.bar(x_n, fed_node_dpd, b_width, label="Standard FedAvg", color="#EF4444")
            max_dpd_val = max(fed_node_dpd + [10.0])

        ax_n1.set_title("Node-Wise Local Accuracy (%)", fontsize=11, fontweight="bold")
        ax_n1.set_ylabel("Accuracy (%)", fontsize=10)
        ax_n1.set_xticks(x_n)
        ax_n1.set_xticklabels(node_labels, rotation=15, ha="right", fontsize=9)
        ax_n1.set_ylim(40.0, 90.0)
        ax_n1.grid(True, linestyle=":", alpha=0.6)
        ax_n1.legend(fontsize=8, loc="lower right")

        ax_n2.set_title("Node-Wise Demographic Parity ΔDP (%)", fontsize=11, fontweight="bold")
        ax_n2.set_ylabel("Disparity (%) [Lower = Fairer]", fontsize=10)
        ax_n2.set_xticks(x_n)
        ax_n2.set_xticklabels(node_labels, rotation=15, ha="right", fontsize=9)
        ax_n2.set_ylim(0.0, max_dpd_val * 1.3)
        ax_n2.grid(True, linestyle=":", alpha=0.6)
        ax_n2.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        st.pyplot(fig_node)
    else:
        df_node_display = pd.DataFrame()

    # Export & Download Results Feature
    st.markdown("---")
    st.markdown("### 📥 Export & Download Results")
    st.markdown("Download experimental artifacts, multi-round trajectories, and demographic subgroup audits:")

    # 1. Summary Comparison CSV
    df_download_summary = df_comp.copy()
    df_download_summary.index.name = "Metric"
    csv_summary = df_download_summary.to_csv().encode("utf-8")

    # 2. Round Trajectories CSV
    traj_records = []
    if mode == "compare":
        rounds_axis = [t["round"] for t in fair_res["trajectory"]]
        fed_map = {t["round"]: t for t in fedavg_res["trajectory"]}
        fair_map = {t["round"]: t for t in fair_res["trajectory"]}
        for r in rounds_axis:
            f_entry = fed_map.get(r, {})
            fa_entry = fair_map.get(r, {})
            traj_records.append({
                "Round": r,
                "FedAvg_Accuracy": f_entry.get("accuracy"),
                "Fairness_Accuracy": fa_entry.get("accuracy"),
                "FedAvg_F1": f_entry.get("f1"),
                "Fairness_F1": fa_entry.get("f1"),
                "FedAvg_DPD": f_entry.get("dpd"),
                "Fairness_DPD": fa_entry.get("dpd"),
                "FedAvg_EOD": f_entry.get("eod"),
                "Fairness_EOD": fa_entry.get("eod"),
                "FedAvg_DIR": f_entry.get("dir"),
                "Fairness_DIR": fa_entry.get("dir"),
                "FedAvg_AUROC": f_entry.get("auroc"),
                "Fairness_AUROC": fa_entry.get("auroc"),
            })
    elif mode == "fairness_only":
        for t in fair_res["trajectory"]:
            traj_records.append({
                "Round": t["round"],
                "Fairness_Accuracy": t.get("accuracy"),
                "Fairness_F1": t.get("f1"),
                "Fairness_DPD": t.get("dpd"),
                "Fairness_EOD": t.get("eod"),
                "Fairness_DIR": t.get("dir"),
                "Fairness_AUROC": t.get("auroc"),
            })
    else: # fedavg_only
        for t in fedavg_res["trajectory"]:
            traj_records.append({
                "Round": t["round"],
                "FedAvg_Accuracy": t.get("accuracy"),
                "FedAvg_F1": t.get("f1"),
                "FedAvg_DPD": t.get("dpd"),
                "FedAvg_EOD": t.get("eod"),
                "FedAvg_DIR": t.get("dir"),
                "FedAvg_AUROC": t.get("auroc"),
            })
    df_trajectories = pd.DataFrame(traj_records)
    csv_trajectories = df_trajectories.to_csv(index=False).encode("utf-8")

    # 3. Subgroup Demographic Audit CSV
    sub_records = []
    if fedavg_res:
        fed_subgroups = fedavg_res["final_metrics"].get("subgroups", {})
        for g_name, g_vals in fed_subgroups.items():
            sub_records.append({
                "Model": "Standard FedAvg",
                "Subgroup": g_name,
                "Sample_Count": g_vals.get("count"),
                "Accuracy": g_vals.get("accuracy"),
                "Positive_Rate": g_vals.get("pos_rate"),
                "TPR_Sensitivity": g_vals.get("tpr"),
                "FPR_Specificity": g_vals.get("fpr"),
                "F1_Score": g_vals.get("f1")
            })
    if fair_res:
        fair_subgroups = fair_res["final_metrics"].get("subgroups", {})
        for g_name, g_vals in fair_subgroups.items():
            sub_records.append({
                "Model": "Fairness-Aware (Proposed)",
                "Subgroup": g_name,
                "Sample_Count": g_vals.get("count"),
                "Accuracy": g_vals.get("accuracy"),
                "Positive_Rate": g_vals.get("pos_rate"),
                "TPR_Sensitivity": g_vals.get("tpr"),
                "FPR_Specificity": g_vals.get("fpr"),
                "F1_Score": g_vals.get("f1")
            })
    df_subgroups_export = pd.DataFrame(sub_records)
    csv_subgroups = df_subgroups_export.to_csv(index=False).encode("utf-8")

    # 4. Full JSON Bundle
    export_json = {
        "metadata": {
            "algorithm": "FedEquiMed (Flower + PyTorch)",
            "task": "30-Day ICU Mortality Prediction (MIMIC-IV)",
            "mode": mode,
            "communication_rounds": rounds,
            "local_epochs": epochs,
            "learning_rate": lr,
            "fairness_beta": fairness_beta,
            "momentum_gamma": momentum_gamma,
            "active_hospitals": list(active_hospitals.keys())
        },
        "comparison_metrics": comp_table,
    }
    if fedavg_res:
        export_json["standard_fedavg"] = {
            "elapsed_seconds": fedavg_res["elapsed_seconds"],
            "trajectory": fedavg_res["trajectory"],
            "final_metrics": fedavg_res["final_metrics"]
        }
    if fair_res:
        export_json["fairness_aware"] = {
            "elapsed_seconds": fair_res["elapsed_seconds"],
            "trajectory": fair_res["trajectory"],
            "weights_history": fair_res["weights_history"],
            "final_metrics": fair_res["final_metrics"]
        }
    json_bytes = json.dumps(export_json, indent=2, default=str).encode("utf-8")

    csv_nodewise = df_node_display.to_csv(index=False).encode("utf-8") if not df_node_display.empty else b""

    dl_col1, dl_col2, dl_col3, dl_col4, dl_col5 = st.columns(5)
    with dl_col1:
        st.download_button(
            label="📄 Summary Matrix",
            data=csv_summary,
            file_name="fedequimed_summary_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="📈 Trajectories",
            data=csv_trajectories,
            file_name="fedequimed_trajectories.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col3:
        st.download_button(
            label="👥 Subgroup Audit",
            data=csv_subgroups,
            file_name="fedequimed_subgroup_fairness.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col4:
        st.download_button(
            label="🏥 Node-Wise CSV",
            data=csv_nodewise,
            file_name="fedequimed_nodewise_metrics.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=df_node_display.empty
        )
    with dl_col5:
        st.download_button(
            label="📦 Full JSON Bundle",
            data=json_bytes,
            file_name="fedequimed_experiment_bundle.json",
            mime="application/json",
            use_container_width=True
        )

    # 6. Prominent PhD Research Technical Report (PDF)
    if os.path.exists("FedEquiMed_PhD_Research_Report.pdf"):
        with open("FedEquiMed_PhD_Research_Report.pdf", "rb") as f_pdf:
            pdf_bytes = f_pdf.read()
        st.markdown("")
        st.download_button(
            label="📑 Download Complete PhD Research Technical Report (PDF with Embedded Figures & Analysis)",
            data=pdf_bytes,
            file_name="FedEquiMed_PhD_Research_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

# ----------------------------------------------------
# TAB 2: Hospital Cohorts
# ----------------------------------------------------
with tab2:
    st.subheader(f"Institutional Client Profiles ({len(all_available_hospitals)} Total Available)")
    st.markdown(
        "Inspect each participating hospital node's sample size, demographic composition, and clinical outcome rates:"
    )

    h_list = list(all_available_hospitals.keys())
    num_h = len(h_list)
    h_cols = st.columns(min(3, num_h))

    for i, hid in enumerate(h_list):
        col = h_cols[i % len(h_cols)]
        h_data = all_available_hospitals[hid]
        total_p = h_data["sample_size"]
        female_count = int((h_data["gender"] == 0).sum().item())
        male_count = int((h_data["gender"] == 1).sum().item())
        mortality_count = int((h_data["y"] == 1).sum().item())

        is_custom = hid in st.session_state["custom_hospitals"] or not hid.startswith("hospital_") or hid not in ["hospital_a", "hospital_b", "hospital_c"]

        with col:
            title_prefix = "🏥" if not is_custom else "🌟"
            badge = " *(Custom Node)*" if is_custom else ""
            st.markdown(f"### {title_prefix} {hid.replace('_', ' ').title()}{badge}")
            st.markdown(f"**Total Records**: `{total_p}`")
            st.markdown(f"**Gender Distribution**: {female_count} Female ({female_count/total_p*100:.1f}%) | {male_count} Male ({male_count/total_p*100:.1f}%)")
            st.markdown(f"**Mortality Rate**: {mortality_count/total_p*100:.1f}% ({mortality_count} events)")

            # Post-FL Local Node Performance
            node_comp = results.get("node_comparison", {})
            if hid in node_comp:
                h_m = node_comp[hid]
                if mode == "compare":
                    st.markdown(
                        f"**Post-FL Node Performance**: "
                        f"Fairness Acc: `{h_m.get('fairness_accuracy', 0)*100:.2f}%` (FedAvg: `{h_m.get('fedavg_accuracy', 0)*100:.2f}%`) | "
                        f"Fairness ΔDP: `{h_m.get('fairness_dpd', 0)*100:.2f}%` (FedAvg: `{h_m.get('fedavg_dpd', 0)*100:.2f}%`)"
                    )
                elif mode == "fairness_only":
                    st.markdown(
                        f"**Post-FL Node Performance**: "
                        f"Acc: `{h_m.get('fairness_accuracy', 0)*100:.2f}%` | "
                        f"F1: `{h_m.get('fairness_f1', 0)*100:.2f}%` | "
                        f"ΔDP: `{h_m.get('fairness_dpd', 0)*100:.2f}%`"
                    )
                else:
                    st.markdown(
                        f"**Post-FL Node Performance**: "
                        f"Acc: `{h_m.get('fedavg_accuracy', 0)*100:.2f}%` | "
                        f"F1: `{h_m.get('fedavg_f1', 0)*100:.2f}%` | "
                        f"ΔDP: `{h_m.get('fedavg_dpd', 0)*100:.2f}%`"
                    )

            # Full hospital patient records table
            h_df = pd.DataFrame(h_data["raw_records"])
            col_subset = ["gender", "age", "ethnicity", "heart_rate", "systolic_bp", "glucose", "creatinine", "oxygen_sat", "lactate", "comorbidity_score", "label"]
            ordered_cols = [c for c in col_subset if c in h_df.columns] + [c for c in h_df.columns if c not in col_subset]
            h_df = h_df[ordered_cols]

            st.dataframe(h_df, height=300, use_container_width=True)
            st.download_button(
                label=f"💾 Download {hid.replace('_', ' ').title()} CSV ({total_p} records)",
                data=h_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{hid}_records_{total_p}.csv",
                mime="text/csv",
                key=f"dl_card_{hid}",
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("📋 Multi-Center Patient Records Explorer (All Data)")
    st.markdown(
        "Inspect, filter, and export the complete patient dataset across all hospital client nodes and holdout evaluation cohorts:"
    )

    # Compile all records into a single consolidated dataframe
    all_combined_records = []
    for hid_name, h_content in all_available_hospitals.items():
        for rec in h_content["raw_records"]:
            r_copy = dict(rec)
            r_copy["hospital_source"] = hid_name.replace("_", " ").title()
            all_combined_records.append(r_copy)

    if global_test and "raw_records" in global_test:
        for rec in global_test["raw_records"]:
            r_copy = dict(rec)
            r_copy["hospital_source"] = "Holdout Evaluation Cohort"
            all_combined_records.append(r_copy)

    df_master = pd.DataFrame(all_combined_records)
    master_cols = ["hospital_source", "gender", "age", "ethnicity", "heart_rate", "systolic_bp", "glucose", "creatinine", "oxygen_sat", "lactate", "comorbidity_score", "label"]
    master_cols = [c for c in master_cols if c in df_master.columns] + [c for c in df_master.columns if c not in master_cols]
    df_master = df_master[master_cols]

    # Filters
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1])
    with f_col1:
        source_opts = ["All Cohorts Combined"] + [hid.replace("_", " ").title() for hid in all_available_hospitals.keys()] + ["Holdout Evaluation Cohort"]
        selected_source = st.selectbox("Select Cohort / Hospital:", source_opts)
    with f_col2:
        selected_gen = st.selectbox("Filter Gender:", ["All", "Female", "Male"])
    with f_col3:
        selected_out = st.selectbox("Filter Outcome:", ["All", "Survived (0)", "Mortality (1)"])
    with f_col4:
        age_min, age_max = int(df_master["age"].min()), int(df_master["age"].max())
        selected_age = st.slider("Age Range:", age_min, age_max, (age_min, age_max))

    # Apply filters
    view_df = df_master.copy()
    if selected_source != "All Cohorts Combined":
        view_df = view_df[view_df["hospital_source"] == selected_source]
    if selected_gen != "All":
        view_df = view_df[view_df["gender"] == selected_gen]
    if selected_out == "Survived (0)":
        view_df = view_df[view_df["label"] == 0]
    elif selected_out == "Mortality (1)":
        view_df = view_df[view_df["label"] == 1]
    view_df = view_df[(view_df["age"] >= selected_age[0]) & (view_df["age"] <= selected_age[1])]

    # Summary metrics row
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.metric("Total Patients Displayed", f"{len(view_df):,} Records")
    with s_col2:
        fem_pct = (view_df["gender"] == "Female").mean() * 100 if len(view_df) > 0 else 0
        st.metric("Female Representation", f"{fem_pct:.1f}%")
    with s_col3:
        mort_rate = (view_df["label"] == 1).mean() * 100 if len(view_df) > 0 else 0
        st.metric("ICU Mortality Incidence", f"{mort_rate:.1f}%")
    with s_col4:
        avg_age = view_df["age"].mean() if len(view_df) > 0 else 0
        st.metric("Average Patient Age", f"{avg_age:.1f} yrs")

    # Display full table
    st.dataframe(view_df, height=450, use_container_width=True)

    # Consolidated download button
    st.download_button(
        label=f"📥 Download Filtered Cohort CSV ({len(view_df):,} records, {len(view_df.columns)} features)",
        data=view_df.to_csv(index=False).encode('utf-8'),
        file_name=f"fedequimed_{selected_source.lower().replace(' ', '_')}_{len(view_df)}_patients.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True
    )

# ----------------------------------------------------
# TAB 3: Add Custom Hospital Data (NEW FEATURE)
# ----------------------------------------------------
with tab3:
    st.subheader("➕ Add Custom Hospital Patient Data")
    st.markdown(
        "Extend the federated healthcare network by adding a new hospital client node. "
        "You can either **upload a custom CSV file** from an external hospital or **synthesize a custom clinic cohort**."
    )

    custom_mode = st.radio(
        "Choose Custom Data Method:",
        ["📁 Upload Hospital CSV File", "🧪 Synthesize Custom Hospital Cohort", "🌐 Load Benchmark Dataset from Old Website"],
        horizontal=True
    )

    # 1. Upload Hospital CSV
    if custom_mode == "📁 Upload Hospital CSV File":
        st.markdown("#### Upload Hospital Electronic Health Record (CSV)")
        st.info(
            "Expected columns (case-insensitive): `gender` (Female/Male or 0/1), `age`, `heart_rate`, "
            "`systolic_bp`, `glucose`, `creatinine`, `oxygen_sat`, `lactate`, `comorbidity_score`, and `label` (0 or 1)."
        )

        uploaded_file = st.file_uploader("Select Patient Records CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                up_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded CSV with {len(up_df)} records and {len(up_df.columns)} columns.")
                
                # Preview Table
                st.dataframe(up_df.head(6), use_container_width=True)

                col_id, col_name = st.columns(2)
                with col_id:
                    up_id = st.text_input("Hospital Identifier (ID)", value=f"hospital_d", help="Unique node ID in lowercase (e.g. hospital_d)")
                with col_name:
                    up_name = st.text_input("Hospital Display Name", value="Hospital D (Community Center)")

                save_disk = st.checkbox("Save CSV permanently to data/ directory", value=True)

                if st.button("✅ Add Uploaded Hospital to Federated Network", type="primary"):
                    parsed_node = parse_patient_dataframe(up_df)
                    st.session_state["custom_hospitals"][up_id] = parsed_node
                    
                    if save_disk:
                        saved_path = save_custom_hospital_to_disk(up_df, up_id, data_dir="data")
                        st.success(f"Saved to disk: `{saved_path}`")
                    
                    st.success(f"🎉 **{up_name}** added to federated network! Total records: {parsed_node['sample_size']}.")
                    st.rerun()

            except Exception as e:
                st.error(f"Error parsing uploaded CSV: {e}")

    # 2. Synthesize Custom Hospital
    elif custom_mode == "🧪 Synthesize Custom Hospital Cohort":
        st.markdown("#### Synthesize Custom Hospital Cohort with Tailored Skews")
        st.markdown("Specify institutional demographics and patient cohort size to simulate a specialized facility:")

        synth_col1, synth_col2 = st.columns(2)
        with synth_col1:
            s_name = st.text_input("Hospital Name", value="Hospital D (Pediatric & Women's Health)")
            s_id = st.text_input("Hospital ID", value="hospital_d")
            s_size = st.slider("Cohort Sample Size (patients)", min_value=200, max_value=2500, value=900, step=50)
        with synth_col2:
            s_female = st.slider("Female Representation (%)", min_value=10, max_value=90, value=75, step=5,
                                 help="Simulates institutions with demographic concentration (e.g., maternity or women's health).")
            s_geriatric = st.slider("Geriatric Patients ≥ 60 Years (%)", min_value=10, max_value=90, value=30, step=5)
            s_bias = st.slider("Injected Institutional Disparity Shift", min_value=-0.4, max_value=0.4, value=0.15, step=0.05,
                               help="Simulates institutional diagnostic variation or resource discrepancy.")

        if st.button("🧪 Generate & Add Synthetic Hospital Node", type="primary"):
            synth_df = generate_synthetic_custom_hospital(
                hospital_name=s_name,
                sample_size=s_size,
                female_pct=s_female / 100.0,
                geriatric_pct=s_geriatric / 100.0,
                bias_shift=s_bias,
                seed=123
            )
            parsed_synth = parse_patient_dataframe(synth_df)
            st.session_state["custom_hospitals"][s_id] = parsed_synth
            save_custom_hospital_to_disk(synth_df, s_id, data_dir="data")
            
            st.success(f"🎉 Created & Added **{s_name}** ({s_size} records, {s_female}% Female)! Ready for FL simulation.")
            st.rerun()

    # 3. Load Preset Benchmark Dataset from Old System
    else:
        st.markdown("#### Clinical Benchmark Datasets from the Original Repository")
        st.markdown(
            "Select and enroll one of the top 5 medical AI benchmarks from the original system "
            "as a hospital client node or use it to test global model generalization:"
        )

        b_keys = list(BENCHMARK_DATASETS_META.keys())
        b_labels = {
            "mimic_iv": "🏥 1. MIMIC-IV (PhysioNet) — ICU Electronic Health Records (~380k patients)",
            "eicu": "🏥 2. eICU Research DB — Multi-center critical care (~200k stays)",
            "medmnist": "🖼️ 3. MedMNIST — Medical Imaging representation proxies (700k+)",
            "covid19_imaging": "🫁 4. COVID-19 Imaging — Pulmonary acute hypoxemia (~50k+)",
            "tcga": "🧬 5. TCGA — The Cancer Genome Atlas cancer genomics (~11k patients)"
        }

        b_col1, b_col2 = st.columns([2, 1])
        with b_col1:
            selected_b_key = st.selectbox(
                "Select Benchmark Repository:",
                options=b_keys,
                format_func=lambda k: b_labels.get(k, k)
            )
        with b_col2:
            b_count = st.select_slider(
                "Sample Size (records):",
                options=[50, 150, 300, 500, 1000],
                value=300
            )

        meta = BENCHMARK_DATASETS_META[selected_b_key]

        # Clinical Profile Banner
        st.markdown(
            f"""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <strong style="color: #166534; font-size: 1.05rem;">{meta['name']}</strong>
                    <a href="{meta['url']}" target="_blank" style="color: #0284C7; font-size: 0.85rem; font-weight: 600; text-decoration: underline;">🔗 Official Repository</a>
                </div>
                <div style="font-size: 0.85rem; color: #374151; margin-bottom: 0.4rem;"><b>Modality & Scope</b>: {meta['type']} ({meta['records_approx']})</div>
                <div style="font-size: 0.85rem; color: #4B5563; margin-bottom: 0.5rem;">{meta['description']}</div>
                <div style="font-size: 0.82rem; color: #DC2626; font-weight: 500;">⚠️ <b>Demographic Gap / Bias Profile</b>: {meta['bias_profile']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        parsed_bench, df_bench = load_preset_benchmark_cohort(selected_b_key, count=b_count, seed=42)

        st.markdown("##### Preview of Generated Benchmark Patient Records:")
        st.dataframe(df_bench.head(5), use_container_width=True)

        b_act1, b_act2, b_act3 = st.columns([1.5, 1.5, 1])
        with b_act1:
            if st.button("➕ Enroll Benchmark as Hospital Client Node", type="primary"):
                b_node_id = f"hospital_{selected_b_key}"
                st.session_state["custom_hospitals"][b_node_id] = parsed_bench
                save_custom_hospital_to_disk(df_bench, b_node_id, data_dir="data")
                st.success(f"🎉 Enrolled **{meta['name']}** as `{b_node_id}` ({b_count} records) into federated network!")
                st.rerun()

        with b_act2:
            if st.button("🧪 Evaluate Global Models on this Benchmark"):
                st.session_state[f"test_eval_{selected_b_key}"] = True

        with b_act3:
            st.download_button(
                label="💾 Download CSV",
                data=df_bench.to_csv(index=False).encode('utf-8'),
                file_name=f"{selected_b_key}_cohort_{b_count}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Show evaluation if triggered
        if st.session_state.get(f"test_eval_{selected_b_key}", False):
            st.markdown("##### 🧪 Out-of-Distribution Evaluation on Selected Benchmark:")
            X_b = parsed_bench["X"]
            y_b = parsed_bench["y"]
            g_b = parsed_bench["gender"]

            eval_cols = {}
            if fedavg_res and "final_weights" in fedavg_res and fedavg_res["final_weights"]:
                m_fed = PyTorchLogisticRegression(num_features=X_b.shape[1])
                m_fed.set_weights(fedavg_res["final_weights"])
                m_fed.eval()
                with torch.no_grad():
                    preds_fed = m_fed.predict(X_b)
                    probs_fed = m_fed.predict_proba(X_b)
                    ev_fed = compute_clinical_fairness_metrics(y_b, preds_fed, probs_fed, g_b)
                eval_cols["Standard FedAvg (Flower)"] = [
                    f"{ev_fed.get('accuracy', 0)*100:.2f}% ({ev_fed.get('accuracy', 0):.4f})",
                    f"{ev_fed.get('f1', 0)*100:.2f}% ({ev_fed.get('f1', 0):.4f})",
                    f"{ev_fed.get('dpd', 0)*100:.2f}% ({ev_fed.get('dpd', 0):.4f})",
                    f"{ev_fed.get('eod', 0)*100:.2f}% ({ev_fed.get('eod', 0):.4f})",
                    f"{ev_fed.get('auroc', 0)*100:.2f}% ({ev_fed.get('auroc', 0):.4f})"
                ]
            if fair_res and "final_weights" in fair_res and fair_res["final_weights"]:
                m_fair = PyTorchLogisticRegression(num_features=X_b.shape[1])
                m_fair.set_weights(fair_res["final_weights"])
                m_fair.eval()
                with torch.no_grad():
                    preds_fair = m_fair.predict(X_b)
                    probs_fair = m_fair.predict_proba(X_b)
                    ev_fair = compute_clinical_fairness_metrics(y_b, preds_fair, probs_fair, g_b)
                eval_cols["Fairness-Aware Adaptive (Proposed)"] = [
                    f"{ev_fair.get('accuracy', 0)*100:.2f}% ({ev_fair.get('accuracy', 0):.4f})",
                    f"{ev_fair.get('f1', 0)*100:.2f}% ({ev_fair.get('f1', 0):.4f})",
                    f"{ev_fair.get('dpd', 0)*100:.2f}% ({ev_fair.get('dpd', 0):.4f})",
                    f"{ev_fair.get('eod', 0)*100:.2f}% ({ev_fair.get('eod', 0):.4f})",
                    f"{ev_fair.get('auroc', 0)*100:.2f}% ({ev_fair.get('auroc', 0):.4f})"
                ]
            bench_eval_df = pd.DataFrame(eval_cols, index=["Accuracy", "F1-Score", "Demographic Parity (ΔDP)", "Equalized Odds (ΔEO)", "AUROC"])
            st.table(bench_eval_df)

    # Active Custom Hospitals List
    if st.session_state["custom_hospitals"]:
        st.markdown("---")
        st.markdown("#### Currently Loaded Custom Hospital Nodes:")
        for c_id, c_data in st.session_state["custom_hospitals"].items():
            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                st.markdown(f"• **{c_id}**: {c_data['sample_size']} patients | Female: {int((c_data['gender']==0).sum())} | Male: {int((c_data['gender']==1).sum())}")
            with del_col2:
                if st.button(f"🗑️ Remove {c_id}", key=f"del_{c_id}"):
                    del st.session_state["custom_hospitals"][c_id]
                    st.rerun()



# ----------------------------------------------------
# TAB 4: Demographic Fairness Audit
# ----------------------------------------------------
with tab4:
    st.subheader("Subgroup Demographic Breakdown (Gender: Female vs Male)")
    
    subgroups = fair_res["final_metrics"].get("subgroups", {}) if fair_res else {}
    fed_subgroups = fedavg_res["final_metrics"].get("subgroups", {}) if fedavg_res else {}

    def format_subgroup_table(sub_dict):
        rows = []
        for g_name, g_vals in sub_dict.items():
            rows.append({
                "Subgroup": g_name,
                "Patients": g_vals.get("count", 0),
                "Accuracy": f"{g_vals.get('accuracy', 0)*100:.2f}%",
                "Mortality Rate": f"{g_vals.get('pos_rate', 0)*100:.2f}%",
                "TPR (Sensitivity)": f"{g_vals.get('tpr', 0)*100:.2f}%",
                "FPR (False Alarm)": f"{g_vals.get('fpr', 0)*100:.2f}%",
                "F1-Score": f"{g_vals.get('f1', 0)*100:.2f}%"
            })
        return pd.DataFrame(rows).set_index("Subgroup")

    if mode == "compare" and subgroups and fed_subgroups:
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.markdown("#### Standard FedAvg Subgroup Performance")
            st.dataframe(format_subgroup_table(fed_subgroups), use_container_width=True)

        with sub_col2:
            st.markdown("#### Fairness-Aware FL Subgroup Performance")
            st.dataframe(format_subgroup_table(subgroups), use_container_width=True)

        st.markdown("#### True Positive Rate (TPR / Sensitivity) Parity")
        fig_tpr, ax_tpr = plt.subplots(figsize=(7, 2.5))
        groups = list(subgroups.keys())
        fed_tprs = [fed_subgroups[g]["tpr"] * 100 for g in groups]
        fair_tprs = [subgroups[g]["tpr"] * 100 for g in groups]

        x_indices = np.arange(len(groups))
        width = 0.35

        ax_tpr.bar(x_indices - width/2, fed_tprs, width, label="Standard FedAvg", color="#9CA3AF")
        ax_tpr.bar(x_indices + width/2, fair_tprs, width, label="Fairness-Aware FL", color="#10B981")
        ax_tpr.set_ylabel("True Positive Rate (%)", fontsize=10)
        ax_tpr.set_xticks(x_indices)
        ax_tpr.set_xticklabels(groups)
        ax_tpr.set_ylim(0.0, 100.0)
        ax_tpr.legend(loc="lower right")
        ax_tpr.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_tpr)

    elif mode == "fairness_only" and subgroups:
        st.markdown("#### Fairness-Aware FL Subgroup Performance")
        st.dataframe(format_subgroup_table(subgroups), use_container_width=True)

        st.markdown("#### True Positive Rate (TPR / Sensitivity) Across Subgroups")
        fig_tpr, ax_tpr = plt.subplots(figsize=(6, 2.5))
        groups = list(subgroups.keys())
        fair_tprs = [subgroups[g]["tpr"] * 100 for g in groups]
        x_indices = np.arange(len(groups))
        ax_tpr.bar(x_indices, fair_tprs, 0.35, label="Fairness-Aware FL", color="#10B981")
        ax_tpr.set_ylabel("True Positive Rate (%)", fontsize=10)
        ax_tpr.set_xticks(x_indices)
        ax_tpr.set_xticklabels(groups)
        ax_tpr.set_ylim(0.0, 100.0)
        ax_tpr.legend(loc="lower right")
        ax_tpr.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_tpr)

    elif mode == "fedavg_only" and fed_subgroups:
        st.markdown("#### Standard FedAvg Subgroup Performance")
        st.dataframe(format_subgroup_table(fed_subgroups), use_container_width=True)

        st.markdown("#### True Positive Rate (TPR / Sensitivity) Across Subgroups")
        fig_tpr, ax_tpr = plt.subplots(figsize=(6, 2.5))
        groups = list(fed_subgroups.keys())
        fed_tprs = [fed_subgroups[g]["tpr"] * 100 for g in groups]
        x_indices = np.arange(len(groups))
        ax_tpr.bar(x_indices, fed_tprs, 0.35, label="Standard FedAvg", color="#9CA3AF")
        ax_tpr.set_ylabel("True Positive Rate (%)", fontsize=10)
        ax_tpr.set_xticks(x_indices)
        ax_tpr.set_xticklabels(groups)
        ax_tpr.set_ylim(0.0, 100.0)
        ax_tpr.legend(loc="lower right")
        ax_tpr.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_tpr)

# ----------------------------------------------------
# TAB 5: Patient Clinical Risk Predictor
# ----------------------------------------------------
with tab5:
    st.subheader("🩺 Single Patient ICU Risk Predictor")
    st.markdown(
        "Enter vital signs and clinical biomarkers for an individual patient to inspect "
        "the calibrated risk score predicted by the trained global model(s)."
    )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        p_gender = st.selectbox("Gender", ["Female", "Male"])
        p_age = st.slider("Age (years)", 18, 95, 68)
        p_hr = st.slider("Heart Rate (bpm)", 45, 160, 88)
    with p_col2:
        p_sbp = st.slider("Systolic BP (mmHg)", 80, 210, 134)
        p_glucose = st.slider("Blood Glucose (mg/dL)", 60, 350, 125)
        p_creatinine = st.slider("Serum Creatinine (mg/dL)", 0.3, 6.0, 1.4, step=0.1)
    with p_col3:
        p_o2 = st.slider("Oxygen Saturation SpO2 (%)", 75.0, 100.0, 93.5, step=0.5)
        p_lactate = st.slider("Serum Lactate (mmol/L)", 0.5, 8.0, 2.2, step=0.1)
        p_comorb = st.slider("Comorbidity Score (Elixhauser)", 0, 8, 3)

    # Standardize input features
    norm_gender = 1.0 if p_gender == "Male" else -1.0
    norm_age = (p_age - 55.0) / 18.0
    norm_ethnicity = 0.0
    norm_hr = (p_hr - 80.0) / 15.0
    norm_sbp = (p_sbp - 125.0) / 20.0
    norm_glucose = (p_glucose - 110.0) / 35.0
    norm_creatinine = (p_creatinine - 1.0) / 0.5
    norm_o2 = (95.0 - p_o2) / 4.0
    norm_lactate = (p_lactate - 1.5) / 0.8
    norm_comorb = (p_comorb - 2.5) / 1.5

    patient_tensor = torch.tensor([[
        norm_gender, norm_age, norm_ethnicity, norm_hr, norm_sbp,
        norm_glucose, norm_creatinine, norm_o2, norm_lactate, norm_comorb
    ]], dtype=torch.float32)

    if mode == "compare":
        fed_model = PyTorchLogisticRegression(num_features=10)
        fair_model = PyTorchLogisticRegression(num_features=10)
        if fedavg_res and "final_weights" in fedavg_res and fedavg_res["final_weights"]:
            fed_model.set_weights(fedavg_res["final_weights"])
        if fair_res and "final_weights" in fair_res and fair_res["final_weights"]:
            fair_model.set_weights(fair_res["final_weights"])

        fed_model.eval()
        fair_model.eval()
        with torch.no_grad():
            fed_prob = float(fed_model.predict_proba(patient_tensor).item())
            fair_prob = float(fair_model.predict_proba(patient_tensor).item())

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown("#### Standard FedAvg Prediction")
            st.metric("30-Day Mortality / Adverse Event Risk", f"{fed_prob*100:.1f}%")
            st.write("Status:", "⚠️ **High Risk**" if fed_prob >= 0.5 else "✅ **Low/Normal Risk**")
        with res_col2:
            st.markdown("#### Fairness-Aware FL Prediction")
            st.metric("30-Day Mortality / Adverse Event Risk", f"{fair_prob*100:.1f}%")
            st.write("Status:", "⚠️ **High Risk**" if fair_prob >= 0.5 else "✅ **Low/Normal Risk**")

    elif mode == "fairness_only":
        fair_model = PyTorchLogisticRegression(num_features=10)
        if fair_res and "final_weights" in fair_res and fair_res["final_weights"]:
            fair_model.set_weights(fair_res["final_weights"])

        fair_model.eval()
        with torch.no_grad():
            fair_prob = float(fair_model.predict_proba(patient_tensor).item())

        st.markdown("#### Fairness-Aware FL Prediction")
        st.metric("30-Day Mortality / Adverse Event Risk", f"{fair_prob*100:.1f}%")
        st.write("Status:", "⚠️ **High Risk**" if fair_prob >= 0.5 else "✅ **Low/Normal Risk**")

    else: # fedavg_only
        fed_model = PyTorchLogisticRegression(num_features=10)
        if fedavg_res and "final_weights" in fedavg_res and fedavg_res["final_weights"]:
            fed_model.set_weights(fedavg_res["final_weights"])

        fed_model.eval()
        with torch.no_grad():
            fed_prob = float(fed_model.predict_proba(patient_tensor).item())

        st.markdown("#### Standard FedAvg Prediction")
        st.metric("30-Day Mortality / Adverse Event Risk", f"{fed_prob*100:.1f}%")
        st.write("Status:", "⚠️ **High Risk**" if fed_prob >= 0.5 else "✅ **Low/Normal Risk**")
