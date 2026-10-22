# FedEquiMed: Fairness-Aware Federated Learning in Healthcare

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies: Standard Library](https://img.shields.io/badge/dependencies-0%20external-brightgreen.svg)](https://docs.python.org/3/)
[![Mobile Responsive](https://img.shields.io/badge/UI-Mobile%20Responsive-teal.svg)](#responsive-ui-support)

FedEquiMed is a fairness-aware federated learning system designed for decentralized healthcare risk prediction across heterogeneous hospital networks. It mitigates demographic disparities across Gender, Age, and Ethnicity without centralizing Protected Health Information (PHI).

---

## 1. Project Overview & Clinical Problem
Healthcare algorithms deployed across diverse hospitals frequently exhibit acute demographic disparities due to non-IID demographic shifts and clinical representation gaps. Traditional Federated Averaging (FedAvg) aggregates parameters proportionally to sample volume, causing global models to absorb and amplify the institutional disparities of larger clinics.

FedEquiMed introduces:
1. **In-Processing Adaptive Reweighting (FedIDA)**: Dynamically balances subgroup representations during local gradient descent.
2. **ClinDisentangleNet**: Dual-stream neural representation separating clinical physiological biomarkers from demographic observational biases.
3. **Covariate-Adjusted Risk Parity (CARP-Loss)**: Conditions fairness penalties on baseline patient acuity, avoiding false penalties for natural biological etiology differences.
4. **D-TAMA Aggregation**: Dual-Tier Adaptive Momentum Aggregation that tracks institutional bias momentum across communication rounds.

---

## 2. Architecture & Technology Stack
- **Zero External Dependencies**: Powered exclusively by Python's standard library (`http.server`, `urllib`, `json`, `math`, `secrets`).
- **Zero-Database Persistence**: All experiment configurations, convergence metrics, and statistical summaries are stored directly as file-based artifacts under `data/experiments/` (`config.json`, `metrics.csv`, `results.json`).
- **Client-Side Interface**: Lightweight vanilla JavaScript, HTML5 Canvas charts, and mobile-first CSS.

---

## 3. Primary Healthcare Task (MIMIC-IV)
- **Primary Task**: 30-Day ICU Adverse Clinical Event & Mortality Risk Prediction.
- **Predictive Features (10)**: Normalized Gender, Standardized Age, Ethnicity, Heart Rate, Systolic BP, Blood Glucose, Serum Creatinine, $S_p O_2$, Serum Lactate, Comorbidity Score.
- **Sensitive Demographics**: Gender (Female / Male), Age Group ($<60$ / $\ge 60$), Ethnicity.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, AUROC, Demographic Parity Difference ($\Delta\text{DP}$), Equalized Odds Difference ($\Delta\text{EO}$), Disparate Impact Ratio ($\text{DIR}$).

---

## 4. RO1 Setup: Flower + PyTorch + Streamlit

This repository includes a full Research Objective 1 (RO1) implementation combining:
- **PyTorch**: Logistic Regression model (`nn.Module`) with BCE loss and subgroup adaptive reweighting.
- **Flower (`flwr`)**: Federated simulation orchestrating simulated hospital nodes (`hospital_a`, `hospital_b`, `hospital_c`).
- **Standard FedAvg vs Fairness-Aware Aggregation**: Comparing sample-volume baseline against adaptive momentum aggregation (D-TAMA).
- **Fairness & Clinical Performance**: Calculating **Accuracy**, **F1-score**, **DPD (Demographic Parity Difference)**, and **EOD (Equalized Odds Difference)** on the holdout MIMIC-IV cohort.
- **Streamlit Interactive Dashboard**: Real-time convergence charts, hospital cohort exploration, and single-patient ICU risk predictor.

### Quick Start (RO1):
```bash
# 1. Run Flower simulation benchmark via CLI
python3 run_flower_simulation.py --rounds 5

# 2. Launch the interactive Streamlit Dashboard
streamlit run streamlit_app.py

# 3. Run all unit tests (Flower + PyTorch + Fairness)
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 5. Running the Original Standard-Library Server

### Prerequisites
- Python 3.8 or higher.

### Start the Application Server
```bash
# Optional: Set environment passkey for administrator controls
export ADMIN_PASSKEY="your_secure_passkey"
export PORT=8085

python3 server.py
```
Open your browser at `http://localhost:8085`.

---

## 5. Running Automated Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

---

## 6. Running Reproducible Experiments via CLI
```bash
# Execute 5-seed benchmark evaluation (Seeds 42, 123, 456, 789, 999)
python3 -c "
from fl_engine.experiments import run_and_persist_experiment
config = {
    'algorithm': 'FedEquiMed',
    'dataset': 'mimic_iv',
    'num_clients': 3,
    'rounds': 5,
    'seeds': [42, 123, 456, 789, 999]
}
results = run_and_persist_experiment(config)
print('Experiment ID:', results['experiment_id'])
print('Accuracy:', results['metrics_summary']['accuracy'])
print('Gender ΔDP:', results['metrics_summary']['delta_dp'])
"
```

---

## 7. 3-Minute Project Demonstration Flow
1. **Step 1 — Review Nodes**: Open Tab 1 ("Hospital Nodes") to inspect simulated hospitals with demographic skews.
2. **Step 2 — Train Local Node**: Click "Train Model" on Hospital A to observe local adaptive reweighting and DP perturbation.
3. **Step 3 — Run Round**: Open Tab 2 ("Aggregation & Charts") and click "Run 1 Round" to observe D-TAMA dynamic weighting and trajectory convergence.
4. **Step 4 — Benchmark**: Open Tab 3 ("Multi-Seed Benchmark") and click "Run 5-Round Benchmark" to compare FedAvg, FairFed, and FedEquiMed.
5. **Step 5 — Pareto Trade-off**: Inspect the Accuracy vs Fairness Trade-off scatter plot.
6. **Step 6 — Ablation**: Open Tab 4 ("Ablation Study") to evaluate the contribution of each module.
7. **Step 7 — Privacy**: Open Tab 5 ("Privacy & DP") to review the $(\epsilon, \delta)$ analytical guarantee.
8. **Step 8 — History**: Open Tab 6 ("Experiment History") to inspect saved experiment files on disk.

---

## 8. Responsive UI Support
FedEquiMed is fully responsive and verified across:
- **320px**: Mobile extra-small (e.g. iPhone SE 1st Gen)
- **375px**: Mobile standard (iPhone 12/13 mini)
- **390px**: iPhone standard
- **414px**: iPhone Plus / Max
- **768px**: Tablet portrait (iPad)
- **1024px**: Laptop / iPad landscape
- **1440px**: Desktop display

---

## 9. Limitations & Research Notes
- This implementation runs hospital nodes as virtualized local processes in Python. In clinical deployment, each hospital node executes as an independent container inside the hospital's intranet.
- Differential privacy perturbation reduces reconstruction risk but involves a utility-privacy trade-off documented in `docs/PRIVACY.md`.
