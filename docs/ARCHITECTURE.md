# FedEquiMed System Architecture

## 1. High-Level Design

FedEquiMed is a fairness-aware federated learning system designed for multi-center clinical healthcare environments. It coordinates decentralized training across heterogeneous hospital nodes without centralizing Protected Health Information (PHI).

```
Hospital Node A              Hospital Node B              Hospital Node C
(Urban Medical Center)      (Rural Geriatric)            (Cardiac Specialty)
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ Local Patient EHR    │    │ Local Patient EHR    │    │ Local Patient EHR    │
│ Subgroup Reweighting │    │ Subgroup Reweighting │    │ Subgroup Reweighting │
│ Local ClinDisentangle│    │ Local ClinDisentangle│    │ Local ClinDisentangle│
│ Local DP Perturb     │    │ Local DP Perturb     │    │ Local DP Perturb     │
└──────────┬───────────┘    └──────────┬───────────┘    └──────────┬───────────┘
           │ Sanitized                 │ Sanitized                 │ Sanitized
           │ Deltas (ΔW, Δb)           │ Deltas (ΔW, Δb)           │ Deltas (ΔW, Δb)
           ▼                           ▼                           ▼
    ════════════════════════════════════════════════════════════════════════════
                        Central Federated Aggregator (Server)
                        - D-TAMA Momentum Aggregation
                        - FairFed Adaptive Weighting
                        - Multi-Center Benchmark Evaluation
                        - Zero Raw Record Storage
    ════════════════════════════════════════════════════════════════════════════
```

## 2. Technology Stack & Zero-Database Architecture
- **Language**: Python 3 standard library (`http.server`, `urllib`, `json`, `math`, `random`, `secrets`).
- **Frontend**: Vanilla JavaScript (ES6+), HTML5, CSS3 with responsive mobile-first grid and flexbox.
- **Persistence Layer**: File-based artifacts (`data/experiments/{experiment_id}/` storing `config.json`, `metrics.csv`, and `results.json`). No external SQL or NoSQL database required.

## 3. Core Modules
- `fl_engine/models.py`: Linear clinical risk classifiers and `ClinDisentangleNet` dual-stream representations with `CARP-Loss`.
- `fl_engine/bias_detector.py`: Evaluates Demographic Parity Difference ($\Delta\text{DP}$), Equalized Odds ($\Delta\text{EO}$), and Disparate Impact Ratio ($\text{DIR}$).
- `fl_engine/privacy_engine.py`: $L_2$ gradient clipping, calibrated Gaussian perturbation, and Renyi DP analytical budgeting.
- `fl_engine/hospital_node.py`: Simulates private institutional silos, local training, and payload export.
- `fl_engine/global_server.py`: Server-side aggregation algorithms (FedAvg, FairFed, D-TAMA).
- `fl_engine/experiments.py`: Experiment pipeline, multi-seed executor, and ablation framework.
