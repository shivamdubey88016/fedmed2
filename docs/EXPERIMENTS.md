# FedEquiMed Experiment Pipeline & Benchmark Guide

## 1. Primary Healthcare Benchmark Task
- **Canonical Dataset**: MIMIC-IV Critical Care EHR Cohort (simulated multi-hospital non-IID environment).
- **Target Variable**: 30-Day In-Hospital Adverse Clinical Event & Mortality Risk ($Y \in \{0, 1\}$).
- **Sensitive Demographics**: Gender ($A \in \{0, 1\}$: Female / Male), Age Group ($A \in \{0, 1\}$: Young $<60$ / Senior $\ge 60$), Ethnicity ($A \in \{0, 1, 2\}$).
- **Predictive Features (10 dimensions)**:
  1. Gender (normalized)
  2. Age (standardized)
  3. Ethnicity (cohort indicator)
  4. Heart rate
  5. Systolic blood pressure
  6. Blood glucose
  7. Serum creatinine
  8. Oxygen saturation ($S_p O_2$)
  9. Serum lactate
  10. Chronic comorbidity index

## 2. Directory Structure for Persisted Experiments
```
data/
└── experiments/
    ├── FEQ-2026-001/
    │   ├── config.json       # Hyperparameters, seeds, split config
    │   ├── metrics.csv       # Per-round convergence metrics
    │   └── results.json      # Summary statistics (mean ± std dev)
    └── ...
```

## 3. Supported Baseline Methods
1. **FedAvg (McMahan et al., 2017)**: Volume-weighted aggregation ($w_k = n_k / N$).
2. **FairFed (Ezzeldin et al., 2023)**: Disparity-penalized dynamic weighting ($\alpha_k = (n_k / N) \exp(-\beta \Delta_k)$).
3. **FedEquiMed (Proposed)**: ClinDisentangleNet + Covariate-Adjusted Risk Parity (CARP-Loss) + D-TAMA dual-tier momentum aggregator.

## 4. Multi-Seed Robustness Protocol
Default random seeds: `42, 123, 456, 789, 999`.
Summary results report:
$$\text{Metric} = \text{Mean} \pm \text{Std Dev}$$
