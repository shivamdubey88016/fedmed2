"""
Flower-based Federated Healthcare System with PyTorch and Fairness-Aware Adaptive Aggregation
"""

from flower_fl.model import PyTorchLogisticRegression
from flower_fl.dataset_loader import (
    parse_patient_csv,
    parse_patient_dataframe,
    load_all_hospital_datasets,
    generate_synthetic_custom_hospital,
    save_custom_hospital_to_disk,
    load_preset_benchmark_cohort,
    BENCHMARK_DATASETS_META
)
from flower_fl.metrics import compute_clinical_fairness_metrics
from flower_fl.client import HospitalFlowerClient, LocalClientProxy
from flower_fl.strategy import FairnessAwareFedAvg
from flower_fl.simulation import run_federated_simulation, run_comparative_benchmark

__all__ = [
    "PyTorchLogisticRegression",
    "parse_patient_csv",
    "parse_patient_dataframe",
    "load_all_hospital_datasets",
    "generate_synthetic_custom_hospital",
    "save_custom_hospital_to_disk",
    "load_preset_benchmark_cohort",
    "BENCHMARK_DATASETS_META",
    "compute_clinical_fairness_metrics",
    "HospitalFlowerClient",
    "LocalClientProxy",
    "FairnessAwareFedAvg",
    "run_federated_simulation",
    "run_comparative_benchmark"
]
