"""
Fairness-Aware Federated Learning in Healthcare Package
"""

from fl_engine.dataset import generate_federated_healthcare_datasets, FEATURE_NAMES, SENSITIVE_ATTRIBUTES
from fl_engine.models import ClinicalRiskModel
from fl_engine.bias_detector import BiasDetector, calculate_subgroup_metrics
from fl_engine.hospital_node import HospitalNode
from fl_engine.global_server import GlobalFederatedServer

__all__ = [
    "generate_federated_healthcare_datasets",
    "FEATURE_NAMES",
    "SENSITIVE_ATTRIBUTES",
    "ClinicalRiskModel",
    "BiasDetector",
    "calculate_subgroup_metrics",
    "HospitalNode",
    "GlobalFederatedServer"
]
