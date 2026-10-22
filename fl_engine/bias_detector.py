"""
Secure Bias Detection and Fairness Auditing Engine for Federated Healthcare
Audits demographic disparities across sensitive attributes (Gender, Age, Ethnicity)
without transmitting or exposing raw patient records.
"""

from typing import List, Dict, Any, Tuple
import math

def calculate_auroc(y_true: List[int], y_score: List[float]) -> float:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (AUROC)
    using the Wilcoxon-Mann-Whitney statistic.
    Handles edge cases safely (all-positive or all-negative cohorts return 0.50).
    """
    if not y_true or not y_score or len(y_true) != len(y_score):
        return 0.50

    pos_scores = [s for y, s in zip(y_true, y_score) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_score) if y == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.50

    # Mann-Whitney U calculation
    # Pairwise comparison with tie handling (0.5 for ties)
    u_stat = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                u_stat += 1.0
            elif ps == ns:
                u_stat += 0.5

    auroc = u_stat / (n_pos * n_neg)
    return round(auroc, 4)

def calculate_subgroup_metrics(
    predictions: List[int],
    ground_truth: List[int],
    attribute_values: List[int],
    attribute_name: str
) -> Dict[str, Any]:
    """
    Computes rigorous fairness and performance metrics for a specific sensitive attribute.
    Groups are split by unique values in attribute_values.

    Demographic Parity Difference:
      ΔDP = max_g P(Ŷ=1 | A=g) - min_g P(Ŷ=1 | A=g)

    Equalized Odds Difference:
      ΔEO = 0.5 * [(max_g TPR_g - min_g TPR_g) + (max_g FPR_g - min_g FPR_g)]

    Disparate Impact Ratio:
      DIR = min_g P(Ŷ=1 | A=g) / (max_g P(Ŷ=1 | A=g) + eps)
    """
    unique_groups = sorted(list(set(attribute_values)))
    group_stats = {}

    for grp in unique_groups:
        idx_list = [i for i, val in enumerate(attribute_values) if val == grp]
        count = len(idx_list)
        if count == 0:
            continue

        tp = sum(1 for i in idx_list if predictions[i] == 1 and ground_truth[i] == 1)
        fp = sum(1 for i in idx_list if predictions[i] == 1 and ground_truth[i] == 0)
        tn = sum(1 for i in idx_list if predictions[i] == 0 and ground_truth[i] == 0)
        fn = sum(1 for i in idx_list if predictions[i] == 0 and ground_truth[i] == 1)

        total_pos = tp + fn
        total_neg = fp + tn
        pred_pos = tp + fp

        acc = (tp + tn) / count if count > 0 else 0.0
        pos_rate = pred_pos / count if count > 0 else 0.0
        tpr = tp / total_pos if total_pos > 0 else 0.0
        fpr = fp / total_neg if total_neg > 0 else 0.0
        fnr = fn / total_pos if total_pos > 0 else 0.0
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        group_stats[grp] = {
            "count": count,
            "accuracy": round(acc, 4),
            "pos_rate": round(pos_rate, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }

    # If no groups found
    if not group_stats:
        return {
            "attribute": attribute_name,
            "group_stats": {},
            "demographic_parity_difference": 0.0,
            "disparate_impact_ratio": 1.0,
            "equalized_odds_difference": 0.0,
            "delta_tpr": 0.0,
            "delta_fpr": 0.0
        }

    # Generalized across all demographic groups present
    pos_rates = [group_stats[g]["pos_rate"] for g in unique_groups if g in group_stats]
    tprs = [group_stats[g]["tpr"] for g in unique_groups if g in group_stats]
    fprs = [group_stats[g]["fpr"] for g in unique_groups if g in group_stats]

    # Demographic Parity Difference: max_g P(Ŷ=1|A=g) - min_g P(Ŷ=1|A=g)
    dpd = (max(pos_rates) - min(pos_rates)) if pos_rates else 0.0

    # Disparate Impact Ratio: min_g P(Ŷ=1|A=g) / max_g P(Ŷ=1|A=g) with safe non-zero handling
    eps = 1e-7
    if pos_rates:
        max_p = max(pos_rates)
        min_p = min(pos_rates)
        if max_p == 0.0 and min_p == 0.0:
            dir_ratio = 1.0
        else:
            dir_ratio = min_p / (max_p + eps)
    else:
        dir_ratio = 1.0

    # Equalized Odds Differences
    delta_tpr = (max(tprs) - min(tprs)) if tprs else 0.0
    delta_fpr = (max(fprs) - min(fprs)) if fprs else 0.0
    # Canonical Equalized Odds disparity: 0.5 * (ΔTPR + ΔFPR) or max(ΔTPR, ΔFPR)
    # Both are reported for transparency
    eod = round(0.5 * (delta_tpr + delta_fpr), 4)

    return {
        "attribute": attribute_name,
        "group_stats": group_stats,
        "demographic_parity_difference": round(dpd, 4),
        "disparate_impact_ratio": round(dir_ratio, 4),
        "equalized_odds_difference": eod,
        "delta_tpr": round(delta_tpr, 4),
        "delta_fpr": round(delta_fpr, 4)
    }

class BiasDetector:
    """
    Hospital-side secure fairness auditor.
    Executes local inference and generates mathematical fairness summaries.
    Ensures raw records leave the auditing module only as aggregated statistical scalars.
    """

    @staticmethod
    def audit_model(
        model,
        dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not dataset:
            return {
                "status": "empty_dataset",
                "sample_size": 0,
                "overall_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auroc": 0.50,
                "composite_disparity": 0.0,
                "gender_audit": {
                    "demographic_parity_difference": 0.0,
                    "equalized_odds_difference": 0.0,
                    "disparate_impact_ratio": 1.0,
                    "delta_tpr": 0.0,
                    "delta_fpr": 0.0,
                    "group_stats": {}
                },
                "age_audit": {
                    "demographic_parity_difference": 0.0,
                    "equalized_odds_difference": 0.0,
                    "disparate_impact_ratio": 1.0,
                    "delta_tpr": 0.0,
                    "delta_fpr": 0.0,
                    "group_stats": {}
                },
                "ethnicity_audit": {
                    "demographic_parity_difference": 0.0,
                    "equalized_odds_difference": 0.0,
                    "disparate_impact_ratio": 1.0,
                    "delta_tpr": 0.0,
                    "delta_fpr": 0.0,
                    "group_stats": {}
                }
            }

        features_list = [item["features"] for item in dataset]
        ground_truth = [item["label"] for item in dataset]
        genders = [item["gender"] for item in dataset]
        age_groups = [item["age_group"] for item in dataset]
        ethnicities = [item["ethnicity"] for item in dataset]

        # Local model prediction
        predictions = model.predict(features_list)
        probs = model.predict_proba(features_list)

        # Overall performance metrics
        total = len(ground_truth)
        tp = sum(1 for p, y in zip(predictions, ground_truth) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(predictions, ground_truth) if p == 1 and y == 0)
        tn = sum(1 for p, y in zip(predictions, ground_truth) if p == 0 and y == 0)
        fn = sum(1 for p, y in zip(predictions, ground_truth) if p == 0 and y == 1)

        overall_accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        auroc = calculate_auroc(ground_truth, probs)

        # Auditing across sensitive subgroups
        gender_audit = calculate_subgroup_metrics(predictions, ground_truth, genders, "gender")
        age_audit = calculate_subgroup_metrics(predictions, ground_truth, age_groups, "age_group")
        ethnicity_audit = calculate_subgroup_metrics(predictions, ground_truth, ethnicities, "ethnicity")

        # Composite disparity score: explicitly non-negative weighted Demographic Parity Differences
        composite_disparity = (
            gender_audit["demographic_parity_difference"] * 0.45 +
            age_audit["demographic_parity_difference"] * 0.35 +
            ethnicity_audit["demographic_parity_difference"] * 0.20
        )

        return {
            "sample_size": total,
            "overall_accuracy": round(overall_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "auroc": auroc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "composite_disparity": round(composite_disparity, 4),
            "gender_audit": gender_audit,
            "age_audit": age_audit,
            "ethnicity_audit": ethnicity_audit,
            "subgroup_labels": {
                "gender": {0: "Female", 1: "Male"},
                "age_group": {0: "Young (<60)", 1: "Senior (>=60)"},
                "ethnicity": {0: "Cohort Alpha", 1: "Cohort Beta", 2: "Cohort Gamma"}
            }
        }
