"""
Fairness and Clinical Performance Metrics Engine
Computes:
- Accuracy, F1-score, Precision, Recall, AUROC
- DPD: Demographic Parity Difference
- EOD: Equalized Odds Difference
- DIR: Disparate Impact Ratio
"""

from typing import Dict, Any, List, Union
import numpy as np
import torch

def compute_clinical_fairness_metrics(
    y_true: Union[torch.Tensor, np.ndarray, List[int]],
    y_pred: Union[torch.Tensor, np.ndarray, List[int]],
    y_prob: Union[torch.Tensor, np.ndarray, List[float]],
    sensitive_attr: Union[torch.Tensor, np.ndarray, List[int]],
    attribute_name: str = "gender"
) -> Dict[str, Any]:
    """
    Computes rigorous clinical performance and demographic fairness metrics.
    Sensitive attribute: 0 = Female (or unprivileged), 1 = Male (or privileged).
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().astype(int)
    else:
        y_true = np.array(y_true, dtype=int)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().astype(int)
    else:
        y_pred = np.array(y_pred, dtype=int)

    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy().astype(float).flatten()
    else:
        y_prob = np.array(y_prob, dtype=float).flatten()

    if isinstance(sensitive_attr, torch.Tensor):
        sensitive_attr = sensitive_attr.detach().cpu().numpy().astype(int)
    else:
        sensitive_attr = np.array(sensitive_attr, dtype=int)

    total_samples = len(y_true)
    if total_samples == 0:
        return {}

    # Overall Performance Metrics
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # AUROC via Mann-Whitney U
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        # Pairwise comparison
        u_stat = float(np.sum(pos_scores[:, None] > neg_scores[None, :]) + 0.5 * np.sum(pos_scores[:, None] == neg_scores[None, :]))
        auroc = u_stat / (len(pos_scores) * len(neg_scores))
    else:
        auroc = 0.50

    # Subgroup Disparities across Sensitive Attribute
    unique_groups = np.unique(sensitive_attr)
    subgroups = {}
    pos_rates = {}
    tprs = {}
    fprs = {}

    for grp in unique_groups:
        mask = (sensitive_attr == grp)
        count = int(np.sum(mask))
        if count == 0:
            continue

        g_true = y_true[mask]
        g_pred = y_pred[mask]

        g_tp = int(np.sum((g_pred == 1) & (g_true == 1)))
        g_fp = int(np.sum((g_pred == 1) & (g_true == 0)))
        g_tn = int(np.sum((g_pred == 0) & (g_true == 0)))
        g_fn = int(np.sum((g_pred == 0) & (g_true == 1)))

        g_acc = (g_tp + g_tn) / count if count > 0 else 0.0
        g_pos_rate = (g_tp + g_fp) / count if count > 0 else 0.0
        g_tpr = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
        g_fpr = g_fp / (g_fp + g_tn) if (g_fp + g_tn) > 0 else 0.0
        g_f1 = (2 * g_tp) / (2 * g_tp + g_fp + g_fn) if (2 * g_tp + g_fp + g_fn) > 0 else 0.0

        grp_label = "Female" if (attribute_name == "gender" and grp == 0) else (
            "Male" if (attribute_name == "gender" and grp == 1) else f"Group_{grp}"
        )

        subgroups[grp_label] = {
            "count": count,
            "accuracy": round(g_acc, 4),
            "pos_rate": round(g_pos_rate, 4),
            "tpr": round(g_tpr, 4),
            "fpr": round(g_fpr, 4),
            "f1": round(g_f1, 4),
            "tp": g_tp, "fp": g_fp, "tn": g_tn, "fn": g_fn
        }
        pos_rates[grp] = g_pos_rate
        tprs[grp] = g_tpr
        fprs[grp] = g_fpr

    # Demographic Parity Difference (DPD): max_g P(Y_hat=1|A=g) - min_g P(Y_hat=1|A=g)
    if len(pos_rates) >= 2:
        dpd = float(max(pos_rates.values()) - min(pos_rates.values()))
    else:
        dpd = 0.0

    # Equalized Odds Difference (EOD): 0.5 * (|TPR_max - TPR_min| + |FPR_max - FPR_min|)
    if len(tprs) >= 2 and len(fprs) >= 2:
        eod = 0.5 * (float(max(tprs.values()) - min(tprs.values())) + float(max(fprs.values()) - min(fprs.values())))
    else:
        eod = 0.0

    # Disparate Impact Ratio (DIR): min P(Y_hat=1|A=g) / (max P(Y_hat=1|A=g) + eps)
    if len(pos_rates) >= 2:
        max_pr = max(pos_rates.values())
        min_pr = min(pos_rates.values())
        dir_val = float(min_pr / (max_pr + 1e-6))
    else:
        dir_val = 1.0

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "auroc": round(auroc, 4),
        "dpd": round(dpd, 4),
        "eod": round(eod, 4),
        "dir": round(dir_val, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "subgroups": subgroups
    }
