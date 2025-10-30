"""
metrics.py - Part of fNIR Base Model.
"""

import numpy as np

# 计算敏感性 (sensitivity) 和 特异性 (specificity)
def compute_sen_spec(y_true, y_pred):
    # sensitivity = TP / (TP + FN)
    # specificity = TN / (TN + FP)
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negative

    sen = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
    return sen, spe