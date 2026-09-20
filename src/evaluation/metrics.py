"""
Comprehensive Evaluation Metrics Module
Computes Dice, IoU, Precision, Recall, Specificity, F1, Accuracy,
ROC-AUC, mAP, MCC, and Confusion Matrix across all 3 classes.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def compute_class_confusion_counts(
    preds: np.ndarray, targets: np.ndarray, num_classes: int = 3
) -> Dict[int, Dict[str, int]]:
    """Calculates TP, FP, FN, TN per class."""
    counts = {}
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        tp = int(np.logical_and(pred_c, target_c).sum())
        fp = int(np.logical_and(pred_c, ~target_c).sum())
        fn = int(np.logical_and(~pred_c, target_c).sum())
        tn = int(np.logical_and(~pred_c, ~target_c).sum())

        counts[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return counts


def compute_all_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray | None = None,
    num_classes: int = 3,
) -> Dict[str, Any]:
    """
    Computes all mandatory metrics required by the PRD.
    Args:
        preds: (N,) 1D flat predicted class labels in {0, 1, 2}
        targets: (N,) 1D flat ground truth class labels in {0, 1, 2}
        probs: (N, 3) predicted softmax probabilities (optional, for ROC-AUC & mAP)
    """
    preds_flat = preds.flatten().astype(np.int64)
    targets_flat = targets.flatten().astype(np.int64)

    # 1. Overall Pixel Accuracy
    acc = float(accuracy_score(targets_flat, preds_flat))

    # 2. Confusion Counts per class
    counts = compute_class_confusion_counts(preds_flat, targets_flat, num_classes=num_classes)

    metrics: Dict[str, Any] = {
        "overall_accuracy": acc,
    }

    class_names = {0: "background", 1: "pancreas", 2: "tumor"}
    dice_list = []
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):
        name = class_names[c]
        tp = counts[c]["tp"]
        fp = counts[c]["fp"]
        fn = counts[c]["fn"]
        tn = counts[c]["tn"]

        # Dice = 2TP / (2TP + FP + FN)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-8)
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp + 1e-8)
        # Recall / Sensitivity = TP / (TP + FN)
        recall = tp / (tp + fn + 1e-8)
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp + 1e-8)
        # F1 Score
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)

        metrics[f"{name}_dice"] = float(dice)
        metrics[f"{name}_iou"] = float(iou)
        metrics[f"{name}_precision"] = float(precision)
        metrics[f"{name}_recall"] = float(recall)
        metrics[f"{name}_specificity"] = float(specificity)
        metrics[f"{name}_f1"] = float(f1)

        if c > 0:  # Foreground classes (pancreas & tumor)
            dice_list.append(dice)
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

    # Macro averages on foreground
    metrics["mean_foreground_dice"] = float(np.mean(dice_list)) if dice_list else 0.0
    metrics["mean_foreground_iou"] = float(np.mean(iou_list)) if iou_list else 0.0
    metrics["mean_foreground_precision"] = float(np.mean(precision_list)) if precision_list else 0.0
    metrics["mean_foreground_recall"] = float(np.mean(recall_list)) if recall_list else 0.0
    metrics["mean_foreground_f1"] = float(np.mean(f1_list)) if f1_list else 0.0

    # 3. Matthews Correlation Coefficient (MCC)
    try:
        metrics["mcc"] = float(matthews_corrcoef(targets_flat, preds_flat))
    except Exception:
        metrics["mcc"] = 0.0

    # 4. ROC-AUC and mAP (Average Precision) if probabilities provided
    if probs is not None:
        probs_reshaped = probs.reshape(-1, num_classes)
        for c in range(num_classes):
            name = class_names[c]
            bin_target = (targets_flat == c).astype(np.int32)
            if len(np.unique(bin_target)) > 1:
                try:
                    metrics[f"{name}_roc_auc"] = float(roc_auc_score(bin_target, probs_reshaped[:, c]))
                except Exception:
                    metrics[f"{name}_roc_auc"] = 0.5

                try:
                    metrics[f"{name}_map"] = float(average_precision_score(bin_target, probs_reshaped[:, c]))
                except Exception:
                    metrics[f"{name}_map"] = 0.0
            else:
                metrics[f"{name}_roc_auc"] = 1.0 if bin_target[0] == 1 else 0.5
                metrics[f"{name}_map"] = 1.0 if bin_target[0] == 1 else 0.0

    # 5. Confusion matrix
    cm = confusion_matrix(targets_flat, preds_flat, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    return metrics
