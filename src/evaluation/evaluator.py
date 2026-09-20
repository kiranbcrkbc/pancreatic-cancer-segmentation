"""
Full Test Evaluation Module
Runs 5-model soft ensemble on the held-out test set, computes all required metrics,
and exports publication-grade plots and CSV/JSON summaries.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.data.datamodule import get_test_loader
from src.inference.ensemble import EnsemblePredictor
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.hausdorff import compute_hd95
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    save_individual_slice_figures,
)
from src.utils.logger import get_logger


def evaluate_held_out_test_set(
    test_ids: List[str],
    checkpoint_paths: List[str | Path],
    data_dir: str = "data/Task07_Pancreas",
    output_dir: str = "results",
    device: torch.device | None = None,
    model_class: Any = CNNPyramidTransformerSeg,
    model_kwargs: Optional[Dict[str, Any]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Runs 5-model soft ensemble evaluation on the held-out test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger("evaluator")
    logger.info(f"Evaluating held-out test set ({len(test_ids)} patients) on device: {device}")

    out_path = Path(output_dir)
    fig_dir = out_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    default_kwargs = {"in_channels": 1, "num_classes": 3}
    if model_kwargs:
        default_kwargs.update(model_kwargs)

    # 1. Load trained models
    models = []
    scores = []
    for ckpt_p in checkpoint_paths:
        p = Path(ckpt_p)
        if p.exists():
            model = model_class(**default_kwargs)
            data = torch.load(p, map_location=device)
            state_dict = data.get("model_state_dict", data)
            model.load_state_dict(state_dict)
            models.append(model)
            score = float(data.get("best_metric", 0.01))
            scores.append(max(0.001, score))
        else:
            logger.warning(f"Checkpoint {p} not found during test eval.")

    if not models:
        logger.warning("No checkpoints found! Initializing single model for evaluation.")
        models.append(model_class(**default_kwargs))
        scores = [1.0]

    ensemble = EnsemblePredictor(models, device, weights=scores)

    # 2. Build test loader
    test_loader = get_test_loader(
        test_ids=test_ids,
        data_dir=data_dir,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
    )

    all_probs = []
    all_preds = []
    all_targets = []

    for images, targets, _ in test_loader:
        probs, preds = ensemble.predict_batch(images)
        all_probs.append(probs)
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    cat_probs = np.concatenate(all_probs, axis=0)
    cat_preds = np.concatenate(all_preds, axis=0)
    cat_targets = np.concatenate(all_targets, axis=0)

    # 3. Compute metrics
    metrics = compute_all_metrics(cat_preds, cat_targets, probs=cat_probs)
    hd_metrics = compute_hd95(cat_preds, cat_targets)
    metrics.update(hd_metrics)

    # 4. Save Confusion Matrix Plot
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        classes=["Background", "Pancreas", "Tumor"],
        save_path=fig_dir / "confusion_matrix.png",
        normalize=True,
        title="Test Set Confusion Matrix (Normalized)",
    )

    # 5. Save Per-Class Dice & IoU Bar Charts
    classes = ["Background", "Pancreas", "Tumor"]
    dice_vals = [metrics["background_dice"], metrics["pancreas_dice"], metrics["tumor_dice"]]
    iou_vals = [metrics["background_iou"], metrics["pancreas_iou"], metrics["tumor_iou"]]

    # Dice Bar Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(classes, [v * 100 for v in dice_vals], color=["#4CAF50", "#2196F3", "#E91E63"])
    plt.ylabel("Dice Score (%)")
    plt.title("Per-Class Dice Score on Held-Out Test Set")
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_dice.png", dpi=300)
    plt.close()

    # IoU Bar Chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(classes, [v * 100 for v in iou_vals], color=["#8BC34A", "#03A9F4", "#FF5722"])
    plt.ylabel("IoU / Jaccard (%)")
    plt.title("Per-Class IoU on Held-Out Test Set")
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_iou.png", dpi=300)
    plt.close()

    # 6. Save ROC Curves Plot
    plot_roc_curves(cat_probs, cat_targets, fig_dir / "roc_curves.png", classes=classes)

    # 7. Save Presentation Slice Figures (Original, Preprocessed, GT, Pred, Overlay)
    sample_img = test_loader.dataset[0][0].squeeze().numpy()
    sample_tgt = cat_targets[0]
    sample_pred = cat_preds[0]
    save_individual_slice_figures(sample_img, sample_tgt, sample_pred, fig_dir)

    # 8. Save final metrics CSV & JSON
    summary_rows = [
        {"metric": "Overall Pixel Accuracy", "actual_value": metrics["overall_accuracy"], "unit": "ratio"},
        {"metric": "Background Dice", "actual_value": metrics["background_dice"], "unit": "ratio"},
        {"metric": "Pancreas Dice", "actual_value": metrics["pancreas_dice"], "unit": "ratio"},
        {"metric": "Tumor Dice", "actual_value": metrics["tumor_dice"], "unit": "ratio"},
        {"metric": "Mean Foreground Dice", "actual_value": metrics["mean_foreground_dice"], "unit": "ratio"},
        {"metric": "Background IoU", "actual_value": metrics["background_iou"], "unit": "ratio"},
        {"metric": "Pancreas IoU", "actual_value": metrics["pancreas_iou"], "unit": "ratio"},
        {"metric": "Tumor IoU", "actual_value": metrics["tumor_iou"], "unit": "ratio"},
        {"metric": "Pancreas HD95", "actual_value": metrics.get("pancreas_hd95", 0.0), "unit": "pixels"},
        {"metric": "Tumor HD95", "actual_value": metrics.get("tumor_hd95", 0.0), "unit": "pixels"},
        {"metric": "MCC", "actual_value": metrics.get("mcc", 0.0), "unit": "score"},
    ]

    csv_name = f"{prefix}final_metrics.csv" if prefix else "final_metrics.csv"
    json_name = f"{prefix}final_metrics.json" if prefix else "final_metrics.json"
    pd.DataFrame(summary_rows).to_csv(out_path / csv_name, index=False)

    with open(out_path / json_name, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Test Set Evaluation Completed! Accuracy: {metrics['overall_accuracy']:.4f}, Tumor Dice: {metrics['tumor_dice']:.4f}")
    return metrics

