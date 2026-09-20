"""
5-Fold Cross-Validation Runner Module
Orchestrates training across 5 folds and logs per-fold metrics.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import pandas as pd
import torch

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.data.datamodule import get_train_val_loaders_for_fold
from src.training.trainer import FoldTrainer
from src.utils.visualization import plot_training_curves
from src.utils.logger import get_logger


def run_5fold_cross_validation(
    kfold_dict: Dict[str, Dict[str, List[str]]],
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    loss_cfg: Dict[str, Any],
    data_dir: str = "data/Task07_Pancreas",
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Runs training across all 5 folds, saves checkpoints, and compiles cross-validation metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger("cv_runner")
    logger.info(f"Starting 5-Fold Cross-Validation on device: {device}")

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(train_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    cv_results = {}
    fold_metrics_rows = []

    for fold_name, fold_data in kfold_dict.items():
        fold_idx = int(fold_name.split("_")[-1])
        train_ids = fold_data["train"]
        val_ids = fold_data["val"]

        logger.info(f"Preparing Fold {fold_idx}: {len(train_ids)} train, {len(val_ids)} val patients")

        train_loader, val_loader = get_train_val_loaders_for_fold(
            fold_train_ids=train_ids,
            fold_val_ids=val_ids,
            data_dir=data_dir,
            batch_size=train_cfg.get("batch_size", 8),
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=train_cfg.get("pin_memory", False),
            patch_size=(128, 128),
        )

        # Build fresh model instance
        model = CNNPyramidTransformerSeg(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=model_cfg.get("num_classes", 3),
            encoder_channels=model_cfg.get("encoder_channels", [64, 128, 256, 512]),
            embed_dim=model_cfg.get("bottleneck", {}).get("embed_dim", 256),
            ppm_pool_sizes=model_cfg.get("bottleneck", {}).get("ppm_pool_sizes", [1, 2, 4, 8]),
            num_heads=model_cfg.get("bottleneck", {}).get("num_heads", 8),
            transformer_depth=model_cfg.get("bottleneck", {}).get("transformer_depth", 4),
            ffn_dim=model_cfg.get("bottleneck", {}).get("ffn_dim", 1024),
            dropout=model_cfg.get("dropout", 0.1),
        )

        trainer = FoldTrainer(
            model=model,
            fold_idx=fold_idx,
            device=device,
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            optimizer_name=train_cfg.get("optimizer", "AdamW"),
            max_epochs=train_cfg.get("epochs_per_fold", 50),
            early_stopping_patience=train_cfg.get("early_stopping", {}).get("patience", 10),
            early_stopping_min_delta=train_cfg.get("early_stopping", {}).get("min_delta", 1e-4),
            grad_clip_norm=train_cfg.get("gradient_clip_norm", 1.0),
            use_amp=train_cfg.get("mixed_precision", True),
            checkpoint_dir=checkpoint_dir,
            loss_kwargs=loss_cfg.get("compound_weights", None),
        )

        final_metrics = trainer.fit(train_loader, val_loader)
        cv_results[fold_name] = final_metrics

        row = {
            "fold": fold_idx,
            "overall_accuracy": final_metrics["overall_accuracy"],
            "background_dice": final_metrics["background_dice"],
            "pancreas_dice": final_metrics["pancreas_dice"],
            "tumor_dice": final_metrics["tumor_dice"],
            "mean_foreground_dice": final_metrics["mean_foreground_dice"],
            "pancreas_iou": final_metrics["pancreas_iou"],
            "tumor_iou": final_metrics["tumor_iou"],
            "mcc": final_metrics.get("mcc", 0.0),
        }
        fold_metrics_rows.append(row)

    # Save fold metrics to CSV
    df_folds = pd.DataFrame(fold_metrics_rows)
    df_folds.to_csv(results_dir / "fold_metrics.csv", index=False)

    # Save JSON summary
    with open(results_dir / "cv_training_logs.json", "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)

    # Save Training & Validation Progression Curves
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if "trainer" in locals() and hasattr(trainer, "history"):
        plot_training_curves(trainer.history, fig_dir)

    logger.info("5-Fold Cross-Validation complete! Metrics saved.")
    return cv_results
