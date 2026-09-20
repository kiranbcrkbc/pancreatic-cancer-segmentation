"""
5-Fold Cross-Validation Training CLI Script
Trains CNNPyramidTransformerSeg across 5 folds and saves best checkpoints.
"""

import argparse
from pathlib import Path
import json
import shutil
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.cv_runner import run_5fold_cross_validation
from src.utils.config import load_all_configs
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-Fold Cross Validation")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--data-dir", type=str, default="data/Task07_Pancreas")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    args = parser.parse_args()

    logger = get_logger("train_cv")
    configs = load_all_configs("configs")

    model_cfg = configs.get("model", {})
    train_cfg = configs.get("training", {})
    loss_cfg = configs.get("loss", {})

    if args.epochs is not None:
        train_cfg["epochs_per_fold"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr

    kfold_file = Path(args.splits_dir) / "kfold_5.json"
    if not kfold_file.exists():
        logger.info("Splits not found. Executing data preparation...")
        from scripts.prepare_data import main as prep_main
        prep_main()

    with open(kfold_file, "r", encoding="utf-8") as f:
        kfold_dict = json.load(f)

    # Run CV
    cv_metrics = run_5fold_cross_validation(
        kfold_dict=kfold_dict,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        loss_cfg=loss_cfg,
        data_dir=args.data_dir,
    )

    # Identify best overall fold checkpoint and save as checkpoints/final_model.pt
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    best_fold_idx = 1
    best_mean_dice = -1.0

    for fold_name, metrics in cv_metrics.items():
        f_idx = int(fold_name.split("_")[-1])
        m_dice = metrics.get("mean_foreground_dice", 0.0)
        if m_dice > best_mean_dice:
            best_mean_dice = m_dice
            best_fold_idx = f_idx

    best_fold_ckpt = ckpt_dir / f"fold{best_fold_idx}_best.pt"
    final_model_ckpt = ckpt_dir / "final_model.pt"
    if best_fold_ckpt.exists():
        shutil.copy(best_fold_ckpt, final_model_ckpt)
        logger.info(f"Copied top checkpoint ({best_fold_ckpt.name}, MeanDice={best_mean_dice:.4f}) to {final_model_ckpt}")

    logger.info("5-Fold Cross-Validation training completed successfully!")


if __name__ == "__main__":
    main()
