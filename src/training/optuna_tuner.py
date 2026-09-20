"""
Hyperparameter Optimization Module via Optuna
Runs fast Bayesian optimization with MedianPruner and exports best_hparams.json.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Subset

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.losses.compound_loss import CompoundLoss
from src.data.dataset import PancreasPatchDataset
from src.augmentation.albumentations import get_training_augmentation, get_validation_augmentation
from src.evaluation.metrics import compute_all_metrics
from src.utils.logger import get_logger


def run_optuna_study(
    train_ids: List[str],
    val_ids: List[str],
    data_dir: str = "data/Task07_Pancreas",
    n_trials: int = 12,
    epochs_per_trial: int = 3,
    subset_size: int = 200,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs Optuna HPO to find optimal hyperparameters and exports best_hparams.json."""
    logger = get_logger("optuna_tuner")
    logger.info(f"Starting Optuna HPO study (Trials={n_trials}, Epochs/Trial={epochs_per_trial})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Prepare datasets
    full_train_ds = PancreasPatchDataset(
        patient_ids=train_ids, data_dir=data_dir, transform=get_training_augmentation(), is_training=True
    )
    full_val_ds = PancreasPatchDataset(
        patient_ids=val_ids, data_dir=data_dir, transform=get_validation_augmentation(), is_training=False
    )

    # Subsets for fast tuning
    n_tr = min(subset_size, len(full_train_ds))
    n_va = min(max(20, subset_size // 4), len(full_val_ds))
    train_sub = Subset(full_train_ds, list(range(n_tr)))
    val_sub = Subset(full_val_ds, list(range(n_va)))

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        transformer_depth = trial.suggest_categorical("transformer_depth", [2, 3, 4])
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        embed_dim = trial.suggest_categorical("embed_dim", [128, 256])
        opt_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        model = CNNPyramidTransformerSeg(
            in_channels=1,
            num_classes=3,
            encoder_channels=[64, 128, 256, 512],
            embed_dim=embed_dim,
            num_heads=num_heads,
            transformer_depth=transformer_depth,
            dropout=dropout,
        ).to(device)

        criterion = CompoundLoss(dice_w=0.6, ce_w=0.3, focal_w=0.1).to(device)
        if opt_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        mean_dice = 0.0
        for ep in range(epochs_per_trial):
            model.train()
            for imgs, tgts, _ in train_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss, _ = criterion(logits, tgts)
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            all_preds, all_tgts = [], []
            with torch.no_grad():
                for imgs, tgts, _ in val_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    logits = model(imgs)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_tgts.append(tgts.cpu().numpy())

            cat_p = np.concatenate(all_preds, axis=0)
            cat_t = np.concatenate(all_tgts, axis=0)
            metrics = compute_all_metrics(cat_p, cat_t)
            mean_dice = (metrics["pancreas_dice"] + metrics["tumor_dice"]) / 2.0

            trial.report(mean_dice, ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return mean_dice

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    logger.info(f"Optuna complete! Best Mean Dice: {best_value:.4f}")
    logger.info(f"Best Params: {best_params}")

    result_payload = {
        "best_mean_dice": float(best_value),
        "best_params": best_params,
        "n_trials": n_trials,
    }

    with open(out_path / "best_hparams.json", "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return result_payload
