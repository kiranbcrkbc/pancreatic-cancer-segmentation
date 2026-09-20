"""
Single-Fold Trainer Module
Orchestrates training, validation, AMP mixed precision, gradient clipping,
early stopping, and atomic checkpointing.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.losses.compound_loss import CompoundLoss
from src.evaluation.metrics import compute_all_metrics
from src.utils.logger import get_logger


class FoldTrainer:
    """Trains and validates a segmentation model for one cross-validation fold."""

    def __init__(
        self,
        model: nn.Module,
        fold_idx: int,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer_name: str = "AdamW",
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        grad_clip_norm: float = 1.0,
        use_amp: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device)
        self.fold_idx = fold_idx
        self.device = device
        self.max_epochs = max_epochs
        self.patience = early_stopping_patience
        self.min_delta = early_stopping_min_delta
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp and (device.type == "cuda")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"trainer_fold_{fold_idx}")

        # Loss function
        loss_cfg = loss_kwargs or {}
        self.criterion = CompoundLoss(**loss_cfg).to(device)

        # Optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )

        # Scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # State tracking
        self.best_metric = -1.0
        self.patience_counter = 0
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_pancreas_dice": [],
            "val_tumor_dice": [],
            "val_mean_dice": [],
            "val_accuracy": [],
        }

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0

        for images, targets, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss, _ = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss, _ = self.criterion(logits, targets)
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        return total_loss / max(1, batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        batches = 0

        all_preds = []
        all_targets = []
        all_probs = []

        for images, targets, _ in dataloader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss, _ = self.criterion(logits, targets)
            total_loss += float(loss.item())
            batches += 1

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.detach().cpu().numpy())

        cat_preds = np.concatenate(all_preds, axis=0)
        cat_targets = np.concatenate(all_targets, axis=0)
        cat_probs = np.concatenate(all_probs, axis=0)

        # In epoch validation, skip ROC-AUC / mAP calculation on 600k pixels for 10x speedup
        metrics = compute_all_metrics(cat_preds, cat_targets, probs=None)
        avg_loss = total_loss / max(1, batches)
        return avg_loss, metrics

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Atomic saving via .tmp file to prevent corrupt checkpoints."""
        tmp_path = path.with_suffix(".pt.tmp")
        checkpoint = {
            "fold": self.fold_idx,
            "epoch": len(self.history["epoch"]),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
        }
        torch.save(checkpoint, tmp_path)
        if path.exists():
            path.unlink()
        tmp_path.rename(path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Runs complete training loop across max_epochs with early stopping."""
        best_checkpoint_path = self.checkpoint_dir / f"fold{self.fold_idx}_best.pt"
        last_checkpoint_path = self.checkpoint_dir / f"fold{self.fold_idx}_last.pt"

        start_epoch = 1
        if last_checkpoint_path.exists():
            try:
                ckpt = torch.load(last_checkpoint_path, map_location=self.device)
                if ckpt.get("fold") == self.fold_idx and len(ckpt.get("history", {}).get("epoch", [])) > 0:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    self.best_metric = float(ckpt.get("best_metric", -1.0))
                    self.history = ckpt.get("history", self.history)
                    start_epoch = len(self.history["epoch"]) + 1
                    self.logger.info(f"Resuming Fold {self.fold_idx} from epoch {start_epoch} (best metric: {self.best_metric:.4f})...")
            except Exception as e:
                self.logger.warning(f"Could not resume from {last_checkpoint_path}: {e}")

        self.logger.info(f"--- Starting Training Fold {self.fold_idx} ({len(train_loader.dataset)} train, {len(val_loader.dataset)} val) ---")

        for epoch in range(start_epoch, self.max_epochs + 1):
            t0 = time.time()
            tr_loss = self.train_one_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            self.scheduler.step()

            p_dice = val_metrics["pancreas_dice"]
            t_dice = val_metrics["tumor_dice"]
            mean_dice = (p_dice + t_dice) / 2.0
            acc = val_metrics["overall_accuracy"]

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_pancreas_dice"].append(p_dice)
            self.history["val_tumor_dice"].append(t_dice)
            self.history["val_mean_dice"].append(mean_dice)
            self.history["val_accuracy"].append(acc)

            duration = time.time() - t0
            self.logger.info(
                f"Fold {self.fold_idx} | Ep {epoch:02d}/{self.max_epochs:02d} | "
                f"TrLoss: {tr_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"P-Dice: {p_dice:.4f} | T-Dice: {t_dice:.4f} | MeanDice: {mean_dice:.4f} | "
                f"Acc: {acc:.4f} | Time: {duration:.1f}s"
            )

            # Checkpoint tracking prioritizing tumor localization
            checkpoint_score = (0.80 * t_dice + 0.20 * p_dice) if t_dice > 0 else (0.20 * p_dice)

            # Check if best model
            if checkpoint_score > self.best_metric + self.min_delta:
                self.best_metric = checkpoint_score
                self.patience_counter = 0
                self.save_checkpoint(best_checkpoint_path, is_best=True)
                self.logger.info(f"==> Fold {self.fold_idx} Saved new BEST checkpoint (Score: {checkpoint_score:.4f}, T-Dice: {t_dice:.4f}, P-Dice: {p_dice:.4f})")
            else:
                self.patience_counter += 1

            # Save last checkpoint
            self.save_checkpoint(last_checkpoint_path, is_best=False)

            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch} (Patience={self.patience}).")
                break

        # Load best weights before returning
        if best_checkpoint_path.exists():
            best_ckpt = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])

        _, final_val_metrics = self.evaluate(val_loader)
        return final_val_metrics
