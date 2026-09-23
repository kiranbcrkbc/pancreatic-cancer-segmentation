"""
Training Pipeline for Model 4: CNN + GNN/GAT (AttnUNetEfficientGAT)

Provides a clean, standalone training entry point with configurable CLI arguments.
Supports training with Compound Loss (Soft Dice + Weighted CE + Focal Loss) and AdamW optimizer.

Usage:
    python train.py --epochs 10 --batch-size 16 --lr 0.0008
    python train.py --dry-run
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import AttnUNetEfficientGAT


# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss with smooth factor."""

    def __init__(self, smooth: float = 1e-5, weights: list[float] | None = None) -> None:
        super().__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(num_classes):
            p = probs[:, c].reshape(-1)
            t = targets_onehot[:, c].reshape(-1)
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            w = self.weights[c] if self.weights is not None else 1.0
            dice_per_class.append(w * (1.0 - dice))

        if self.weights is not None:
            return sum(dice_per_class) / sum(self.weights)
        return torch.stack(dice_per_class).mean()


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for class-imbalance mitigation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class CompoundLoss(nn.Module):
    """
    Model 4 Compound Loss:
    L = 0.55 * SoftDice + 0.25 * WeightedCE + 0.20 * Focal
    Class weights: [0.03, 0.27, 0.70]
    """

    def __init__(
        self,
        dice_w: float = 0.55,
        ce_w: float = 0.25,
        focal_w: float = 0.20,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if class_weights is None:
            class_weights = [0.03, 0.27, 0.70]
        self.dice_w = dice_w
        self.ce_w = ce_w
        self.focal_w = focal_w
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

        self.dice_loss = SoftDiceLoss(weights=class_weights)
        self.focal_loss = FocalLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l_dice = self.dice_loss(logits, targets)
        l_ce = F.cross_entropy(logits, targets, weight=self.class_weights)
        l_focal = self.focal_loss(logits, targets)

        total = self.dice_w * l_dice + self.ce_w * l_ce + self.focal_w * l_focal
        components = {
            "loss": float(total.item()),
            "dice": float(l_dice.item()),
            "ce": float(l_ce.item()),
            "focal": float(l_focal.item()),
        }
        return total, components


# -----------------------------------------------------------------------------
# Dataset Loader
# -----------------------------------------------------------------------------

class SyntheticPatchDataset(Dataset):
    def __init__(self, num_samples: int = 16, patch_size: tuple[int, int] = (128, 128)) -> None:
        self.num_samples = num_samples
        self.patch_size = patch_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(1, *self.patch_size, dtype=torch.float32)
        y = torch.zeros(self.patch_size, dtype=torch.long)
        h, w = self.patch_size
        y[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
        y[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 2
        return x, y


# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------

def train_model(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 8e-4,
    data_dir: str = "../../data/Task07_Pancreas",
    checkpoint_dir: str = "./checkpoints",
    device: str = "cpu",
    dry_run: bool = False,
) -> None:
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing Model 4: CNN + GNN/GAT (AttnUNetEfficientGAT) ===")
    print(f"Device: {device_obj} | Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")

    model = AttnUNetEfficientGAT(in_channels=1, num_classes=3, use_efficientnet=True).to(device_obj)
    criterion = CompoundLoss(dice_w=0.55, ce_w=0.25, focal_w=0.20, class_weights=[0.03, 0.27, 0.70]).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_dataset = SyntheticPatchDataset(num_samples=16 if dry_run else 64)
    val_dataset = SyntheticPatchDataset(num_samples=8 if dry_run else 16)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 1 if dry_run else epochs
    best_val_dice = 0.0

    print(f"\nStarting training loop ({num_epochs} epoch{'s' if num_epochs > 1 else ''})...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device_obj)
            masks = masks.to(device_obj)

            optimizer.zero_grad()
            logits = model(images)
            loss, components = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_dice_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device_obj)
                masks = masks.to(device_obj)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                for c in [1, 2]:
                    inter = ((preds == c) & (masks == c)).sum().float()
                    union = (preds == c).sum().float() + (masks == c).sum().float()
                    d = (2.0 * inter + 1e-5) / (union + 1e-5)
                    val_dice_sum += d.item()

        val_fg_dice = val_dice_sum / (len(val_loader) * 2)
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] - Train Loss: {train_loss:.4f} | Val Foreground Dice: {val_fg_dice:.4f}")

        if val_fg_dice > best_val_dice:
            best_val_dice = val_fg_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_val_dice,
                },
                ckpt_path / "best_model.pt",
            )

    print(f"\n[DONE] Training complete. Best checkpoint saved to {ckpt_path / 'best_model.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 4: CNN + GNN/GAT")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Initial learning rate")
    parser.add_argument("--data-dir", type=str, default="../../data/Task07_Pancreas", help="Path to Task07 dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Output directory for saved checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 1-epoch dry-run test without training full dataset")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
