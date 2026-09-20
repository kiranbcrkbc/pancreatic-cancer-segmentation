"""
Authoritative Compound Loss Module
Combines Soft Dice Loss (0.6), Class-Weighted Cross-Entropy (0.3), and Focal Loss (0.1).
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.dice_loss import SoftDiceLoss
from src.losses.focal_loss import FocalLoss
from src.losses.boundary_loss import BoundaryLoss


class CompoundLoss(nn.Module):
    """
    Authoritative composite loss function:
    L = dice_w * Dice + ce_w * Weighted_CE + focal_w * Focal + boundary_w * Boundary
    Defaults: dice_w = 0.6, ce_w = 0.3, focal_w = 0.1, CLASS_W = [0.1, 0.3, 0.6]
    """

    def __init__(
        self,
        dice_w: float = 0.6,
        ce_w: float = 0.3,
        focal_w: float = 0.1,
        boundary_w: float = 0.0,
        focal_tversky_w: float = 0.0,
        class_weights: List[float] = [0.1, 0.3, 0.6],
        dice_smooth: float = 1e-5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dice_w = dice_w
        self.ce_w = ce_w
        self.focal_w = focal_w
        self.boundary_w = boundary_w
        self.focal_tversky_w = focal_tversky_w

        self.dice_loss = SoftDiceLoss(smooth=dice_smooth, include_background=True, weights=class_weights)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss() if boundary_w > 0 else None

        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits: (B, num_classes, H, W)
            targets: (B, H, W)
        Returns:
            total_loss: scalar tensor
            loss_components: dict of individual loss values for logging
        """
        # 1. Soft Dice Loss
        l_dice = self.dice_loss(logits, targets) if self.dice_w > 0 else torch.tensor(0.0, device=logits.device)

        # 2. Weighted Cross-Entropy Loss
        l_ce = F.cross_entropy(logits, targets, weight=self.class_weights) if self.ce_w > 0 else torch.tensor(0.0, device=logits.device)

        # 3. Focal Loss
        l_focal = self.focal_loss(logits, targets) if self.focal_w > 0 else torch.tensor(0.0, device=logits.device)

        # 4. Optional Boundary Loss
        l_boundary = self.boundary_loss(logits, targets) if (self.boundary_loss and self.boundary_w > 0) else torch.tensor(0.0, device=logits.device)

        total_loss = (
            self.dice_w * l_dice
            + self.ce_w * l_ce
            + self.focal_w * l_focal
            + self.boundary_w * l_boundary
        )

        components = {
            "loss": float(total_loss.item()),
            "loss_dice": float(l_dice.item()),
            "loss_ce": float(l_ce.item()),
            "loss_focal": float(l_focal.item()),
        }

        return total_loss, components
