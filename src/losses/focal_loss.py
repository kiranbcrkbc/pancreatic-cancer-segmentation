"""
Multi-Class Focal Loss
Down-weights well-classified background pixels to focus gradients on challenging tumor boundaries.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class Focal Loss."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits
            targets: (B, H, W) integer ground truth {0..C-1}
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot: (B, C, H, W)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # p_t: probability of true class
        pt = (probs * target_one_hot).sum(dim=1) + 1e-8
        log_pt = pt.log()

        focal_term = -self.alpha * ((1.0 - pt) ** self.gamma) * log_pt

        if self.reduction == "mean":
            return focal_term.mean()
        elif self.reduction == "sum":
            return focal_term.sum()
        return focal_term
