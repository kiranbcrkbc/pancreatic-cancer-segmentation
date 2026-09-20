from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice loss with optional background inclusion and class weighting."""

    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = True,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        if weights is not None:
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W) raw unnormalized logits
            targets: (B, H, W) integer ground truth masks {0, 1, ..., C-1}
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, num_classes, H, W)
        target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        start_idx = 0 if self.include_background else 1
        dice_per_class = []

        for c in range(start_idx, num_classes):
            p_c = probs[:, c].contiguous().view(-1)
            t_c = target_one_hot[:, c].contiguous().view(-1)

            intersection = (p_c * t_c).sum()
            cardinality = p_c.sum() + t_c.sum()

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            loss_c = 1.0 - dice
            if self.weights is not None:
                loss_c = loss_c * self.weights[c]
            dice_per_class.append(loss_c)

        if self.weights is not None:
            w_sum = self.weights[start_idx:].sum() + 1e-8
            return torch.stack(dice_per_class).sum() / w_sum
        return torch.stack(dice_per_class).mean()

