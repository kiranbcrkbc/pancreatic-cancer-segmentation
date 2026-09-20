"""
Boundary Loss Module
Penalizes boundary contour discrepancies using Euclidean distance transforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(nn.Module):
    """Boundary loss for penalizing contour divergence."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes

    def compute_sdf(self, mask: np.ndarray) -> np.ndarray:
        """Compute signed distance field for binary mask."""
        pos = distance_transform_edt(mask)
        neg = distance_transform_edt(1 - mask)
        sdf = neg - pos
        return sdf.astype(np.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        b, c, h, w = probs.shape

        # Precompute distance maps for foreground classes (1, 2)
        device = logits.device
        loss = torch.tensor(0.0, device=device)

        targets_np = targets.detach().cpu().numpy()
        for cls_idx in range(1, c):
            sdfs = []
            for batch_i in range(b):
                bin_mask = (targets_np[batch_i] == cls_idx).astype(np.uint8)
                if bin_mask.sum() == 0:
                    sdf = np.zeros((h, w), dtype=np.float32)
                else:
                    sdf = self.compute_sdf(bin_mask)
                sdfs.append(sdf)
            sdfs_tensor = torch.from_numpy(np.stack(sdfs)).to(device)
            loss += (probs[:, cls_idx] * sdfs_tensor).mean()

        return loss / max(1, c - 1)
