"""
Soft Ensemble Module
Averages softmax probability maps across all 5 trained fold checkpoints.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class EnsemblePredictor:
    """Combines predictions from multiple fold models using soft probability averaging or performance weighting."""

    def __init__(self, models: List[nn.Module], device: torch.device, weights: Optional[List[float]] = None) -> None:
        self.models = [m.to(device).eval() for m in models]
        self.device = device
        if weights is not None and len(weights) == len(models):
            norm_w = np.array(weights, dtype=np.float32)
            if norm_w.sum() > 0:
                norm_w = norm_w / norm_w.sum()
            self.weights = norm_w.tolist()
        else:
            self.weights = [1.0 / max(1, len(models))] * len(models)

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs soft ensemble inference on a batch of slices.
        Args:
            x: (B, 1, H, W) tensor
        Returns:
            mean_probs: (B, num_classes, H, W) averaged softmax probabilities
            pred_classes: (B, H, W) discrete class predictions {0, 1, 2}
        """
        x = x.to(self.device)
        prob_sum = None

        for i, model in enumerate(self.models):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            w = self.weights[i]
            if prob_sum is None:
                prob_sum = probs * w
            else:
                prob_sum += probs * w

        mean_probs = prob_sum.cpu().numpy()
        pred_classes = np.argmax(mean_probs, axis=1)

        return mean_probs, pred_classes

