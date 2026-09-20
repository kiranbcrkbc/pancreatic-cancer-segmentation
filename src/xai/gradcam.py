"""
Grad-CAM Module for Medical Semantic Segmentation
Hooks target convolutional layers, computes class activation gradients,
and generates spatial heatmaps.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMSeg:
    """Grad-CAM generator for convolutional segmentation models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradients(self, module: nn.Module, grad_in: Tuple, grad_out: Tuple) -> None:
        self.gradients = grad_out[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 2,
    ) -> np.ndarray:
        """
        Generates normalized (H, W) Grad-CAM heatmap for a given target class.
        Args:
            input_tensor: (1, 1, H, W) single CT slice tensor
            target_class: integer class ID (2 = tumor, 1 = pancreas)
        """
        # Temporarily freeze model parameters so backward() only tracks gradients for target_layer activations
        # (prevents allocating 82MB of weight gradients in memory on resource-constrained containers)
        prev_states = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)

        # Target score is the sum of logits for the target class across all spatial locations
        score = logits[0, target_class].sum()
        score.backward(retain_graph=False)

        # Restore parameter grad states
        for p, state in zip(self.model.parameters(), prev_states):
            p.requires_grad = state


        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling on gradients: (C,)
        weights = torch.mean(self.gradients[0], dim=(1, 2))

        # Weighted combination of forward activation maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        # ReLU to keep positive influence
        cam = F.relu(cam)

        # Bilinear interpolation back to input slice resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Min-max normalize heatmap to [0, 1]
        cam_min, cam_max = cam_upsampled.min(), cam_upsampled.max()
        if cam_max > cam_min:
            cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_upsampled)

        return cam_norm.astype(np.float32)

    def close(self) -> None:
        """Remove hooks."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()
