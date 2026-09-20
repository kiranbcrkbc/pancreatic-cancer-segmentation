"""
Transformer Attention Visualization Module
Intercepts multi-head self-attention matrices from the transformer bottleneck
and projects them into spatial attention heatmaps.
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def extract_bottleneck_attention_maps(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Runs forward pass requesting attention weights, averages across heads and tokens,
    and upsamples to target slice size.
    Returns:
        heatmap: (H, W) normalized attention intensity map in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(input_tensor, return_attention=True)

    if not attn_weights or len(attn_weights) == 0:
        return np.zeros(target_size, dtype=np.float32)

    # Take the last transformer layer's attention weights: (B, H*W, H*W) = (1, 64, 64)
    last_attn = attn_weights[-1][0].cpu().numpy()  # (64, 64)

    # Mean attention received by each spatial token across all other query tokens
    spatial_token_attn = np.mean(last_attn, axis=0)  # (64,)

    # Reshape token sequence (64,) back to feature grid (8, 8)
    grid_size = int(round(np.sqrt(len(spatial_token_attn))))
    attn_grid = spatial_token_attn.reshape(grid_size, grid_size)

    # Convert to tensor and upsample to target resolution (128, 128)
    grid_tensor = torch.from_numpy(attn_grid).unsqueeze(0).unsqueeze(0).float()
    upsampled = F.interpolate(grid_tensor, size=target_size, mode="bilinear", align_corners=False)
    heatmap = upsampled.squeeze().numpy()

    # Normalize to [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    return heatmap_norm.astype(np.float32)
