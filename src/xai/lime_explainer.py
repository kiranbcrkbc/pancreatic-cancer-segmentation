"""
LIME Superpixel Explainability Module
Generates superpixel-level perturbation explanations for pancreatic and tumor segmentations.
"""

from typing import Tuple, Callable
import numpy as np
import torch
from skimage.segmentation import quickshift
from sklearn.linear_model import Ridge


def explain_slice_with_lime(
    model: torch.nn.Module,
    image_2d: np.ndarray,
    target_class: int = 2,
    num_samples: int = 150,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes LIME superpixel importance weights for an input CT slice.
    Returns:
        segments: (H, W) superpixel segmentation map
        heatmap: (H, W) importance heatmap normalized to [0, 1]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # Convert 1-channel grayscale to 3-channel pseudo-RGB for quickshift
    img_rgb = np.stack([image_2d, image_2d, image_2d], axis=-1)
    segments = quickshift(img_rgb, kernel_size=4, max_dist=10, ratio=0.2)
    num_segments = len(np.unique(segments))

    # Binary perturbation matrix: (num_samples, num_segments)
    rng = np.random.RandomState(42)
    perturbations = rng.binomial(1, 0.5, size=(num_samples, num_segments))
    # Guarantee base sample is included
    perturbations[0] = 1

    # Predict probability of target class for each perturbation
    predictions = []
    h, w = image_2d.shape

    batch_imgs = []
    for i in range(num_samples):
        mask_active = perturbations[i]
        perturbed_img = image_2d.copy()
        # Zero out inactive superpixels
        for seg_val in range(num_segments):
            if mask_active[seg_val] == 0:
                perturbed_img[segments == seg_val] = 0.0
        batch_imgs.append(perturbed_img)

    # Run inference in batches of 16
    batch_size = 16
    with torch.no_grad():
        for b_start in range(0, num_samples, batch_size):
            b_end = min(num_samples, b_start + batch_size)
            sub_batch = np.array(batch_imgs[b_start:b_end])
            t = torch.from_numpy(sub_batch).unsqueeze(1).float().to(device)
            logits = model(t)
            probs = torch.softmax(logits, dim=1)
            # Sum probability of target class
            cls_probs = probs[:, target_class].mean(dim=(1, 2)).cpu().numpy()
            predictions.extend(cls_probs.tolist())

    # Fit Ridge regressor on perturbations -> predictions
    distances = np.sum((perturbations - 1) ** 2, axis=1)
    weights = np.exp(-distances / (num_segments * 0.25))

    reg = Ridge(alpha=1.0)
    reg.fit(perturbations, predictions, sample_weight=weights)
    seg_weights = reg.coef_

    # Map superpixel coefficients back to 2D heatmap
    heatmap = np.zeros((h, w), dtype=np.float32)
    for seg_val in range(num_segments):
        heatmap[segments == seg_val] = seg_weights[seg_val]

    # Normalize positive influence to [0, 1]
    heatmap = np.maximum(0, heatmap)
    h_max = heatmap.max()
    if h_max > 0:
        heatmap = heatmap / h_max

    return segments, heatmap
