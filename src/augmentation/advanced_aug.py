"""
Advanced / Standalone Augmentation Classes
Implements TargetedZoomAug, ElasticDeformer, MultiScaleDataset, and AttentionGuidedAug
as specified in the PRD modular experimental layer.
"""

from typing import Tuple, List, Optional
import random
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.transform import resize
import torch
from torch.utils.data import Dataset


class TargetedZoomAug:
    """Scale-aware zoom focused on pancreas and tumor coordinates."""

    def __init__(self, zoom_range: Tuple[float, float] = (0.5, 1.5), p: float = 0.5) -> None:
        self.zoom_range = zoom_range
        self.p = p

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() > self.p:
            return image, mask

        h, w = image.shape[:2]
        zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # Find centroid of foreground or use center
        if mask is not None and np.any(mask > 0):
            fg_coords = np.argwhere(mask > 0)
            cy, cx = fg_coords.mean(axis=0)
        else:
            cy, cx = h / 2.0, w / 2.0

        new_h, new_w = int(round(h * zoom_factor)), int(round(w * zoom_factor))
        resized_img = resize(image, (new_h, new_w), order=1, mode="reflect", anti_aliasing=True)

        if mask is not None:
            resized_mask = resize(
                mask, (new_h, new_w), order=0, preserve_range=True, anti_aliasing=False
            ).astype(mask.dtype)
        else:
            resized_mask = None

        # Re-crop or pad to original (h, w) around the scaled centroid
        new_cy = cy * zoom_factor
        new_cx = cx * zoom_factor

        ymin = int(round(new_cy - h / 2.0))
        xmin = int(round(new_cx - w / 2.0))
        ymax = ymin + h
        xmax = xmin + w

        # Pad if outside bounds
        pad_top = max(0, -ymin)
        pad_bottom = max(0, ymax - new_h)
        pad_left = max(0, -xmin)
        pad_right = max(0, xmax - new_w)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            resized_img = np.pad(
                resized_img,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            if resized_mask is not None:
                resized_mask = np.pad(
                    resized_mask,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                    constant_values=0,
                )
            ymin += pad_top
            ymax += pad_top
            xmin += pad_left
            xmax += pad_left

        out_img = resized_img[ymin:ymax, xmin:xmax]
        out_mask = resized_mask[ymin:ymax, xmin:xmax] if resized_mask is not None else None

        # Safety resize if minor off-by-one pixel occurred
        if out_img.shape != (h, w):
            out_img = resize(out_img, (h, w), order=1, mode="reflect", anti_aliasing=True)
        if out_mask is not None and out_mask.shape != (h, w):
            out_mask = resize(out_mask, (h, w), order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

        return out_img.astype(np.float32), out_mask


class ElasticDeformer:
    """Non-rigid spatial elastic deformation using random displacement fields."""

    def __init__(self, alpha: float = 120.0, sigma: float = 10.0, p: float = 0.4) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() > self.p:
            return image, mask

        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        deformed_img = map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
        if mask is not None:
            deformed_mask = map_coordinates(mask, indices, order=0, mode="constant", cval=0).reshape(shape)
        else:
            deformed_mask = None

        return deformed_img.astype(np.float32), deformed_mask


class AttentionGuidedAug:
    """Applies higher noise to salient/attention regions and low noise to background."""

    def __init__(
        self,
        noise_inside: float = 0.04,
        noise_outside: float = 0.005,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        threshold_pct: float = 75.0,
        p: float = 0.5,
    ) -> None:
        self.noise_inside = noise_inside
        self.noise_outside = noise_outside
        self.brightness_range = brightness_range
        self.threshold_pct = threshold_pct
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        attention_map: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() > self.p:
            return image, mask

        if attention_map is None:
            if mask is not None and np.any(mask > 0):
                attention_map = (mask > 0).astype(np.float32)
            else:
                return image, mask

        thresh = np.percentile(attention_map, self.threshold_pct)
        high_attn_mask = (attention_map >= thresh).astype(np.float32)

        # Apply variable Gaussian noise
        noise_in = np.random.normal(0, self.noise_inside, size=image.shape).astype(np.float32)
        noise_out = np.random.normal(0, self.noise_outside, size=image.shape).astype(np.float32)

        aug_image = (
            image
            + high_attn_mask * noise_in
            + (1.0 - high_attn_mask) * noise_out
        )

        # Random brightness modulation
        factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        aug_image = aug_image * factor

        return np.clip(aug_image, 0.0, 1.0).astype(np.float32), mask
