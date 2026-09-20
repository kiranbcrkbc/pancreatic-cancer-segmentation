"""
ROI Patch Extractor Module
Extracts context-expanded 128x128 patches focused on pancreatic and tumor regions.
"""

from typing import Tuple, Optional
import numpy as np
from skimage.transform import resize


class ROIPatchExtractor:
    """Extracts square Region-of-Interest (ROI) patches around pancreas/tumor masks."""

    def __init__(
        self,
        min_area: int = 50,
        margin: int = 10,
        context_scale: float = 1.5,
        patch_size: Tuple[int, int] = (128, 128),
    ) -> None:
        self.min_area = min_area
        self.margin = margin
        self.context_scale = context_scale
        self.patch_size = patch_size

    def has_sufficient_foreground(self, mask_2d: np.ndarray) -> bool:
        """Check if slice mask contains at least min_area foreground pixels."""
        return int(np.sum(mask_2d > 0)) >= self.min_area

    def compute_bounding_box(
        self, mask_2d: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute bounding box [ymin, ymax, xmin, xmax] around foreground pixels."""
        fg_indices = np.argwhere(mask_2d > 0)
        if len(fg_indices) == 0:
            return None

        ymin, xmin = fg_indices.min(axis=0)
        ymax, xmax = fg_indices.max(axis=0)

        # Apply margin
        h, w = mask_2d.shape
        ymin = max(0, ymin - self.margin)
        ymax = min(h, ymax + self.margin)
        xmin = max(0, xmin - self.margin)
        xmax = min(w, xmax + self.margin)

        return int(ymin), int(ymax), int(xmin), int(xmax)

    def extract_patch(
        self,
        image_2d: np.ndarray,
        mask_2d: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int]]:
        """Extract square context-expanded ROI patch resized to patch_size (128, 128)."""
        h, w = image_2d.shape

        if mask_2d is not None and self.has_sufficient_foreground(mask_2d):
            bbox = self.compute_bounding_box(mask_2d)
        else:
            bbox = None

        if bbox is None:
            # Fallback: full slice resized to patch_size
            img_patch = resize(
                image_2d,
                self.patch_size,
                order=1,
                mode="reflect",
                anti_aliasing=True,
            ).astype(np.float32)

            mask_patch = None
            if mask_2d is not None:
                mask_patch = resize(
                    mask_2d,
                    self.patch_size,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.int64)

            return img_patch, mask_patch, (0, h, 0, w)

        ymin, ymax, xmin, xmax = bbox
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0
        larger_side = max(ymax - ymin, xmax - xmin)
        box_side = larger_side * self.context_scale

        half_side = box_side / 2.0
        crop_ymin = max(0, int(np.floor(cy - half_side)))
        crop_ymax = min(h, int(np.ceil(cy + half_side)))
        crop_xmin = max(0, int(np.floor(cx - half_side)))
        crop_xmax = min(w, int(np.ceil(cx + half_side)))

        # Crop
        img_crop = image_2d[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        img_patch = resize(
            img_crop,
            self.patch_size,
            order=1,
            mode="reflect",
            anti_aliasing=True,
        ).astype(np.float32)

        mask_patch = None
        if mask_2d is not None:
            mask_crop = mask_2d[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            mask_patch = resize(
                mask_crop,
                self.patch_size,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.int64)

        return img_patch, mask_patch, (crop_ymin, crop_ymax, crop_xmin, crop_xmax)
