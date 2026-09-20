"""
Albumentations Augmentation Pipelines
Implements the authoritative 12-transform training augmentation pipeline
and deterministic validation/test padding transform.
"""

from typing import Tuple
import albumentations as A
import numpy as np


def get_training_augmentation(patch_size: Tuple[int, int] = (128, 128)) -> A.Compose:
    """Authoritative training augmentation pipeline on 128x128 ROI patches."""
    h, w = patch_size
    crop_h = min(96, int(h * 0.75))
    crop_w = min(96, int(w * 0.75))

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=60.0,
            sigma=4.0,
            alpha_affine=0.0,
            border_mode=0,
            p=0.3,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=0,
            p=0.2,
        ),
        A.RandomCrop(height=crop_h, width=crop_w, p=0.3),
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=0),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        A.CoarseDropout(
            max_holes=4,
            max_height=12,
            max_width=12,
            min_holes=1,
            fill_value=0,
            mask_fill_value=0,
            p=0.2,
        ),
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=0),
    ])


def get_validation_augmentation(patch_size: Tuple[int, int] = (128, 128)) -> A.Compose:
    """Deterministic validation / test pipeline: only pad if needed, zero stochasticity."""
    h, w = patch_size
    return A.Compose([
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=0),
    ])


def apply_augmentation(
    transform: A.Compose,
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Applies albumentations transform to a 2D image and optional integer mask."""
    if mask is not None:
        augmented = transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]
    else:
        augmented = transform(image=image)
        return augmented["image"], None
