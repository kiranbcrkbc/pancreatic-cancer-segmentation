"""
CT Preprocessing Module
Performs volume-level HU windowing, min-max normalization, isotropic resampling,
and slice-level Gaussian smoothing and CLAHE enhancement.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage.transform import resize
from skimage.exposure import equalize_adapthist


class CTPreprocessor:
    """Preprocesses 3D CT volumes and 2D axial slices for pancreatic segmentation."""

    def __init__(
        self,
        hu_min: float = -150.0,
        hu_max: float = 250.0,
        target_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        slice_size: Tuple[int, int] = (256, 256),
        gaussian_sigma: float = 0.5,
        clahe_clip_limit: float = 0.03,
    ) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_spacing_mm = target_spacing_mm
        self.slice_size = slice_size
        self.gaussian_sigma = gaussian_sigma
        self.clahe_clip_limit = clahe_clip_limit

    def clip_hu(self, volume: np.ndarray) -> np.ndarray:
        """Clip Hounsfield Units to soft tissue / pancreas window [-150, +250]."""
        return np.clip(volume, self.hu_min, self.hu_max)

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """Min-Max scale intensity values to [0.0, 1.0] as float32."""
        clipped = self.clip_hu(volume)
        norm = (clipped - self.hu_min) / (self.hu_max - self.hu_min)
        return norm.astype(np.float32)

    def resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        is_mask: bool = False,
    ) -> np.ndarray:
        """Resample volume to isotropic 1.0 mm spacing."""
        zoom_factors = [
            current_spacing[i] / self.target_spacing_mm[i]
            for i in range(3)
        ]
        order = 0 if is_mask else 1
        resampled = zoom(volume, zoom_factors, order=order, prefilter=not is_mask)
        return resampled

    def preprocess_volume(
        self,
        volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """Full volume-level preprocessing: HU clipping, normalization, optional resampling."""
        norm_vol = self.normalize(volume)
        if spacing is not None and tuple(spacing) != self.target_spacing_mm:
            norm_vol = self.resample_volume(norm_vol, spacing, is_mask=False)
        return norm_vol

    def process_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Process a single 2D axial slice: resize to 256x256, Gaussian filter, CLAHE."""
        # 1. Resize to intermediate 256x256
        if slice_2d.shape != self.slice_size:
            slice_resized = resize(
                slice_2d,
                self.slice_size,
                order=1,
                mode="reflect",
                anti_aliasing=True,
            ).astype(np.float32)
        else:
            slice_resized = slice_2d.astype(np.float32)

        # 2. Gaussian smoothing (sigma = 0.5)
        if self.gaussian_sigma > 0:
            slice_smoothed = gaussian_filter(slice_resized, sigma=self.gaussian_sigma)
        else:
            slice_smoothed = slice_resized

        # 3. CLAHE (clip_limit = 0.03)
        # Ensure values are strictly in [0, 1] for equalize_adapthist
        slice_clamped = np.clip(slice_smoothed, 0.0, 1.0)
        slice_clahe = equalize_adapthist(slice_clamped, clip_limit=self.clahe_clip_limit)
        return slice_clahe.astype(np.float32)
