"""
Hausdorff Distance Evaluation Module
Computes directed and 95th percentile Hausdorff Distance (HD95) for pancreas and tumor masks.
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def compute_directed_hausdorff_2d(
    pred_binary: np.ndarray, target_binary: np.ndarray
) -> float:
    """Computes directed Hausdorff distance in pixels/mm between two 2D binary masks."""
    p_pts = np.argwhere(pred_binary > 0)
    t_pts = np.argwhere(target_binary > 0)

    if len(p_pts) == 0 and len(t_pts) == 0:
        return 0.0
    if len(p_pts) == 0 or len(t_pts) == 0:
        return 128.0  # Max bounding box penalty

    d_p_to_t = directed_hausdorff(p_pts, t_pts)[0]
    d_t_to_p = directed_hausdorff(t_pts, p_pts)[0]
    return float(max(d_p_to_t, d_t_to_p))


def compute_hd95(
    preds: np.ndarray, targets: np.ndarray, voxel_spacing: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, float]:
    """
    Computes 95th percentile Hausdorff Distance for Pancreas (Class 1) and Tumor (Class 2).
    """
    results: Dict[str, float] = {}

    for c, name in [(1, "pancreas"), (2, "tumor")]:
        p_c = (preds == c).astype(np.uint8)
        t_c = (targets == c).astype(np.uint8)

        # Batch iteration
        if p_c.ndim == 3:
            distances = [
                compute_directed_hausdorff_2d(p_c[i], t_c[i])
                for i in range(len(p_c))
            ]
            hd95_val = float(np.percentile(distances, 95))
        else:
            hd95_val = float(compute_directed_hausdorff_2d(p_c, t_c))

        results[f"{name}_hd95"] = hd95_val

    return results
