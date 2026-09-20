"""
Inference Predictor Module
Ingests 3D NIfTI CT scans or 2D slices, executes preprocessing,
runs 5-fold ensemble inference, and exports predictions.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import nibabel as nib

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.preprocessing.ct_preprocessor import CTPreprocessor
from src.preprocessing.roi_extractor import ROIPatchExtractor
from src.inference.ensemble import EnsemblePredictor
from src.utils.logger import get_logger


class ClinicalPredictor:
    """End-to-end clinical inference pipeline for pancreatic CT segmentation."""

    def __init__(
        self,
        checkpoint_paths: List[str | Path],
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.logger = get_logger("clinical_predictor")
        self.preprocessor = CTPreprocessor()
        self.roi_extractor = ROIPatchExtractor(patch_size=(128, 128))

        # Load models
        models = []
        for ckpt_path in checkpoint_paths:
            path = Path(ckpt_path)
            if not path.exists():
                self.logger.warning(f"Checkpoint not found: {path}")
                continue
            model = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)
            data = torch.load(path, map_location=self.device)
            state_dict = data.get("model_state_dict", data)
            model.load_state_dict(state_dict)
            models.append(model)

        if not models:
            self.logger.warning("No checkpoints loaded! Initializing fresh default model.")
            models.append(CNNPyramidTransformerSeg(in_channels=1, num_classes=3))

        self.ensemble = EnsemblePredictor(models, self.device)

    def predict_slice(self, slice_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runs inference on a single 2D axial slice. Returns (pred_mask, softmax_probs)."""
        proc_slice = self.preprocessor.process_slice(slice_2d)
        img_patch, _, _ = self.roi_extractor.extract_patch(proc_slice)

        t = torch.from_numpy(img_patch).unsqueeze(0).unsqueeze(0).float()
        probs, preds = self.ensemble.predict_batch(t)

        return preds[0], probs[0]

    def predict_nifti_volume(
        self, nifti_path: str | Path, output_mask_path: Optional[str | Path] = None
    ) -> np.ndarray:
        """Predicts full 3D segmentation volume from a NIfTI file."""
        nii = nib.load(str(nifti_path))
        vol = nii.get_fdata().astype(np.float32)
        affine = nii.affine

        norm_vol = self.preprocessor.preprocess_volume(vol)
        h, w, d = norm_vol.shape
        pred_volume = np.zeros((h, w, d), dtype=np.uint8)

        for z in range(d):
            sl = norm_vol[:, :, z]
            pred_mask, _ = self.predict_slice(sl)
            # Resize predicted 128x128 mask back to original slice dimension (h, w)
            from skimage.transform import resize
            pred_full = resize(
                pred_mask,
                (h, w),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.uint8)
            pred_volume[:, :, z] = pred_full

        if output_mask_path is not None:
            out_p = Path(output_mask_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_nii = nib.Nifti1Image(pred_volume, affine)
            nib.save(out_nii, str(out_p))
            self.logger.info(f"Saved predicted NIfTI mask to: {out_p}")

        return pred_volume
