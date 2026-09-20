"""
Clinical Inference CLI Script
Runs segmentation inference on a provided NIfTI scan or numpy slice.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import ClinicalPredictor
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical Segmentation Inference")
    parser.add_argument("--input", type=str, required=False, default=None, help="Path to input NIfTI (.nii.gz) or image")
    parser.add_argument("--output-dir", type=str, default="results/inference", help="Directory to save predictions")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    logger = get_logger("infer_cli")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = [Path(args.checkpoints_dir) / f"fold{i}_best.pt" for i in range(1, 6)]
    predictor = ClinicalPredictor(ckpt_paths)

    if args.input and Path(args.input).exists():
        in_p = Path(args.input)
        if in_p.suffix == ".gz" or in_p.suffix == ".nii":
            out_mask_p = out_dir / f"{in_p.stem}_predicted_mask.nii.gz"
            pred_vol = predictor.predict_nifti_volume(in_p, output_mask_path=out_mask_p)
            logger.info(f"Generated 3D segmentation volume: {pred_vol.shape}")
    else:
        logger.info("No input file provided. Generating sample inference demonstration...")
        # Create synthetic test slice
        h, w = 256, 256
        rng = np.random.RandomState(42)
        sample_slice = np.clip(rng.normal(0.45, 0.08, size=(h, w)), 0, 1).astype(np.float32)

        pred_mask, probs = predictor.predict_slice(sample_slice)

        # Plot result
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(sample_slice, cmap="gray")
        axes[0].set_title("Input CT Axial Slice")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="viridis", vmin=0, vmax=2)
        axes[1].set_title("Predicted Mask (0=BG, 1=Panc, 2=Tum)")
        axes[1].axis("off")

        plt.tight_layout()
        save_file = out_dir / "sample_inference_demo.png"
        plt.savefig(save_file, dpi=300)
        plt.close()
        logger.info(f"Sample inference demonstration saved to {save_file}")


if __name__ == "__main__":
    main()
