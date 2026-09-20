"""
Explainable AI (XAI) Generation CLI Script
Generates Grad-CAM, Transformer Self-Attention, and LIME interpretability panels.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.data.dataset import PancreasPatchDataset
from src.xai.gradcam import GradCAMSeg
from src.xai.attention_vis import extract_bottleneck_attention_maps
from src.xai.lime_explainer import explain_slice_with_lime
from src.utils.visualization import save_segmentation_comparison
from src.utils.logger import get_logger


def main() -> None:
    logger = get_logger("generate_xai")
    logger.info("Starting Explainable AI (XAI) diagnostic generation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    xai_dir = Path("xai")
    xai_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)
    ckpt_path = Path("checkpoints/final_model.pt")
    if not ckpt_path.exists():
        # Fallback to fold 1
        ckpt_path = Path("checkpoints/fold1_best.pt")

    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location=device)
        state_dict = data.get("model_state_dict", data)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {ckpt_path}")
    else:
        logger.warning("No checkpoint found, using initialized model.")
    model.to(device).eval()

    # 2. Load a sample test patch
    split_file = Path("data/splits/split_70_15_15.json")
    if split_file.exists():
        with open(split_file, "r") as f:
            test_ids = json.load(f)["test_patients"]
    else:
        test_ids = ["pancreas_001"]

    ds = PancreasPatchDataset(patient_ids=test_ids[:3], is_training=False)
    img_tensor, mask_tensor, meta = ds[0]
    img_np = img_tensor.squeeze().numpy()
    mask_np = mask_tensor.numpy()

    # 3. Model forward prediction
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # 4. Grad-CAM for Tumor (Class 2)
    cam_layer = model.get_cam_target_layer()
    gradcam_engine = GradCAMSeg(model, cam_layer)
    cam_heatmap = gradcam_engine.generate_heatmap(img_tensor.unsqueeze(0).to(device), target_class=2)
    gradcam_engine.close()

    # 5. Transformer Bottleneck Self-Attention Map
    attn_heatmap = extract_bottleneck_attention_maps(model, img_tensor.unsqueeze(0).to(device))

    # 6. LIME Superpixel Perturbation
    segments, lime_heatmap = explain_slice_with_lime(model, img_np, target_class=2, num_samples=60, device=device)

    # 7. Save Figures
    # 7.1 Authoritative 5-Panel Diagnostic Report (Raw CT, Ground Truth, Prediction, Grad-CAM, Overlay)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input CT (Axial ROI)")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="viridis", vmin=0, vmax=2)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="viridis", vmin=0, vmax=2)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(cam_heatmap, cmap="jet")
    axes[3].set_title("Grad-CAM Heatmap (Tumor)")
    axes[3].axis("off")

    axes[4].imshow(img_np, cmap="gray")
    axes[4].imshow(cam_heatmap, cmap="jet", alpha=0.5)
    axes[4].set_title("Blended Grad-CAM Overlay")
    axes[4].axis("off")

    plt.suptitle("Clinical Diagnostic Interpretability Panel", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(xai_dir / "gradcam_xai_interpretability.png", dpi=300)
    plt.savefig(fig_dir / "gradcam_heatmap.png", dpi=300)
    plt.close()

    # 7.2 Transformer Attention Map Plot
    plt.figure(figsize=(5, 5))
    plt.imshow(img_np, cmap="gray")
    plt.imshow(attn_heatmap, cmap="plasma", alpha=0.55)
    plt.title("Transformer Bottleneck Self-Attention Overlay", fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "transformer_attention.png", dpi=300)
    plt.close()

    # 7.3 LIME Explanation Plot
    plt.figure(figsize=(5, 5))
    plt.imshow(img_np, cmap="gray")
    plt.imshow(lime_heatmap, cmap="magma", alpha=0.55)
    plt.title("LIME Superpixel Importance (Tumor)", fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "lime_visualization.png", dpi=300)
    plt.close()

    # 7.4 Ground Truth vs Prediction Comparison Plot
    save_segmentation_comparison(
        image=img_np,
        target=mask_np,
        prediction=pred_mask,
        save_path=fig_dir / "ground_truth_vs_prediction.png",
        heatmap=cam_heatmap,
        title="Comparative Segmentation & Interpretability",
    )

    logger.info("XAI Figures successfully generated and saved to results/figures/ and xai/!")


if __name__ == "__main__":
    main()
