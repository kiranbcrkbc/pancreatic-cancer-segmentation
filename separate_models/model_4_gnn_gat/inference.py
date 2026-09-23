"""
Inference & Interpretability Pipeline for Model 4: CNN + GNN/GAT (AttnUNetEfficientGAT)

Demonstrates end-to-end clinical inference:
Input CT Slice
    ↓
Preprocessing (HU windowing [-150, +250], Min-Max scaling [0.0, 1.0], Resizing)
    ↓
Model Forward Pass (AttnUNetEfficientGAT)
    ↓
Segmentation Prediction (Argmax -> 0: Background, 1: Pancreas, 2: Tumor)
    ↓
Grad-CAM Interpretability (Target layer: gat_bottleneck.conv_out[0])
    ↓
Saved Output Masks & Diagnostic Panels

Usage:
    python inference.py --checkpoint ../../checkpoints/gnn/final_model.pt --output-dir ./outputs
    python inference.py --input /path/to/slice.png --checkpoint ../../checkpoints/gnn/final_model.pt
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AttnUNetEfficientGAT


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

class CTPreprocessor:
    """Preprocesses CT slices with Hounsfield Unit windowing and normalization."""

    def __init__(self, hu_min: float = -150.0, hu_max: float = 250.0, target_size: Tuple[int, int] = (128, 128)) -> None:
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_size = target_size

    def preprocess_slice(self, slice_array: np.ndarray) -> np.ndarray:
        clipped = np.clip(slice_array, self.hu_min, self.hu_max)
        norm = (clipped - self.hu_min) / (self.hu_max - self.hu_min)
        norm = norm.astype(np.float32)

        if norm.shape != self.target_size:
            pil_img = Image.fromarray((norm * 255).astype(np.uint8))
            resized = pil_img.resize(self.target_size, resample=Image.BILINEAR)
            norm = np.array(resized, dtype=np.float32) / 255.0

        return norm


# -----------------------------------------------------------------------------
# Grad-CAM Engine
# -----------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for segmentation models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, inp: Tuple, out: torch.Tensor) -> None:
        self.activations = out.detach()

    def _save_gradients(self, module: nn.Module, grad_in: Tuple, grad_out: Tuple) -> None:
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 2) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(x)

        score = logits[0, target_class].sum()
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = F.relu(cam)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        cam_min, cam_max = cam_upsampled.min(), cam_upsampled.max()
        if cam_max > cam_min:
            cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_upsampled)

        return cam_norm.astype(np.float32)

    def close(self) -> None:
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# -----------------------------------------------------------------------------
# Inference Execution
# -----------------------------------------------------------------------------

def resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    script_dir = Path(__file__).resolve().parent
    cand1 = (script_dir / path_str).resolve()
    if cand1.exists():
        return cand1
    cand2 = (script_dir.parent.parent / "checkpoints" / "gnn" / p.name).resolve()
    if cand2.exists():
        return cand2
    return p


def run_inference(
    input_path: Optional[str] = None,
    checkpoint_path: str = "../../checkpoints/gnn/final_model.pt",
    output_dir: str = "./outputs",
    device: str = "cpu",
) -> None:
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Model 4: CNN + GNN/GAT Inference ===")
    print(f"Device: {device_obj}")

    model = AttnUNetEfficientGAT(in_channels=1, num_classes=3, use_efficientnet=True).to(device_obj)

    ckpt_file = resolve_checkpoint(checkpoint_path)
    if ckpt_file.exists():
        print(f"Loading checkpoint from: {ckpt_file}")
        data = torch.load(ckpt_file, map_location=device_obj)
        state_dict = data.get("model_state_dict", data)
        model.load_state_dict(state_dict, strict=True)
        print("[PASS] Checkpoint loaded successfully with strict=True.")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_file}. Running with initialized weights.")

    model.eval()
    preprocessor = CTPreprocessor()

    if input_path and Path(input_path).exists():
        p = Path(input_path)
        if p.suffix.lower() == ".npy":
            raw_slice = np.load(p)
        else:
            pil_img = Image.open(p).convert("L")
            raw_slice = np.array(pil_img, dtype=np.float32)
        print(f"Loaded input CT slice from: {input_path}")
    else:
        print("No input image specified. Generating synthetic abdominal CT slice patch...")
        raw_slice = np.ones((128, 128), dtype=np.float32) * -100.0
        y, x = np.ogrid[:128, :128]
        pancreas_mask = ((x - 64) ** 2 / 30**2 + (y - 64) ** 2 / 16**2) <= 1
        tumor_mask = ((x - 70) ** 2 / 8**2 + (y - 64) ** 2 / 8**2) <= 1
        raw_slice[pancreas_mask] = 40.0
        raw_slice[tumor_mask] = 80.0

    processed_slice = preprocessor.preprocess_slice(raw_slice)
    tensor_in = torch.from_numpy(processed_slice).unsqueeze(0).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        logits = model(tensor_in)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_mask = np.argmax(probs, axis=0)

    cam_layer = model.get_cam_target_layer()
    cam_engine = GradCAM(model, cam_layer)
    cam_heatmap = cam_engine.generate_heatmap(tensor_in, target_class=2)
    cam_engine.close()

    plt.imsave(out_dir / "input_ct_slice.png", processed_slice, cmap="gray")
    plt.imsave(out_dir / "predicted_mask.png", pred_mask, cmap="viridis", vmin=0, vmax=2)
    plt.imsave(out_dir / "gradcam_heatmap.png", cam_heatmap, cmap="jet")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(processed_slice, cmap="gray")
    axes[0].set_title("Input CT Slice (Windowed)")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="viridis", vmin=0, vmax=2)
    axes[1].set_title("Predicted Mask (0:BG, 1:Panc, 2:Tumor)")
    axes[1].axis("off")

    axes[2].imshow(cam_heatmap, cmap="jet")
    axes[2].set_title("Grad-CAM Heatmap (Tumor Class 2)")
    axes[2].axis("off")

    axes[3].imshow(processed_slice, cmap="gray")
    axes[3].imshow(cam_heatmap, cmap="jet", alpha=0.45)
    axes[3].set_title("Blended XAI Overlay")
    axes[3].axis("off")

    plt.suptitle("Model 4 (CNN + GNN/GAT) Clinical Diagnostic Inference", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "clinical_inference_panel.png", dpi=200)
    plt.close()

    print(f"\n[PASS] Inference completed successfully!")
    print(f"Outputs exported to: {out_dir.resolve()}")
    print(f"  - Input CT Slice      : {out_dir / 'input_ct_slice.png'}")
    print(f"  - Predicted Mask      : {out_dir / 'predicted_mask.png'}")
    print(f"  - Grad-CAM Heatmap    : {out_dir / 'gradcam_heatmap.png'}")
    print(f"  - Clinical Panel      : {out_dir / 'clinical_inference_panel.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for Model 4: CNN + GNN/GAT")
    parser.add_argument("--input", type=str, default=None, help="Path to input CT slice image or .npy file")
    parser.add_argument("--checkpoint", type=str, default="../../checkpoints/gnn/final_model.pt", help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save output figures")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
