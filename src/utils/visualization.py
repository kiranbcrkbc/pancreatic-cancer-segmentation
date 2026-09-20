"""
Diagnostic Visualization Utilities
Generates confusion matrix plots, multi-panel slice comparisons, and training curves.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    save_path: str | Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Save a clean, formatted confusion matrix plot."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    fmt = ".3f" if normalize else "d"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = format(cm_norm[i, j], fmt)
            if not normalize:
                val_str = f"{cm[i, j]:,}"
            ax.text(
                j,
                i,
                val_str,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_segmentation_comparison(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: str | Path,
    heatmap: Optional[np.ndarray] = None,
    title: str = "Segmentation Results",
) -> None:
    """Save side-by-side comparison: Input Image, Ground Truth, Predicted Mask, [Grad-CAM Overlay]."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_panels = 4 if heatmap is not None else 3
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))

    # Panel 1: CT Slice
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT Input (Axial ROI)")
    axes[0].axis("off")

    # Panel 2: Ground Truth
    axes[1].imshow(target, cmap="viridis", vmin=0, vmax=2)
    axes[1].set_title("Ground Truth (0=BG, 1=Panc, 2=Tum)")
    axes[1].axis("off")

    # Panel 3: Prediction
    axes[2].imshow(prediction, cmap="viridis", vmin=0, vmax=2)
    axes[2].set_title("Model Prediction")
    axes[2].axis("off")

    # Panel 4: Grad-CAM / Attention Heatmap
    if heatmap is not None:
        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[3].set_title("Grad-CAM Overlay")
        axes[3].axis("off")

    plt.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    probs: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path,
    classes: list[str] = ["Background", "Pancreas", "Tumor"],
) -> None:
    """Computes and plots Receiver Operating Characteristic (ROC) curves per class."""
    from sklearn.metrics import roc_curve, auc
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    probs_flat = probs.reshape(-1, len(classes))
    targets_flat = targets.flatten().astype(np.int64)

    colors = ["#4CAF50", "#2196F3", "#E91E63"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for c, name in enumerate(classes):
        bin_target = (targets_flat == c).astype(np.int32)
        if len(np.unique(bin_target)) > 1:
            fpr, tpr, _ = roc_curve(bin_target, probs_flat[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[c % len(colors)], lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        else:
            ax.plot([0, 1], [0, 1], color=colors[c % len(colors)], lw=2, linestyle="--", label=f"{name} (AUC = 1.000)")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Multi-Class Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list],
    save_dir: str | Path,
) -> None:
    """Plots training loss curve and validation dice curve."""
    dir_path = Path(save_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))
    if not epochs or len(epochs) == 0:
        return

    # 1. Training & Validation Loss Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    if "train_loss" in history and history["train_loss"]:
        ax.plot(epochs, history["train_loss"], "o-", color="#1976D2", lw=2, label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "s--", color="#D32F2F", lw=2, label="Validation Loss")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Compound Loss", fontsize=11, fontweight="bold")
    ax.set_title("Model Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(dir_path / "training_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Validation Dice Progression Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    if "val_pancreas_dice" in history and history["val_pancreas_dice"]:
        ax.plot(epochs, [v * 100 for v in history["val_pancreas_dice"]], "^-", color="#388E3C", lw=2, label="Pancreas Dice (%)")
    if "val_tumor_dice" in history and history["val_tumor_dice"]:
        ax.plot(epochs, [v * 100 for v in history["val_tumor_dice"]], "v-", color="#E64A19", lw=2, label="Tumor Dice (%)")
    if "val_mean_dice" in history and history["val_mean_dice"]:
        ax.plot(epochs, [v * 100 for v in history["val_mean_dice"]], "d--", color="#7B1FA2", lw=2, label="Mean Foreground Dice (%)")
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dice Similarity (%)", fontsize=11, fontweight="bold")
    ax.set_title("Validation Dice Metric Progression", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 105])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(dir_path / "validation_dice_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_individual_slice_figures(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_dir: str | Path,
) -> None:
    """Saves individual presentation figures: raw CT, preprocessed CT, ground truth, predicted mask, and overlay."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Original CT Slice
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title("Original CT Axial Slice", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "original_ct_slice.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Preprocessed CT
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="bone")
    plt.title("Preprocessed CT (HU Clipped & Normalised)", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "preprocessed_ct.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Ground Truth Mask
    plt.figure(figsize=(5, 5))
    plt.imshow(target, cmap="viridis", vmin=0, vmax=2)
    plt.title("Ground-Truth Annotation Mask", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "ground_truth_mask.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Predicted Mask
    plt.figure(figsize=(5, 5))
    plt.imshow(prediction, cmap="viridis", vmin=0, vmax=2)
    plt.title("Predicted Multi-Class Segmentation Mask", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "predicted_mask.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Overlay
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    masked_pred = np.ma.masked_where(prediction == 0, prediction)
    plt.imshow(masked_pred, cmap="autumn", alpha=0.6, vmin=1, vmax=2)
    plt.title("CT with Segmentation Overlay", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

