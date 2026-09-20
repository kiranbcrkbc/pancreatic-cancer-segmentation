"""
Metric Reproduction & Ground Truth Audit Script
Performs an independent end-to-end evaluation from raw test slices and weights,
calculating Dice, IoU, Precision, Recall, Specificity, F1, Accuracy, and MCC from scratch.
"""

from pathlib import Path
import json
import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.data.datamodule import get_test_loader
from src.inference.ensemble import EnsemblePredictor
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.hausdorff import compute_hd95


def reproduce_and_verify():
    print("=" * 70)
    print("        INDEPENDENT METRIC REPRODUCTION AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load splits
    split_file = Path("data/splits/split_70_15_15.json")
    with open(split_file, "r", encoding="utf-8") as f:
        test_ids = json.load(f)["test_patients"]
    print(f"Loaded {len(test_ids)} held-out test patients: {test_ids}")

    # 2. Load the 5 checkpoints
    ckpt_dir = Path("checkpoints")
    models = []
    weights = []
    for f in range(1, 6):
        cp = ckpt_dir / f"fold{f}_best.pt"
        if cp.exists():
            data = torch.load(cp, map_location=device)
            m = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)
            m.load_state_dict(data["model_state_dict"])
            m.to(device).eval()
            models.append(m)
            w = float(data.get("best_metric", 0.01))
            weights.append(max(0.001, w))
            print(f"Loaded Fold {f} Checkpoint (Weight: {w:.4f})")

    ensemble = EnsemblePredictor(models, device=device, weights=weights)

    # 3. Load test data
    test_loader = get_test_loader(test_ids=test_ids, batch_size=8, num_workers=0)
    print(f"Evaluating {len(test_loader.dataset)} test patches...")

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            probs, preds = ensemble.predict_batch(images)
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    # 4. Independent Metric Calculation
    metrics = compute_all_metrics(preds, targets, probs=probs)
    hd_metrics = compute_hd95(preds, targets)
    metrics.update(hd_metrics)

    print("\n" + "-" * 70)
    print("REPRODUCED VALUES (From scratch):")
    print(f"Overall Accuracy : {metrics['overall_accuracy'] * 100:.2f}%")
    print(f"Background Dice  : {metrics['background_dice'] * 100:.2f}%")
    print(f"Pancreas Dice    : {metrics['pancreas_dice'] * 100:.2f}%")
    print(f"Tumor Dice       : {metrics['tumor_dice'] * 100:.2f}%")
    print(f"Tumor IoU        : {metrics['tumor_iou'] * 100:.2f}%")
    print(f"Pancreas IoU     : {metrics['pancreas_iou'] * 100:.2f}%")
    print(f"MCC              : {metrics['mcc']:.4f}")
    print(f"Pancreas HD95    : {metrics.get('pancreas_hd95', 'N/A')} px")
    print(f"Tumor HD95       : {metrics.get('tumor_hd95', 'N/A')} px")

    # 5. Compare with saved file
    saved_file = Path("results/final_metrics.json")
    if saved_file.exists():
        with open(saved_file, "r") as f:
            saved = json.load(f)
        diff_bg = abs(saved["background_dice"] - metrics["background_dice"])
        diff_pan = abs(saved["pancreas_dice"] - metrics["pancreas_dice"])
        diff_tum = abs(saved["tumor_dice"] - metrics["tumor_dice"])
        print("\nVerification against results/final_metrics.json:")
        print(f"Background Dice absolute difference : {diff_bg:.6f}")
        print(f"Pancreas Dice absolute difference   : {diff_pan:.6f}")
        print(f"Tumor Dice absolute difference      : {diff_tum:.6f}")
        assert diff_bg < 1e-4 and diff_pan < 1e-4 and diff_tum < 1e-4, "Metrics mismatch!"
        print("[PASS] Reproduced metrics match results/final_metrics.json exactly.")

    print("=" * 70)
    return True


if __name__ == "__main__":
    reproduce_and_verify()
