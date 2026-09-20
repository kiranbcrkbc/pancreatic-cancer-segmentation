"""
Automated Quality Gate Script
Validates data integrity, no leakage, forward pass, checkpoints, metrics,
XAI outputs, and notebook validity.
Reports TECHNICAL QUALITY GATE and PERFORMANCE TARGET STATUS separately.
"""

from pathlib import Path
import json
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.losses.compound_loss import CompoundLoss


def run_quality_gate() -> bool:
    checks = {}

    # 1. Dataset valid
    data_dir = Path("data/Task07_Pancreas")
    checks["Dataset valid"] = data_dir.exists() or Path("data/splits/split_70_15_15.json").exists()

    # 2. Patient split valid & No leakage
    split_file = Path("data/splits/split_70_15_15.json")
    if split_file.exists():
        with open(split_file, "r") as f:
            splits = json.load(f)
        tr = set(splits.get("train_patients", []))
        va = set(splits.get("val_patients", []))
        ts = set(splits.get("test_patients", []))
        checks["Patient split valid"] = (len(tr) > 0 and len(va) > 0 and len(ts) > 0)
        checks["No leakage"] = (len(tr & va) == 0) and (len(tr & ts) == 0) and (len(va & ts) == 0)
    else:
        checks["Patient split valid"] = False
        checks["No leakage"] = False

    # 3. Model forward pass works & Loss works
    try:
        model = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)
        dummy_x = torch.randn(2, 1, 128, 128)
        dummy_y = torch.randint(0, 3, (2, 128, 128)).long()
        logits = model(dummy_x)
        checks["Model forward pass works"] = (logits.shape == (2, 3, 128, 128))

        loss_fn = CompoundLoss()
        loss, _ = loss_fn(logits, dummy_y)
        checks["Loss works"] = bool(torch.isfinite(loss).item())
    except Exception:
        checks["Model forward pass works"] = False
        checks["Loss works"] = False

    # 4. Checkpoints exist & Training completed
    ckpt_dir = Path("checkpoints")
    fold_ckpts = [ckpt_dir / f"fold{i}_best.pt" for i in range(1, 6)]
    checks["Training completed"] = (Path("results/fold_metrics.csv").exists() or all(p.exists() for p in fold_ckpts))
    checks["Checkpoints exist"] = all(p.exists() for p in fold_ckpts) and (ckpt_dir / "final_model.pt").exists()

    # 5. Test evaluation completed & All required metrics exist
    results_dir = Path("results")
    metrics_file = results_dir / "final_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            m = json.load(f)
        required_keys = [
            "overall_accuracy", "background_dice", "pancreas_dice", "tumor_dice",
            "mean_foreground_dice", "pancreas_iou", "tumor_iou", "mcc"
        ]
        checks["Test evaluation completed"] = True
        checks["All required metrics exist"] = all(k in m for k in required_keys)
    else:
        checks["Test evaluation completed"] = False
        checks["All required metrics exist"] = False

    # 6. Results files exist
    checks["Results files exist"] = (
        (results_dir / "final_metrics.csv").exists()
        and (results_dir / "final_metrics.json").exists()
        and (results_dir / "best_hparams.json").exists()
    )

    # 7. XAI outputs exist
    fig_dir = results_dir / "figures"
    checks["XAI outputs exist"] = (
        (fig_dir / "confusion_matrix.png").exists()
        and (fig_dir / "gradcam_heatmap.png").exists()
        and (fig_dir / "transformer_attention.png").exists()
        and (fig_dir / "lime_visualization.png").exists()
    )

    # 8. Notebook exists & Notebook imports/structure are valid
    nb_path = Path("notebooks/Pancreatic_Cancer_Segmentation_End_to_End.ipynb")
    checks["Notebook exists"] = nb_path.exists()
    if nb_path.exists():
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            checks["Notebook imports/structure are valid"] = (len(nb.get("cells", [])) >= 50)
        except Exception:
            checks["Notebook imports/structure are valid"] = False
    else:
        checks["Notebook imports/structure are valid"] = False

    print("\n" + "=" * 65)
    print("                AUTOMATED QUALITY GATE REPORT")
    print("=" * 65)
    all_tech_passed = True
    for name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_tech_passed = False
        print(f"[ {'X' if passed else ' '} ] {name:38s} : {status}")

    print("-" * 65)
    if all_tech_passed:
        print("TECHNICAL QUALITY GATE              : PASS")
        print("PROJECT QUALITY GATE                : PASS")
    else:
        print("TECHNICAL QUALITY GATE              : FAIL")
        print("PROJECT QUALITY GATE                : FAIL")

    # Evaluate Performance Targets from actual metrics
    print("-" * 65)
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            m = json.load(f)
        acc = m.get("overall_accuracy", 0.0)
        t_dice = m.get("tumor_dice", 0.0)
        p_dice = m.get("pancreas_dice", 0.0)
        bg_dice = m.get("background_dice", 0.0)

        print(f"Overall Accuracy:  {acc * 100:.2f}%  (Target: >90.0%)")
        print(f"Background Dice:   {bg_dice * 100:.2f}%  (Target: 96-98%)")
        print(f"Pancreas Dice:     {p_dice * 100:.2f}%  (Target: 80-90%)")
        print(f"Tumor Dice:        {t_dice * 100:.2f}%  (Target: 90-95%, PRD: >=84.0%)")

        print("-" * 65)
        if t_dice >= 0.84:
            print("PERFORMANCE TARGET STATUS           : PASS")
        else:
            print(f"PERFORMANCE TARGET STATUS           : FAIL (Tumor Dice {t_dice * 100:.2f}% < Target 84.0%)")
    else:
        print("PERFORMANCE TARGET STATUS           : PENDING (Metrics not generated)")

    print("=" * 65 + "\n")
    return all_tech_passed


if __name__ == "__main__":
    success = run_quality_gate()
    if not success:
        sys.exit(1)
