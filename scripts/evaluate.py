"""
Held-Out Test Set Evaluation CLI Script
Executes 5-model soft ensemble on the untouched 15% test set and verifies acceptance criteria.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import evaluate_held_out_test_set
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Held-Out Test Set")
    parser.add_argument("--data-dir", type=str, default="data/Task07_Pancreas")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    logger = get_logger("evaluate")
    logger.info("Starting Held-Out Test Set Evaluation...")

    splits_path = Path(args.splits_dir) / "split_70_15_15.json"
    if not splits_path.exists():
        logger.error("Splits file not found! Run prepare_data first.")
        sys.exit(1)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = splits["test_patients"]
    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_paths = [
        ckpt_dir / f"fold{i}_best.pt" for i in range(1, 6)
    ]

    metrics = evaluate_held_out_test_set(
        test_ids=test_ids,
        checkpoint_paths=ckpt_paths,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    tumor_dice = metrics["tumor_dice"]
    panc_dice = metrics["pancreas_dice"]
    bg_dice = metrics["background_dice"]
    acc = metrics["overall_accuracy"]

    print("\n" + "=" * 60)
    print("           HELD-OUT TEST SET EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy:       {acc * 100:.2f}%")
    print(f"Background Dice:        {bg_dice * 100:.2f}% (Target: 96-98%)")
    print(f"Pancreas Dice:          {panc_dice * 100:.2f}% (Target: 80-90%)")
    print(f"Tumor Dice:             {tumor_dice * 100:.2f}% (Target: 90-95%, PRD Acceptance: >=84.0%)")
    print(f"Mean Foreground Dice:   {metrics['mean_foreground_dice'] * 100:.2f}%")
    print(f"Tumor IoU (Jaccard):    {metrics['tumor_iou'] * 100:.2f}%")
    print(f"Pancreas IoU:           {metrics['pancreas_iou'] * 100:.2f}%")
    print(f"Matthews Corr (MCC):    {metrics.get('mcc', 0.0):.4f}")
    print("=" * 60)

    # Acceptance Gate evaluation
    if tumor_dice >= 0.84:
        print("TUMOR DICE ACCEPTANCE GATE: PASS (>= 0.84 achieved)")
    else:
        print(f"TUMOR DICE ACCEPTANCE GATE: FAIL (Actual: {tumor_dice:.4f} < Target: 0.8400)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
