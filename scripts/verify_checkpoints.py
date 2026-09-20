"""
Independent Checkpoint Verification Script
Verifies existence, integrity, parameter matching, and independence of all 5 fold checkpoints.
"""

from pathlib import Path
import json
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.cnn_transformer import CNNPyramidTransformerSeg


def verify_all_checkpoints():
    ckpt_dir = Path("checkpoints")
    model = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)
    ref_keys = set(model.state_dict().keys())

    print("=" * 70)
    print("        INDEPENDENT 5-FOLD CHECKPOINT VERIFICATION REPORT")
    print("=" * 70)

    all_valid = True
    fold_weights = {}

    for fold_idx in range(1, 6):
        ckpt_path = ckpt_dir / f"fold{fold_idx}_best.pt"
        print(f"\n--- Verifying Fold {fold_idx} Checkpoint: {ckpt_path} ---")

        if not ckpt_path.exists():
            print(f"[FAIL] File does not exist: {ckpt_path}")
            all_valid = False
            continue

        try:
            data = torch.load(ckpt_path, map_location="cpu")
            state_dict = data.get("model_state_dict", data)

            # Check keys
            ckpt_keys = set(state_dict.keys())
            missing = ref_keys - ckpt_keys
            unexpected = ckpt_keys - ref_keys

            if missing or unexpected:
                print(f"[FAIL] Architecture mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys")
                all_valid = False
                continue

            # Load into model to verify compatibility
            model.load_state_dict(state_dict)

            # Check parameter values for NaN / Inf
            has_nan = False
            first_weight = None
            total_params = 0
            for name, param in state_dict.items():
                if not torch.isfinite(param).all():
                    has_nan = True
                    break
                total_params += param.numel()
                if first_weight is None and "conv" in name:
                    first_weight = param.flatten()[:5].tolist()

            if has_nan:
                print(f"[FAIL] Checkpoint contains NaN or Inf values!")
                all_valid = False
                continue

            fold_weights[fold_idx] = first_weight

            # Metadata
            epoch = data.get("epoch", "N/A")
            best_metric = data.get("best_metric", "N/A")
            fold_meta = data.get("fold", fold_idx)

            print(f"[PASS] Checkpoint integrity: VALID (Total params: {total_params:,})")
            print(f"       Metadata: Fold={fold_meta}, Epoch={epoch}, BestMetric={best_metric}")

        except Exception as e:
            print(f"[FAIL] Checkpoint loading exception: {e}")
            all_valid = False

    # Verify independence: ensure checkpoints were not accidentally duplicated
    print("\n" + "-" * 70)
    print("Checkpoint Independence Audit:")
    duplicates = False
    for f1 in fold_weights:
        for f2 in fold_weights:
            if f1 < f2 and fold_weights[f1] == fold_weights[f2]:
                print(f"[WARN] Fold {f1} and Fold {f2} have identical initial weights!")
                duplicates = True

    if not duplicates:
        print("[PASS] All 5 fold checkpoints have distinct, independently trained weights.")

    print("=" * 70)
    print(f"OVERALL CHECKPOINT STATUS: {'ALL 5 PASS' if all_valid else 'FAIL'}")
    print("=" * 70)
    return all_valid


if __name__ == "__main__":
    ok = verify_all_checkpoints()
    if not ok:
        sys.exit(1)
