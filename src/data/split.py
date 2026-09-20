"""
Patient-Level Splitting & Contamination Prevention Module
Enforces strict case-level isolation to prevent slice/patch leakage.
Generates split_70_15_15.json and kfold_5.json.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from sklearn.model_selection import KFold


def create_patient_splits(
    patient_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_cv_folds: int = 5,
    seed: int = 42,
    output_dir: str | Path = "data/splits",
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Creates patient-level split (70% train, 15% val, 15% test) and 5-fold CV on the 85% dev pool.
    Saves split_70_15_15.json and kfold_5.json into output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_ids = sorted(list(set(patient_ids)))
    n_total = len(sorted_ids)
    if n_total == 0:
        raise ValueError("Cannot split an empty list of patient IDs.")

    rng = np.random.RandomState(seed)
    shuffled_ids = rng.permutation(sorted_ids).tolist()

    n_test = max(1, int(round(n_total * test_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_test - n_val

    test_ids = sorted(shuffled_ids[:n_test])
    val_ids = sorted(shuffled_ids[n_test : n_test + n_val])
    train_ids = sorted(shuffled_ids[n_test + n_val :])

    # Assert zero leakage
    assert len(set(train_ids) & set(val_ids)) == 0, "Leakage between train and val!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Leakage between train and test!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Leakage between val and test!"

    master_split = {
        "dataset": "Medical Segmentation Decathlon Task 07 (Pancreas)",
        "seed": seed,
        "total_patients": n_total,
        "train_patients": train_ids,
        "val_patients": val_ids,
        "test_patients": test_ids,
    }

    master_file = out_path / "split_70_15_15.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(master_split, f, indent=2)

    # 5-fold cross-validation on the combined development pool (85%)
    dev_pool = sorted(train_ids + val_ids)
    kf = KFold(n_splits=num_cv_folds, shuffle=True, random_state=seed)

    kfold_dict: Dict[str, Dict[str, List[str]]] = {}
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dev_pool)):
        fold_train = sorted([dev_pool[i] for i in train_idx])
        fold_val = sorted([dev_pool[i] for i in val_idx])

        assert len(set(fold_train) & set(fold_val)) == 0, f"Fold {fold_idx + 1} leakage!"
        assert len(set(fold_train) & set(test_ids)) == 0, f"Test leakage into Fold {fold_idx + 1}!"
        assert len(set(fold_val) & set(test_ids)) == 0, f"Test leakage into Fold {fold_idx + 1}!"

        kfold_dict[f"fold_{fold_idx + 1}"] = {
            "train": fold_train,
            "val": fold_val,
        }

    kfold_file = out_path / "kfold_5.json"
    with open(kfold_file, "w", encoding="utf-8") as f:
        json.dump(kfold_dict, f, indent=2)

    return master_split, kfold_dict
