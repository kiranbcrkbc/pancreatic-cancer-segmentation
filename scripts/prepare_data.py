"""
Data Preparation & Leakage Verification Script
Discovers MSD Task 07 Pancreas data, creates patient splits, and verifies zero leakage.
"""

import argparse
from pathlib import Path
import json
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.split import create_patient_splits
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset and patient splits")
    parser.add_argument("--data-dir", type=str, default="data/Task07_Pancreas", help="Path to raw MSD Task 07 data")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split JSONs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    logger = get_logger("prepare_data")
    logger.info("Initializing Data Preparation Pipeline...")

    data_path = Path(args.data_dir)
    images_tr = data_path / "imagesTr"

    # Discover patients
    patient_ids = []
    if images_tr.exists():
        for f in sorted(images_tr.glob("*.nii*")):
            pid = f.name.replace(".nii.gz", "").replace(".nii", "")
            patient_ids.append(pid)

    if not patient_ids:
        logger.warning(f"No NIfTI files found in {images_tr}. Generating standard 50-patient cohort IDs for setup.")
        patient_ids = [f"pancreas_{i:03d}" for i in range(1, 51)]

    logger.info(f"Identified {len(patient_ids)} unique patient cases.")

    # Create patient splits
    master_split, kfold_dict = create_patient_splits(
        patient_ids=patient_ids,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        num_cv_folds=5,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    tr_set = set(master_split["train_patients"])
    val_set = set(master_split["val_patients"])
    ts_set = set(master_split["test_patients"])

    # Strict Leakage Checks
    leak_tv = len(tr_set & val_set)
    leak_tt = len(tr_set & ts_set)
    leak_vt = len(val_set & ts_set)

    if leak_tv == 0 and leak_tt == 0 and leak_vt == 0:
        print("DATA LEAKAGE CHECK: PASS")
        logger.info("DATA LEAKAGE CHECK: PASS (Zero patient overlap across Train, Val, and Test partitions).")
    else:
        print("DATA LEAKAGE CHECK: FAIL")
        logger.error(f"DATA LEAKAGE CHECK: FAIL! Overlaps: TR/VAL={leak_tv}, TR/TS={leak_tt}, VAL/TS={leak_vt}")
        sys.exit(1)


if __name__ == "__main__":
    main()
