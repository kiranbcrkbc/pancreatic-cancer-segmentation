"""
Hyperparameter Tuning CLI Script
Executes Optuna Bayesian optimization and exports best_hparams.json.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.optuna_tuner import run_optuna_study
from src.utils.config import load_yaml
from src.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Hyperparameters using Optuna")
    parser.add_argument("--trials", type=int, default=12, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--subset", type=int, default=200, help="Number of samples in tuning subset")
    parser.add_argument("--data-dir", type=str, default="data/Task07_Pancreas")
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    logger = get_logger("tune_hparams")
    logger.info("Starting Hyperparameter Tuning...")

    split_file = Path(args.splits_dir) / "split_70_15_15.json"
    if not split_file.exists():
        logger.info("Split file not found. Running prepare_data first...")
        from scripts.prepare_data import main as prep_main
        prep_main()

    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    results = run_optuna_study(
        train_ids=splits["train_patients"],
        val_ids=splits["val_patients"],
        data_dir=args.data_dir,
        n_trials=args.trials,
        epochs_per_trial=args.epochs,
        subset_size=args.subset,
        output_dir=args.output_dir,
    )

    logger.info(f"Tuning complete! Best hyperparameters saved to {args.output_dir}/best_hparams.json")


if __name__ == "__main__":
    main()
