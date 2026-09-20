"""
One-Command Master Pipeline Orchestrator
Executes data preparation, hyperparameter search, 5-fold CV training,
test set evaluation, XAI generation, and automated quality gate.
"""

from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger


def run_command(cmd: list[str], desc: str) -> None:
    logger = get_logger("orchestrator")
    logger.info(f"=== [STARTING] {desc} ===")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        logger.error(f"Command failed with code {res.returncode}: {' '.join(cmd)}")
        sys.exit(res.returncode)
    dur = time.time() - t0
    logger.info(f"=== [FINISHED] {desc} in {dur:.1f}s ===")


def main() -> None:
    py_exec = sys.executable

    # 1. Prepare Data & Splits
    run_command([py_exec, "scripts/prepare_data.py"], "Step 1: Dataset & Patient Splitting")

    # 2. Hyperparameter Tuning (skip if best_hparams.json already exists)
    if not Path("results/best_hparams.json").exists():
        run_command([py_exec, "scripts/tune_hparams.py", "--trials", "4", "--epochs", "2", "--subset", "40"], "Step 2: Optuna Hyperparameter Optimization")
    else:
        print("[Step 2] Optuna best hyperparameters already present in results/best_hparams.json (Skipping redundant search).")

    # 3. 5-Fold Cross Validation Training
    fold_ckpts = [Path(f"checkpoints/fold{i}_best.pt") for i in range(1, 6)]
    if not all(p.exists() for p in fold_ckpts):
        run_command([py_exec, "scripts/train_cv.py", "--epochs", "3", "--batch-size", "8"], "Step 3: 5-Fold Cross Validation Training")
    else:
        print("[Step 3] 5-Fold Checkpoints already exist in checkpoints/ (Skipping redundant training).")

    # 4. Held-out Test Set Evaluation
    run_command([py_exec, "scripts/evaluate.py"], "Step 4: Held-Out Test Set Evaluation")

    # 5. Explainable AI Generation
    run_command([py_exec, "scripts/generate_xai.py"], "Step 5: Explainable AI Diagnostics")

    # 6. Sample Inference
    run_command([py_exec, "scripts/infer.py"], "Step 6: Sample Clinical Inference")

    # 7. Automated Quality Gate
    run_command([py_exec, "scripts/quality_gate.py"], "Step 7: Automated Quality Gate")

    print("\n" + "*" * 60)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()
