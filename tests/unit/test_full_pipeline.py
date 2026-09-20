"""Comprehensive pipeline tests covering imports, configs, dataset, models, losses, metrics, ensemble, and XAI."""

from pathlib import Path
import json
import pytest
import numpy as np
import torch

# A. Imports
from src.utils.config import load_all_configs
from src.data.dataset import PancreasPatchDataset
from src.data.split import create_patient_splits
from src.preprocessing.ct_preprocessor import CTPreprocessor
from src.preprocessing.roi_extractor import ROIPatchExtractor
from src.augmentation.albumentations import get_training_augmentation
from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.losses.compound_loss import CompoundLoss
from src.evaluation.metrics import compute_all_metrics
from src.inference.ensemble import EnsemblePredictor
from src.inference.predictor import ClinicalPredictor
from src.xai.gradcam import GradCAMSeg
from src.xai.attention_vis import extract_bottleneck_attention_maps
from generate_notebook import build_notebook


def test_configuration_loading():
    # B. Configuration
    configs = load_all_configs("configs")
    assert "model" in configs
    assert "training" in configs
    assert "loss" in configs
    assert "preprocessing" in configs


def test_dataset_and_split(tmp_path):
    # C. Dataset loading & patient split
    pids = [f"test_pat_{i:03d}" for i in range(1, 21)]
    master, kfolds = create_patient_splits(pids, output_dir=tmp_path)
    assert len(master["train_patients"]) > 0
    assert len(master["val_patients"]) > 0
    assert len(master["test_patients"]) > 0


def test_preprocessing_and_augmentation():
    # D & E. Preprocessing & Augmentation
    prep = CTPreprocessor(hu_min=-150, hu_max=250)
    raw = np.array([-200.0, 0.0, 100.0, 300.0], dtype=np.float32)
    norm = prep.normalize(prep.clip_hu(raw))
    assert norm.min() >= 0.0 and norm.max() <= 1.0

    aug = get_training_augmentation(patch_size=(64, 64))
    res = aug(image=np.zeros((64, 64), dtype=np.float32), mask=np.zeros((64, 64), dtype=np.int64))
    assert res["image"].shape == (64, 64)


def test_model_forward_and_loss():
    # F, G, H. Model construction, forward pass, compound loss
    model = CNNPyramidTransformerSeg(
        in_channels=1,
        num_classes=3,
        encoder_channels=[16, 32, 64, 128],
        embed_dim=64,
        num_heads=4,
        transformer_depth=2,
        ffn_dim=128,
        dropout=0.0,
    )
    x = torch.randn(2, 1, 64, 64)
    y = torch.randint(0, 3, (2, 64, 64)).long()
    logits = model(x)
    assert logits.shape == (2, 3, 64, 64)

    criterion = CompoundLoss(dice_w=0.6, ce_w=0.3, focal_w=0.1)
    loss, components = criterion(logits, y)
    assert torch.isfinite(loss)
    assert "loss_dice" in components


def test_checkpoint_loading_and_ensemble():
    # I, J, K, L. Checkpoint loading, inference, metrics, ensemble
    model = CNNPyramidTransformerSeg(in_channels=1, num_classes=3)
    ckpt_path = Path("checkpoints/final_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/fold1_best.pt")

    if ckpt_path.exists():
        data = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(data["model_state_dict"])

    ensemble = EnsemblePredictor([model], device=torch.device("cpu"), weights=[1.0])
    dummy_input = torch.randn(2, 1, 128, 128)
    probs, preds = ensemble.predict_batch(dummy_input)
    assert probs.shape == (2, 3, 128, 128)
    assert preds.shape == (2, 128, 128)

    # Metrics
    dummy_tgt = np.random.randint(0, 3, size=(2, 128, 128))
    metrics = compute_all_metrics(preds, dummy_tgt, probs=probs)
    assert "overall_accuracy" in metrics
    assert "tumor_dice" in metrics


def test_xai_and_notebook():
    # M & N. XAI generation & Notebook structure
    model = CNNPyramidTransformerSeg(
        in_channels=1,
        num_classes=3,
        encoder_channels=[16, 32, 64, 128],
        embed_dim=64,
        num_heads=4,
        transformer_depth=2,
        ffn_dim=128,
    )
    x = torch.randn(1, 1, 64, 64)
    cam_engine = GradCAMSeg(model, model.get_cam_target_layer())
    hm = cam_engine.generate_heatmap(x, target_class=2)
    cam_engine.close()
    assert hm.shape == (64, 64)

    # Notebook verification
    build_notebook()
    nb_path = Path("notebooks/Pancreatic_Cancer_Segmentation_End_to_End.ipynb")
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert len(nb["cells"]) >= 50
