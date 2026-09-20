"""Unit tests for preprocessing, augmentation, losses, and data leakage."""

import numpy as np
import pytest
import torch

from src.preprocessing.ct_preprocessor import CTPreprocessor
from src.preprocessing.roi_extractor import ROIPatchExtractor
from src.augmentation.albumentations import get_training_augmentation, get_validation_augmentation
from src.losses.compound_loss import CompoundLoss
from src.data.split import create_patient_splits


def test_ct_preprocessor():
    prep = CTPreprocessor(hu_min=-150, hu_max=250)
    raw = np.array([-200, -150, 0, 100, 250, 300], dtype=np.float32)
    norm = prep.normalize(raw)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_roi_extractor():
    extractor = ROIPatchExtractor(min_area=10, patch_size=(64, 64))
    img = np.random.rand(100, 100).astype(np.float32)
    mask = np.zeros((100, 100), dtype=np.int64)
    mask[30:50, 40:60] = 1  # 400 pixels foreground

    img_p, lbl_p, bbox = extractor.extract_patch(img, mask)
    assert img_p.shape == (64, 64)
    assert lbl_p.shape == (64, 64)
    assert np.any(lbl_p == 1)


def test_albumentations():
    aug = get_training_augmentation(patch_size=(64, 64))
    img = np.random.rand(64, 64).astype(np.float32)
    mask = np.random.randint(0, 3, size=(64, 64)).astype(np.int64)
    res = aug(image=img, mask=mask)
    assert res["image"].shape == (64, 64)
    assert res["mask"].shape == (64, 64)


def test_compound_loss():
    loss_fn = CompoundLoss(dice_w=0.6, ce_w=0.3, focal_w=0.1)
    logits = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 3, (2, 32, 32)).long()

    loss, comp = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert "loss_dice" in comp


def test_data_leakage(tmp_path):
    pids = [f"patient_{i:03d}" for i in range(1, 31)]
    master, kfolds = create_patient_splits(pids, output_dir=tmp_path)

    tr = set(master["train_patients"])
    va = set(master["val_patients"])
    ts = set(master["test_patients"])

    assert len(tr & va) == 0
    assert len(tr & ts) == 0
    assert len(va & ts) == 0

    for fname, fdata in kfolds.items():
        assert len(set(fdata["train"]) & set(fdata["val"])) == 0
        assert len(set(fdata["train"]) & ts) == 0
        assert len(set(fdata["val"]) & ts) == 0
