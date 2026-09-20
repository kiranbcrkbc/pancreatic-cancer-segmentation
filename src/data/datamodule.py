"""
DataModule Module
Provides DataLoader factories for 5-fold cross-validation and held-out test evaluation.
"""

from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader

from src.data.dataset import PancreasPatchDataset
from src.augmentation.albumentations import get_training_augmentation, get_validation_augmentation


def get_train_val_loaders_for_fold(
    fold_train_ids: List[str],
    fold_val_ids: List[str],
    data_dir: str = "data/Task07_Pancreas",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    patch_size: Tuple[int, int] = (128, 128),
) -> Tuple[DataLoader, DataLoader]:
    """Builds training and validation DataLoaders for a given fold."""
    train_dataset = PancreasPatchDataset(
        patient_ids=fold_train_ids,
        data_dir=data_dir,
        transform=get_training_augmentation(patch_size=patch_size),
        is_training=True,
        patch_size=patch_size,
    )

    val_dataset = PancreasPatchDataset(
        patient_ids=fold_val_ids,
        data_dir=data_dir,
        transform=get_validation_augmentation(patch_size=patch_size),
        is_training=False,
        patch_size=patch_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(
    test_ids: List[str],
    data_dir: str = "data/Task07_Pancreas",
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    patch_size: Tuple[int, int] = (128, 128),
) -> DataLoader:
    """Builds DataLoader for the held-out test set."""
    test_dataset = PancreasPatchDataset(
        patient_ids=test_ids,
        data_dir=data_dir,
        transform=get_validation_augmentation(patch_size=patch_size),
        is_training=False,
        patch_size=patch_size,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
