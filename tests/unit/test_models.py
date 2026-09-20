"""Unit tests for neural network architectures."""

import pytest
import torch
from src.models.cnn_transformer import CNNPyramidTransformerSeg
from src.models.cbam_net import CBAMNet
from src.models.att_unet_gat import AttnUNetGAT


def test_cnn_pyramid_transformer_forward():
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
    out = model(x)
    assert out.shape == (2, 3, 64, 64)


def test_cbam_net_forward():
    model = CBAMNet(in_channels=1, num_classes=3)
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert out.shape == (1, 3, 32, 32)


def test_att_unet_gat_forward():
    model = AttnUNetGAT(in_channels=1, num_classes=3)
    x = torch.randn(1, 1, 32, 32)
    out = model(x)
    assert out.shape == (1, 3, 32, 32)
