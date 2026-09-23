"""
Model 2 Architecture: CNN + CBAM (CBAMNet)
Implements Channel Attention and Spatial Attention within a Residual CNN U-Net
with a Multi-Scale Dilated CBAM Bottleneck.

Extracted directly from the verified production implementation in src/models/cbam_net.py.
"""

from typing import Optional, List
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module using average and max pooling."""

    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module across channel statistics."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return scale * x


class CBAMBlock(nn.Module):
    """Residual Block with sequential Channel Attention and Spatial Attention."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int = 16, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ca = ChannelAttention(out_channels, ratio)
        self.sa = SpatialAttention()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        return self.relu(out + res)


class MultiScaleCBAMBottleneck(nn.Module):
    """Multi-Scale CBAM Bottleneck refining features across receptive fields."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        half_c = max(1, channels // 2)
        self.branch1 = CBAMBlock(channels, half_c, ratio=ratio, dilation=1)
        self.branch2 = CBAMBlock(channels, half_c, ratio=ratio, dilation=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ChannelAttention(channels, ratio),
            SpatialAttention(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        fused = torch.cat([b1, b2], dim=1)
        return x + self.fuse(fused)


class CBAMNet(nn.Module):
    """
    Authoritative CNN + CBAM segmentation network with multi-scale CBAM bottleneck.
    Features:
    - Single CBAM blocks per stage
    - Multi-scale dilated CBAM bottleneck (dilation 1 and 2)
    - Channel attention and Spatial attention
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        encoder_channels: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]
        c1, c2, c3, c4 = encoder_channels

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder with integrated CBAM attention blocks
        self.enc1 = CBAMBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = CBAMBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = CBAMBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = CBAMBlock(c3, c4)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Multi-Scale CBAM Bottleneck (Dilation 1 and 2)
        self.bottleneck = MultiScaleCBAMBottleneck(c4)

        # Decoder with skip connections and CBAM refinement
        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec4 = CBAMBlock(c3 + c4, c3)

        self.up3 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec3 = CBAMBlock(c2 + c3, c2)

        self.up2 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec2 = CBAMBlock(c1 + c2, c1)

        self.up1 = nn.ConvTranspose2d(c1, 32, 2, stride=2)
        self.dec1 = CBAMBlock(32 + c1, 32)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.final_head = nn.Conv2d(32, num_classes, 1)

    def get_cam_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM interpretability."""
        return self.enc4.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.dropout(self.bottleneck(self.pool4(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final_head(d1)
