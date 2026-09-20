"""
Pyramid Pooling Module (PPM)
Aggregates multi-scale spatial context across multiple pooling bins.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    """PSPNet-style Pyramid Pooling Module for multi-scale contextual aggregation."""

    def __init__(
        self,
        in_channels: int,
        pool_sizes: List[int] = [1, 2, 4, 8],
    ) -> None:
        super().__init__()
        self.pool_sizes = pool_sizes
        out_channels_per_bin = max(1, in_channels // len(pool_sizes))

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(bin_size, bin_size)),
                nn.Conv2d(in_channels, out_channels_per_bin, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels_per_bin),
                nn.ReLU(inplace=True),
            )
            for bin_size in pool_sizes
        ])

        total_concat_channels = in_channels + out_channels_per_bin * len(pool_sizes)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_concat_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        priors = [x]
        for stage in self.stages:
            pooled = stage(x)
            upsampled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            priors.append(upsampled)
        concat = torch.cat(priors, dim=1)
        return self.bottleneck(concat)
