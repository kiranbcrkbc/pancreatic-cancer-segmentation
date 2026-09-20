"""
Authoritative Model Architecture: CNNPyramidTransformerSeg
Combines a 4-stage Residual CNN encoder, Pyramid Pooling Module (PPM),
Multi-Head Self-Attention (MHSA) transformer bottleneck, and a U-Net residual decoder.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from src.models.modules.res_block import ResBlock
from src.models.modules.ppm import PyramidPoolingModule
from src.models.modules.attention import MultiHeadSelfAttention2D


class TransformerBottleneck(nn.Module):
    """Bottleneck containing 1x1 projection, PPM, and stacked MultiHeadSelfAttention2D blocks."""

    def __init__(
        self,
        in_channels: int = 512,
        embed_dim: int = 256,
        ppm_pool_sizes: List[int] = [1, 2, 4, 8],
        num_heads: int = 8,
        transformer_depth: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.ppm = PyramidPoolingModule(embed_dim, pool_sizes=ppm_pool_sizes)

        self.attn_blocks = nn.ModuleList([
            MultiHeadSelfAttention2D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(transformer_depth)
        ])

        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.proj_in(x)
        x = self.ppm(x)

        attn_weights = []
        for block in self.attn_blocks:
            x, w = block(x, need_weights=return_attention)
            if return_attention and w is not None:
                attn_weights.append(w)

        out = self.proj_out(x)
        return out, (attn_weights if return_attention else None)


class CNNPyramidTransformerSeg(nn.Module):
    """
    Authoritative Primary Model:
    Residual CNN Encoder (64 -> 128 -> 256 -> 512)
    + Transformer Bottleneck (PPM + N x MHSA)
    + U-Net Residual Decoder (512 -> 256 -> 128 -> 64 -> 32)
    + Final 1x1 Conv (32 -> 3 logits).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        encoder_channels: List[int] = [64, 128, 256, 512],
        embed_dim: int = 256,
        ppm_pool_sizes: List[int] = [1, 2, 4, 8],
        num_heads: int = 8,
        transformer_depth: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        c1, c2, c3, c4 = encoder_channels

        # Encoder Stage 1: 1 -> c1
        self.enc1 = nn.Sequential(
            ResBlock(in_channels, c1, dropout=dropout),
            ResBlock(c1, c1, dropout=dropout),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Encoder Stage 2: c1 -> c2
        self.enc2 = nn.Sequential(
            ResBlock(c1, c2, dropout=dropout),
            ResBlock(c2, c2, dropout=dropout),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Encoder Stage 3: c2 -> c3
        self.enc3 = nn.Sequential(
            ResBlock(c2, c3, dropout=dropout),
            ResBlock(c3, c3, dropout=dropout),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Encoder Stage 4: c3 -> c4
        self.enc4 = nn.Sequential(
            ResBlock(c3, c4, dropout=dropout),
            ResBlock(c4, c4, dropout=dropout),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck on pool4 output
        self.bottleneck = TransformerBottleneck(
            in_channels=c4,
            embed_dim=embed_dim,
            ppm_pool_sizes=ppm_pool_sizes,
            num_heads=num_heads,
            transformer_depth=transformer_depth,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Decoder Stage 4: c4 (512) -> c3 (256), concat with enc4 (512) -> 768 -> 256
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ResBlock(c3 + c4, c3, dropout=dropout),
            ResBlock(c3, c3, dropout=dropout),
        )

        # Decoder Stage 3: c3 (256) -> c2 (128), concat with enc3 (256) -> 384 -> 128
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ResBlock(c2 + c3, c2, dropout=dropout),
            ResBlock(c2, c2, dropout=dropout),
        )

        # Decoder Stage 2: c2 (128) -> c1 (64), concat with enc2 (128) -> 192 -> 64
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ResBlock(c1 + c2, c1, dropout=dropout),
            ResBlock(c1, c1, dropout=dropout),
        )

        # Decoder Stage 1: c1 (64) -> 32, concat with enc1 (64) -> 96 -> 32
        self.up1 = nn.ConvTranspose2d(c1, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ResBlock(32 + c1, 32, dropout=dropout),
            ResBlock(32, 32, dropout=dropout),
        )

        # Final classification head: 32 -> num_classes logits
        self.final_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def get_cam_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM interpretability."""
        return self.enc4[-1].conv1[0]

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # Encoder forward
        e1 = self.enc1(x)       # (B, 64, H, W)
        p1 = self.pool1(e1)     # (B, 64, H/2, W/2)

        e2 = self.enc2(p1)      # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)     # (B, 128, H/4, W/4)

        e3 = self.enc3(p2)      # (B, 256, H/4, W/4)
        p3 = self.pool3(e3)     # (B, 256, H/8, W/8)

        e4 = self.enc4(p3)      # (B, 512, H/8, W/8)
        p4 = self.pool4(e4)     # (B, 512, H/16, W/16)

        # Bottleneck
        b, attn_weights = self.bottleneck(p4, return_attention=return_attention)

        # Decoder forward with exact matching skip concatenation
        d4 = self.up4(b)        # (B, 256, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1) # (B, 256+512, H/8, W/8)
        d4 = self.dec4(d4)      # (B, 256, H/8, W/8)

        d3 = self.up3(d4)       # (B, 128, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1) # (B, 128+256, H/4, W/4)
        d3 = self.dec3(d3)      # (B, 128, H/4, W/4)

        d2 = self.up2(d3)       # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1) # (B, 64+128, H/2, W/2)
        d2 = self.dec2(d2)      # (B, 64, H/2, W/2)

        d1 = self.up1(d2)       # (B, 32, H, W)
        d1 = torch.cat([d1, e1], dim=1) # (B, 32+64, H, W)
        d1 = self.dec1(d1)      # (B, 32, H, W)

        logits = self.final_head(d1)

        if return_attention:
            return logits, attn_weights
        return logits
