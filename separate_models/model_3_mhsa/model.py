"""
Model 3 Architecture: CNN + Multi-Head Self-Attention (CNNMHSASeg)
Residual CNN encoder + 4-Layer 8-Head Multi-Head Self-Attention (MHSA) bottleneck + U-Net residual decoder.
Distinct from Model 1 (CNNPyramidTransformerSeg) by omitting the Pyramid Pooling Module (PPM).

Extracted directly from the verified production implementation in src/models/cnn_mhsa.py.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions, BatchNorm, ReLU, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


class MultiHeadSelfAttention2D(nn.Module):
    """2D spatial multi-head self-attention transformer block with pre-norm LayerNorm and GELU FFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, c, h, w = x.shape
        # Flatten spatial dimensions: (H*W, B, C)
        seq = x.flatten(2).permute(2, 0, 1)

        # Pre-norm + Multi-head self-attention
        normed_seq = self.norm1(seq)
        attn_out, weights = self.mha(
            normed_seq, normed_seq, normed_seq, need_weights=need_weights, average_attn_weights=True
        )
        seq = seq + attn_out

        # Pre-norm + Feed-forward network
        ffn_out = self.ffn(self.norm2(seq))
        seq = seq + ffn_out

        if need_weights:
            self.last_attention_weights = weights

        # Reshape back to (B, C, H, W)
        out = seq.permute(1, 2, 0).view(b, c, h, w)
        return out, weights


class MHSABottleneck(nn.Module):
    """Bottleneck containing 1x1 projection and stacked MultiHeadSelfAttention2D blocks (without PPM)."""

    def __init__(
        self,
        in_channels: int = 512,
        embed_dim: int = 256,
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

        attn_weights = []
        for block in self.attn_blocks:
            x, w = block(x, need_weights=return_attention)
            if return_attention and w is not None:
                attn_weights.append(w)

        out = self.proj_out(x)
        return out, (attn_weights if return_attention else None)


class CNNMHSASeg(nn.Module):
    """
    Model 3 Specification Architecture:
    Residual CNN Encoder (64 -> 128 -> 256 -> 512)
    + Multi-Head Self-Attention Bottleneck (4 Layers, 8 Heads, no PPM)
    + U-Net Residual Decoder (512 -> 256 -> 128 -> 64 -> 32)
    + Final 1x1 Conv (32 -> 3 logits).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        encoder_channels: List[int] = [64, 128, 256, 512],
        embed_dim: int = 256,
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

        # MHSA Bottleneck (pure Multi-Head Self-Attention, without PPM)
        self.bottleneck = MHSABottleneck(
            in_channels=c4,
            embed_dim=embed_dim,
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
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b, attn_weights = self.bottleneck(p4, return_attention=return_attention)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.final_head(d1)

        if return_attention:
            return logits, attn_weights
        return logits
