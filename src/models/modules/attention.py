"""
Multi-Head Self-Attention 2D Module
Pre-norm standard transformer block operating on flattened spatial feature tokens.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


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

        # Cache for interpretability / XAI attention inspection
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input feature map of shape (B, C, H, W) where C == embed_dim
            need_weights: Whether to compute and return attention weight tensor
        Returns:
            out: Feature map of shape (B, C, H, W)
            weights: Optional attention weights (B, H*W, H*W)
        """
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
