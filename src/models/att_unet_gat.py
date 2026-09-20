"""
Comparative Architecture: AttnUNetGAT
Attention U-Net with Graph Attention Network (GAT) bottleneck emulation and attention gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections."""

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class GATBottleneck(nn.Module):
    """Multi-Head Graph Attention Network (GAT) layer operating on spatial tokens."""

    def __init__(self, in_channels: int = 512, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.gat = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj_back = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        proj = self.proj(x)
        tokens = proj.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        attn_out, _ = self.gat(tokens, tokens, tokens)
        tokens_back = attn_out.transpose(1, 2).view(b, -1, h, w)
        out = self.proj_back(tokens_back)
        return self.relu(self.norm(x + out))


class AttnUNetGAT(nn.Module):
    """Attention U-Net with GAT bottleneck."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3) -> None:
        super().__init__()
        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.e1 = conv_block(in_channels, 64)
        self.p1 = nn.MaxPool2d(2, 2)
        self.e2 = conv_block(64, 128)
        self.p2 = nn.MaxPool2d(2, 2)
        self.e3 = conv_block(128, 256)
        self.p3 = nn.MaxPool2d(2, 2)
        self.e4 = conv_block(256, 512)
        self.p4 = nn.MaxPool2d(2, 2)

        self.gat_bottleneck = GATBottleneck(512, hidden_dim=128, num_heads=4)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = AttentionGate(f_g=256, f_l=512, f_int=128)
        self.d4 = conv_block(256 + 512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(f_g=128, f_l=256, f_int=64)
        self.d3 = conv_block(128 + 256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(f_g=64, f_l=128, f_int=32)
        self.d2 = conv_block(64 + 128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(f_g=32, f_l=64, f_int=16)
        self.d1 = conv_block(32 + 64, 32)

        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))

        b = self.gat_bottleneck(self.p4(e4))

        u4 = self.up4(b)
        x4 = self.att4(g=u4, x=e4)
        d4 = self.d4(torch.cat([u4, x4], dim=1))

        u3 = self.up3(d4)
        x3 = self.att3(g=u3, x=e3)
        d3 = self.d3(torch.cat([u3, x3], dim=1))

        u2 = self.up2(d3)
        x2 = self.att2(g=u2, x=e2)
        d2 = self.d2(torch.cat([u2, x2], dim=1))

        u1 = self.up1(d2)
        x1 = self.att1(g=u1, x=e1)
        d1 = self.d1(torch.cat([u1, x1], dim=1))

        return self.final(d1)
