"""
EURS-Net V2 Shared Multi-Scale Backbone
Includes:
- MBConvBlock: MobileNetV2 inverted residual block with optional Coordinate Attention.
- SharedBackbone: Shared multi-scale feature extractor outputting Stem and F256, F128, F64, F32 feature maps.
"""

from typing import Tuple
import torch
import torch.nn as nn
from models.common import CoordinateAttention


class MBConvBlock(nn.Module):
    """
    MobileNetV2-style Inverted Residual Block:
        1x1 expansion -> 3x3 depthwise conv -> (optional Coordinate Attention) -> 1x1 linear projection.
    Uses BatchNorm2d and HardSwish activations throughout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 6,
        use_ca: bool = False
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        layers = []
        # Expansion phase
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish()
            ])

        # Depthwise spatial filtering
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish()
        ])

        # Direction-aware Coordinate Attention (if enabled)
        if use_ca:
            layers.append(CoordinateAttention(hidden_dim))

        # Pointwise linear projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SharedBackbone(nn.Module):
    """
    Multi-Scale Shared Feature Extractor:
    - Stem: Conv 3x3, stride=2 -> [B, 16, 512, 512]
    - Stage 1 (F256): MBConv exp=1, stride=2 -> [B, 24, 256, 256]
    - Stage 2 (F128): MBConv exp=6, stride=2 + CoordAttn -> [B, 40, 128, 128]
    - Stage 3 (F64):  MBConv exp=6, stride=2 + CoordAttn -> [B, 64, 64, 64]
    - Stage 4 (F32):  MBConv exp=6, stride=2 -> [B, 96, 32, 32] (Semantic Bottleneck)

    Returns:
        tuple: (stem, f256, f128, f64, f32)
    """
    def __init__(self):
        super().__init__()
        # Stem: immediate 2x downsampling from 1024 to 512
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        # Stage 1: 512 -> 256
        self.stage1 = MBConvBlock(16, 24, stride=2, expansion=1, use_ca=False)

        # Stage 2: 256 -> 128 + Coordinate Attention (2 blocks)
        self.stage2a = MBConvBlock(24, 40, stride=2, expansion=6, use_ca=True)
        self.stage2b = MBConvBlock(40, 40, stride=1, expansion=6, use_ca=False)

        # Stage 3: 128 -> 64 + Coordinate Attention (2 blocks)
        self.stage3a = MBConvBlock(40, 64, stride=2, expansion=6, use_ca=True)
        self.stage3b = MBConvBlock(64, 64, stride=1, expansion=6, use_ca=False)

        # Stage 4: 64 -> 32 (Semantic Bottleneck, 2 blocks)
        self.stage4a = MBConvBlock(64, 96, stride=2, expansion=6, use_ca=False)
        self.stage4b = MBConvBlock(96, 96, stride=1, expansion=6, use_ca=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stem(x)        # [B, 16, 512, 512]
        f256 = self.stage1(stem)   # [B, 24, 256, 256]
        f128 = self.stage2b(self.stage2a(f256)) # [B, 40, 128, 128]
        f64  = self.stage3b(self.stage3a(f128)) # [B, 64, 64, 64]
        f32  = self.stage4b(self.stage4a(f64))  # [B, 96, 32, 32]

        return stem, f256, f128, f64, f32
