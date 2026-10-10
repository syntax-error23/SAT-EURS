"""
EURS-Net V2 Decoders:
Includes:
- ProgressiveRestorationDecoder: Multi-scale guided restoration decoder operating from 128 -> 256 -> 512 -> 1024.
- ProgressiveSegmentationDecoder: Top-down FPN with Small-Object Neck and 512x512 Boundary Refinement Neck.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import CoordinateAttention


class ProgressiveRestorationDecoder(nn.Module):
    """
    Progressive Multi-Scale Guided Restoration Decoder:
    Starts at 128x128 (F128) -> Fuses F256 (256x256) -> Fuses Stem (512x512).
    Projects skips to matching channel dimensions, applies depthwise-separable convs,
    and predicts high-frequency & radiometric residual Delta I at 512x512.
    Residual is gently upsampled to 1024x1024 and added to the white-balanced image wb_x.
    Initializes head weights and bias to strictly zero for safe baseline initialization.
    """
    def __init__(self, bounded_residual: bool = True, bound_scale: float = 0.20):
        super().__init__()
        self.bounded_residual = bounded_residual
        self.bound_scale = bound_scale
        # Level 1: 128x128 -> Project F128 to 48 ch
        self.proj_f128 = nn.Sequential(
            nn.Conv2d(40, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Level 2: 256x256 -> Project F256 skip to 48 ch
        self.proj_f256 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.dw_conv256 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Level 3: 512x512 -> Project Stem skip to 32 ch
        self.proj_stem = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.dw_conv512 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Final residual prediction head at 512x512
        self.head = nn.Conv2d(24, 3, kernel_size=3, padding=1)

        # Zero-Init Head Fix: strictly zero out head weights and bias
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        f128: torch.Tensor,
        f256: torch.Tensor,
        stem: torch.Tensor,
        raw_input: torch.Tensor,
        wb_input: torch.Tensor = None
    ) -> torch.Tensor:
        # Step 1: 128x128 -> 256x256
        x = self.proj_f128(f128)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = x + self.proj_f256(f256)
        x = self.dw_conv256(x)

        # Step 2: 256x256 -> 512x512
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = x + self.proj_stem(stem)
        x = self.dw_conv512(x)

        # Step 3: Residual map at 512x512 -> 1024x1024
        if self.bounded_residual:
            delta_512 = self.bound_scale * torch.tanh(self.head(x))
        else:
            delta_512 = self.head(x)
        delta_1024 = F.interpolate(delta_512, size=raw_input.shape[2:], mode='bilinear', align_corners=False)

        # Add residual to white-balanced image (or raw_input if wb_input not given)
        base = wb_input if wb_input is not None else raw_input
        restored = torch.clamp(base + delta_1024, 0.0, 1.0)
        return restored


# Alias for backward compatibility
ProgressiveRestorationHead = ProgressiveRestorationDecoder


class ProgressiveSegmentationDecoder(nn.Module):
    """
    Progressive Multi-Scale Segmentation Decoder:
    - Top-down FPN fusing F32 -> F64 -> F128 -> F256 with explicit channel matching before addition.
    - Small-Object Neck with Coordinate Attention at 256x256.
    - Boundary Refinement Neck with Stem skip and Coordinate Attention at 512x512.
    - 5-class logit prediction at 512x512 followed by 2x bilinear upsampling (align_corners=False) to 1024x1024.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # FPN Top-Down Lateral Projections (matching channel dimensions before additions)
        self.lat_f32 = nn.Conv2d(96, 64, kernel_size=1)
        self.lat_f64 = nn.Conv2d(64, 64, kernel_size=1)
        self.lat_f128 = nn.Conv2d(40, 48, kernel_size=1)
        self.lat_f256 = nn.Conv2d(24, 32, kernel_size=1)

        self.smooth_64 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish()
        )
        self.smooth_128 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish()
        )

        # Small-Object Neck at 256x256
        self.small_obj_neck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            CoordinateAttention(32)
        )

        # Boundary Refinement Level at 512x512 (fusing Stem skip)
        self.lat_stem = nn.Conv2d(16, 24, kernel_size=1)
        self.boundary_refine_512 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish()
        )
        self.dw_boundary_512 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
            CoordinateAttention(24),
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        # Segmentation Head predicting logits at 512x512
        self.head_512 = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(
        self,
        f32: torch.Tensor,
        f64: torch.Tensor,
        f128: torch.Tensor,
        f256: torch.Tensor,
        stem: torch.Tensor
    ) -> torch.Tensor:
        # 1. 32x32 -> 64x64
        p32 = self.lat_f32(f32)
        p64 = F.interpolate(p32, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f64(f64)
        p64 = self.smooth_64(p64)

        # 2. 64x64 -> 128x128
        p128 = F.interpolate(p64, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f128(f128)
        p128 = self.smooth_128(p128)

        # 3. 128x128 -> 256x256
        p256 = F.interpolate(p128, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f256(f256)
        p256 = self.small_obj_neck(p256)

        # 4. 256x256 -> 512x512 (Boundary Refinement with Stem skip)
        p512 = F.interpolate(p256, scale_factor=2.0, mode='bilinear', align_corners=False)
        p512 = self.boundary_refine_512(p512) + self.lat_stem(stem)
        p512 = self.dw_boundary_512(p512)

        # 5. Predict at 512x512, then gently upsample 2x to full 1024x1024
        logits_512 = self.head_512(p512)
        logits_1024 = F.interpolate(logits_512, scale_factor=2.0, mode='bilinear', align_corners=False)
        return logits_1024
