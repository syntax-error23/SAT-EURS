"""
EURS-Net V2: Complete Dual-Task Unified Network
Designed for Embedded Underwater Restoration & Segmentation (EURS @ MaCVi WACV 2027)
Target Platform: NVIDIA Jetson AGX Orin (Track 2)
Constraints: Parameters <= 1.0M, FLOPs <= 12.0 GFLOPs at 1024x1024.
"""

from typing import Tuple
import torch
import torch.nn as nn
from models.common import FastWhiteBalance, CoordinateAttention
from models.backbone import SharedBackbone
from models.decoders import ProgressiveRestorationDecoder, ProgressiveSegmentationDecoder


class EURSNetV2(nn.Module):
    """
    EURS-Net V2:
    - Input: [B, 3, 1024, 1024]
    - Stage 0: FastWhiteBalance learnable affine color correction
    - Backbone: Shared MobileNetV2-style backbone with Coordinate Attention
    - Restoration Branch: Progressive Multi-Scale Guided Restoration Decoder
    - Segmentation Branch: Progressive Top-Down FPN with Small-Object & Boundary Refinement Necks
    - Output: (restored_rgb, seg_logits)
        restored_rgb: [B, 3, 1024, 1024] clamped in [0.0, 1.0]
        seg_logits:   [B, num_classes, 1024, 1024]
    """
    def __init__(self, num_classes: int = 5, bounded_restoration: bool = True, bound_scale: float = 0.20):
        super().__init__()
        self.num_classes = num_classes

        # Color Correction Layer
        self.fwb = FastWhiteBalance()

        # Shared Feature Extractor
        self.backbone = SharedBackbone()

        # Dual Task Decoders
        self.restoration_decoder = ProgressiveRestorationDecoder(bounded_residual=bounded_restoration, bound_scale=bound_scale)
        self.segmentation_decoder = ProgressiveSegmentationDecoder(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        Args:
            x (torch.Tensor): Degraded input RGB tensor of shape [B, 3, 1024, 1024], range [0.0, 1.0].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - restored_rgb: [B, 3, 1024, 1024]
                - seg_logits:   [B, num_classes, 1024, 1024]
        """
        # 1. Color Pre-balancing
        wb_x = self.fwb(x)

        # 2. Shared Multi-Scale Feature Extraction
        stem, f256, f128, f64, f32 = self.backbone(wb_x)

        # 3. Restoration Branch (Guided residual addition to white-balanced input wb_x)
        restored_rgb = self.restoration_decoder(f128, f256, stem, raw_input=x, wb_input=wb_x)

        # 4. Segmentation Branch (Top-down FPN with Small-Object & Boundary Refinements)
        seg_logits = self.segmentation_decoder(f32, f64, f128, f256, stem)

        return restored_rgb, seg_logits
