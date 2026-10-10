"""
EURS-Net V3 LiteJoint Architecture:
- Shared Multi-Scale MobileNetV3-style Backbone with Coordinate Attention & ASPP-lite
- Restoration Branch: U-Shaped Progressive Guided Restoration Decoder with high-res detail path
- Segmentation Branch: Top-Down FPN with Small-Object Neck, Gated Cross-Task Fusion, and Boundary Refinement
- Hard Constraints: Parameters <= 1.0M, FLOPs <= 12.0 GFLOPs at static [1, 3, 1024, 1024]
- Target: MaCVi @ WACV 2027 EURS Track 2 (Embedded Jetson AGX Orin)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import CoordinateAttention, FastWhiteBalance


class InvertedResidual(nn.Module):
    """
    MobileNetV3 / V2 style Inverted Residual block with HardSwish and optional Coordinate Attention.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        use_ca: bool = False
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expansion)

        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish(inplace=True)
            ])
        else:
            hidden_dim = in_channels

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True)
        ])

        if use_ca:
            layers.append(CoordinateAttention(hidden_dim))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class DilatedContextBlock(nn.Module):
    """
    Compact Dilated-Context Block (ASPP-Lite) at 1/32 resolution (32x32).
    Captures multi-scale global underwater contextual features without spatial compute bloat.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_ch = out_channels // 4

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True)
        )
        # 3x3 depthwise dilated rate 2
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=2, dilation=2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True)
        )
        # 3x3 depthwise dilated rate 4
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=4, dilation=4, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True)
        )
        # Global pooling branch
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_ch, 1, bias=True),
            nn.Hardswish(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = F.interpolate(self.b4(x), size=(h, w), mode='nearest')
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        return self.project(concat)


class GatedCrossTaskFusion(nn.Module):
    """
    Cross-task interaction module at 1/4 resolution (256x256).
    Enables segmentation to benefit from clean, edge-enhanced restoration features
    via adaptive gating without full-resolution attention overhead.
    """
    def __init__(self, rest_channels: int, seg_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(rest_channels, seg_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.proj_conv = nn.Sequential(
            nn.Conv2d(rest_channels, seg_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(seg_channels)
        )

    def forward(self, rest_feat: torch.Tensor, seg_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate_conv(rest_feat)
        proj = self.proj_conv(rest_feat)
        return seg_feat + gate * proj


class SharedEncoderV3(nn.Module):
    """
    Multi-Scale Shared Pyramid Encoder:
    - Stem: 3 -> 20 ch, stride 2 (512x512)
    - Stage 1: 20 -> 28 ch, stride 2 (256x256) + CoordAttn
    - Stage 2: 28 -> 44 ch, stride 2 (128x128) + CoordAttn (2 blocks)
    - Stage 3: 44 -> 72 ch, stride 2 (64x64) + CoordAttn (2 blocks)
    - Stage 4: 72 -> 112 ch, stride 2 (32x32) + CoordAttn
    - Stage 5: 112 -> 128 ch + DilatedContextBlock (ASPP-lite at 32x32)
    """
    def __init__(self):
        super().__init__()
        # Stem: 1024 -> 512
        self.stem = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.Hardswish(inplace=True)
        )

        # Stage 1: 512 -> 256 (1/4)
        self.stage1 = InvertedResidual(20, 28, stride=2, expansion=2, use_ca=True)

        # Stage 2: 256 -> 128 (1/8)
        self.stage2a = InvertedResidual(28, 44, stride=2, expansion=3, use_ca=True)
        self.stage2b = InvertedResidual(44, 44, stride=1, expansion=3, use_ca=False)

        # Stage 3: 128 -> 64 (1/16)
        self.stage3a = InvertedResidual(44, 72, stride=2, expansion=4, use_ca=True)
        self.stage3b = InvertedResidual(72, 72, stride=1, expansion=4, use_ca=False)

        # Stage 4: 64 -> 32 (1/32)
        self.stage4a = InvertedResidual(72, 112, stride=2, expansion=4, use_ca=True)
        self.stage4b = InvertedResidual(112, 112, stride=1, expansion=4, use_ca=False)

        # Stage 5: 32x32 Context Enrichment (ASPP-lite)
        self.stage5_context = DilatedContextBlock(112, 128)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stem(x)                              # [B, 20, 512, 512] (1/2)
        c1 = self.stage1(stem)                           # [B, 28, 256, 256] (1/4)
        c2 = self.stage2b(self.stage2a(c1))              # [B, 44, 128, 128] (1/8)
        c3 = self.stage3b(self.stage3a(c2))              # [B, 72, 64, 64]   (1/16)
        c4 = self.stage4b(self.stage4a(c3))              # [B, 112, 32, 32]  (1/32)
        c5 = self.stage5_context(c4)                     # [B, 128, 32, 32]  (1/32)
        return stem, c1, c2, c3, c4, c5


class RestorationDecoderV3(nn.Module):
    """
    Lightweight U-Shaped Progressive Reconstruction Decoder:
    - Skips at 1/16 (64x64), 1/8 (128x128), 1/4 (256x256), 1/2 (512x512), and 1/1 (1024x1024)
    - Emits intermediate 1/4 feature (32 ch) for Cross-Task Fusion into segmentation
    - Shallow detail path directly from raw image at 1024x1024 (8 channels)
    - Supports bounded residual (default) and sigmoid residual modes
    - Zero-initialized output head for perfect identity at initialization
    """
    def __init__(self, output_mode: str = 'bounded', bound_scale: float = 0.20):
        super().__init__()
        self.output_mode = output_mode
        self.bound_scale = bound_scale

        # 1/16 -> 1/8: upsample c3 (72 ch) and fuse c2 (44 ch) -> 36 ch
        self.up_16_8 = nn.Sequential(
            nn.Conv2d(72, 36, 1, bias=False),
            nn.BatchNorm2d(36),
            nn.Hardswish(inplace=True)
        )
        self.fuse_8 = nn.Sequential(
            nn.Conv2d(36 + 44, 36, 1, bias=False),
            nn.BatchNorm2d(36),
            nn.Hardswish(inplace=True),
            nn.Conv2d(36, 36, 3, padding=1, groups=36, bias=False),
            nn.BatchNorm2d(36),
            nn.Hardswish(inplace=True),
            nn.Conv2d(36, 36, 1, bias=False),
            nn.BatchNorm2d(36),
            nn.Hardswish(inplace=True)
        )

        # 1/8 -> 1/4: upsample to 256x256 and fuse c1 (28 ch) -> 32 ch
        self.up_8_4 = nn.Sequential(
            nn.Conv2d(36, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True)
        )
        self.fuse_4 = nn.Sequential(
            nn.Conv2d(32 + 28, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True)
        )

        # 1/4 -> 1/2: upsample to 512x512 and fuse stem (20 ch) -> 20 ch
        self.up_4_2 = nn.Sequential(
            nn.Conv2d(32, 20, 1, bias=False),
            nn.BatchNorm2d(20),
            nn.Hardswish(inplace=True)
        )
        self.fuse_2 = nn.Sequential(
            nn.Conv2d(20 + 20, 20, 1, bias=False),
            nn.BatchNorm2d(20),
            nn.Hardswish(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1, groups=20, bias=False),
            nn.BatchNorm2d(20),
            nn.Hardswish(inplace=True),
            nn.Conv2d(20, 20, 1, bias=False),
            nn.BatchNorm2d(20),
            nn.Hardswish(inplace=True)
        )

        # 1/2 -> 1/1: upsample to 1024x1024 (8 ch) + shallow detail from raw input (8 ch)
        self.up_2_1 = nn.Sequential(
            nn.Conv2d(20, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(inplace=True)
        )
        self.shallow_detail = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(inplace=True)
        )
        self.fuse_1 = nn.Sequential(
            nn.Conv2d(8 + 8, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(inplace=True),
            nn.Conv2d(8, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(inplace=True)
        )

        # Output Residual Head at 1024x1024
        self.head = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        stem: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        raw_x: torch.Tensor,
        wb_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Up 16 -> 8
        u8 = F.interpolate(self.up_16_8(c3), size=c2.shape[2:], mode='bilinear', align_corners=False)
        f8 = self.fuse_8(torch.cat([u8, c2], dim=1))

        # Up 8 -> 4
        u4 = F.interpolate(self.up_8_4(f8), size=c1.shape[2:], mode='bilinear', align_corners=False)
        f4 = self.fuse_4(torch.cat([u4, c1], dim=1))  # [B, 32, 256, 256] -> feature for cross-task fusion!

        # Up 4 -> 2
        u2 = F.interpolate(self.up_4_2(f4), size=stem.shape[2:], mode='bilinear', align_corners=False)
        f2 = self.fuse_2(torch.cat([u2, stem], dim=1))

        # Up 2 -> 1
        u1 = F.interpolate(self.up_2_1(f2), size=raw_x.shape[2:], mode='bilinear', align_corners=False)
        s1 = self.shallow_detail(raw_x)
        f1 = self.fuse_1(torch.cat([u1, s1], dim=1))

        # Residual Head
        residual = self.head(f1)

        if self.output_mode == 'bounded':
            restored = torch.clamp(wb_x + self.bound_scale * torch.tanh(residual), 0.0, 1.0)
        elif self.output_mode == 'sigmoid':
            wb_logit = torch.logit(torch.clamp(wb_x, 1e-4, 1.0 - 1e-4))
            restored = torch.sigmoid(wb_logit + residual)
        else:
            restored = torch.clamp(wb_x + residual, 0.0, 1.0)

        return restored, f4


class SegmentationDecoderV3(nn.Module):
    """
    Feature Pyramid Network (FPN) Segmentation Decoder with Gated Restoration Guidance:
    - Lateral 1x1 projections to C_fpn = 36
    - Top-down fusion from 1/32 down to 1/4
    - Gated Cross-Task Fusion at 1/4 resolution (fusing rest_feat_4)
    - Small-Object Neck with Coordinate Attention at 1/4 resolution (256x256)
    - Boundary Refinement Neck fusing stem skip at 1/2 resolution (512x512)
    - 5-class logit prediction at 512x512, followed by smooth 2x bilinear upsampling to 1024x1024
    - Emits 5 raw logits at [B, 5, 1024, 1024]
    """
    def __init__(self, num_classes: int = 5, fpn_dim: int = 36, aux_head: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.aux_head = aux_head
        self.fpn_dim = fpn_dim

        # Lateral 1x1 projections
        self.lat5 = nn.Conv2d(128, fpn_dim, 1, bias=False)
        self.lat4 = nn.Conv2d(112, fpn_dim, 1, bias=False)
        self.lat3 = nn.Conv2d(72, fpn_dim, 1, bias=False)
        self.lat2 = nn.Conv2d(44, fpn_dim, 1, bias=False)
        self.lat1 = nn.Conv2d(28, fpn_dim, 1, bias=False)

        # Smooth convolutions (depthwise-separable)
        self.smooth4 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.Hardswish(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim)
        )
        self.smooth3 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.Hardswish(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.Hardswish(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim)
        )

        # Gated Cross-Task Fusion at 1/4 resolution (restoration features -> segmentation)
        self.cross_fusion = GatedCrossTaskFusion(rest_channels=32, seg_channels=fpn_dim)

        # Small-Object Neck at 1/4 resolution (256x256)
        self.small_obj_neck = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.Hardswish(inplace=True),
            CoordinateAttention(fpn_dim)
        )

        # Boundary Refinement Level at 1/2 resolution (512x512) fusing Stem skip
        self.lat_stem = nn.Conv2d(20, fpn_dim, kernel_size=1, bias=False)
        self.boundary_neck = nn.Sequential(
            nn.Conv2d(fpn_dim * 2, fpn_dim, 1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.Hardswish(inplace=True),
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.Hardswish(inplace=True),
            CoordinateAttention(fpn_dim),
            nn.Conv2d(fpn_dim, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(inplace=True)
        )

        # Classifier Head at 512x512
        self.classifier = nn.Conv2d(24, num_classes, kernel_size=1)

        # Optional training-only auxiliary head at 1/8 resolution
        if aux_head:
            self.aux_classifier = nn.Conv2d(fpn_dim, num_classes, 1)
        else:
            self.aux_classifier = None

    def forward(
        self,
        stem: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
        rest_feat_4: torch.Tensor,
        target_size: Tuple[int, int] = (1024, 1024)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Top-down FPN pathway
        p5 = self.lat5(c5)  # [B, fpn_dim, 32, 32]
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False))

        # 1/4 resolution: fuse with lateral c1, then cross-task fusion with restoration feature
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        p1_fused = self.cross_fusion(rest_feat_4, p1)
        p1_refined = self.small_obj_neck(p1_fused)  # [B, fpn_dim, 256, 256]

        # 1/2 resolution: Boundary Refinement fusing Stem skip
        u_stem = F.interpolate(p1_refined, size=stem.shape[2:], mode='bilinear', align_corners=False)
        stem_proj = self.lat_stem(stem)
        boundary_feat = self.boundary_neck(torch.cat([u_stem, stem_proj], dim=1))  # [B, 24, 512, 512]

        # Logits at 512x512
        logits_512 = self.classifier(boundary_feat)

        # 2x Bilinear upsample to target resolution (1024, 1024)
        seg_logits = F.interpolate(logits_512, size=target_size, mode='bilinear', align_corners=False)

        aux_logits = None
        if self.aux_classifier is not None and self.training:
            aux_logits = F.interpolate(self.aux_classifier(p3), size=target_size, mode='bilinear', align_corners=False)

        return seg_logits, aux_logits


class EURSNetV3LiteJoint(nn.Module):
    """
    EURS-Net V3 LiteJoint: Unified Multi-Task Architecture
    - Color correction: FastWhiteBalance
    - Shared Multi-Scale Encoder: SharedEncoderV3 (stem, c1..c5)
    - Task 1: RestorationDecoderV3 (output: 'restored_image' [1, 3, 1024, 1024])
    - Task 2: SegmentationDecoderV3 (output: 'seg_logits' [1, 5, 1024, 1024])
    - Cross-Task Gated Interaction at 1/4 resolution (256x256)
    - Small-Object & Boundary Refinement Necks for rare marine organisms
    """
    def __init__(
        self,
        num_classes: int = 5,
        restoration_mode: str = 'bounded',
        bound_scale: float = 0.20,
        aux_head: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.aux_head = aux_head

        self.fwb = FastWhiteBalance()
        self.encoder = SharedEncoderV3()
        self.restoration_decoder = RestorationDecoderV3(output_mode=restoration_mode, bound_scale=bound_scale)
        self.segmentation_decoder = SegmentationDecoderV3(num_classes=num_classes, fpn_dim=36, aux_head=aux_head)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wb_x = self.fwb(x)
        stem, c1, c2, c3, c4, c5 = self.encoder(wb_x)

        # 1. Restoration forward (also yields intermediate feature f4 at 1/4 resolution)
        restored, rest_feat_4 = self.restoration_decoder(stem, c1, c2, c3, raw_x=x, wb_x=wb_x)

        # 2. Segmentation forward (receives rest_feat_4 for gated cross-task guidance)
        seg_logits, aux_logits = self.segmentation_decoder(stem, c1, c2, c3, c4, c5, rest_feat_4, target_size=x.shape[2:])

        if self.training and self.aux_head and aux_logits is not None:
            return restored, seg_logits, aux_logits

        return restored, seg_logits
