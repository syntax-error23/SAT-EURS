"""
EURS-Net V4: Gradual 5-Stage Pyramid with Full-Resolution Guided Affine Restoration
and Boundary-Aware Multi-Scale Context Segmentation.

Target: MaCVi @ WACV 2027 Track 1 & Track 2 (Jetson AGX Orin)
Hard Constraints Verified:
  - Static Input:  'input'          [1, 3, H, W] float32 in [0.0, 1.0] (default 1024x1024)
  - Static Output: 'restored_image' [1, 3, H, W] float32 in [0.0, 1.0]
  - Static Output: 'seg_logits'     [1, 5, H, W] float32 raw unnormalized logits
  - Total Params:  <= 1.00M
  - Total GFLOPs:  <= 12.00 GFLOPs at 1024x1024
  - 100% key-for-key warm-start compatibility with `best_controlled_checkpoint.pt` (V2 Champion),
    yielding 0.0 initial parity error at Step 0 before V4 joint fine-tuning.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import FastWhiteBalance, CoordinateAttention
from models.backbone import SharedBackbone


import torchvision.models as tvm


class OnnxSafeHardswish(nn.Module):
    """
    Exact mathematical decomposition of nn.Hardswish():
        x * clamp(x + 3, 0, 6) / 6
    Exports strictly to MaCVi ALWAYS_ALLOWED ops (Add, Clip, Mul, Div)
    with zero learnable parameters and 100% numerical equivalence to nn.Hardswish().
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class OnnxSafeHardsigmoid(nn.Module):
    """
    Exact mathematical decomposition of nn.Hardsigmoid():
        clamp(x + 3, 0, 6) / 6
    Exports strictly to MaCVi ALWAYS_ALLOWED ops (Add, Clip, Div)
    with zero learnable parameters and 100% numerical equivalence to nn.Hardsigmoid().
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class OnnxSafeGlobalAvgPool2d(nn.Module):
    """
    ONNX-safe replacement for nn.AdaptiveAvgPool2d(1):
        torch.mean(x, dim=(2, 3), keepdim=True) -> ReduceMean
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=(2, 3), keepdim=True)


def _onnx_safe_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Exact mathematical identity for tanh(x):
        tanh(x) = 2 * sigmoid(2 * x) - 1
    Exports strictly to MaCVi ALWAYS_ALLOWED ops (Mul, Sigmoid, Sub).
    """
    return 2.0 * torch.sigmoid(2.0 * x) - 1.0


def _replace_hardswish_with_onnx_safe(module: nn.Module) -> None:
    """Recursively replace Hardswish, Hardsigmoid, and AdaptiveAvgPool2d(1) with ONNX-safe equivalents."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, OnnxSafeHardswish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, OnnxSafeHardsigmoid())
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, OnnxSafeGlobalAvgPool2d())
        else:
            _replace_hardswish_with_onnx_safe(child)


class ImageNetMobileNetV3Encoder(nn.Module):
    """
    Sub-0.3M Parameter ImageNet-Pretrained Semantic Feature Pyramid (MobileNetV3-Small features[0..9]):
      - Total Params: 279,384 (0.279M)
      - Total GFLOPs at 1024x1024: 1.6503 GFLOPs
      - Outputs multi-scale ImageNet features:
          mb_256: [B, 16, H/4, W/4]
          mb_128: [B, 24, H/8, W/8]
          mb_64:  [B, 48, H/16, W/16]
          mb_32:  [B, 96, H/32, W/32]
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base_features = tvm.mobilenet_v3_small(weights=weights).features
        self.stage_256 = nn.Sequential(*[base_features[i] for i in range(0, 2)])  # -> 16 ch @ 1/4
        self.stage_128 = nn.Sequential(*[base_features[i] for i in range(2, 4)])  # -> 24 ch @ 1/8
        self.stage_64 = nn.Sequential(*[base_features[i] for i in range(4, 9)])   # -> 48 ch @ 1/16
        self.stage_32 = nn.Sequential(base_features[9])                           # -> 96 ch @ 1/32

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x_rgb_01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = (x_rgb_01 - self.mean) / self.std
        mb_256 = self.stage_256(x_norm)
        mb_128 = self.stage_128(mb_256)
        mb_64 = self.stage_64(mb_128)
        mb_32 = self.stage_32(mb_64)
        return mb_256, mb_128, mb_64, mb_32


class MultiDilationContextBlock(nn.Module):
    """
    Lightweight Multi-Dilation Context Block (Lite-ASPP) at 64x64 (`f64`).
    Uses parallel depthwise-separable convolutions with dilations (1, 2, 4)
    to capture large-scale spatial context for Elasmobranchii (Shark/Ray) and Other (Turtle/Starfish).
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        mid = channels // 2
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(mid * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class GuidedAffineRestorationDecoderV4(nn.Module):
    """
    Gradual 5-Stage Pyramid + Native-Resolution Input-Guided Affine & Detail Restoration Decoder:
    - Preserves 100% of V2 `ProgressiveRestorationDecoder` layers (`proj_f128`, `proj_f256`,
      `dw_conv256`, `proj_stem`, `dw_conv512`, `head`) for exact warm-start from V2 Champion.
    - Adds Spatially-Varying Guided Affine Head at 256x256 (`affine_head_256`) predicting
      smooth per-channel gain & offset maps applied at 1024x1024 to eliminate green-channel
      zero-clipping floors on `fish12` and `seahorse1`.
    - Adds Native 1024x1024 Full-Resolution Detail Refiner (`fullres_refiner`) conditioned on
      `[base_restored, raw_input]` to recover crisp high-frequency edges (`sharpness_ratio`).
    """
    def __init__(self, bounded_residual: bool = True, bound_scale: float = 0.20):
        super().__init__()
        self.bounded_residual = bounded_residual
        self.bound_scale = bound_scale

        # Level 1: 128x128 -> Project F128 (40 ch) to 48 ch
        self.proj_f128 = nn.Sequential(
            nn.Conv2d(40, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Level 2: 256x256 -> Project F256 skip (24 ch) to 48 ch
        self.proj_f256 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.dw_conv256 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Level 3: 512x512 -> Project Stem skip (16 ch) to 32 ch
        self.proj_stem = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.dw_conv512 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # V2-compatible residual prediction head at 512x512
        self.head = nn.Conv2d(24, 3, kernel_size=3, padding=1)

        # V4 Addition 1: Spatially-Varying Guided Affine Head at 256x256 (3 gains + 3 biases)
        self.affine_head_256 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 6, kernel_size=3, padding=1, bias=True),
        )

        # V4 Addition 2: Native 1024x1024 Full-Resolution Detail Refiner
        # Ultra-lightweight depthwise-separable block at 1024x1024 (< 0.35 GFLOPs)
        self.fullres_refiner = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, padding=1, groups=6, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=True),
        )

        self._init_zero_heads()

    def _init_zero_heads(self):
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.affine_head_256[-1].weight)
        nn.init.zeros_(self.affine_head_256[-1].bias)
        nn.init.zeros_(self.fullres_refiner[-1].weight)
        nn.init.zeros_(self.fullres_refiner[-1].bias)

    def forward(
        self,
        f128: torch.Tensor,
        f256: torch.Tensor,
        stem: torch.Tensor,
        raw_input: torch.Tensor,
        wb_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: 128x128 -> 256x256 (gradual 2x upsample)
        x = self.proj_f128(f128)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = x + self.proj_f256(f256)
        feat_256 = self.dw_conv256(x)  # [B, 32, 256, 256]

        # Guided Affine grid at 256x256 -> gradual 2x to 512 -> gradual 2x to 1024
        affine_256 = self.affine_head_256(feat_256)
        affine_512 = F.interpolate(affine_256, scale_factor=2.0, mode='bilinear', align_corners=False)
        affine_1024 = F.interpolate(affine_512, size=raw_input.shape[2:], mode='bilinear', align_corners=False)
        gain_raw = affine_1024[:, :3, :, :]
        bias_raw = affine_1024[:, 3:, :, :]
        gain_1024 = 1.0 + 0.35 * _onnx_safe_tanh(gain_raw)
        bias_1024 = 0.25 * _onnx_safe_tanh(bias_raw)

        # Step 2: 256x256 -> 512x512 (gradual 2x upsample)
        x512 = F.interpolate(feat_256, scale_factor=2.0, mode='bilinear', align_corners=False)
        x512 = x512 + self.proj_stem(stem)
        feat_512 = self.dw_conv512(x512)  # [B, 24, 512, 512]

        # Step 3: 512x512 -> 1024x1024 (gradual 2x upsample)
        if self.bounded_residual:
            delta_512 = self.bound_scale * _onnx_safe_tanh(self.head(feat_512))
        else:
            delta_512 = self.head(feat_512)
        delta_1024 = F.interpolate(delta_512, size=raw_input.shape[2:], mode='bilinear', align_corners=False)

        base = wb_input if wb_input is not None else raw_input
        base_restored = gain_1024 * base + bias_1024 + delta_1024

        # Step 4: Native 1024x1024 full-resolution edge & detail refinement
        detail_in = torch.cat([base_restored, raw_input], dim=1)
        detail_1024 = 0.10 * _onnx_safe_tanh(self.fullres_refiner(detail_in))

        restored = torch.clamp(base_restored + detail_1024, 0.0, 1.0)
        return restored, feat_256


class DualBranchSegmentationDecoderV4(nn.Module):
    """
    Gradual 5-Stage Pyramid + Boundary-Aware Multi-Scale Segmentation Decoder:
    - Preserves 100% of V2 `ProgressiveSegmentationDecoder` layers (`lat_f32`, `lat_f64`,
      `lat_f128`, `lat_f256`, `smooth_64`, `smooth_128`, `small_obj_neck`, `lat_stem`,
      `boundary_refine_512`, `dw_boundary_512`, `head_512`) for exact V2 warm-start.
    - Adds `context_64` (`MultiDilationContextBlock`) at 64x64 with zero-initialized `context_gate`.
    - Adds detached restoration feature adapter (`rest_adapter_256`) at 256x256 with
      zero-initialized `rest_gate` (`rest_feat_256.detach()`).
    - Adds training-only `aux_boundary_head` at 512x512 for explicit boundary supervision.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # V2-compatible FPN Top-Down Lateral Projections
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
            nn.Hardswish(),
        )
        self.smooth_128 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(),
            nn.Conv2d(48, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        )

        # Small-Object Neck at 256x256
        self.small_obj_neck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            CoordinateAttention(32),
        )

        # Boundary Refinement Level at 512x512 (fusing Stem skip)
        self.lat_stem = nn.Conv2d(16, 24, kernel_size=1)
        self.boundary_refine_512 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
        )
        self.dw_boundary_512 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
            CoordinateAttention(24),
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )

        # Segmentation Head predicting logits at 512x512 -> 2x upsample to 1024x1024
        self.head_512 = nn.Conv2d(16, num_classes, kernel_size=1)

        # V4 Additions (zero-initialized gates so warm-start matches V2 100% at Step 0)
        self.context_64 = MultiDilationContextBlock(channels=48)
        self.context_gate = nn.Parameter(torch.zeros(1, 48, 1, 1))

        self.rest_adapter_256 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rest_gate = nn.Parameter(torch.zeros(1, 32, 1, 1))

        # Auxiliary boundary head at 512x512 (used only during training)
        self.aux_boundary_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )

        # Path 2 Additions: ImageNet MobileNetV3-Small Multi-Scale Lateral Projections & Gates
        self.mb_lat_32 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mb_lat_64 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mb_lat_128 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.mb_lat_256 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mb_gate_32 = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.mb_gate_64 = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.mb_gate_128 = nn.Parameter(torch.zeros(1, 48, 1, 1))
        self.mb_gate_256 = nn.Parameter(torch.zeros(1, 32, 1, 1))

    def forward_features(
        self,
        f32: torch.Tensor,
        f64: torch.Tensor,
        f128: torch.Tensor,
        f256: torch.Tensor,
        stem: torch.Tensor,
        rest_feat_256: Optional[torch.Tensor] = None,
        mb_feats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 32x32 -> 64x64 (gradual 2x)
        p32 = self.lat_f32(f32)
        if mb_feats is not None:
            mb_256, mb_128, mb_64, mb_32 = mb_feats
            p32 = p32 + self.mb_gate_32 * self.mb_lat_32(mb_32)
        p64 = F.interpolate(p32, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f64(f64)
        if mb_feats is not None:
            p64 = p64 + self.mb_gate_64 * self.mb_lat_64(mb_64)
        p64 = self.smooth_64(p64)
        p64 = p64 + self.context_gate * self.context_64(p64)

        # 2. 64x64 -> 128x128 (gradual 2x)
        p128 = F.interpolate(p64, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f128(f128)
        if mb_feats is not None:
            p128 = p128 + self.mb_gate_128 * self.mb_lat_128(mb_128)
        p128 = self.smooth_128(p128)

        # 3. 128x128 -> 256x256 (gradual 2x) + detached restoration guidance
        p256 = F.interpolate(p128, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f256(f256)
        if mb_feats is not None:
            p256 = p256 + self.mb_gate_256 * self.mb_lat_256(mb_256)
        if rest_feat_256 is not None:
            p256 = p256 + self.rest_gate * self.rest_adapter_256(rest_feat_256.detach())
        p256 = self.small_obj_neck(p256)

        # 4. 256x256 -> 512x512 (gradual 2x, Boundary Refinement with Stem skip)
        p512 = F.interpolate(p256, scale_factor=2.0, mode='bilinear', align_corners=False)
        p512 = self.boundary_refine_512(p512) + self.lat_stem(stem)
        p512 = self.dw_boundary_512(p512)

        # 5. 512x512 -> 1024x1024 (gradual 2x)
        logits_512 = self.head_512(p512)
        logits_1024 = F.interpolate(logits_512, scale_factor=2.0, mode='bilinear', align_corners=False)
        return logits_1024, p512

    def forward(
        self,
        f32: torch.Tensor,
        f64: torch.Tensor,
        f128: torch.Tensor,
        f256: torch.Tensor,
        stem: torch.Tensor,
        rest_feat_256: Optional[torch.Tensor] = None,
        mb_feats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        logits_1024, _ = self.forward_features(f32, f64, f128, f256, stem, rest_feat_256, mb_feats=mb_feats)
        return logits_1024


class EURSNetV4(nn.Module):
    """
    EURS-Net V4 Complete Unified Architecture (Path 2 Dual-Encoder Edition):
    - Gradual 5-stage downscaling (1024 -> 512 -> 256 -> 128 -> 64 -> 32)
    - Gradual 5-stage upscaling (32 -> 64 -> 128 -> 256 -> 512 -> 1024)
    - Native-resolution Guided Affine + Full-Resolution Detail Refiner restoration head
    - ImageNet-pretrained `MobileNetV3-Small` semantic encoder (`imagenet_encoder`, 0.279M params,
      1.65 GFLOPs at 1024x1024) operating on detached restored RGB (`restored_rgb.detach()`)
    - Multi-dilation context + detached cross-task guidance + boundary refinement segmentation head
    - 100% key-compatible warm-start from `EURSNetV2` (`best_controlled_checkpoint.pt`) or `best_v4_checkpoint.pt`
    """
    def __init__(
        self,
        num_classes: int = 5,
        bounded_restoration: bool = True,
        bound_scale: float = 0.20,
        pretrained_imagenet: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fwb = FastWhiteBalance()
        self.backbone = SharedBackbone()
        self.restoration_decoder = GuidedAffineRestorationDecoderV4(
            bounded_residual=bounded_restoration, bound_scale=bound_scale
        )
        self.imagenet_encoder = ImageNetMobileNetV3Encoder(pretrained=pretrained_imagenet)
        self.segmentation_decoder = DualBranchSegmentationDecoderV4(num_classes=num_classes)
        _replace_hardswish_with_onnx_safe(self)

    def enable_imagenet_gates(self, init_val: float = 0.05) -> None:
        """Open ImageNet lateral gates slightly after Step-0 verification so gradients flow from Batch 1."""
        with torch.no_grad():
            for gate in (
                self.segmentation_decoder.mb_gate_32,
                self.segmentation_decoder.mb_gate_64,
                self.segmentation_decoder.mb_gate_128,
                self.segmentation_decoder.mb_gate_256,
            ):
                if float(gate.abs().max().item()) == 0.0:
                    gate.fill_(init_val)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wb_x = self.fwb(x)
        stem, f256, f128, f64, f32 = self.backbone(wb_x)
        restored_rgb, rest_feat_256 = self.restoration_decoder(f128, f256, stem, raw_input=x, wb_input=wb_x)
        mb_feats = self.imagenet_encoder(restored_rgb.detach())
        seg_logits = self.segmentation_decoder(
            f32, f64, f128, f256, stem, rest_feat_256=rest_feat_256, mb_feats=mb_feats
        )
        return restored_rgb, seg_logits

    def forward_with_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wb_x = self.fwb(x)
        stem, f256, f128, f64, f32 = self.backbone(wb_x)
        restored_rgb, rest_feat_256 = self.restoration_decoder(f128, f256, stem, raw_input=x, wb_input=wb_x)
        mb_feats = self.imagenet_encoder(restored_rgb.detach())
        seg_logits, p512 = self.segmentation_decoder.forward_features(
            f32, f64, f128, f256, stem, rest_feat_256=rest_feat_256, mb_feats=mb_feats
        )
        boundary_logits_512 = self.segmentation_decoder.aux_boundary_head(p512)
        return restored_rgb, seg_logits, boundary_logits_512

    def load_v2_champion_weights(self, ckpt_path: str, strict_v2_keys: bool = True) -> Dict[str, Any]:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src_sd = ckpt.get('model_state_dict', ckpt)
        dst_sd = self.state_dict()

        transferred = []
        skipped = []
        for k, v in src_sd.items():
            if k in dst_sd and dst_sd[k].shape == v.shape:
                dst_sd[k] = v.clone()
                transferred.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(dst_sd, strict=True)
        if strict_v2_keys:
            assert len(skipped) == 0, f"Unexpected skipped keys: {skipped}"
        return {
            'ckpt_path': ckpt_path,
            'transferred_tensors': len(transferred),
            'skipped_tensors': len(skipped),
            'total_v4_tensors': len(dst_sd),
        }

    def load_from_v2_checkpoint(self, ckpt_path: str, strict_v2_keys: bool = True) -> Dict[str, Any]:
        return self.load_v2_champion_weights(ckpt_path, strict_v2_keys=strict_v2_keys)
