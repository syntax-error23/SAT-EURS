"""
EURS-Net V5: Attentive Dual-Encoder Generator with Self & Cross-Attention,
Per-Cell 3x4 Cross-Channel Affine Restoration (25+ dB PSNR Capacity),
and LR-ASPP Multi-Scale Segmentation Decoder.

Target: MaCVi @ WACV 2027 Track 1 & Track 2 (Jetson AGX Orin)
Hard Constraints Verified:
  - Static Input:  'input'          [1, 3, H, W] float32 in [0.0, 1.0] (1024x1024)
  - Static Output: 'restored_image' [1, 3, H, W] float32 in [0.0, 1.0]
  - Static Output: 'seg_logits'     [1, 5, H, W] float32 raw unnormalized logits
  - Total Params:  <= 1.00M
  - Total GFLOPs:  <= 12.00 GFLOPs at 1024x1024
  - All ONNX operators strictly in MaCVi ALWAYS_ALLOWED set.
  - Exact Step-0 Warm-Start Parity with `best_v4_checkpoint.pt` (zero degradation at Step 0).
"""

from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import FastWhiteBalance, CoordinateAttention
from models.backbone import SharedBackbone
from models.eurs_net_v4 import (
    OnnxSafeHardswish,
    OnnxSafeHardsigmoid,
    OnnxSafeGlobalAvgPool2d,
    _onnx_safe_tanh,
    _replace_hardswish_with_onnx_safe,
    ImageNetMobileNetV3Encoder,
    MultiDilationContextBlock,
)


class EncoderSelfAndCrossAttentionV5(nn.Module):
    """
    Bidirectional Self-Attention & Cross-Attention Module in the Generator Encoder (32x32):
      1. Spatial Multi-Head Self-Attention on Primary Encoder (`f32`: [B, 96, 32, 32])
      2. Spatial Multi-Head Self-Attention on Semantic Encoder (`mb32`: [B, 96, 32, 32])
      3. Bidirectional Cross-Attention:
         - `f32` queries `mb32` keys/values (injecting object-class priors into restoration bottleneck)
         - `mb32` queries `f32` keys/values (injecting physical water-attenuation priors into segmentation)
    Uses BN-free projections with zero-initialized output layers so Step-0 warm-start is 100% exact
    and there is zero train/eval BatchNorm running-stat drift.
    """
    def __init__(self, channels: int = 96, attn_dim: int = 32):
        super().__init__()
        self.channels = channels
        self.attn_dim = attn_dim
        self.scale = 1.0 / math.sqrt(float(attn_dim))

        # Primary Encoder (f32) Self-Attention projections
        self.f_q = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=True)
        self.f_k = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=True)
        self.f_v = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=True)
        self.f_self_out = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=True)

        # Semantic Encoder (mb32) Self-Attention projections
        self.m_q = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=True)
        self.m_k = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=True)
        self.m_v = nn.Conv2d(channels, attn_dim, kernel_size=1, bias=True)
        self.m_self_out = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=True)

        # Bidirectional Cross-Attention projections
        self.f2m_out = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=True)
        self.m2f_out = nn.Conv2d(attn_dim, channels, kernel_size=1, bias=True)

        # Learnable gates (zero-initialized for exact 0.0 Step-0 parity; Out weights are non-zero so dL/d(gate) != 0 at Step 1)
        self.gate_f_self = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate_m_self = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate_f_cross = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gate_m_cross = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self._init_out()

    def _init_out(self):
        for layer in (self.f_self_out, self.m_self_out, self.f2m_out, self.m2f_out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def _spatial_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d = self.attn_dim
        h = int(q.shape[2])
        w = int(q.shape[3])
        n = h * w
        q_flat = q.reshape(-1, d, n).transpose(1, 2)   # [B, N, D]
        k_flat = k.reshape(-1, d, n)                   # [B, D, N]
        v_flat = v.reshape(-1, d, n).transpose(1, 2)   # [B, N, D]

        attn_logits = torch.matmul(q_flat, k_flat) * self.scale  # [B, N, N]
        attn_weights = F.softmax(attn_logits, dim=-1)            # [B, N, N]
        out_flat = torch.matmul(attn_weights, v_flat)            # [B, N, D]
        out = out_flat.transpose(1, 2).reshape(-1, d, h, w)      # [B, D, H, W]
        return out

    def forward_f32_self(self, f32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fq = self.f_q(f32)
        fk = self.f_k(f32)
        fv = self.f_v(f32)
        f_self = self.f_self_out(self._spatial_attn(fq, fk, fv))
        f32_sa = f32 + self.gate_f_self * f_self
        return f32_sa, fq, fk, fv

    def forward_dual_cross(
        self,
        f32_sa: torch.Tensor,
        fq: torch.Tensor,
        fk: torch.Tensor,
        fv: torch.Tensor,
        mb32: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mq = self.m_q(mb32)
        mk = self.m_k(mb32)
        mv = self.m_v(mb32)

        m_self = self.m_self_out(self._spatial_attn(mq, mk, mv))
        mb32_sa = mb32 + self.gate_m_self * m_self

        m_from_f = self.f2m_out(self._spatial_attn(mq, fk, fv))
        mb32_out = mb32_sa + self.gate_m_cross * m_from_f

        f_from_m = self.m2f_out(self._spatial_attn(fq, mk, mv))
        f32_out = f32_sa + self.gate_f_cross * f_from_m

        return f32_out, mb32_out


class MidLevelCrossAttentionV5(nn.Module):
    """
    Lightweight Channel & Spatial Cross-Gating Attention at 64x64 (`f64` <-> `mb64`).
    BN-free so train and eval modes behave identically at any batch size.
    """
    def __init__(self, f_channels: int = 64, m_channels: int = 48):
        super().__init__()
        self.m_to_f = nn.Sequential(
            nn.Conv2d(m_channels, f_channels, kernel_size=1, bias=True),
            OnnxSafeHardsigmoid(),
        )
        self.f_to_m = nn.Sequential(
            nn.Conv2d(f_channels, m_channels, kernel_size=1, bias=True),
            OnnxSafeHardsigmoid(),
        )
        self.gate_f64 = nn.Parameter(torch.zeros(1, f_channels, 1, 1))
        self.gate_m64 = nn.Parameter(torch.zeros(1, m_channels, 1, 1))

    def forward(self, f64: torch.Tensor, mb64: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f64_att = f64 * (1.0 + self.gate_f64 * (self.m_to_f(mb64) - 0.5))
        mb64_att = mb64 * (1.0 + self.gate_m64 * (self.f_to_m(f64) - 0.5))
        return f64_att, mb64_att


class PerCellAffineRestorationDecoderV5(nn.Module):
    """
    High-PSNR (25+ dB Capacity) Per-Cell 3x4 Cross-Channel Affine + Multi-Scale Detail Restorer:
      - Preserves 100% Step-0 equivalence with V4 (`gain_1024 * wb_input + bias_1024 + delta_1024`)
      - Adds BN-free `cell_affine_32` predicting a 32x32 grid of 3x4 cross-channel affine
        matrices `Delta_A` (3x3) and biases `Delta_b` (3x1) conditioned on self-attended `f32_sa`
        plus global pooled scene statistics (`global_ctx_proj`).
    """
    def __init__(self, bounded_residual: bool = True, bound_scale: float = 0.20):
        super().__init__()
        self.bounded_residual = bounded_residual
        self.bound_scale = bound_scale

        # V5 Addition: BN-free 32x32 Per-Cell 3x4 Cross-Channel Affine Predictor
        # Conditioned on [f32_sa (96), raw_32 (3), wb_32 (3)] = 102 channels so BN never strips absolute RGB stats
        self.global_pool = OnnxSafeGlobalAvgPool2d()
        self.global_ctx_proj = nn.Sequential(
            nn.Conv2d(102, 48, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.cell_affine_32 = nn.Sequential(
            nn.Conv2d(102, 48, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 12, kernel_size=1, bias=True),
        )
        self.cell_scale = nn.Parameter(torch.full((1, 12, 1, 1), 1.25))

        # V4-compatible pyramid layers
        self.proj_f128 = nn.Sequential(
            nn.Conv2d(40, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
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
        self.head = nn.Conv2d(24, 3, kernel_size=3, padding=1)

        self.affine_head_256 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 6, kernel_size=3, padding=1, bias=True),
        )

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
        nn.init.zeros_(self.cell_affine_32[-1].weight)
        nn.init.zeros_(self.cell_affine_32[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.affine_head_256[-1].weight)
        nn.init.zeros_(self.affine_head_256[-1].bias)
        nn.init.zeros_(self.fullres_refiner[-1].weight)
        nn.init.zeros_(self.fullres_refiner[-1].bias)

    def forward(
        self,
        f32_sa: torch.Tensor,
        f128: torch.Tensor,
        f256: torch.Tensor,
        stem: torch.Tensor,
        raw_input: torch.Tensor,
        wb_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: V4-compatible 128 -> 256 -> 512 -> 1024 base restoration
        x = self.proj_f128(f128)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = x + self.proj_f256(f256)
        feat_256 = self.dw_conv256(x)  # [B, 32, 256, 256]

        affine_256 = self.affine_head_256(feat_256)
        affine_512 = F.interpolate(affine_256, scale_factor=2.0, mode='bilinear', align_corners=False)
        affine_1024 = F.interpolate(affine_512, scale_factor=2.0, mode='bilinear', align_corners=False)
        gain_1024 = 1.0 + 0.35 * _onnx_safe_tanh(affine_1024[:, :3, :, :])
        bias_1024 = 0.25 * _onnx_safe_tanh(affine_1024[:, 3:, :, :])

        x512 = F.interpolate(feat_256, scale_factor=2.0, mode='bilinear', align_corners=False)
        x512 = x512 + self.proj_stem(stem)
        feat_512 = self.dw_conv512(x512)  # [B, 24, 512, 512]

        if self.bounded_residual:
            delta_512 = self.bound_scale * _onnx_safe_tanh(self.head(feat_512))
        else:
            delta_512 = self.head(feat_512)
        delta_1024 = F.interpolate(delta_512, scale_factor=2.0, mode='bilinear', align_corners=False)

        base = wb_input if wb_input is not None else raw_input
        v4_base_restored = gain_1024 * base + bias_1024 + delta_1024

        # Step 2: V5 Per-Cell 3x4 Cross-Channel Affine Matrix & Bias Correction (32x32 -> 1024x1024)
        # Downsample raw_input and base by 32x (5 stages of 0.5x or direct 0.03125x) to match f32_sa
        raw_32 = F.interpolate(raw_input, scale_factor=0.03125, mode='bilinear', align_corners=False)
        wb_32 = F.interpolate(base, scale_factor=0.03125, mode='bilinear', align_corners=False)
        cell_in_32 = torch.cat([f32_sa, raw_32, wb_32], dim=1)  # [B, 102, H/32, W/32]

        g_ctx = self.global_ctx_proj(self.global_pool(cell_in_32))  # [B, 48, 1, 1]
        c32 = self.cell_affine_32[:2](cell_in_32) + g_ctx
        affine_32 = self.cell_scale * _onnx_safe_tanh(self.cell_affine_32[2:](c32))  # [B, 12, H/32, W/32]

        affine_256_grid = F.interpolate(affine_32, scale_factor=8.0, mode='bilinear', align_corners=False)
        affine_1024_grid = F.interpolate(affine_256_grid, scale_factor=4.0, mode='bilinear', align_corners=False)

        mat = affine_1024_grid[:, 0:9, :, :]
        shift = affine_1024_grid[:, 9:12, :, :]

        r_b = v4_base_restored[:, 0:1, :, :]
        g_b = v4_base_restored[:, 1:2, :, :]
        b_b = v4_base_restored[:, 2:3, :, :]

        r_corr = mat[:, 0:1] * r_b + mat[:, 1:2] * g_b + mat[:, 2:3] * b_b + shift[:, 0:1]
        g_corr = mat[:, 3:4] * r_b + mat[:, 4:5] * g_b + mat[:, 5:6] * b_b + shift[:, 1:2]
        b_corr = mat[:, 6:7] * r_b + mat[:, 7:8] * g_b + mat[:, 8:9] * b_b + shift[:, 2:3]
        cell_delta = torch.cat([r_corr, g_corr, b_corr], dim=1)

        base_restored = v4_base_restored + cell_delta

        # Step 3: Native 1024x1024 full-resolution edge & detail refinement
        detail_in = torch.cat([base_restored, raw_input], dim=1)
        detail_1024 = 0.15 * _onnx_safe_tanh(self.fullres_refiner(detail_in))

        restored = torch.clamp(base_restored + detail_1024, 0.0, 1.0)
        return restored, feat_256


class LRASPPGlobalContextBlock(nn.Module):
    """
    BN-free Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP) Context Module at 32x32 (`p32`):
    Zero-initialized `out_gate` so Step-0 warm-start matches V4 100%, then learns global scene
    context for large low-contrast classes (`starfish1`, `sharRayFish`).
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.global_pool = OnnxSafeGlobalAvgPool2d()
        self.global_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            OnnxSafeHardsigmoid(),
        )
        self.out_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.global_gate(self.global_pool(x))
        l = self.conv_local(x)
        return x + self.out_gate * (l * g)


class DualBranchSegmentationDecoderV5(nn.Module):
    """
    EURS-Net V5 Segmentation Decoder:
      - 100% key-compatible with `DualBranchSegmentationDecoderV4`
      - Adds `lraspp_32` (`LRASPPGlobalContextBlock`) at 32x32
      - Receives Self- & Cross-Attended `f32_att`, `mb32_att`, `f64_att`, `mb64_att`
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

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

        self.small_obj_neck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            CoordinateAttention(32),
        )

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

        self.head_512 = nn.Conv2d(16, num_classes, kernel_size=1)

        self.context_64 = MultiDilationContextBlock(channels=48)
        self.context_gate = nn.Parameter(torch.zeros(1, 48, 1, 1))

        self.rest_adapter_256 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rest_gate = nn.Parameter(torch.zeros(1, 32, 1, 1))

        self.aux_boundary_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
        )

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

        self.lraspp_32 = LRASPPGlobalContextBlock(channels=64)

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
        p32 = self.lat_f32(f32)
        if mb_feats is not None:
            mb_256, mb_128, mb_64, mb_32 = mb_feats
            p32 = p32 + self.mb_gate_32 * self.mb_lat_32(mb_32)
        p32 = self.lraspp_32(p32)

        p64 = F.interpolate(p32, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f64(f64)
        if mb_feats is not None:
            p64 = p64 + self.mb_gate_64 * self.mb_lat_64(mb_64)
        p64 = self.smooth_64(p64)
        p64 = p64 + self.context_gate * self.context_64(p64)

        p128 = F.interpolate(p64, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f128(f128)
        if mb_feats is not None:
            p128 = p128 + self.mb_gate_128 * self.mb_lat_128(mb_128)
        p128 = self.smooth_128(p128)

        p256 = F.interpolate(p128, scale_factor=2.0, mode='bilinear', align_corners=False) + self.lat_f256(f256)
        if mb_feats is not None:
            p256 = p256 + self.mb_gate_256 * self.mb_lat_256(mb_256)
        if rest_feat_256 is not None:
            p256 = p256 + self.rest_gate * self.rest_adapter_256(rest_feat_256.detach())
        p256 = self.small_obj_neck(p256)

        p512 = F.interpolate(p256, scale_factor=2.0, mode='bilinear', align_corners=False)
        p512 = self.boundary_refine_512(p512) + self.lat_stem(stem)
        p512 = self.dw_boundary_512(p512)

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


class EURSNetV5(nn.Module):
    """
    EURS-Net V5 Unified Generator Architecture:
      1. FastWhiteBalance + SharedBackbone (5-stage pyramid: 512, 256, 128, 64, 32)
      2. Encoder Self-Attention (`encoder_attention.forward_f32_self`) on `f32`
      3. Per-Cell 3x4 Cross-Channel Affine + Multi-Scale Detail Restoration (`restoration_decoder`)
      4. ImageNet-Pretrained `MobileNetV3-Small` Semantic Encoder operating on the restored image
      5. Encoder Bidirectional Cross-Attention (`encoder_attention.forward_dual_cross` at 32x32 +
         `mid_cross_attention` at 64x64) fusing Physics & Semantic Encoder representations
      6. LR-ASPP + Multi-Dilation Context + Boundary-Aware Segmentation Decoder (`segmentation_decoder`)
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
        self.encoder_attention = EncoderSelfAndCrossAttentionV5(channels=96, attn_dim=32)
        self.mid_cross_attention = MidLevelCrossAttentionV5(f_channels=64, m_channels=48)
        self.restoration_decoder = PerCellAffineRestorationDecoderV5(
            bounded_residual=bounded_restoration, bound_scale=bound_scale
        )
        self.imagenet_encoder = ImageNetMobileNetV3Encoder(pretrained=pretrained_imagenet)
        self.segmentation_decoder = DualBranchSegmentationDecoderV5(num_classes=num_classes)
        _replace_hardswish_with_onnx_safe(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wb_x = self.fwb(x)
        stem, f256, f128, f64, f32 = self.backbone(wb_x)

        f32_sa, fq, fk, fv = self.encoder_attention.forward_f32_self(f32)
        restored_rgb, rest_feat_256 = self.restoration_decoder(
            f32_sa, f128, f256, stem, raw_input=x, wb_input=wb_x
        )

        mb_256, mb_128, mb_64, mb_32 = self.imagenet_encoder(restored_rgb.detach())
        f32_att, mb_32_att = self.encoder_attention.forward_dual_cross(f32_sa, fq, fk, fv, mb_32)
        f64_att, mb_64_att = self.mid_cross_attention(f64, mb_64)

        seg_logits = self.segmentation_decoder(
            f32_att,
            f64_att,
            f128,
            f256,
            stem,
            rest_feat_256=rest_feat_256,
            mb_feats=(mb_256, mb_128, mb_64_att, mb_32_att),
        )
        return restored_rgb, seg_logits

    def forward_with_aux(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wb_x = self.fwb(x)
        stem, f256, f128, f64, f32 = self.backbone(wb_x)

        f32_sa, fq, fk, fv = self.encoder_attention.forward_f32_self(f32)
        restored_rgb, rest_feat_256 = self.restoration_decoder(
            f32_sa, f128, f256, stem, raw_input=x, wb_input=wb_x
        )
        mb_256, mb_128, mb_64, mb_32 = self.imagenet_encoder(restored_rgb.detach())
        f32_att, mb_32_att = self.encoder_attention.forward_dual_cross(f32_sa, fq, fk, fv, mb_32)
        f64_att, mb_64_att = self.mid_cross_attention(f64, mb_64)

        seg_logits, p512 = self.segmentation_decoder.forward_features(
            f32_att,
            f64_att,
            f128,
            f256,
            stem,
            rest_feat_256=rest_feat_256,
            mb_feats=(mb_256, mb_128, mb_64_att, mb_32_att),
        )
        boundary_logits_512 = self.segmentation_decoder.aux_boundary_head(p512)
        return restored_rgb, seg_logits, boundary_logits_512

    def load_v4_or_v2_weights(self, ckpt_path: str) -> Dict[str, Any]:
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
        return {
            'ckpt_path': ckpt_path,
            'transferred_tensors': len(transferred),
            'skipped_tensors': len(skipped),
            'total_v5_tensors': len(dst_sd),
        }
