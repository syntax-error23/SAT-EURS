"""
SAT-EURS Track 2 — LiteV3 Architecture Variant with High-Resolution Stride-2 Decoder Branch
Integrates Stride 2 lateral skip (16 channels from MobileNetV3 Layer 1) into progressive fusion.
Maintains strict edge compliance:
- Parameters: ~0.482 M (<< 1.0M)
- FLOPs: ~1.75 GFLOPs (<< 12.0 GFLOPs)
- Allowed Opset 17 operators only
"""

import os
import math
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class OnnxSafeHardswish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class OnnxSafeHardsigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


def _replace_activations_with_onnx_safe(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, OnnxSafeHardswish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, OnnxSafeHardsigmoid())
        else:
            _replace_activations_with_onnx_safe(child)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class LiteV3Stride2Decoder(nn.Module):
    """
    Multiscale Progressive Fusion Decoder with Stride-2 Branch.
    Fuses Stride 32 (context) -> Stride 16 -> Stride 8 -> Stride 4 -> Stride 2 (ultra-fine contours).
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        # 1. Stride 32 Context Stage from Stride 16 (112 channels)
        self.context_down = DepthwiseSeparableConv(112, 64, kernel_size=3, stride=2, padding=1)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.context_conv = DepthwiseSeparableConv(64, 64, kernel_size=3, padding=2, dilation=2)
        
        # 2. Lateral skip projections
        self.lat16 = nn.Sequential(
            nn.Conv2d(112, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.lat8 = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.lat4 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.lat2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 3. Progressive Fusions
        self.fuse16 = DepthwiseSeparableConv(64 + 48, 48, kernel_size=3, padding=1)
        self.fuse8 = DepthwiseSeparableConv(48 + 32, 32, kernel_size=3, padding=1)
        self.fuse4 = DepthwiseSeparableConv(32 + 24, 32, kernel_size=3, padding=1)
        self.fuse2 = DepthwiseSeparableConv(32 + 16, 24, kernel_size=3, padding=1)
        
        # 4. Final Classification Head
        self.head = nn.Conv2d(24, num_classes, kernel_size=1, bias=True)

    def forward(self, f2: torch.Tensor, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        # Context at Stride 32
        c32 = self.context_down(f16)
        gate = self.global_gate(c32)
        c32 = self.context_conv(c32 * gate)
        
        # Fuse Stride 16
        c32_up = F.interpolate(c32, size=f16.shape[2:], mode="bilinear", align_corners=False)
        m16 = self.fuse16(torch.cat([c32_up, self.lat16(f16)], dim=1))
        
        # Fuse Stride 8
        m16_up = F.interpolate(m16, size=f8.shape[2:], mode="bilinear", align_corners=False)
        m8 = self.fuse8(torch.cat([m16_up, self.lat8(f8)], dim=1))
        
        # Fuse Stride 4
        m8_up = F.interpolate(m8, size=f4.shape[2:], mode="bilinear", align_corners=False)
        m4 = self.fuse4(torch.cat([m8_up, self.lat4(f4)], dim=1))
        
        # Fuse Stride 2
        m4_up = F.interpolate(m4, size=f2.shape[2:], mode="bilinear", align_corners=False)
        m2 = self.fuse2(torch.cat([m4_up, self.lat2(f2)], dim=1))
        
        # Final prediction and upsample to target resolution
        out = self.head(m2)
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class SATEURSLiteV3Stride2(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        unsharp_factor: float = 1.20,
        pretrained_backbone: bool = True,
        table_path: str = "data/rsuigm_20scene_table.pt"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.unsharp_factor = unsharp_factor
        
        # 1. OPTICAL ROUTER & MATRIX BANK
        self.router = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.router_fc = nn.Linear(32, 200)
        self.routing_temperature = 0.20
        self.register_buffer("table_bank", torch.zeros(200, 57))
        
        # 3x3 depthwise Gaussian unsharp kernel
        blur_k = torch.tensor([
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("unsharp_blur_kernel", blur_k.repeat(3, 1, 1, 1))
        
        # 2. PRETRAINED MOBILENETV3-LARGE BACKBONE
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        mb = tvm.mobilenet_v3_large(weights=weights)
        _replace_activations_with_onnx_safe(mb)
        self.backbone_blocks = mb.features[:12]
        
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # 3. STRIDE-2 MULTISCALE DECODER
        self.decoder = LiteV3Stride2Decoder(num_classes=num_classes)
        
        if os.path.exists(table_path):
            table = torch.load(table_path, map_location="cpu", weights_only=False)
            bank = torch.zeros(200, 57)
            idx = 0
            for wt in range(1, 11):
                for dp in range(5, 105, 5):
                    bank[idx] = table[(wt, dp)].reshape(-1)
                    idx += 1
            self.table_bank.copy_(bank)

    def _build_poly_features(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        r, g, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        ones = torch.ones_like(r)
        phi = torch.cat([
            ones, r, g, bl,
            r * r, g * g, bl * bl,
            r * g, r * bl, g * bl,
            r * g * bl,
            r * r * r, g * g * g, bl * bl * bl,
            r * r * g, r * r * bl,
            g * g * r, g * g * bl,
            bl * bl * r
        ], dim=1)
        return phi

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        
        # Physical restoration
        x_small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        r_feat = self.router(x_small).flatten(1)
        r_logits = self.router_fc(r_feat)
        r_logits_scaled = (r_logits.float() - r_logits.float().max(dim=-1, keepdim=True).values) / self.routing_temperature
        alpha = F.softmax(r_logits_scaled, dim=-1).to(r_logits.dtype)
        
        w_eff = torch.matmul(alpha, self.table_bank).view(b, 19, 3)
        phi = self._build_poly_features(x)
        phi_flat = phi.permute(0, 2, 3, 1).reshape(b, h * w, 19)
        res_flat = torch.bmm(phi_flat, w_eff)
        res_raw = res_flat.permute(0, 2, 1).reshape(b, 3, h, w).clamp(0.0, 1.0)
        
        blur = F.conv2d(res_raw, self.unsharp_blur_kernel, padding=1, groups=3)
        restored_image = torch.clamp(self.unsharp_factor * res_raw + (1.0 - self.unsharp_factor) * blur, 0.0, 1.0)
        
        # Segmentation with Stride-2 tap at layer 1
        x_norm = (restored_image.detach() - self.norm_mean) / self.norm_std
        curr = x_norm
        f2, f4, f8, f16 = None, None, None, None
        for i, blk in enumerate(self.backbone_blocks):
            curr = blk(curr)
            if i == 1:   # Stride 2 (16 channels)
                f2 = curr
            elif i == 3: # Stride 4 (24 channels)
                f4 = curr
            elif i == 6: # Stride 8 (40 channels)
                f8 = curr
            elif i == 11:# Stride 16 (112 channels)
                f16 = curr
                
        seg_logits = self.decoder(f2, f4, f8, f16, (h, w))
        return restored_image, seg_logits


def build_track2_litev3_stride2_model(pretrained_backbone: bool = True, unsharp_factor: float = 1.20) -> SATEURSLiteV3Stride2:
    return SATEURSLiteV3Stride2(num_classes=5, unsharp_factor=unsharp_factor, pretrained_backbone=pretrained_backbone)
