"""
Track 2 SOTA Architecture for NVIDIA Jetson AGX Orin: `Track2SOTAOrin`
Target: MaCVi @ WACV 2027 EURS Challenge (Track 2: Embedded Edge)

Properties:
  - Input:  'input'          [1, 3, 1024, 1024] float32 in [0.0, 1.0]
  - Output: 'restored_image' [1, 3, 1024, 1024] float32 in [0.0, 1.0]
  - Output: 'seg_logits'     [1, 5, 1024, 1024] float32 raw unnormalized logits
  - Complexity: <= 1.00M parameters, <= 12.00 GFLOPs at 1024x1024
  - Backbone: ImageNet-pretrained MobileNetV3-Large (Features 0..11)
  - Decoder: Progressive Multi-Scale Pyramid Decoder (Stride 32 Context, 16, 8, 4)
  - Restoration: 19-term analytical physical RSUIGM basis with condition router + gentle unsharp filter
  - ZERO negative bias offsets (eliminates Class 3 & Class 4 blindness)
"""

import os
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class OnnxSafeHardswish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0) / 6.0


class OnnxSafeHardsigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0) / 6.0


def _replace_activations_with_onnx_safe(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, OnnxSafeHardswish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, OnnxSafeHardsigmoid())
        else:
            _replace_activations_with_onnx_safe(child)


class DepthwiseSeparableConv(nn.Module):
    """
    Standard Depthwise-Separable Convolution:
    DW Conv 3x3 -> BN -> ReLU -> PW Conv 1x1 -> BN -> ReLU
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_dw(self.dw(x)))
        x = self.act(self.bn_pw(self.pw(x)))
        return x


class ProgressivePyramidDecoder(nn.Module):
    """
    Multiscale Progressive Fusion Decoder:
    Aggregates Stride 32 (semantic context), Stride 16, Stride 8, and Stride 4 (fine boundaries).
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        # 1. Stride 32 Context Block from Stride 16 (112 channels)
        self.context_down = DepthwiseSeparableConv(112, 128, kernel_size=3, stride=2, padding=1)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.context_conv = DepthwiseSeparableConv(128, 128, kernel_size=3, padding=2, dilation=2)

        # 2. Lateral Projections
        self.lat16 = nn.Sequential(
            nn.Conv2d(112, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.lat8 = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.lat4 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 3. Progressive Fusions
        self.fuse16 = DepthwiseSeparableConv(128 + 96, 96, kernel_size=3, padding=1)
        self.fuse8 = DepthwiseSeparableConv(96 + 64, 64, kernel_size=3, padding=1)
        self.fuse4 = DepthwiseSeparableConv(64 + 48, 48, kernel_size=3, padding=1)

        # 4. Final Classification Head
        self.head = nn.Conv2d(48, num_classes, kernel_size=1, bias=True)

    def forward(
        self,
        f4: torch.Tensor,
        f8: torch.Tensor,
        f16: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        # Stride 32 Context
        f32 = self.context_down(f16)
        gate = self.global_gate(f32)
        f32 = self.context_conv(f32) * gate

        # Fusion at Stride 16
        up16 = F.interpolate(f32, size=(f16.shape[2], f16.shape[3]), mode="bilinear", align_corners=False)
        p16 = self.fuse16(torch.cat([up16, self.lat16(f16)], dim=1))

        # Fusion at Stride 8
        up8 = F.interpolate(p16, size=(f8.shape[2], f8.shape[3]), mode="bilinear", align_corners=False)
        p8 = self.fuse8(torch.cat([up8, self.lat8(f8)], dim=1))

        # Fusion at Stride 4
        up4 = F.interpolate(p8, size=(f4.shape[2], f4.shape[3]), mode="bilinear", align_corners=False)
        p4 = self.fuse4(torch.cat([up4, self.lat4(f4)], dim=1))

        # Head & final upsample to target resolution
        logits_stride4 = self.head(p4)
        return F.interpolate(logits_stride4, size=target_size, mode="bilinear", align_corners=False)


class Track2SOTAOrin(nn.Module):
    """
    Official Track 2 Production Model:
    Input:  'input'          [1, 3, 1024, 1024] in [0, 1]
    Output: 'restored_image' [1, 3, 1024, 1024] in [0, 1]
    Output: 'seg_logits'     [1, 5, 1024, 1024] raw logits
    """
    def __init__(
        self,
        num_classes: int = 5,
        unsharp_factor: float = 0.20,
        pretrained_backbone: bool = True,
        table_path: str = "checkpoints/track1_screening/tbl19_20sc_clean.pt"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.unsharp_factor = unsharp_factor

        # --- 1. RESTORATION SUBNET ---
        self.router = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.router_fc = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 200)
        )
        self.register_buffer("table_bank", torch.zeros(200, 57))

        # 3x3 depthwise Gaussian unsharp kernel
        blur_k = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0]
        ]) / 16.0
        self.register_buffer("unsharp_blur_kernel", blur_k.repeat(3, 1, 1, 1))

        # --- 2. BACKBONE (MobileNetV3-Large truncated to stage 11) ---
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        mb = tvm.mobilenet_v3_large(weights=weights)
        _replace_activations_with_onnx_safe(mb)
        self.backbone_blocks = mb.features[:12]

        # Normalization constants (ImageNet standard)
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # --- 3. MULTISCALE DECODER ---
        self.decoder = ProgressivePyramidDecoder(num_classes=num_classes)

        # Load pre-fitted 20-scene RSUIGM table into buffer
        if os.path.exists(table_path):
            table = torch.load(table_path, map_location="cpu", weights_only=False)
            bank = torch.zeros(200, 57)
            idx = 0
            for wt in range(1, 11):
                for dp in range(5, 105, 5):
                    bank[idx] = table[(wt, dp)].reshape(-1)
                    idx += 1
            self.table_bank.copy_(bank)
        else:
            print(f"[Warning] RSUIGM table not found at {table_path}; buffer initialized to zeros.")

    def _build_poly_features(self, x: torch.Tensor) -> torch.Tensor:
        """Constructs 19-dimensional physical RSUIGM feature basis natively in ONNX."""
        b, _, h, w = x.shape
        b1 = F.interpolate(F.avg_pool2d(x, kernel_size=16, stride=16), size=(h, w), mode="bilinear", align_corners=False)
        b2 = F.interpolate(F.avg_pool2d(x, kernel_size=64, stride=64), size=(h, w), mode="bilinear", align_corners=False)

        r, g, bl = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        ones = torch.ones_like(r)
        x2 = x * x
        rg, rb, gb = r * g, r * bl, g * bl
        x3 = x2 * x
        return torch.cat([ones, x, x2, rg, rb, gb, x3, b1, b2], dim=1)  # [B, 19, H, W]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape

        # --- 1. PHYSICAL RESTORATION ---
        x_small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        r_feat = self.router(x_small).flatten(1)
        r_logits = self.router_fc(r_feat)
        alpha = F.softmax(r_logits, dim=-1)

        w_eff = torch.matmul(alpha, self.table_bank).view(b, 19, 3)
        phi = self._build_poly_features(x)
        phi_flat = phi.permute(0, 2, 3, 1).reshape(b, h * w, 19)
        res_flat = torch.bmm(phi_flat, w_eff)
        res_raw = res_flat.permute(0, 2, 1).reshape(b, 3, h, w).clamp(0.0, 1.0)

        # Gentle linear unsharp mask (preserves high UIQM/UCIQE)
        blur = F.conv2d(res_raw, self.unsharp_blur_kernel, padding=1, groups=3)
        restored_image = torch.clamp(res_raw + self.unsharp_factor * (res_raw - blur), 0.0, 1.0)

        # --- 2. MULTI-SCALE SEGMENTATION ---
        # ImageNet normalization on restored image
        x_norm = (restored_image.detach() - self.norm_mean) / self.norm_std

        cur = x_norm
        f4 = None
        f8 = None
        f16 = None
        for i, block in enumerate(self.backbone_blocks):
            cur = block(cur)
            if i == 3:
                f4 = cur  # Stride 4: [B, 24, 256, 256]
            elif i == 6:
                f8 = cur  # Stride 8: [B, 40, 128, 128]
            elif i == 11:
                f16 = cur  # Stride 16: [B, 112, 64, 64]

        seg_logits = self.decoder(f4, f8, f16, (h, w))
        return restored_image, seg_logits


def build_track2_sota_model(pretrained_backbone: bool = True) -> Track2SOTAOrin:
    return Track2SOTAOrin(num_classes=5, unsharp_factor=0.20, pretrained_backbone=pretrained_backbone)
