"""
SAT-EURS Track 2 — LiteV3 Production Architecture
Hardware Target: NVIDIA Jetson AGX Orin (Embedded Edge)
Complexity Budget: Total Parameters <= 1.0M, Static Compute <= 12.0 GFLOPs

Architecture Components:
1. Restoration Subnet:
   - 3-stage lightweight Optical Router CNN (128x128 input -> 200 condition logits)
   - 19-term analytical RSUIGM polynomial feature basis
   - Static 200-condition matrix bank [200, 57]
   - Calibrated 3x3 Gaussian unsharp edge enhancement (factor = 1.20)
   - Zero metadata dependency (fully blind inference from degraded RGB)
2. Pretrained Semantic Backbone:
   - TorchVision ImageNet-pretrained MobileNetV3-Large truncated to features[:12]
   - Stages: Stride 2 (16 ch), Stride 4 (24 ch), Stride 8 (40 ch), Stride 16 (112 ch)
   - Hardswish/Hardsigmoid converted to ONNX-safe primitives for TensorRT compatibility
3. Multiscale Progressive Pyramid Decoder:
   - Stride-32 lightweight context stage (depthwise-separable conv 112 -> 64, stride 2)
   - Global context gating (ReduceMean -> 1x1 Conv -> Sigmoid)
   - Lateral skip projections from Strides 16, 8, and 4
   - Progressive depthwise-separable fusion to retain boundary contours for marine fauna
   - Direct 5-class calibrated logits
"""

from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class OnnxSafeHardswish(nn.Module):
    """x * clamp(x + 3, 0, 6) / 6 strictly using ALWAYS_ALLOWED ops."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class OnnxSafeHardsigmoid(nn.Module):
    """clamp(x + 3, 0, 6) / 6 strictly using ALWAYS_ALLOWED ops."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


def _replace_activations_with_onnx_safe(module: nn.Module) -> None:
    """Recursively replaces Hardswish and Hardsigmoid with ONNX-safe equivalents."""
    for name, child in module.named_children():
        if isinstance(child, nn.Hardswish):
            setattr(module, name, OnnxSafeHardswish())
        elif isinstance(child, nn.Hardsigmoid):
            setattr(module, name, OnnxSafeHardsigmoid())
        else:
            _replace_activations_with_onnx_safe(child)


class DepthwiseSeparableConv(nn.Module):
    """Lightweight 2D Depthwise-Separable Convolution (DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU)."""
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


class LiteV3Decoder(nn.Module):
    """
    Multiscale Progressive Fusion Decoder.
    Combines Stride 32 (semantic context), Stride 16, Stride 8, and Stride 4 (high-res boundary).
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        # 1. Stride 32 Context Stage from Stride 16 (112 channels)
        self.context_down = DepthwiseSeparableConv(112, 64, kernel_size=3, stride=2, padding=1)
        
        # Global context gating
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
        
        # 3. Progressive Fusions
        self.fuse16 = DepthwiseSeparableConv(64 + 48, 48, kernel_size=3, padding=1)
        self.fuse8 = DepthwiseSeparableConv(48 + 32, 32, kernel_size=3, padding=1)
        self.fuse4 = DepthwiseSeparableConv(32 + 24, 32, kernel_size=3, padding=1)
        
        # 4. Final Classification Head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

    def forward(self, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        # Context extraction at Stride 32
        f32 = self.context_down(f16) # [B, 64, H/32, W/32]
        gate = self.global_gate(f32)
        f32 = self.context_conv(f32) * gate
        
        # Progressive fusion: 32 -> 16
        u16 = F.interpolate(f32, size=f16.shape[2:], mode="bilinear", align_corners=False)
        m16 = self.fuse16(torch.cat([u16, self.lat16(f16)], dim=1))
        
        # Progressive fusion: 16 -> 8
        u8 = F.interpolate(m16, size=f8.shape[2:], mode="bilinear", align_corners=False)
        m8 = self.fuse8(torch.cat([u8, self.lat8(f8)], dim=1))
        
        # Progressive fusion: 8 -> 4
        u4 = F.interpolate(m8, size=f4.shape[2:], mode="bilinear", align_corners=False)
        m4 = self.fuse4(torch.cat([u4, self.lat4(f4)], dim=1))
        
        # Classification logits
        logits_stride4 = self.head(m4)
        return F.interpolate(logits_stride4, size=target_size, mode="bilinear", align_corners=False)


class SATEURSLiteV3(nn.Module):
    """
    Unified Track 2 Production Model: SAT-EURS-LiteV3
    Input:  'input'          [1, 3, H, W] float32 in [0, 1]
    Output: 'restored_image' [1, 3, H, W] float32 in [0, 1]
    Output: 'seg_logits'     [1, 5, H, W] float32 raw logits
    """
    def __init__(
        self,
        num_classes: int = 5,
        unsharp_factor: float = 1.20,
        use_edge_gated: bool = True,
        routing_temperature: float = 0.10,
        pretrained_backbone: bool = True,
        table_path: str = "checkpoints/track1_screening/tbl19_20sc_clean.pt"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.unsharp_factor = unsharp_factor
        self.use_edge_gated = use_edge_gated
        self.routing_temperature = routing_temperature
        
        # --- 1. RESTORATION SUBNET ---
        self.router = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False), # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False), # 16x16
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
        blur_weight = blur_k.repeat(3, 1, 1, 1)
        self.register_buffer("unsharp_blur_kernel", blur_weight)
        
        # Sobel gradient filters for edge-gated sharpening
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        
        # --- 2. PRETRAINED MOBILENETV3-LARGE BACKBONE ---
        weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        mb = tvm.mobilenet_v3_large(weights=weights)
        _replace_activations_with_onnx_safe(mb)
        self.backbone_blocks = mb.features[:12]
        
        # Normalization constants (ImageNet standard)
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # --- 3. MULTISCALE DECODER ---
        self.decoder = LiteV3Decoder(num_classes=num_classes)
        
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
        return torch.cat([ones, x, x2, rg, rb, gb, x3, b1, b2], dim=1) # [B, 19, H, W]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        
        # --- 1. PHYSICAL RESTORATION ---
        x_small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        r_feat = self.router(x_small).flatten(1)
        r_logits = self.router_fc(r_feat)
        # Numerically stable temperature routing in FP32 (prevents FP16 exp overflow)
        r_logits_scaled = (r_logits.float() - r_logits.float().max(dim=-1, keepdim=True).values) / self.routing_temperature
        alpha = F.softmax(r_logits_scaled, dim=-1).to(r_logits.dtype)
        
        w_eff = torch.matmul(alpha, self.table_bank).view(b, 19, 3)
        phi = self._build_poly_features(x)
        phi_flat = phi.permute(0, 2, 3, 1).reshape(b, h * w, 19)
        res_flat = torch.bmm(phi_flat, w_eff)
        res_raw = res_flat.permute(0, 2, 1).reshape(b, 3, h, w).clamp(0.0, 1.0)
        
        blur = F.conv2d(res_raw, self.unsharp_blur_kernel, padding=1, groups=3)
        if self.use_edge_gated:
            gray = 0.299 * res_raw[:, 0:1] + 0.587 * res_raw[:, 1:2] + 0.114 * res_raw[:, 2:3]
            gx = F.conv2d(gray, self.sobel_x, padding=1)
            gy = F.conv2d(gray, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
            edge_mask = torch.clamp((grad_mag - 0.08) / (0.30 - 0.08 + 1e-6), 0.0, 1.0)
            unsharp_delta = res_raw - blur
            restored_image = torch.clamp(res_raw + 0.45 * edge_mask * unsharp_delta, 0.0, 1.0)
        else:
            restored_image = torch.clamp(self.unsharp_factor * res_raw + (1.0 - self.unsharp_factor) * blur, 0.0, 1.0)
        
        # --- 2. SEGMENTATION ENCODER ---
        # Decouple optical physical reconstruction from segmentation backprop
        x_norm = (restored_image.detach() - self.norm_mean) / self.norm_std
        curr = x_norm
        f4, f8, f16 = None, None, None
        for i, blk in enumerate(self.backbone_blocks):
            curr = blk(curr)
            if i == 3:   # Stride 4 (24 channels)
                f4 = curr
            elif i == 6: # Stride 8 (40 channels)
                f8 = curr
            elif i == 11:# Stride 16 (112 channels)
                f16 = curr
                
        # --- 3. MULTISCALE DECODER ---
        seg_logits = self.decoder(f4, f8, f16, (h, w))
        
        return restored_image, seg_logits


def build_track2_litev3_model(pretrained_backbone: bool = True, unsharp_factor: float = 1.20, use_edge_gated: bool = True) -> SATEURSLiteV3:
    """Builds and returns a production SAT-EURS-LiteV3 model instance."""
    return SATEURSLiteV3(num_classes=5, unsharp_factor=unsharp_factor, use_edge_gated=use_edge_gated, pretrained_backbone=pretrained_backbone)
