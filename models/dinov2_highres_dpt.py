"""
DINOv2HighResDPT: High-Resolution Vision Transformer with Progressive Detail Adapter
Stage 3 of SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Key Architectural Features:
1. Backbone: DINOv2-ViT-L/14 with blocks 0..15 frozen to preserve universal features.
   Fine-tunes top blocks 16..23 at lr=1.5e-5.
2. High-Res RGB Detail Stem:
   - Stride-2 stage (64 ch at H/2 x W/2): Conv3x3-GN-GELU.
   - Stride-4 stage (128 ch at H/4 x W/4): Conv3x3-GN-GELU.
3. Top-Down DPT Progressive Fusion:
   - Multi-layer DINOv2 reassembly (blocks 5, 11, 17, 23).
   - Fusion at Stride 4 with Depthwise-Separable Conv block.
   - Fusion at Stride 2 with Conv refinement block.
   - Full resolution prediction at native input resolution.
4. Warm-Start Zero-Residual Guarantee:
   - Loads existing 5-fold V6 checkpoints.
   - Zero-initializes residual detail projection so Step 0 matches baseline identically.
"""

import math
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))

class HighResRGBStem(nn.Module):
    """Parallel RGB Detail Stem extracting Stride-2 and Stride-4 edge representations."""
    def __init__(self, in_channels: int = 3, feat2_dim: int = 64, feat4_dim: int = 128):
        super().__init__()
        self.stage2 = nn.Sequential(
            ConvGNAct(in_channels, feat2_dim, kernel_size=3, stride=2, padding=1),
            ConvGNAct(feat2_dim, feat2_dim, kernel_size=3, stride=1, padding=1)
        )
        self.stage4 = nn.Sequential(
            ConvGNAct(feat2_dim, feat4_dim, kernel_size=3, stride=2, padding=1),
            ConvGNAct(feat4_dim, feat4_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f2 = self.stage2(x)
        f4 = self.stage4(f2)
        return f2, f4

class DINOv2HighResDPT(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        freeze_blocks_up_to: int = 15,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_blocks_up_to = freeze_blocks_up_to

        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=pretrained)
        embed_dim = self.encoder.embed_dim  # 1024
        self.intermediate_blocks = (5, 11, 17, 23)

        # Base 1x1 reassembly projections (strictly matching V6 checkpoint shapes)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in self.intermediate_blocks
        ])

        # Base fusion head (layers 0..2: Conv(1024,256), layers 3..5: Conv(256,128), layer 6: Conv(128,5))
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * len(self.intermediate_blocks), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        # High-Res Conv Detail Stem
        self.stem = HighResRGBStem(in_channels=3, feat2_dim=64, feat4_dim=128)

        # Progressive DPT Decoders with Depthwise-Separable Convolutions
        self.fuse_stage4 = nn.Sequential(
            DepthwiseSeparableConv(128 + 128, 128),
            ConvGNAct(128, 96, kernel_size=3, padding=1)
        )
        self.fuse_stage2 = nn.Sequential(
            DepthwiseSeparableConv(96 + 64, 64),
            ConvGNAct(64, 48, kernel_size=3, padding=1)
        )

        # High-Res Zero-Initialized Residual Projection
        self.detail_head = nn.Conv2d(48, num_classes, kernel_size=1)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)

        # Freeze early blocks
        self._freeze_early_blocks()

    def _freeze_early_blocks(self):
        # Freeze patch embedding and cls token
        for p in self.encoder.patch_embed.parameters():
            p.requires_grad = False
        if hasattr(self.encoder, "cls_token"):
            self.encoder.cls_token.requires_grad = False
        if hasattr(self.encoder, "pos_embed"):
            self.encoder.pos_embed.requires_grad = False

        # Freeze blocks up to freeze_blocks_up_to
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.freeze_blocks_up_to:
                for p in blk.parameters():
                    p.requires_grad = False
            else:
                for p in blk.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_feat, w_feat = H // 14, W // 14

        # 1. ViT Multi-Level Features
        vit_feats = self.encoder.get_intermediate_layers(x, n=self.intermediate_blocks, return_class_token=False)
        projected = []
        for feat, p in zip(vit_feats, self.proj):
            f_map = feat.permute(0, 2, 1).reshape(B, -1, h_feat, w_feat)
            projected.append(p(f_map))
        cat_feat = torch.cat(projected, dim=1)  # [B, 1024, H/14, W/14]

        # 2. Base Model Coarse Logits
        z_256 = self.fusion[0:3](cat_feat)      # [B, 256, H/14, W/14]
        z_128 = self.fusion[3:6](z_256)        # [B, 128, H/14, W/14]
        base_logits_low = self.fusion[6](z_128) # [B, 5, H/14, W/14]
        base_logits = F.interpolate(base_logits_low, size=(H, W), mode="bilinear", align_corners=False)

        # 3. High-Res RGB Detail Stem
        f2, f4 = self.stem(x)  # f2: [B, 64, H/2, W/2], f4: [B, 128, H/4, W/4]

        # 4. Progressive Top-Down Fusion
        z_up4 = F.interpolate(z_128, size=f4.shape[2:], mode="bilinear", align_corners=False)
        fuse4 = self.fuse_stage4(torch.cat([z_up4, f4], dim=1))

        fuse4_up2 = F.interpolate(fuse4, size=f2.shape[2:], mode="bilinear", align_corners=False)
        fuse2 = self.fuse_stage2(torch.cat([fuse4_up2, f2], dim=1))

        # 5. Detail Residual Correction
        detail_low = self.detail_head(fuse2)
        detail_logits = F.interpolate(detail_low, size=(H, W), mode="bilinear", align_corners=False)

        return base_logits + detail_logits
