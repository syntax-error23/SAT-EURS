"""
DINOv2HighResDecoder Module (Experiment 3.1)
MaCVi @ WACV 2027 Challenge (Track 1)

Architecture:
1. Base Backbone: Pretrained DINOv2-ViT-L/14 with multi-level feature reassembly (Blocks 5, 11, 17, 23).
2. Parallel High-Resolution RGB Detail Stem:
   - Stride-2 stage (48 channels at H/2 x W/2)
   - Stride-4 stage (96 channels at H/4 x W/4)
3. Progressive Top-Down Fusion:
   - Fuses ViT semantically rich H/14 features with Stride-4 detail features.
   - Fuses with Stride-2 detail features.
   - High-Res Residual Projection head initialized to ZERO.
4. Mathematical Guarantee:
   - At step 0 (before fine-tuning), the zero-initialized projection outputs exactly 0.
   - Total Logits = Base Logits + 0 = Base Logits identically.
   - Zero catastrophic forgetting or regression at initialization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighResRGBStem(nn.Module):
    """
    Lightweight convolutional stem extracting high-frequency edge and texture details
    directly from the input RGB image at Stride 2 and Stride 4.
    """
    def __init__(self, in_channels: int = 3, feat2_dim: int = 48, feat4_dim: int = 96):
        super().__init__()
        # Stride 2: H/2 x W/2
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels, feat2_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat2_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat2_dim, feat2_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat2_dim),
            nn.ReLU(inplace=True)
        )
        # Stride 4: H/4 x W/4
        self.stage4 = nn.Sequential(
            nn.Conv2d(feat2_dim, feat4_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat4_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat4_dim, feat4_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat4_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        f2 = self.stage2(x)
        f4 = self.stage4(f2)
        return f2, f4

class DINOv2HighResDecoder(nn.Module):
    """
    Full Architecture with Pretrained DINOv2 Backbone + Parallel Detail Stem + Zero-Init Fusion.
    """
    def __init__(self, num_classes: int = 5, pretrained_backbone: bool = True):
        super().__init__()
        if pretrained_backbone:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
        else:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)

        embed_dim = self.encoder.embed_dim  # 1024
        self.intermediate_blocks = (5, 11, 17, 23)

        # Base 1x1 reassembly projections (matches checkpoints/track1_screening exactly)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in self.intermediate_blocks
        ])

        # Base fusion head (layers 0..2: Conv(1024,256)+BN+ReLU; layers 3..5: Conv(256,128)+BN+ReLU; layer 6: Conv(128,5,1))
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * len(self.intermediate_blocks), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

        # High-Resolution Stem
        self.stem = HighResRGBStem(in_channels=3, feat2_dim=48, feat4_dim=96)

        # Progressive Fusion Decoders
        self.fuse_stage4 = nn.Sequential(
            nn.Conv2d(128 + 96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.fuse_stage2 = nn.Sequential(
            nn.Conv2d(96 + 48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Residual High-Res Projection (Zero-Initialized)
        self.zero_conv = nn.Conv2d(48, num_classes, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_feat, w_feat = H // 14, W // 14

        # 1. ViT Backbone multi-level feature extraction
        vit_feats = self.encoder.get_intermediate_layers(x, n=self.intermediate_blocks, return_class_token=False)
        projected = []
        for feat, p in zip(vit_feats, self.proj):
            f_map = feat.permute(0, 2, 1).reshape(B, -1, h_feat, w_feat)
            projected.append(p(f_map))
        cat_feat = torch.cat(projected, dim=1)  # [B, 1024, H/14, W/14]

        # 2. Base Model Forward
        # Intermediate feature at layer 5 (128 channels)
        z_256 = self.fusion[0:3](cat_feat)      # [B, 256, H/14, W/14]
        z_128 = self.fusion[3:6](z_256)        # [B, 128, H/14, W/14]
        base_logits_low = self.fusion[6](z_128) # [B, num_classes, H/14, W/14]
        base_logits = F.interpolate(base_logits_low, size=(H, W), mode="bilinear", align_corners=False)

        # 3. Parallel High-Res RGB Detail Stem
        f2, f4 = self.stem(x)  # f2: [B, 48, H/2, W/2], f4: [B, 96, H/4, W/4]

        # 4. Progressive Detail Fusion
        # Fuse z_128 upsampled to Stride 4
        z_up4 = F.interpolate(z_128, size=f4.shape[2:], mode="bilinear", align_corners=False)
        fuse4 = self.fuse_stage4(torch.cat([z_up4, f4], dim=1))

        # Fuse fuse4 upsampled to Stride 2
        fuse4_up2 = F.interpolate(fuse4, size=f2.shape[2:], mode="bilinear", align_corners=False)
        fuse2 = self.fuse_stage2(torch.cat([fuse4_up2, f2], dim=1))

        # 5. Zero-Initialized Residual Correction
        stem_logits_stride2 = self.zero_conv(fuse2)  # [B, num_classes, H/2, W/2]
        stem_logits = F.interpolate(stem_logits_stride2, size=(H, W), mode="bilinear", align_corners=False)

        # 6. Combined Prediction
        return base_logits + stem_logits
