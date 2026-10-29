"""
SAT-EURS Track 1 — V21-B Physics-Anchored Restoration Correction Module
Architecture:
- Consumes raw degraded RGB and frozen V18 restored RGB output.
- Predicts a smooth, low-resolution per-cell RGB affine correction grid [24, 32].
- Slices/interpolates smoothly to full resolution.
- Bounded parameterization: delta_A in [-scale_A, scale_A], delta_b in [-scale_b, scale_b] via tanh.
- Exact identity initialization: starts with delta_A = 0, delta_b = 0 (bit-for-bit identical to V18).
- Extremely lightweight: ~63k parameters (246 KB).
"""

from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchoredAffineCorrectionNet(nn.Module):
    """
    Physics-Anchored Affine Correction Network.
    Applies a smooth, low-frequency, bounded affine correction field to the
    frozen V18 restored output:
        I_corrected(x,y) = (I_3 + delta_A(x,y)) @ I_v18(x,y) + delta_b(x,y)
    """
    def __init__(
        self,
        h_grid: int = 24,
        w_grid: int = 32,
        scale_A: float = 0.25,
        scale_b: float = 0.15
    ):
        super().__init__()
        self.h_grid = h_grid
        self.w_grid = w_grid
        self.scale_A = scale_A
        self.scale_b = scale_b

        # 1. Conditioning Subsystem (water type 1..10, depth 5..100)
        self.wt_embed = nn.Embedding(16, 16)
        self.dp_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16)
        )
        self.film = nn.Linear(32, 64 * 2)

        # 2. Low-Resolution Convolutional Feature Extractor on [deg (3ch), v18 (3ch)]
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Affine Correction Grid Output Head (12 channels: 9 for delta A, 3 for delta b)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((h_grid, w_grid)),
            nn.Conv2d(64, 12, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Zero-initializes the output head so correction starts at exact identity."""
        with torch.no_grad():
            last_conv = self.head[-1]
            last_conv.weight.zero_()
            last_conv.bias.zero_()

    def forward(
        self,
        deg_rgb: torch.Tensor,
        v18_rgb: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            deg_rgb: [B, 3, H, W] degraded image in [0, 1]
            v18_rgb: [B, 3, H, W] frozen V18 restored image in [0, 1]
            wt: [B] integer water type in [1..10]
            depth: [B] integer depth in [5..100]
        Returns:
            Tuple of (corrected_rgb [B, 3, H, W], raw_grid [B, 12, h_grid, w_grid])
        """
        B, _, H, W = v18_rgb.shape

        # 1. Conditioning FiLM modulation
        emb_w = self.wt_embed(wt.clamp(1, 10))
        emb_d = self.dp_embed(depth.float().view(-1, 1) / 100.0)
        cond = torch.cat([emb_w, emb_d], dim=-1)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.view(B, 64, 1, 1)
        beta = beta.view(B, 64, 1, 1)

        # 2. Low-res feature extraction (256x256)
        inp_low = F.interpolate(
            torch.cat([deg_rgb, v18_rgb], dim=1),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        feat = self.conv1(inp_low)
        feat = self.conv2(feat)
        feat = feat * (1.0 + gamma) + beta
        feat = self.conv3(feat)

        # 3. Predict smooth low-res affine correction grid [B, 12, h_grid, w_grid]
        raw_grid = self.head(feat)

        # 4. Smoothly upsample affine grid to full resolution [B, 12, H, W]
        grid_up = F.interpolate(raw_grid, size=(H, W), mode='bilinear', align_corners=False)

        # 5. Bounded parameterization
        delta_A = self.scale_A * torch.tanh(grid_up[:, :9]).view(B, 3, 3, H, W)
        delta_b = self.scale_b * torch.tanh(grid_up[:, 9:12])  # [B, 3, H, W]

        # 6. Apply bounded affine correction to V18 base:
        # I_corrected = (I_3 + delta_A) @ I_v18 + delta_b
        # delta_term = delta_A @ I_v18 + delta_b
        delta_term = torch.einsum('bcdhw,bdhw->bchw', delta_A, v18_rgb) + delta_b
        out = (v18_rgb + delta_term).clamp(0.0, 1.0)

        return out, raw_grid
