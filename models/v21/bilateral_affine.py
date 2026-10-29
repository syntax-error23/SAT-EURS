"""
SAT-EURS Track 1 — V21 ConditionedBilateralAffineNet
Architecture:
- Native PyTorch Condition-Aware Per-Cell Bilateral Affine Color Restorer
- Conditioning on Water Type (categorical embedding) and Depth (continuous projection)
- Low-resolution convolutional backbone predicting a 3D bilateral grid of affine transforms
- Full-resolution luminance guidance for edge-preserving bilateral slicing
- Exact identity-preserving initialization (starts at zero reconstruction error)
- Zero external dependencies; fully compliant with Windows PyTorch environment
"""

import math
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionedBilateralAffineNet(nn.Module):
    """
    Conditioned Bilateral Affine Network for single-image underwater color restoration.
    Predicts a spatially smooth 3D grid of local 3x3 affine transforms and 3x1 biases,
    conditioned on water type and water depth, and slices it at full resolution using
    edge-preserving luminance guidance.
    """
    def __init__(
        self,
        d_grid: int = 8,
        h_grid: int = 24,
        w_grid: int = 32,
        num_water_types: int = 16,
        cond_dim: int = 64
    ):
        super().__init__()
        self.d_grid = d_grid
        self.h_grid = h_grid
        self.w_grid = w_grid
        self.cond_dim = cond_dim

        # 1. Conditioning Subsystem
        half_cond = cond_dim // 2
        self.wt_embed = nn.Embedding(num_water_types, half_cond)
        self.dp_embed = nn.Sequential(
            nn.Linear(1, half_cond),
            nn.GELU(),
            nn.Linear(half_cond, half_cond)
        )
        self.film = nn.Linear(cond_dim, 128 * 2)

        # 2. Low-Resolution Convolutional Backbone (operates on 256x256)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Bilateral Grid Output Head (predicts 12 * d_grid channels)
        out_channels = 12 * d_grid
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((h_grid, w_grid)),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )

        # 4. Full-Resolution Guidance Head (1x1 conv mapping RGB to scalar guide)
        self.guide_conv = nn.Conv2d(3, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initializes the network to an exact identity transform."""
        with torch.no_grad():
            # Guidance head: Standard grayscale luminance coefficients
            self.guide_conv.weight.copy_(torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]]))
            self.guide_conv.bias.zero_()

            # Output head: Zero out weights and initialize bias to identity matrix
            # Channel ordering per cell: [A11, A12, A13, b1, A21, A22, A23, b2, A31, A32, A33, b3]
            # Diagonal elements are: A11 (c=0), A22 (c=5), A33 (c=10)
            last_conv = self.head[-1]
            last_conv.weight.zero_()
            last_conv.bias.zero_()
            for d in range(self.d_grid):
                last_conv.bias.data[0 * self.d_grid + d] = 1.0   # A11
                last_conv.bias.data[5 * self.d_grid + d] = 1.0   # A22
                last_conv.bias.data[10 * self.d_grid + d] = 1.0  # A33

    def forward(
        self,
        x: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor,
        return_grid: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Degraded RGB tensor [B, 3, H, W] in [0, 1]
            wt: Water type tensor [B] (integer 1..10)
            depth: Depth tensor [B] (integer 5..100)
            return_grid: If True, returns the 3D bilateral grid for regularization
        Returns:
            Dict containing 'restored_image' [B, 3, H, W] and optionally 'grid'
        """
        B, _, H, W = x.shape

        # 1. Conditioning Embedding
        emb_w = self.wt_embed(wt)
        emb_d = self.dp_embed(depth.float().view(-1, 1) / 100.0)
        cond = torch.cat([emb_w, emb_d], dim=-1)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.view(B, 128, 1, 1)
        beta = beta.view(B, 128, 1, 1)

        # 2. Low-Resolution Feature Extraction (256x256)
        x_low = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        feat = self.stem(x_low)
        feat = self.b1(feat)
        feat = self.b2(feat)
        feat = feat * (1.0 + gamma) + beta
        feat = self.b3(feat)

        # 3. Bilateral Grid Prediction [B, 12, d_grid, h_grid, w_grid]
        grid_flat = self.head(feat)
        grid = grid_flat.view(B, 12, self.d_grid, self.h_grid, self.w_grid)

        # 4. Full-Resolution Guidance Map [B, 1, H, W] normalized to [-1, 1]
        guide = self.guide_conv(x).clamp(0.0, 1.0)
        gz = 2.0 * guide - 1.0

        # 5. Bilateral Query Coordinates [B, 1, H, W, 3]
        gx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        gy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        query = torch.stack([gx, gy, gz], dim=-1)

        # 6. Bilateral Slicing via 5D Trilinear Grid Sample
        # Output coeffs: [B, 12, H, W]
        coeffs = F.grid_sample(grid, query, mode='bilinear', padding_mode='border', align_corners=True).squeeze(2)

        # 7. Apply Full-Resolution Affine Color Correction
        A = coeffs.view(B, 3, 4, H, W)
        ones = torch.ones(B, 1, H, W, dtype=x.dtype, device=x.device)
        x_homo = torch.cat([x, ones], dim=1).unsqueeze(1) # [B, 1, 4, H, W]
        restored = (A * x_homo).sum(dim=2).clamp(0.0, 1.0)

        out = {"restored_image": restored}
        if return_grid:
            out["grid"] = grid
            out["coeffs"] = coeffs
        return out
