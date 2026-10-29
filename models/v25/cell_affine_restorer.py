"""
SAT-EURS Track 1 — V25 Conditioned Cell-Affine Restoration Network (R1 & R2)
Protocol Compliance: Strict Single-Image Inference
Inputs per forward pass: (x: [B, 3, H, W], wt: [B], depth: [B])

Architectural Details:
- R1 (Conditioned Cell-Affine Restorer):
  Predicts spatially varying 3x4 affine color transforms at a coarse grid (e.g. 32x32),
  conditioned on physical water type and depth via continuous FiLM modulation.
  Identity-preserving initialization (starts exactly at identity transform A=I, b=0).
  Smooth bilinear upsampling to full image resolution.
- R2 (Conditioned Cell-Affine + Full-Resolution Detail Refiner):
  Augments R1 with a full-resolution residual detail network that processes
  concatenated [I_affine, I_deg] conditioned on metadata embeddings.
  Zero-initialized residual head (starts exactly at Delta_I = 0).
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, num_features * 2),
            nn.GELU(),
            nn.Linear(num_features * 2, num_features * 2)
        )
        # Initialize scale to 0 (so 1 + gamma = 1) and shift to 0
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlockFiLM(nn.Module):
    """NAFBlock augmented with FiLM metadata conditioning."""
    def __init__(self, c: int, cond_dim: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True)
        )

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)

        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)

        self.film = FiLMBlock(cond_dim, c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.film(x, cond)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma


class ConditionedCellAffineRestorer(nn.Module):
    """
    R1 and R2 Unified Model:
    - R1: use_detail_refiner=False (only Conditioned Cell-Affine Color Restorer)
    - R2: use_detail_refiner=True  (Conditioned Cell-Affine + Full-Resolution Detail Refiner)
    """
    def __init__(
        self,
        grid_size: Tuple[int, int] = (32, 32),
        backbone_channels: int = 64,
        cond_dim: int = 64,
        use_detail_refiner: bool = False,
        refiner_channels: int = 32,
        refiner_blocks: int = 4
    ):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.cond_dim = cond_dim
        self.use_detail_refiner = use_detail_refiner

        # 1. Conditioning Embeddings
        half_dim = cond_dim // 2
        self.wt_embed = nn.Embedding(16, half_dim)
        self.dp_mlp = nn.Sequential(
            nn.Linear(1, half_dim),
            nn.GELU(),
            nn.Linear(half_dim, half_dim)
        )
        self.cond_fusion = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # 2. Coarse Feature Backbone for Affine Parameter Prediction
        # Operates on downsampled 256x256 image to predict smooth spatial grid
        self.down_stem = nn.Sequential(
            nn.Conv2d(3, backbone_channels // 2, kernel_size=3, stride=2, padding=1),  # 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(backbone_channels // 2, backbone_channels, kernel_size=3, stride=2, padding=1),  # 64
            nn.GroupNorm(8, backbone_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.film_b1 = FiLMBlock(cond_dim, backbone_channels)
        self.mid_conv = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, stride=2, padding=1),  # 32
            nn.GroupNorm(8, backbone_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.film_b2 = FiLMBlock(cond_dim, backbone_channels)

        # Affine Grid Output Head: predicts 12 channels per cell:
        # [A11, A12, A13, b1, A21, A22, A23, b2, A31, A32, A33, b3]
        self.grid_pool = nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w))
        self.affine_head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(backbone_channels, 12, kernel_size=1)
        )

        # 3. Optional R2 Full-Resolution Detail Refiner
        if self.use_detail_refiner:
            self.refiner_in = nn.Conv2d(6, refiner_channels, kernel_size=3, padding=1)
            self.refiner_blocks = nn.ModuleList([
                NAFBlockFiLM(refiner_channels, cond_dim) for _ in range(refiner_blocks)
            ])
            self.refiner_out = nn.Conv2d(refiner_channels, 3, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Strict identity-preserving initialization with bounded parameterization."""
        with torch.no_grad():
            # Zero-init affine head weights and biases so initially Delta A = 0, b = 0
            final_affine_conv = self.affine_head[-1]
            final_affine_conv.weight.zero_()
            final_affine_conv.bias.zero_()

            # Zero-init detail refiner if present
            if self.use_detail_refiner:
                self.refiner_out.weight.zero_()
                self.refiner_out.bias.zero_()

    def get_condition_embedding(self, wt: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Constructs fused metadata condition vector."""
        # wt in [1..10], depth in [5..100]
        wt_clamped = wt.clamp(0, 15)
        dp_norm = ((depth.float() - 5.0) / 95.0).clamp(0.0, 1.0).view(-1, 1)

        w_emb = self.wt_embed(wt_clamped)
        d_emb = self.dp_mlp(dp_norm)
        cond = self.cond_fusion(torch.cat([w_emb, d_emb], dim=-1))
        return cond

    def forward(
        self,
        x: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Degraded RGB tensor [B, 3, H, W] in [0, 1]
            wt: Water type tensor [B] (int)
            depth: Depth tensor [B] (int)
        Returns:
            Dict containing:
            - 'restored_image': Final output [B, 3, H, W] in [0, 1]
            - 'affine_image': Coarse cell-affine output [B, 3, H, W] in [0, 1]
            - 'affine_grid': Affine parameter maps [B, 12, grid_h, grid_w]
        """
        B, _, H, W = x.shape

        # 1. Condition Vector
        cond = self.get_condition_embedding(wt, depth)

        # 2. Downsample input to fixed 256x256 for fast, stable cell-affine prediction
        x_low = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        feat = self.down_stem(x_low)
        feat = self.film_b1(feat, cond)
        feat = self.mid_conv(feat)
        feat = self.film_b2(feat, cond)
        feat_grid = self.grid_pool(feat)

        # Affine Grid [B, 12, grid_h, grid_w]
        raw_grid = self.affine_head(feat_grid)

        # Bounded parameterization:
        # Predict delta_A in [-1.0, 1.0] and bias b in [-0.25, 0.25]
        # Diagonal elements: A11 = 1 + delta_A[0], A22 = 1 + delta_A[4], A33 = 1 + delta_A[8]
        # Off-diagonal elements: A_ij = delta_A[k]
        delta_A = 1.0 * torch.tanh(raw_grid[:, 0:9, :, :])
        bias_b  = 0.25 * torch.tanh(raw_grid[:, 9:12, :, :])

        # Form full 12-channel affine parameter map
        # Order: [A11, A12, A13, b1, A21, A22, A23, b2, A31, A32, A33, b3]
        a11 = 1.0 + delta_A[:, 0:1, :, :]
        a12 = delta_A[:, 1:2, :, :]
        a13 = delta_A[:, 2:3, :, :]
        b1  = bias_b[:, 0:1, :, :]

        a21 = delta_A[:, 3:4, :, :]
        a22 = 1.0 + delta_A[:, 4:5, :, :]
        a23 = delta_A[:, 5:6, :, :]
        b2  = bias_b[:, 1:2, :, :]

        a31 = delta_A[:, 6:7, :, :]
        a32 = delta_A[:, 7:8, :, :]
        a33 = 1.0 + delta_A[:, 8:9, :, :]
        b3  = bias_b[:, 2:3, :, :]

        affine_grid = torch.cat([a11, a12, a13, b1, a21, a22, a23, b2, a31, a32, a33, b3], dim=1)

        # 3. Bilinear Upsampling to Full Resolution [B, 12, H, W]
        affine_full = F.interpolate(affine_grid, size=(H, W), mode="bilinear", align_corners=False)

        a11_f = affine_full[:, 0:1, :, :]
        a12_f = affine_full[:, 1:2, :, :]
        a13_f = affine_full[:, 2:3, :, :]
        b1_f  = affine_full[:, 3:4, :, :]

        a21_f = affine_full[:, 4:5, :, :]
        a22_f = affine_full[:, 5:6, :, :]
        a23_f = affine_full[:, 6:7, :, :]
        b2_f  = affine_full[:, 7:8, :, :]

        a31_f = affine_full[:, 8:9, :, :]
        a32_f = affine_full[:, 9:10, :, :]
        a33_f = affine_full[:, 10:11, :, :]
        b3_f  = affine_full[:, 11:12, :, :]

        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B_ch = x[:, 2:3, :, :]

        R_aff = a11_f * R + a12_f * G + a13_f * B_ch + b1_f
        G_aff = a21_f * R + a22_f * G + a23_f * B_ch + b2_f
        B_aff = a31_f * R + a32_f * G + a33_f * B_ch + b3_f

        affine_img = torch.clamp(torch.cat([R_aff, G_aff, B_aff], dim=1), 0.0, 1.0)

        # 4. Optional R2 Detail Refinement
        if self.use_detail_refiner:
            ref_in = torch.cat([affine_img, x], dim=1)  # [B, 6, H, W]
            f_ref = self.refiner_in(ref_in)
            for blk in self.refiner_blocks:
                f_ref = blk(f_ref, cond)
            res_detail = 0.25 * torch.tanh(self.refiner_out(f_ref))
            final_img = torch.clamp(affine_img + res_detail, 0.0, 1.0)
        else:
            final_img = affine_img

        out = {
            "restored_image": final_img,
            "affine_image": affine_img,
            "affine_grid": affine_grid
        }
        return out


def build_r1_cell_affine(grid_size: Tuple[int, int] = (32, 32)) -> ConditionedCellAffineRestorer:
    """Builds R1 (Standalone Conditioned Cell-Affine Model)."""
    return ConditionedCellAffineRestorer(
        grid_size=grid_size,
        backbone_channels=64,
        cond_dim=64,
        use_detail_refiner=False
    )


def build_r2_cell_affine_refiner(
    grid_size: Tuple[int, int] = (32, 32),
    refiner_channels: int = 32,
    refiner_blocks: int = 4
) -> ConditionedCellAffineRestorer:
    """Builds R2 (Conditioned Cell-Affine + Full-Resolution Detail Refiner)."""
    return ConditionedCellAffineRestorer(
        grid_size=grid_size,
        backbone_channels=64,
        cond_dim=64,
        use_detail_refiner=True,
        refiner_channels=refiner_channels,
        refiner_blocks=refiner_blocks
    )
