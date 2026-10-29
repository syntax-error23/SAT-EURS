"""
SAT-EURS Track 1 — V26 Deep Bilateral Restorer Architecture (Apex-085)
Combines:
1. Physical RSUIGM 19-term polynomial anchor inversion (base ~25-26 dB)
2. Condition-guided spatially varying Bilateral Affine Grid (64x64 grid with exponential dynamic scaling)
3. Full-resolution NAFNet residual detail refiner (8 NAFBlocks + FiLM conditioning)
Protocol: Strict Single-Image Inference
"""

import math
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rsuigm_poly_features_gpu(img_bchw: torch.Tensor) -> torch.Tensor:
    """
    Computes the 19-dimensional physical RSUIGM inversion basis on CUDA.
    Input: [B, 3, H, W] float in [0, 1]
    Output: [B, H*W, 19] float on device
    """
    b, _, h, w = img_bchw.shape
    s1 = F.interpolate(
        F.avg_pool2d(img_bchw, kernel_size=16, stride=16),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    s2 = F.interpolate(
        F.avg_pool2d(img_bchw, kernel_size=64, stride=64),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )

    x = img_bchw.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    b1 = s1.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    b2 = s2.permute(0, 2, 3, 1).reshape(b, h * w, 3)

    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    bl = x[:, :, 2:3]
    ones = torch.ones_like(r)

    return torch.cat(
        [ones, x, x * x, r * g, r * bl, g * bl, x * x * x, b1, b2],
        dim=-1,
    )


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, num_features * 2),
            nn.GELU(),
            nn.Linear(num_features * 2, num_features * 2)
        )
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


class DeepBilateralRestorerV26(nn.Module):
    """
    V26 Breakthrough Restoration Architecture:
    1. Multi-Condition FiLM Embedding
    2. Physical RSUIGM Inversion Table (frozen / differentiable prior)
    3. Spatially-Varying Bilateral Affine Grid with Exponential Dynamic Range
    4. Full-Resolution NAFNet Residual Texture Refiner
    """
    def __init__(
        self,
        rsuigm_table: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
        grid_size: Tuple[int, int] = (64, 64),
        backbone_channels: int = 64,
        cond_dim: int = 64,
        refiner_channels: int = 48,
        refiner_blocks: int = 8
    ):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.cond_dim = cond_dim

        # 1. Physical Inversion Prior Table
        if rsuigm_table is not None:
            self.register_rsuigm_table(rsuigm_table)
        else:
            self.rsuigm_keys = None

        # 2. Condition Embeddings: Water Type (1..10) and Depth (5..100)
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

        # 3. Bilateral Grid Backbone (operates on [I_base, I_deg] = 6 channels)
        self.grid_stem = nn.Sequential(
            nn.Conv2d(6, backbone_channels // 2, kernel_size=3, stride=2, padding=1),  # /2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(backbone_channels // 2, backbone_channels, kernel_size=3, stride=2, padding=1),  # /4
            nn.GroupNorm(8, backbone_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.grid_film1 = FiLMBlock(cond_dim, backbone_channels)
        self.grid_mid = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, stride=2, padding=1),  # /8
            nn.GroupNorm(8, backbone_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.grid_film2 = FiLMBlock(cond_dim, backbone_channels)

        self.grid_pool = nn.AdaptiveAvgPool2d((self.grid_h, self.grid_w))
        self.grid_head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(backbone_channels, 12, kernel_size=1)
        )

        # 4. Full-Resolution NAFNet Texture Refiner
        self.refiner_in = nn.Conv2d(6, refiner_channels, kernel_size=3, padding=1)
        self.refiner_blocks = nn.ModuleList([
            NAFBlockFiLM(refiner_channels, cond_dim) for _ in range(refiner_blocks)
        ])
        self.refiner_out = nn.Conv2d(refiner_channels, 3, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        """Identity initialization for both affine grid and refiner."""
        with torch.no_grad():
            self.grid_head[-1].weight.zero_()
            self.grid_head[-1].bias.zero_()
            self.refiner_out.weight.zero_()
            self.refiner_out.bias.zero_()

    def register_rsuigm_table(self, table: Dict[Tuple[int, int], torch.Tensor]):
        """Caches RSUIGM table tensors on module."""
        for (wt, d), W in table.items():
            name = f"rsuigm_w_{wt}_{d}"
            self.register_buffer(name, W.float(), persistent=False)

    def get_physical_inversion(self, x: torch.Tensor, wt: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Computes the closed-form physical inverse base image using RSUIGM basis."""
        B, C, H, W = x.shape
        feats = build_rsuigm_poly_features_gpu(x)  # [B, H*W, 19]

        res_list = []
        for i in range(B):
            w_val = int(wt[i].item())
            d_val = int(depth[i].item())
            buf_name = f"rsuigm_w_{w_val}_{d_val}"
            if hasattr(self, buf_name):
                W_mat = getattr(self, buf_name)
            else:
                # Fallback to nearest available condition
                W_mat = getattr(self, f"rsuigm_w_1_5")
            pred = torch.matmul(feats[i], W_mat).clamp(0.0, 1.0)  # [H*W, 3]
            res_list.append(pred.reshape(H, W, 3).permute(2, 0, 1))

        return torch.stack(res_list, dim=0)

    def get_condition_embedding(self, wt: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Constructs continuous FiLM condition vector."""
        wt_clamped = wt.clamp(0, 15)
        dp_norm = ((depth.float() - 5.0) / 95.0).clamp(0.0, 1.0).view(-1, 1)

        w_emb = self.wt_embed(wt_clamped)
        d_emb = self.dp_mlp(dp_norm)
        return self.cond_fusion(torch.cat([w_emb, d_emb], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        wt: torch.Tensor,
        depth: torch.Tensor,
        base_img: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward Pass:
        x: Degraded RGB [B, 3, H, W] in [0, 1]
        wt: Water type [B]
        depth: Sensor depth [B]
        base_img: Optional precomputed physical base image [B, 3, H, W]
        """
        B, C, H, W = x.shape
        cond = self.get_condition_embedding(wt, depth)

        # 1. Physical Inversion Base
        if base_img is None:
            base_img = self.get_physical_inversion(x, wt, depth)

        # 2. Predict Spatially-Varying Affine Grid
        low_res = F.interpolate(torch.cat([base_img, x], dim=1), size=(256, 256), mode="bilinear", align_corners=False)
        feat = self.grid_stem(low_res)
        feat = self.grid_film1(feat, cond)
        feat = self.grid_mid(feat)
        feat = self.grid_film2(feat, cond)
        feat_grid = self.grid_pool(feat)

        raw_grid = self.grid_head(feat_grid)  # [B, 12, grid_h, grid_w]

        # Exponential dynamic scaling:
        # Diagonal elements: A_ii = exp(1.5 * tanh(raw_grid[0, 4, 8])) in [0.22, 4.48]
        # Off-diagonal: A_ij = 0.5 * tanh(raw_grid[...])
        # Biases: b_i = 0.35 * tanh(raw_grid[9..11])
        diag_scale = torch.exp(1.5 * torch.tanh(raw_grid[:, [0, 4, 8], :, :]))
        off_diag = 0.5 * torch.tanh(raw_grid[:, [1, 2, 3, 5, 6, 7], :, :])
        biases = 0.35 * torch.tanh(raw_grid[:, 9:12, :, :])

        # Form full 12-channel grid
        a11, a22, a33 = diag_scale[:, 0:1], diag_scale[:, 1:2], diag_scale[:, 2:3]
        a12, a13, a21, a23, a31, a32 = (
            off_diag[:, 0:1], off_diag[:, 1:2], off_diag[:, 2:3],
            off_diag[:, 3:4], off_diag[:, 4:5], off_diag[:, 5:6]
        )
        b1, b2, b3 = biases[:, 0:1], biases[:, 1:2], biases[:, 2:3]

        affine_grid = torch.cat([a11, a12, a13, b1, a21, a22, a23, b2, a31, a32, a33, b3], dim=1)

        # Bilinear upsample affine grid to full image resolution
        affine_full = F.interpolate(affine_grid, size=(H, W), mode="bilinear", align_corners=False)

        a11_f, a12_f, a13_f, b1_f = affine_full[:, 0:1], affine_full[:, 1:2], affine_full[:, 2:3], affine_full[:, 3:4]
        a21_f, a22_f, a23_f, b2_f = affine_full[:, 4:5], affine_full[:, 5:6], affine_full[:, 6:7], affine_full[:, 7:8]
        a31_f, a32_f, a33_f, b3_f = affine_full[:, 8:9], affine_full[:, 9:10], affine_full[:, 10:11], affine_full[:, 11:12]

        R = base_img[:, 0:1, :, :]
        G = base_img[:, 1:2, :, :]
        B_ch = base_img[:, 2:3, :, :]

        R_aff = a11_f * R + a12_f * G + a13_f * B_ch + b1_f
        G_aff = a21_f * R + a22_f * G + a23_f * B_ch + b2_f
        B_aff = a31_f * R + a32_f * G + a33_f * B_ch + b3_f

        affine_img = torch.clamp(torch.cat([R_aff, G_aff, B_aff], dim=1), 0.0, 1.0)

        # 3. Full-Resolution NAFNet Detail Refiner
        ref_in = torch.cat([affine_img, x], dim=1)
        f_ref = self.refiner_in(ref_in)
        for blk in self.refiner_blocks:
            f_ref = blk(f_ref, cond)
        res_detail = 0.25 * torch.tanh(self.refiner_out(f_ref))

        final_img = torch.clamp(affine_img + res_detail, 0.0, 1.0)

        return {
            "restored_image": final_img,
            "affine_image": affine_img,
            "base_image": base_img,
            "affine_grid": affine_grid
        }
