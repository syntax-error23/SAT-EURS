"""
SAT-EURS Track 1 — V21-B Anchored Correction Loss Formulation
Combines:
1. Charbonnier Fidelity Loss (weight 1.0)
2. Structural Similarity Loss (weight 0.30)
3. Grid Spatial Smoothness / TV Loss (weight 0.01)
4. Anchor Regularization Loss (weight 0.05) - keeps correction near identity when V18 is already accurate
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1 Charbonnier loss: sqrt((x - y)^2 + eps^2)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for [B, 3, H, W] images in [0, 1]."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2.0 * sigma**2))
        g = g / g.sum()
        w = torch.outer(g, g).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("window", w)
        self.c1 = (0.01) ** 2
        self.c2 = (0.03) ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.window.to(dtype=pred.dtype, device=pred.device)
        mu1 = F.conv2d(pred, w, padding=self.window_size // 2, groups=3)
        mu2 = F.conv2d(target, w, padding=self.window_size // 2, groups=3)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sig1_sq = F.conv2d(pred * pred, w, padding=self.window_size // 2, groups=3) - mu1_sq
        sig2_sq = F.conv2d(target * target, w, padding=self.window_size // 2, groups=3) - mu2_sq
        sig12 = F.conv2d(pred * target, w, padding=self.window_size // 2, groups=3) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sig12 + self.c2)) / (
            (mu1_sq + mu2_sq + self.c1) * (sig1_sq + sig2_sq + self.c2) + 1e-10
        )
        return 1.0 - torch.mean(ssim_map)


class GridSpatialSmoothnessLoss(nn.Module):
    """L1 Total Variation on 2D affine parameter grid [B, 12, H_g, W_g]."""
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        tv_w = torch.mean(torch.abs(grid[:, :, :, 1:] - grid[:, :, :, :-1]))
        tv_h = torch.mean(torch.abs(grid[:, :, 1:, :] - grid[:, :, :-1, :]))
        return tv_w + tv_h


class AnchoredCorrectionLossV21B(nn.Module):
    """Composite loss for V21-B physics-anchored affine correction."""
    def __init__(
        self,
        w_charb: float = 1.0,
        w_ssim: float = 0.30,
        w_tv: float = 0.01,
        w_anchor: float = 0.05
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_tv = w_tv
        self.w_anchor = w_anchor

        self.charb_fn = CharbonnierLoss()
        self.ssim_fn = SSIMLoss()
        self.tv_fn = GridSpatialSmoothnessLoss()

    def forward(
        self,
        pred_corrected: torch.Tensor,
        target_clean: torch.Tensor,
        raw_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_charb = self.charb_fn(pred_corrected, target_clean)
        l_ssim = self.ssim_fn(pred_corrected, target_clean)
        l_tv = self.tv_fn(raw_grid)
        l_anchor = torch.mean(raw_grid ** 2)

        loss = (
            self.w_charb * l_charb +
            self.w_ssim * l_ssim +
            self.w_tv * l_tv +
            self.w_anchor * l_anchor
        )

        metrics = {
            "loss_charb": l_charb.item(),
            "loss_ssim": l_ssim.item(),
            "loss_tv": l_tv.item(),
            "loss_anchor": l_anchor.item(),
            "loss_total": loss.item()
        }
        return loss, metrics
