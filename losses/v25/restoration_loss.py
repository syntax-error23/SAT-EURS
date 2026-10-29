"""
SAT-EURS Track 1 — V25 Restoration Loss Formulation
Combines:
1. Charbonnier Pixel Reconstruction Loss (L1-like robust metric)
2. Structural Similarity (SSIM) Loss
3. Laplacian High-Frequency Detail Loss
4. Spatial Total Variation Loss on Affine Parameter Maps (prevents grid discontinuities)
"""

from typing import Dict, Tuple, Optional

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


class LaplacianEdgeLoss(nn.Module):
    """High-frequency edge difference using 3x3 Laplacian operator."""
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0.0, 1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0, 1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.register_buffer("lap_kernel", lap)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.lap_kernel.to(dtype=pred.dtype, device=pred.device)
        edge_pred = F.conv2d(pred, k, padding=1, groups=3)
        edge_target = F.conv2d(target, k, padding=1, groups=3)
        return F.l1_loss(edge_pred, edge_target)


class GridSpatialTVLoss(nn.Module):
    """Total Variation loss on spatial grid to enforce spatial smoothness."""
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # grid: [B, C, H, W]
        tv_h = torch.mean(torch.abs(grid[:, :, 1:, :] - grid[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(grid[:, :, :, 1:] - grid[:, :, :, :-1]))
        return tv_h + tv_w


class RestorationLossV25(nn.Module):
    def __init__(
        self,
        w_charb: float = 1.0,
        w_ssim: float = 0.20,
        w_lap: float = 0.05,
        w_tv: float = 1e-4
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_lap = w_lap
        self.w_tv = w_tv

        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.lap = LaplacianEdgeLoss()
        self.tv = GridSpatialTVLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_charb = self.charb(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_lap = self.lap(pred, target)

        total_loss = self.w_charb * loss_charb + self.w_ssim * loss_ssim + self.w_lap * loss_lap

        loss_dict = {
            "loss_charb": loss_charb.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_lap": loss_lap.item(),
        }

        if grid is not None and self.w_tv > 0:
            loss_tv = self.tv(grid)
            total_loss = total_loss + self.w_tv * loss_tv
            loss_dict["loss_tv"] = loss_tv.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
