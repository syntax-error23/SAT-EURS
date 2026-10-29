"""
SAT-EURS Track 1 — V21 Restoration Loss Formulation
Combines:
1. Charbonnier Fidelity Loss (weight 1.0)
2. Multi-Channel Structural Similarity Loss (weight 0.30)
3. Laplacian Edge Gradient Loss (weight 0.10)
4. Bilateral Grid Smoothness / Total Variation Loss (weight 1e-3)

Notice: Free from artificial identity-penalties (||A - I||^2) to allow complete
recovery of underwater dynamic range and deep color restoration.
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
        lap_p = F.conv2d(pred, k, padding=1, groups=3)
        lap_t = F.conv2d(target, k, padding=1, groups=3)
        return F.l1_loss(lap_p, lap_t)


class GridTotalVariationLoss(nn.Module):
    """
    Total variation loss on the 3D bilateral grid [B, 12, D, H, W]
    Enforces spatial and range smoothness across neighboring cells.
    """
    def __init__(self):
        super().__init__()

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        # Differences along width (dim 4), height (dim 3), and depth (dim 2)
        tv_w = torch.mean(torch.abs(grid[..., :, :, 1:] - grid[..., :, :, :-1]))
        tv_h = torch.mean(torch.abs(grid[..., :, 1:, :] - grid[..., :, :-1, :]))
        tv_d = torch.mean(torch.abs(grid[..., 1:, :, :] - grid[..., :-1, :, :]))
        return tv_w + tv_h + tv_d


class RestorationLossV21(nn.Module):
    """Composite loss function for V21 restoration training."""
    def __init__(
        self,
        w_charb: float = 1.0,
        w_ssim: float = 0.30,
        w_lap: float = 0.10,
        w_grid_tv: float = 1e-3
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_lap = w_lap
        self.w_grid_tv = w_grid_tv

        self.charb_fn = CharbonnierLoss()
        self.ssim_fn = SSIMLoss()
        self.lap_fn = LaplacianEdgeLoss()
        self.tv_fn = GridTotalVariationLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        grid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_charb = self.charb_fn(pred, target)
        l_ssim = self.ssim_fn(pred, target)
        l_lap = self.lap_fn(pred, target)

        loss = self.w_charb * l_charb + self.w_ssim * l_ssim + self.w_lap * l_lap
        metrics = {
            "loss_charb": l_charb.item(),
            "loss_ssim": l_ssim.item(),
            "loss_lap": l_lap.item(),
            "loss_tv": 0.0
        }

        if grid is not None and self.w_grid_tv > 0:
            l_tv = self.tv_fn(grid)
            loss = loss + self.w_grid_tv * l_tv
            metrics["loss_tv"] = l_tv.item()

        metrics["loss_total"] = loss.item()
        return loss, metrics
