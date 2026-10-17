"""
SAT-EURS Track 1 — V18 Restoration Loss Functions
Components:
1. Charbonnier / L1 Reconstruction Loss (50%)
2. Multi-Scale SSIM Loss (30%)
3. Laplacian High-Frequency Texture Loss (10%)
4. Affine Smoothness & Identity Penalty (5%)
5. Residual Magnitude L2 Penalty (5%)
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1 / Charbonnier loss: sqrt((x - y)^2 + eps^2)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


def _gaussian_window(window_size: int, sigma: float, channels: int = 3, device=None) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    g_2d = torch.outer(g, g)
    window = g_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    channels = pred.shape[1]
    mu1 = F.conv2d(pred, window, padding=window.shape[-1] // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window.shape[-1] // 2, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window.shape[-1] // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window.shape[-1] // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window.shape[-1] // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
    return ssim_map.mean()


class MSSSIMLoss(nn.Module):
    """Multi-Scale SSIM loss over 3 scales: 1x, 0.5x, 0.25x."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.weights = [0.4, 0.3, 0.3]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        window = _gaussian_window(self.window_size, self.sigma, channels=c, device=pred.device)

        levels_loss = []
        p_curr = pred
        t_curr = target

        for i, weight in enumerate(self.weights):
            if min(p_curr.shape[-2:]) < self.window_size:
                break
            ssim_val = ssim(p_curr, t_curr, window)
            levels_loss.append(weight * (1.0 - ssim_val))

            if i < len(self.weights) - 1:
                p_curr = F.avg_pool2d(p_curr, kernel_size=2, stride=2)
                t_curr = F.avg_pool2d(t_curr, kernel_size=2, stride=2)

        return sum(levels_loss)


class LaplacianLoss(nn.Module):
    """L1 difference on Laplacian filtered outputs to encourage high-frequency edge restoration."""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0.0, 1.0, 0.0],
                               [1.0, -4.0, 1.0],
                               [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lap_p = F.conv2d(pred, self.kernel, padding=1, groups=3)
        lap_t = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(lap_p, lap_t)


class AffineRegularizationLoss(nn.Module):
    """
    Penalizes deviations from Identity matrix and non-smooth affine fields:
      - Identity penalty: ||A(x) - I_3||_F^2
      - Bias magnitude penalty: ||b(x)||_2^2
      - Smoothness penalty: Total Variation ||grad(A)||_1 + ||grad(b)||_1
    """
    def __init__(self):
        super().__init__()
        # Target identity: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        target = torch.zeros(12)
        target[0] = 1.0
        target[4] = 1.0
        target[8] = 1.0
        self.register_buffer("identity_target", target.view(1, 12, 1, 1))

    def forward(self, affine_params_low: torch.Tensor) -> torch.Tensor:
        # affine_params_low: [B, 12, H_low, W_low]
        id_err = affine_params_low - self.identity_target
        id_loss = torch.mean(id_err * id_err)

        # Total Variation smoothness across spatial grid
        diff_h = torch.abs(affine_params_low[:, :, 1:, :] - affine_params_low[:, :, :-1, :])
        diff_w = torch.abs(affine_params_low[:, :, :, 1:] - affine_params_low[:, :, :, :-1])
        tv_loss = torch.mean(diff_h) + torch.mean(diff_w)

        return id_loss + 0.5 * tv_loss


class ResidualMagnitudeLoss(nn.Module):
    """Constrains residual magnitude to prevent neural restorer from diverging."""
    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.mean(residual * residual)


class RestorationV18Loss(nn.Module):
    """
    Full composite loss for ConditionedRestorationRefinerV18:
      0.50 * Charbonnier + 0.30 * MS-SSIM + 0.10 * Laplacian + 0.05 * AffineReg + 0.05 * ResMag
    """
    def __init__(
        self,
        w_charb: float = 0.50,
        w_msssim: float = 0.30,
        w_lap: float = 0.10,
        w_affine: float = 0.05,
        w_res: float = 0.05,
        w_laplacian: Optional[float] = None,
        w_affine_reg: Optional[float] = None,
        w_residual_mag: Optional[float] = None
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_msssim = w_msssim
        self.w_lap = w_laplacian if w_laplacian is not None else w_lap
        self.w_affine = w_affine_reg if w_affine_reg is not None else w_affine
        self.w_res = w_residual_mag if w_residual_mag is not None else w_res

        self.loss_charb = CharbonnierLoss()
        self.loss_msssim = MSSSIMLoss()
        self.loss_lap = LaplacianLoss()
        self.loss_affine = AffineRegularizationLoss()
        self.loss_res = ResidualMagnitudeLoss()

    def forward(
        self,
        pred_dict: dict,
        target_clean_rgb: torch.Tensor
    ) -> dict:
        pred_refined = pred_dict["refined_image"]
        residual = pred_dict.get("residual", torch.zeros_like(pred_refined))
        affine_params = pred_dict.get("affine_params", torch.zeros((1, 12, 8, 8), device=pred_refined.device))

        l_charb = self.loss_charb(pred_refined, target_clean_rgb)
        l_msssim = self.loss_msssim(pred_refined, target_clean_rgb)
        l_lap = self.loss_lap(pred_refined, target_clean_rgb)
        l_affine = self.loss_affine(affine_params)
        l_res = self.loss_res(residual)

        total_loss = (
            self.w_charb * l_charb
            + self.w_msssim * l_msssim
            + self.w_lap * l_lap
            + self.w_affine * l_affine
            + self.w_res * l_res
        )

        return {
            "loss": total_loss,
            "l_charb": l_charb,
            "l_msssim": l_msssim,
            "l_lap": l_lap,
            "l_affine": l_affine,
            "l_res": l_res
        }
