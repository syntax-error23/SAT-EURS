"""
Restoration V7 Loss: Joint Q_FR and Q_NR Optimization Loss Suite
Stage 2 of SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Loss Components:
1. Charbonnier Loss (robust L1 fidelity)
2. MS-SSIM / SSIM Loss (structural coherence)
3. Image Gradient Loss (edge sharpness)
4. ChromaContrast Loss:
   - Differentiable chromaticity statistics: matches mean and standard deviation of
     Red-Green (RG) and Yellow-Blue (YB) color channels to the clean reference.
   - Local luminance contrast: matches local standard deviation of luminance.
   - Directly drives UIQM and UCIQE restoration without degrading PSNR or SSIM.
"""

from typing import Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)

class SSIMLoss(nn.Module):
    """Differentiable fast GPU SSIM loss."""
    def __init__(self, window_size: int = 11, in_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        # Gaussian window
        sigma = 1.5
        gauss = torch.tensor([
            math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
            for x in range(window_size)
        ], dtype=torch.float32)
        gauss = gauss / gauss.sum()
        _1d_window = gauss.unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", _2d_window.expand(in_channels, 1, window_size, window_size))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        device = img1.device
        window = self.window.to(device)
        p = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=p, groups=self.in_channels)
        mu2 = F.conv2d(img2, window, padding=p, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=p, groups=self.in_channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=p, groups=self.in_channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=p, groups=self.in_channels) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        return 1.0 - torch.clamp(ssim_map.mean(), 0.0, 1.0)

import math

class GradientLoss(nn.Module):
    """First-order spatial image gradient loss."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        kx = self.kernel_x.to(pred.device).expand(c, 1, 3, 3)
        ky = self.kernel_y.to(pred.device).expand(c, 1, 3, 3)

        pred_gx = F.conv2d(pred, kx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, ky, padding=1, groups=c)
        tgt_gx = F.conv2d(target, kx, padding=1, groups=c)
        tgt_gy = F.conv2d(target, ky, padding=1, groups=c)

        loss = F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)
        return loss

class ChromaContrastLoss(nn.Module):
    """
    Differentiable Chroma & Contrast Loss.
    Matches color distribution statistics (RG and YB chromaticity) and local luminance contrast.
    """
    def __init__(self, block_size: int = 7):
        super().__init__()
        self.block_size = block_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Chromaticity Matching
        r_p, g_p, b_p = pred[:, 0], pred[:, 1], pred[:, 2]
        r_t, g_t, b_t = target[:, 0], target[:, 1], target[:, 2]

        # Red-Green
        rg_p = r_p - g_p
        rg_t = r_t - g_t
        # Yellow-Blue
        yb_p = 0.5 * (r_p + g_p) - b_p
        yb_t = 0.5 * (r_t + g_t) - b_t

        loss_chroma = (
            F.l1_loss(rg_p.mean(dim=(-2, -1)), rg_t.mean(dim=(-2, -1))) +
            F.l1_loss(rg_p.std(dim=(-2, -1)), rg_t.std(dim=(-2, -1))) +
            F.l1_loss(yb_p.mean(dim=(-2, -1)), yb_t.mean(dim=(-2, -1))) +
            F.l1_loss(yb_p.std(dim=(-2, -1)), yb_t.std(dim=(-2, -1)))
        )

        # 2. Local Luminance Contrast
        y_p = (0.299 * r_p + 0.587 * g_p + 0.114 * b_p).unsqueeze(1)
        y_t = (0.299 * r_t + 0.587 * g_t + 0.114 * b_t).unsqueeze(1)

        k = self.block_size
        mean_yp = F.avg_pool2d(y_p, k, stride=k // 2, padding=k // 2)
        mean_yt = F.avg_pool2d(y_t, k, stride=k // 2, padding=k // 2)

        var_yp = F.avg_pool2d(y_p ** 2, k, stride=k // 2, padding=k // 2) - mean_yp ** 2
        var_yt = F.avg_pool2d(y_t ** 2, k, stride=k // 2, padding=k // 2) - mean_yt ** 2

        std_yp = torch.sqrt(torch.clamp(var_yp, min=1e-6))
        std_yt = torch.sqrt(torch.clamp(var_yt, min=1e-6))

        loss_contrast = F.l1_loss(std_yp, std_yt)
        return loss_chroma + 0.5 * loss_contrast

class RestorationV7Loss(nn.Module):
    """
    Combined Stage 2 Objective:
    L = L_Charb + 0.40 * L_SSIM + 0.20 * L_Grad + 0.15 * L_ChromaContrast
    """
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()
        self.chroma = ChromaContrastLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l_charb = self.charb(pred, target)
        l_ssim = self.ssim(pred, target)
        l_grad = self.grad(pred, target)
        l_chroma = self.chroma(pred, target)

        total_loss = l_charb + 0.40 * l_ssim + 0.20 * l_grad + 0.15 * l_chroma

        losses = {
            "loss_charb": l_charb.item(),
            "loss_ssim": l_ssim.item(),
            "loss_grad": l_grad.item(),
            "loss_chroma": l_chroma.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, losses
