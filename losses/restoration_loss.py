"""
Restoration Losses for EURS-Net V2
Includes:
- CharbonnierLoss: Smooth L1 surrogate with element-wise reduction.
- LaplacianLoss: High-frequency edge preservation loss with reflection padding.
- RestorationLoss: Unified composite restoration loss combining Charbonnier, MS-SSIM, and Laplacian edge loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.msssim import MSSSIMLoss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1 surrogate):
        L_charb = mean(sqrt((I_p - I_t)^2 + eps^2))
    Strictly element-wise with arithmetic mean reduction over [B, C, H, W].
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred.float() - target.float()
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class LaplacianLoss(nn.Module):
    """
    Laplacian Edge Loss:
    Applies fixed 3x3 Laplacian kernel depthwise across each RGB channel independently
    with reflection padding to eliminate boundary artifacts.
    Penalizes blurred edges to directly maximize the official sharpness_ratio.
    """
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, 3, 3]
        # Expand for 3 RGB channels (depthwise)
        kernel_3ch = kernel.repeat(3, 1, 1, 1) # [3, 1, 3, 3]
        self.register_buffer('kernel', kernel_3ch)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: [B, 3, H, W] in [0, 1]
        """
        pred_f = pred.float()
        target_f = target.float()
        channels = pred_f.shape[1]
        k = (self.kernel if channels == 3 else self.kernel[:channels]).float()

        # Use reflection padding for natural boundary gradients
        p_padded = F.pad(pred_f, (1, 1, 1, 1), mode='reflect')
        t_padded = F.pad(target_f, (1, 1, 1, 1), mode='reflect')

        lap_pred = F.conv2d(p_padded, k, groups=channels)
        lap_target = F.conv2d(t_padded, k, groups=channels)

        return F.l1_loss(lap_pred, lap_target)


class ColorAngleLoss(nn.Module):
    """
    Luminance-Gated Color Angle / Cosine Loss (FP32):
    Measures pixel-wise angular cosine deviation between predicted RGB vectors
    and target RGB vectors on pixels with sufficient luminance (||target||_2 > min_lum),
    avoiding false penalties and gradient singularities on dark/black pixels.
    """
    def __init__(self, eps: float = 1e-6, min_lum: float = 0.02):
        super().__init__()
        self.eps = eps
        self.min_lum = min_lum

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()
        norm_p = torch.norm(pred_f, p=2, dim=1)
        norm_t = torch.norm(target_f, p=2, dim=1)
        valid_mask = (norm_t > self.min_lum).float()
        if valid_mask.sum() < 1.0:
            # Fallback to zero angular error when entire target image is dark
            return (pred_f * 0.0).sum()
        dot = (pred_f * target_f).sum(dim=1)
        cos_sim = dot / (norm_p.clamp(min=self.min_lum) * norm_t.clamp(min=self.min_lum) + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        ang_err = (1.0 - cos_sim) * valid_mask
        return ang_err.sum() / (valid_mask.sum() + self.eps)



class RestorationLoss(nn.Module):
    """
    Unified Restoration Loss:
        L_rest = weight_charb * L_charb + weight_ssim * L_msssim + weight_color * L_color + weight_lap * L_lap
    Default weights:
        weight_charb = 1.0
        weight_ssim = 0.5 (reduced from 2.0 to avoid initial MS-SSIM plateau)
        weight_color = 0.5 (color angle/cosine loss for primary radiometric white balance)
        weight_lap = 0.1
    """
    def __init__(
        self,
        weight_charb: float = 1.0,
        weight_ssim: float = 0.5,
        weight_color: float = 0.5,
        weight_lap: float = 0.1,
        charb_eps: float = 1e-3
    ):
        super().__init__()
        self.charb = CharbonnierLoss(eps=charb_eps)
        self.msssim = MSSSIMLoss(data_range=1.0)
        self.color = ColorAngleLoss()
        self.lap = LaplacianLoss()

        self.weight_charb = weight_charb
        self.weight_ssim = weight_ssim
        self.weight_color = weight_color
        self.weight_lap = weight_lap
        self.current_epoch = 1

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        # Keep MS-SSIM weight 0.5 for first 15 epochs, then scale to 1.0
        if epoch <= 15:
            self.weight_ssim = 0.5
        else:
            self.weight_ssim = 1.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_charb = self.charb(pred, target)
        l_ssim = self.msssim(pred, target)
        l_color = self.color(pred, target)
        l_lap = self.lap(pred, target)

        return (
            self.weight_charb * l_charb +
            self.weight_ssim * l_ssim +
            self.weight_color * l_color +
            self.weight_lap * l_lap
        )
