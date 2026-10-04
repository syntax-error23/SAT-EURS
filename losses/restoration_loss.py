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
        diff = pred - target
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
        channels = pred.shape[1]
        k = self.kernel if channels == 3 else self.kernel[:channels]

        # Use reflection padding for natural boundary gradients
        p_padded = F.pad(pred, (1, 1, 1, 1), mode='reflect')
        t_padded = F.pad(target, (1, 1, 1, 1), mode='reflect')

        lap_pred = F.conv2d(p_padded, k, groups=channels)
        lap_target = F.conv2d(t_padded, k, groups=channels)

        return F.l1_loss(lap_pred, lap_target)


class RestorationLoss(nn.Module):
    """
    Unified Restoration Loss:
        L_rest = L_charbonnier + 2.0 * L_msssim + 0.1 * L_laplacian
    """
    def __init__(
        self,
        weight_charb: float = 1.0,
        weight_ssim: float = 2.0,
        weight_lap: float = 0.1,
        charb_eps: float = 1e-3
    ):
        super().__init__()
        self.charb = CharbonnierLoss(eps=charb_eps)
        self.msssim = MSSSIMLoss(data_range=1.0)
        self.lap = LaplacianLoss()

        self.weight_charb = weight_charb
        self.weight_ssim = weight_ssim
        self.weight_lap = weight_lap

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_charb = self.charb(pred, target)
        l_ssim = self.msssim(pred, target)
        l_lap = self.lap(pred, target)

        return (
            self.weight_charb * l_charb +
            self.weight_ssim * l_ssim +
            self.weight_lap * l_lap
        )
