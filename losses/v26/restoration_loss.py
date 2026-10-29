"""
SAT-EURS Track 1 — V26 Restoration Loss (Apex-085)
Objective:
1. Charbonnier Reconstruction Loss on final restored image (w_charb = 0.45)
2. MS-SSIM Loss on final restored image (w_msssim = 0.30)
3. Laplacian High-Frequency Loss on final restored image (w_lap = 0.15)
4. Auxiliary Charbonnier Loss on coarse affine image (w_aux = 0.08)
5. Affine Spatial Smoothness / TV Loss (w_tv = 0.02, ZERO identity trap)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.v18.restoration_losses import CharbonnierLoss, MSSSIMLoss, LaplacianLoss


class AffineSpatialSmoothnessLoss(nn.Module):
    """Total Variation penalty to ensure spatial smoothness of predicted affine grid."""
    def forward(self, affine_grid: torch.Tensor) -> torch.Tensor:
        # affine_grid: [B, 12, H_g, W_g]
        diff_h = torch.abs(affine_grid[:, :, 1:, :] - affine_grid[:, :, :-1, :])
        diff_w = torch.abs(affine_grid[:, :, :, 1:] - affine_grid[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


class RestorationV26Loss(nn.Module):
    def __init__(
        self,
        w_charb: float = 0.45,
        w_msssim: float = 0.30,
        w_lap: float = 0.15,
        w_aux: float = 0.08,
        w_tv: float = 0.02
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_msssim = w_msssim
        self.w_lap = w_lap
        self.w_aux = w_aux
        self.w_tv = w_tv

        self.charb = CharbonnierLoss(eps=1e-3)
        self.msssim = MSSSIMLoss()
        self.lap = LaplacianLoss()
        self.tv = AffineSpatialSmoothnessLoss()

    def forward(
        self,
        pred_dict: dict,
        target: torch.Tensor
    ) -> dict:
        restored = pred_dict["restored_image"]
        affine = pred_dict["affine_image"]
        affine_grid = pred_dict["affine_grid"]

        l_charb = self.charb(restored, target)
        l_msssim = self.msssim(restored, target)
        l_lap = self.lap(restored, target)
        l_aux = self.charb(affine, target)
        l_tv = self.tv(affine_grid)

        total_loss = (
            self.w_charb * l_charb +
            self.w_msssim * l_msssim +
            self.w_lap * l_lap +
            self.w_aux * l_aux +
            self.w_tv * l_tv
        )

        return {
            "loss": total_loss,
            "charb": l_charb.item(),
            "msssim": l_msssim.item(),
            "lap": l_lap.item(),
            "aux": l_aux.item(),
            "tv": l_tv.item()
        }
