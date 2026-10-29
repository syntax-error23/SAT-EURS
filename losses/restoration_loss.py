"""
Restoration Losses for EURS-Net V2
Includes:
- CharbonnierLoss: Smooth L1 surrogate with element-wise reduction.
- LaplacianLoss: High-frequency edge preservation loss with reflection padding.
- RestorationLoss: Unified composite restoration loss combining Charbonnier, MS-SSIM, and Laplacian edge loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.msssim import MSSSIMLoss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1 surrogate):
        L_charb = mean(sqrt((I_p - I_t)^2 + eps^2))
    Strictly element-wise with arithmetic mean reduction over valid pixels.
    Supports optional valid_mask for letterbox padding exclusion.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = pred.float() - target.float()
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if valid_mask is None:
            return loss.mean()

        vm = valid_mask.float()
        if vm.dim() == 2:
            vm = vm.unsqueeze(0).unsqueeze(0)
        elif vm.dim() == 3:
            vm = vm.unsqueeze(1)
        vm = vm.expand_as(loss)
        return (loss * vm).sum() / (vm.sum() + 1e-8)


class LaplacianLoss(nn.Module):
    """
    Laplacian Edge Loss:
    Applies fixed 3x3 Laplacian kernel depthwise across each RGB channel independently
    with reflection padding to eliminate boundary artifacts.
    Penalizes blurred edges to directly maximize the official sharpness_ratio.
    Supports optional valid_mask for letterbox padding exclusion.
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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred, target: [B, 3, H, W] in [0, 1]
            valid_mask: Optional boolean mask [H, W], [B, H, W], or [B, 1, H, W] for padding exclusion.
        """
        pred_f = pred.float()
        target_f = target.float()
        channels = pred_f.shape[1]
        k = (self.kernel if channels == 3 else self.kernel[:channels]).float()

        if valid_mask is None:
            p_padded = F.pad(pred_f, (1, 1, 1, 1), mode='reflect')
            t_padded = F.pad(target_f, (1, 1, 1, 1), mode='reflect')

            lap_pred = F.conv2d(p_padded, k, groups=channels)
            lap_target = F.conv2d(t_padded, k, groups=channels)

            return F.l1_loss(lap_pred, lap_target)

        b = pred.shape[0]
        vm = valid_mask.bool()
        if vm.dim() == 2:
            vm = vm.unsqueeze(0).expand(b, -1, -1)
        elif vm.dim() == 4 and vm.shape[1] == 1:
            vm = vm.squeeze(1)

        # Vectorized fast path: if all images in the batch share identical padding bounds
        y_indices, x_indices = torch.where(vm[0])
        if len(y_indices) > 0:
            ymin, ymax = y_indices.min(), y_indices.max() + 1
            xmin, xmax = x_indices.min(), x_indices.max() + 1
            if (vm[:, ymin:ymax, xmin:xmax]).all() and vm.sum(dim=(-2, -1)).unique().numel() == 1:
                p_crop = pred[:, :, ymin:ymax, xmin:xmax]
                t_crop = target[:, :, ymin:ymax, xmin:xmax]
                p_padded = F.pad(p_crop, (1, 1, 1, 1), mode='reflect')
                t_padded = F.pad(t_crop, (1, 1, 1, 1), mode='reflect')
                lap_pred = F.conv2d(p_padded, k, groups=channels)
                lap_target = F.conv2d(t_padded, k, groups=channels)
                return F.l1_loss(lap_pred, lap_target)

        losses = []
        for i in range(b):
            vm_i = vm[i]
            y_i, x_i = torch.where(vm_i)
            if len(y_i) == 0:
                continue
            ymin, ymax = y_i.min(), y_i.max() + 1
            xmin, xmax = x_i.min(), x_i.max() + 1

            p_crop = pred[i:i+1, :, ymin:ymax, xmin:xmax]
            t_crop = target[i:i+1, :, ymin:ymax, xmin:xmax]

            p_padded = F.pad(p_crop, (1, 1, 1, 1), mode='reflect')
            t_padded = F.pad(t_crop, (1, 1, 1, 1), mode='reflect')

            lap_pred = F.conv2d(p_padded, k, groups=channels)
            lap_target = F.conv2d(t_padded, k, groups=channels)

            losses.append(F.l1_loss(lap_pred, lap_target))

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.stack(losses).mean()


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
    Supports optional valid_mask for letterbox padding exclusion.
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

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        l_charb = self.charb(pred, target, valid_mask=valid_mask)
        l_ssim = self.msssim(pred, target, valid_mask=valid_mask)
        l_color = self.color(pred, target)
        l_lap = self.lap(pred, target, valid_mask=valid_mask)

        return (
            self.weight_charb * l_charb +
            self.weight_ssim * l_ssim +
            self.weight_color * l_color +
            self.weight_lap * l_lap
        )
