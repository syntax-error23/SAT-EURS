"""
EURS-Net V4 Loss Suite (Gate D Verified, Strict FP32 Execution)
Includes:
  - BoundaryTargetGenerator & BoundaryAuxLoss (for crisp segmentation contours)
  - PresentClassLovaszSoftmaxLoss (direct mIoU surrogate over present classes)
  - V4SegmentationLoss (Focal with damped alpha applied ONCE + Batch Present-Class SoftDice + Aux Boundary + Lovasz)
  - V4MultiTaskLoss (Unified joint restoration + segmentation objective in strict FP32)
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.segmentation_loss import ClassWeightedFocalLoss, SoftDiceLoss
from losses.restoration_loss import RestorationLoss


def extract_boundary_targets(mask: torch.Tensor, target_hw: Tuple[int, int], kernel_size: int = 3) -> torch.Tensor:
    """
    Computes binary morphological boundary targets [B, 1, H_t, W_t] in {0.0, 1.0}
    from integer semantic mask [B, H, W] using nearest-neighbor downsampling and
    max-pool != min-pool morphological gradient.
    """
    with torch.no_grad():
        m = mask.unsqueeze(1).float()  # [B, 1, H, W]
        if (m.shape[2], m.shape[3]) != target_hw:
            m = F.interpolate(m, size=target_hw, mode='nearest')
        pad = kernel_size // 2
        m_pad = F.pad(m, (pad, pad, pad, pad), mode='replicate')
        dilated = F.max_pool2d(m_pad, kernel_size=kernel_size, stride=1)
        eroded = -F.max_pool2d(-m_pad, kernel_size=kernel_size, stride=1)
        boundary = (dilated != eroded).float()
        return boundary


class BoundaryAuxLoss(nn.Module):
    """
    Class-balanced Binary Cross-Entropy + Soft Dice loss for auxiliary boundary prediction (FP32).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, boundary_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device_type = boundary_logits.device.type if boundary_logits.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits_f = boundary_logits.float()
            targets_b = extract_boundary_targets(mask, (logits_f.shape[2], logits_f.shape[3]))

            pos_frac = targets_b.mean().clamp(min=0.01, max=0.99)
            pos_weight = ((1.0 - pos_frac) / pos_frac).clamp(min=1.0, max=10.0)
            bce = F.binary_cross_entropy_with_logits(logits_f, targets_b, pos_weight=pos_weight)

            probs = torch.sigmoid(logits_f)
            inter = (probs * targets_b).sum(dim=(1, 2, 3))
            denom = (probs ** 2).sum(dim=(1, 2, 3)) + (targets_b ** 2).sum(dim=(1, 2, 3))
            dice_loss = 1.0 - ((2.0 * inter + self.eps) / (denom + self.eps)).mean()
            return 0.5 * bce + 0.5 * dice_loss


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovasz extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class PresentClassLovaszSoftmaxLoss(nn.Module):
    """
    Multi-class Lovasz-Softmax loss in strict FP32 averaged ONLY over classes present
    in the mini-batch ground truth. Directly optimizes the Jaccard / mIoU index.
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device_type = logits.device.type if logits.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits_f = logits.float()
            probs = F.softmax(logits_f, dim=1)  # [B, C, H, W]
            # Subsample spatial grid by 2x if > 256x256 to keep O(N log N) sort fast
            if probs.shape[2] > 256 or probs.shape[3] > 256:
                probs = F.avg_pool2d(probs, kernel_size=2, stride=2)
                targets_sub = F.interpolate(
                    targets.unsqueeze(1).float(), size=probs.shape[2:], mode='nearest'
                ).squeeze(1).long()
            else:
                targets_sub = targets.long()

            probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # [P, C]
            labels_flat = targets_sub.reshape(-1)  # [P]

            losses = []
            for c in range(self.num_classes):
                fg = (labels_flat == c).float()
                if fg.sum() == 0.0:
                    continue
                class_pred = probs_flat[:, c]
                errors = (fg - class_pred).abs()
                errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
                fg_sorted = fg[perm]
                grad = _lovasz_grad(fg_sorted)
                losses.append(torch.dot(F.relu(errors_sorted), grad))

            if not losses:
                return (logits_f * 0.0).sum()
            return torch.stack(losses).mean()


class V4MultiTaskLoss(nn.Module):
    """
    Unified Multi-Task Loss for EURS-Net V4 (Gate D Verified):
      - Class alpha applied ONLY ONCE (in Focal Loss), damped ratio <= 4.4x:
        default alpha = [0.50, 0.70, 1.40, 2.20, 2.20]
      - Batch-accumulated Present-Class SoftDiceLoss (unweighted over present classes)
      - Present-Class LovaszSoftmaxLoss (direct mIoU optimization)
      - BoundaryAuxLoss on 256x256 boundary logits
      - FP32 RestorationLoss (Charbonnier + FP32 MS-SSIM + Luminance-Gated ColorAngle + Laplacian)
    """
    def __init__(
        self,
        num_classes: int = 5,
        alpha: Optional[torch.Tensor] = None,
        class_alpha: Optional[torch.Tensor] = None,
        weight_seg: float = 1.0,
        weight_rest: float = 0.6,
        lambda_dice: float = 1.0,
        lambda_lovasz: float = 0.25,
        lovasz_start_epoch: int = 3,
        lovasz_max_weight: Optional[float] = None,
        lambda_boundary: float = 0.15,
        weight_boundary: Optional[float] = None,
        weight_charb: float = 1.0,
        weight_ssim: float = 0.5,
        weight_color: float = 0.3,
        weight_lap: float = 0.15,
    ):
        super().__init__()
        eff_alpha = alpha if alpha is not None else class_alpha
        if eff_alpha is None:
            # Moderate, damped inverse-sqrt class weights (max/min ratio = 4.4x, avoiding V3's 30x collapse)
            eff_alpha = torch.tensor([0.50, 0.70, 1.40, 2.20, 2.20], dtype=torch.float32)

        self.focal = ClassWeightedFocalLoss(alpha=eff_alpha, gamma=2.0)
        # Alpha is None in Dice so class weights are applied ONLY ONCE (Gate D requirement)
        self.dice = SoftDiceLoss(num_classes=num_classes, alpha=None, batch_dice=True)
        self.lovasz = PresentClassLovaszSoftmaxLoss(num_classes=num_classes)
        self.boundary = BoundaryAuxLoss()

        self.rest_loss = RestorationLoss(
            weight_charb=weight_charb,
            weight_ssim=weight_ssim,
            weight_color=weight_color,
            weight_lap=weight_lap,
        )

        self.weight_seg = weight_seg
        self.weight_rest = weight_rest
        self.lambda_dice = lambda_dice
        self.lovasz_start_epoch = lovasz_start_epoch
        self.lambda_lovasz = lovasz_max_weight if lovasz_max_weight is not None else lambda_lovasz
        self.lambda_boundary = weight_boundary if weight_boundary is not None else lambda_boundary
        self.current_epoch = 1

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self.rest_loss.set_epoch(epoch)

    def forward(
        self,
        restored_rgb: torch.Tensor,
        seg_logits: torch.Tensor,
        clean_target: torch.Tensor,
        mask_target: torch.Tensor,
        boundary_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device_type = restored_rgb.device.type if restored_rgb.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            l_focal = self.focal(seg_logits.float(), mask_target)
            l_dice = self.dice(seg_logits.float(), mask_target)

            active_lovasz_w = self.lambda_lovasz if self.current_epoch >= self.lovasz_start_epoch else 0.0
            if active_lovasz_w > 0.0:
                l_lovasz = self.lovasz(seg_logits.float(), mask_target)
            else:
                l_lovasz = (seg_logits.float() * 0.0).sum()

            if boundary_logits is not None and self.lambda_boundary > 0.0:
                l_bound = self.boundary(boundary_logits.float(), mask_target)
            else:
                l_bound = (seg_logits.float() * 0.0).sum()

            l_seg_total = (
                l_focal
                + self.lambda_dice * l_dice
                + active_lovasz_w * l_lovasz
                + self.lambda_boundary * l_bound
            )
            l_rest_total = self.rest_loss(restored_rgb.float(), clean_target.float())

            total_loss = self.weight_seg * l_seg_total + self.weight_rest * l_rest_total

            metrics_dict = {
                'loss_total': float(total_loss.detach().item()),
                'loss_seg': float(l_seg_total.detach().item()),
                'loss_focal': float(l_focal.detach().item()),
                'loss_dice': float(l_dice.detach().item()),
                'loss_lovasz': float(l_lovasz.detach().item()),
                'loss_boundary': float(l_bound.detach().item()),
                'loss_rest': float(l_rest_total.detach().item()),
            }
            return total_loss, metrics_dict
