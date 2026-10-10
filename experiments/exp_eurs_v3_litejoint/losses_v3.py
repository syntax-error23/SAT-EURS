"""
EURS-Net V3 LiteJoint Loss Functions
Combines:
1. Class-Weighted Focal Loss & Multiclass Soft Dice Loss for segmentation
2. Multi-component restoration loss: Charbonnier + MS-SSIM + Color Angle + Laplacian edge loss
3. Joint Multi-Task Loss with configurable weighting
"""

from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.restoration_loss import RestorationLoss
from losses.segmentation_loss import ClassWeightedFocalLoss, SoftDiceLoss


class SegmentationLossV3(nn.Module):
    """
    Segmentation loss for V3 LiteJoint:
    - Combines Class-Weighted Focal Loss (gamma=2.0) with Macro Soft Dice Loss
    - Class weights alpha heavily boost rare marine classes (Fish, Shark/Ray, Other)
    """
    def __init__(
        self,
        num_classes: int = 5,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        lambda_dice: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.focal = ClassWeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = SoftDiceLoss(num_classes=num_classes, alpha=alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        total_seg_loss = loss_focal + self.lambda_dice * loss_dice
        return total_seg_loss, loss_focal, loss_dice


class MultiTaskLossV3(nn.Module):
    """
    Unified Multi-Task Joint Loss for EURS-Net V3:
    L_total = weight_seg * L_seg + weight_rest * L_rest
    Default:
        weight_seg = 1.0 (60% competition priority)
        weight_rest = 0.5 (30% competition priority)
    """
    def __init__(
        self,
        num_classes: int = 5,
        alpha_seg: Optional[torch.Tensor] = None,
        weight_seg: float = 1.0,
        weight_rest: float = 0.5,
        lambda_dice: float = 1.0
    ):
        super().__init__()
        self.weight_seg = weight_seg
        self.weight_rest = weight_rest
        self.seg_loss_fn = SegmentationLossV3(num_classes=num_classes, alpha=alpha_seg, lambda_dice=lambda_dice)
        self.rest_loss_fn = RestorationLoss()

    def set_epoch(self, epoch: int):
        if hasattr(self.rest_loss_fn, 'set_epoch'):
            self.rest_loss_fn.set_epoch(epoch)

    def forward(
        self,
        restored_rgb: torch.Tensor,
        seg_logits: torch.Tensor,
        clean_target: torch.Tensor,
        mask_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_seg, l_focal, l_dice = self.seg_loss_fn(seg_logits, mask_target)
        loss_rest = self.rest_loss_fn(restored_rgb, clean_target)

        loss_total = self.weight_seg * loss_seg + self.weight_rest * loss_rest

        loss_dict = {
            'loss_total': float(loss_total.item()),
            'loss_seg': float(loss_seg.item()),
            'loss_focal': float(l_focal.item()),
            'loss_dice': float(l_dice.item()),
            'loss_rest': float(loss_rest.item()),
        }
        return loss_total, loss_dict


def get_v3_class_weights() -> torch.Tensor:
    """
    Authoritative rare-boosted class weights for EURS-Net V3:
    Classes:
      0: Background (~72.8%)
      1: Coral (~24.3%)
      2: Fish (~1.50%)
      3: Elasmobranch / Shark (~0.17%)
      4: Other (~1.18%)
    Balanced to give strong gradient signal to rare marine classes while preserving background/coral.
    Mean is normalized to exactly 1.0.
    """
    # Raw weights emphasizing rare classes
    raw_w = np.array([0.25, 0.75, 3.50, 7.50, 7.00], dtype=np.float32)
    normalized_w = raw_w / float(np.mean(raw_w))
    return torch.tensor(normalized_w, dtype=torch.float32)
