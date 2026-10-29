"""
Multi-Task Unified Loss for EURS-Net V2
Combines Segmentation and Restoration losses with 2:1 ratio (1.0 * L_seg + 0.5 * L_rest),
reflecting the official 60% segmentation / 30% restoration competition weighting.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
from losses.segmentation_loss import SegmentationLoss
from losses.restoration_loss import RestorationLoss


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss Module:
        L_total = weight_seg * L_seg + weight_rest * L_rest
    Default weights:
        weight_seg = 1.0
        weight_rest = 0.5
    """
    def __init__(
        self,
        weight_seg: float = 1.0,
        weight_rest: float = 0.5,
        alpha_seg: Optional[torch.Tensor] = None,
        num_classes: int = 5
    ):
        super().__init__()
        self.weight_seg = weight_seg
        self.weight_rest = weight_rest

        self.seg_loss_fn = SegmentationLoss(num_classes=num_classes, alpha=alpha_seg)
        self.rest_loss_fn = RestorationLoss()

    def set_epoch(self, epoch: int):
        if hasattr(self.rest_loss_fn, 'set_epoch'):
            self.rest_loss_fn.set_epoch(epoch)

    def forward(
        self,
        restored_rgb: torch.Tensor,
        seg_logits: torch.Tensor,
        clean_target: torch.Tensor,
        mask_target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward computation returning scalar total loss and detached dictionary of component losses.
        Supports optional valid_mask for letterbox training padding exclusion.
        """
        loss_seg = self.seg_loss_fn(seg_logits, mask_target, valid_mask=valid_mask)
        loss_rest = self.rest_loss_fn(restored_rgb, clean_target, valid_mask=valid_mask)

        loss_total = self.weight_seg * loss_seg + self.weight_rest * loss_rest

        # Detailed dictionary for tracking
        loss_dict = {
            'loss_total': loss_total.item(),
            'loss_seg': loss_seg.item(),
            'loss_rest': loss_rest.item(),
        }

        return loss_total, loss_dict
