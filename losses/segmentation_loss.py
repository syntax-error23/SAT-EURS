"""
Segmentation Losses for EURS-Net V2
Includes:
- ClassWeightedFocalLoss: Outlier-robust, class-frequency-smoothed focal loss.
- SoftDiceLoss: Macro-averaged region overlap optimizer.
- SegmentationLoss: Unified weighted composite segmentation loss.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Class Semantic Segmentation:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Args:
        alpha (torch.Tensor, optional): Per-class weighting coefficients [C].
        gamma (float): Focusing parameter (default: 2.0).
        reduction (str): Reduction method ('mean' or 'sum').
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        eps: float = 1e-7
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if alpha is not None:
            self.register_buffer('alpha', alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): [B, C, H, W] unnormalized class predictions.
            targets (torch.Tensor): [B, H, W] integer class labels in {0, ..., C-1}.
            valid_mask (torch.Tensor, optional): [H, W] or [B, H, W] boolean mask.
        Returns:
            torch.Tensor: Scalar focal loss in FP32.
        """
        num_classes = logits.shape[1]
        logits_f = logits.float()
        targets_clamped = targets.long().clamp(0, num_classes - 1)
        ce_loss = F.cross_entropy(logits_f, targets_clamped, reduction='none')
        p = torch.exp(-ce_loss)

        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits_f.device)[targets_clamped]
            focal_loss = alpha_t * focal_loss

        if valid_mask is None:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss

        vm = valid_mask.bool()
        if vm.dim() == 2:
            vm = vm.unsqueeze(0).expand_as(targets)
        elif vm.dim() == 4 and vm.shape[1] == 1:
            vm = vm.squeeze(1)

        focal_loss = torch.where(vm, focal_loss, torch.zeros_like(focal_loss))
        if self.reduction == 'mean':
            return focal_loss.sum() / (vm.sum().float() + self.eps)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SoftDiceLoss(nn.Module):
    """
    Batch-Accumulated Present-Class Soft Dice Loss (FP32):
        Dice_c = 2 * sum_{b,h,w}(P_c * Y_c) / (sum_{b,h,w}(P_c^2) + sum_{b,h,w}(Y_c^2) + eps)
    Supports optional valid_mask for letterbox padding exclusion.
    """
    def __init__(
        self,
        num_classes: int = 5,
        eps: float = 1e-6,
        alpha: Optional[torch.Tensor] = None,
        batch_dice: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.batch_dice = batch_dice
        if alpha is not None:
            self.register_buffer('alpha', alpha.float().view(num_classes))
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): [B, C, H, W] unnormalized class predictions.
            targets (torch.Tensor): [B, H, W] integer class labels.
            valid_mask (torch.Tensor, optional): [H, W] or [B, H, W] boolean mask.
        Returns:
            torch.Tensor: Scalar soft dice loss in FP32.
        """
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=1)
        targets_clamped = targets.long().clamp(0, self.num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if valid_mask is not None:
            vm = valid_mask.bool()
            if vm.dim() == 2:
                vm = vm.unsqueeze(0).expand(probs.shape[0], -1, -1)
            elif vm.dim() == 4 and vm.shape[1] == 1:
                vm = vm.squeeze(1)
            vm_c = vm.unsqueeze(1).expand_as(probs).float()
            probs = probs * vm_c
            targets_one_hot = targets_one_hot * vm_c

        probs_flat = probs.reshape(probs.shape[0], self.num_classes, -1)
        targets_flat = targets_one_hot.reshape(targets_one_hot.shape[0], self.num_classes, -1)

        if self.batch_dice:
            # Accumulate over batch and spatial dims -> [C]
            intersection = (probs_flat * targets_flat).sum(dim=(0, 2))
            target_sum = (targets_flat ** 2).sum(dim=(0, 2))
            pred_sum = (probs_flat ** 2).sum(dim=(0, 2))
            present_mask = target_sum > 0.0  # [C]
            if not torch.any(present_mask):
                return (logits_f * 0.0).sum()
            dice_c = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
            if self.alpha is not None:
                w = self.alpha.to(logits_f.device) * present_mask.float()
                return 1.0 - (dice_c * w).sum() / (w.sum() + self.eps)
            else:
                return 1.0 - dice_c[present_mask].mean()
        else:
            # Per-image masked over present classes in each image -> [B, C]
            intersection = (probs_flat * targets_flat).sum(dim=2)
            target_sum = (targets_flat ** 2).sum(dim=2)
            pred_sum = (probs_flat ** 2).sum(dim=2)
            present_mask = (target_sum > 0.0).float()  # [B, C]
            dice_bc = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)
            if self.alpha is not None:
                w = self.alpha.to(logits_f.device).unsqueeze(0) * present_mask
            else:
                w = present_mask
            valid_per_img = w.sum(dim=1)
            img_dice = (dice_bc * w).sum(dim=1) / (valid_per_img + self.eps)
            return (1.0 - img_dice[valid_per_img > 0]).mean()


class SegmentationLoss(nn.Module):
    """
    Unified Segmentation Loss combining ClassWeightedFocalLoss and Present-Class Batch SoftDiceLoss:
        L_seg = L_focal(alpha) + lambda_dice * L_dice
    Supports optional valid_mask for letterbox padding exclusion.
    """
    def __init__(
        self,
        num_classes: int = 5,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        lambda_dice: float = 1.0,
        dice_use_alpha: bool = False,
    ):
        super().__init__()
        self.focal = ClassWeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = SoftDiceLoss(
            num_classes=num_classes,
            alpha=alpha if dice_use_alpha else None,
            batch_dice=True,
        )
        self.lambda_dice = lambda_dice

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss_f = self.focal(logits, targets, valid_mask=valid_mask)
        loss_d = self.dice(logits, targets, valid_mask=valid_mask)
        return loss_f + self.lambda_dice * loss_d

