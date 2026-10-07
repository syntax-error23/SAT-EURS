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
            torch.Tensor: Scalar focal loss.
        """
        num_classes = logits.shape[1]
        targets_clamped = targets.clamp(0, num_classes - 1)
        # Cross entropy loss with log-softmax
        ce_loss = F.cross_entropy(logits, targets_clamped, reduction='none') # [B, H, W]
        p = torch.exp(-ce_loss) # p_t in [0, 1]

        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Gather alpha corresponding to true class targets
            alpha_t = self.alpha[targets_clamped] # [B, H, W]
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
    Soft Dice Loss for Multi-Class Semantic Segmentation:
        Dice_c = 2 * sum(P_c * Y_c) / (sum(P_c^2) + sum(Y_c^2) + eps)
        Loss = 1 - mean(Dice_c)
    Macro-averaged over all C classes, including background class 0.
    Supports optional valid_mask for letterbox padding exclusion.
    """
    def __init__(self, num_classes: int = 5, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

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
            torch.Tensor: Scalar soft dice loss.
        """
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        targets_clamped = targets.clamp(0, self.num_classes - 1)
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float() # [B, C, H, W]

        if valid_mask is not None:
            vm = valid_mask.bool()
            if vm.dim() == 2:
                vm = vm.unsqueeze(0).expand(probs.shape[0], -1, -1)
            elif vm.dim() == 4 and vm.shape[1] == 1:
                vm = vm.squeeze(1)
            vm_c = vm.unsqueeze(1).expand_as(probs).float()
            probs = probs * vm_c
            targets_one_hot = targets_one_hot * vm_c

        # Flatten spatial dimensions: [B, C, H*W]
        probs_flat = probs.view(probs.shape[0], self.num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], self.num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2) # [B, C]
        denominator = (probs_flat ** 2).sum(dim=2) + (targets_flat ** 2).sum(dim=2) # [B, C]

        dice_per_class = (2.0 * intersection + self.eps) / (denominator + self.eps) # [B, C]
        # Macro average over classes, then average over batch
        macro_dice = dice_per_class.mean(dim=1) # [B]
        return (1.0 - macro_dice).mean()


class SegmentationLoss(nn.Module):
    """
    Unified Segmentation Loss combining Focal Loss and Soft Dice Loss:
        L_seg = L_focal + lambda_dice * L_dice
    Supports optional valid_mask for letterbox padding exclusion.
    """
    def __init__(
        self,
        num_classes: int = 5,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        lambda_dice: float = 1.0
    ):
        super().__init__()
        self.focal = ClassWeightedFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = SoftDiceLoss(num_classes=num_classes)
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
