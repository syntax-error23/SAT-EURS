"""
EURS-Net V2 Loss Package
"""

from losses.segmentation_loss import ClassWeightedFocalLoss, SoftDiceLoss, SegmentationLoss
from losses.msssim import ms_ssim, MSSSIMLoss
from losses.restoration_loss import CharbonnierLoss, LaplacianLoss, RestorationLoss
from losses.multi_task_loss import MultiTaskLoss

__all__ = [
    "ClassWeightedFocalLoss",
    "SoftDiceLoss",
    "SegmentationLoss",
    "ms_ssim",
    "MSSSIMLoss",
    "CharbonnierLoss",
    "LaplacianLoss",
    "RestorationLoss",
    "MultiTaskLoss"
]
