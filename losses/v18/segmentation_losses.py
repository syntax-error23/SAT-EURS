"""
SAT-EURS Track 1 — V18 Segmentation Loss Functions
Recipe:
1. Scene-Balanced Capped Weighted Cross-Entropy (40%)
2. Multi-Class Lovasz-Softmax Loss (30%)
3. Soft Dice Loss (20%)
4. Boundary Band Loss (5%)
5. Binary Marine Objectness Loss (5%)

All losses strictly support ignore_index = 255 (masked with zero gradient).
Class weights are strictly capped to prevent numerical explosion on rare classes.
"""

from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


def compute_capped_class_weights(
    class_pixel_counts: List[int],
    min_weight: float = 0.50,
    max_weight: float = 4.00,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Computes capped class weights from scene-balanced pixel counts.
    Prevents unbounded inverse-frequency weights from destabilizing background / common classes.
    """
    counts = np.array(class_pixel_counts, dtype=np.float32)
    total = counts.sum()
    freqs = counts / max(total, 1.0)
    median_freq = float(np.median(freqs))

    raw_weights = median_freq / (freqs + 1e-6)
    capped = np.clip(raw_weights, min_weight, max_weight)
    return torch.from_numpy(capped).float().to(device=device)


# -----------------------------------------------------------------------------
# LOVASZ-SOFTMAX LOSS
# -----------------------------------------------------------------------------
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovasz extension for sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes: list) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss on flattened valid pixels.
    Args:
        probas: [P, C] class probabilities on valid pixels
        labels: [P] ground truth class IDs in {0..C-1}
        classes: list of class indices to average over
    """
    if probas.numel() == 0:
        return probas * 0.0

    C = probas.size(1)
    losses = []
    for c in classes:
        target_c = (labels == c).float()
        if target_c.sum() == 0:
            continue
        prob_c = probas[:, c]
        errors = (target_c - prob_c).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        gt_sorted = target_c[perm]
        grad = lovasz_grad(gt_sorted)
        loss_c = torch.dot(errors_sorted, grad)
        losses.append(loss_c)

    if not losses:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, num_classes: int = 5, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], target: [B, H, W]
        B, C, H, W = logits.shape
        probas = F.softmax(logits, dim=1)
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)  # [P, C]
        target_flat = target.view(-1)                            # [P]

        valid_mask = (target_flat != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        probas_valid = probas_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        return lovasz_softmax_flat(probas_valid, target_valid, list(range(self.num_classes)))


# -----------------------------------------------------------------------------
# SOFT DICE LOSS
# -----------------------------------------------------------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 5, smooth: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probas = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target_clamped = target.clamp(0, self.num_classes - 1)
        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Mask out ignore index
        mask_expanded = valid.unsqueeze(1).float()
        probas = probas * mask_expanded
        target_one_hot = target_one_hot * mask_expanded

        dims = (0, 2, 3)
        intersection = torch.sum(probas * target_one_hot, dims)
        cardinality = torch.sum(probas + target_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)


# -----------------------------------------------------------------------------
# BOUNDARY BAND LOSS
# -----------------------------------------------------------------------------
class BoundaryBandLoss(nn.Module):
    """
    Supervises the 1-channel boundary head within the 15px transition band around object boundaries.
    """
    def __init__(self, band_width: int = 15):
        super().__init__()
        self.band_width = band_width
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, boundary_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        # boundary_logits: [B, 1, H, W]
        # target_mask: [B, H, W]
        b, c, h, w = boundary_logits.shape
        valid = (target_mask != IGNORE_INDEX).float()

        # Compute empirical binary boundary GT using morphological gradient on GPU
        pad = self.band_width // 2
        # Max pool - Min pool on one-hot slices
        with torch.no_grad():
            classes = target_mask.clamp(0, 4).unsqueeze(1).float()
            dilated = F.max_pool2d(classes, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool2d(-classes, kernel_size=3, stride=1, padding=1)
            gt_boundary = ((dilated - eroded) > 0).float() * valid.unsqueeze(1)

        loss = self.bce(boundary_logits, gt_boundary)
        return loss


# -----------------------------------------------------------------------------
# FULL V18 COMPOSITE SEGMENTATION LOSS
# -----------------------------------------------------------------------------
class SegmentationV18Loss(nn.Module):
    """
    Full composite loss for EURSAPEXV18Segmenter:
      0.40 * Capped WCE + 0.30 * Lovasz + 0.20 * SoftDice + 0.05 * Boundary + 0.05 * Objectness
    """
    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        w_ce: float = 0.40,
        w_lovasz: float = 0.30,
        w_dice: float = 0.20,
        w_bnd: float = 0.05,
        w_obj: float = 0.05,
        w_boundary: Optional[float] = None,
        w_objectness: Optional[float] = None,
        ignore_index: int = IGNORE_INDEX
    ):
        super().__init__()
        self.w_ce = w_ce
        self.w_lovasz = w_lovasz
        self.w_dice = w_dice
        self.w_bnd = w_boundary if w_boundary is not None else w_bnd
        self.w_obj = w_objectness if w_objectness is not None else w_obj
        self.ignore_index = ignore_index

        self.loss_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.loss_lovasz = LovaszSoftmaxLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.loss_dice = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.loss_boundary = BoundaryBandLoss(band_width=15)
        self.loss_objectness = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        aux_objectness_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        logits = pred_dict["logits"]
        aux_bnd = pred_dict["aux_boundary"]
        aux_obj = pred_dict["aux_objectness"]

        # 1. Main 5-Class Losses (Guarded for partial-label datasets with zero main classes)
        has_valid_main = (target_mask != self.ignore_index).any()
        if has_valid_main:
            l_ce = self.loss_ce(logits, target_mask)
            l_lovasz = self.loss_lovasz(logits, target_mask)
            l_dice = self.loss_dice(logits, target_mask)
        else:
            l_ce = torch.tensor(0.0, device=logits.device, requires_grad=True)
            l_lovasz = torch.tensor(0.0, device=logits.device, requires_grad=True)
            l_dice = torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 2. Auxiliary Boundary Loss
        l_bnd = self.loss_boundary(aux_bnd, target_mask)

        # 3. Auxiliary Objectness Loss
        if aux_objectness_gt is None:
            # Derive binary objectness (classes 1, 2, 3, 4 = foreground, 0 = background, 255 = ignore)
            valid = (target_mask != self.ignore_index)
            gt_obj = ((target_mask > 0) & valid).float().unsqueeze(1)
        else:
            gt_obj = aux_objectness_gt

        l_obj = self.loss_objectness(aux_obj, gt_obj)

        total_loss = (
            self.w_ce * l_ce
            + self.w_lovasz * l_lovasz
            + self.w_dice * l_dice
            + self.w_bnd * l_bnd
            + self.w_obj * l_obj
        )

        return {
            "loss": total_loss,
            "l_ce": l_ce,
            "l_lovasz": l_lovasz,
            "l_dice": l_dice,
            "l_bnd": l_bnd,
            "l_obj": l_obj
        }
