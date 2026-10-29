"""
SAT-EURS Track 1 — V20 Segmentation Loss Functions
Recipe:
1. Scene-Balanced Capped Weighted Cross-Entropy (40%)
2. Multi-Class Class-Weighted Lovasz-Softmax Loss (30%)
3. Soft Dice Loss (20%)
4. Boundary Band Loss (5%)
5. Binary Marine Objectness Loss (5%)

Controlled Single Ablation:
- Class weights derived directly from true 16-scene training pixel distribution.
- Class weights propagated into Lovasz-Softmax loss to directly optimize rare-class Jaccard index.
- All component weights (40/30/20/5/5), Dice, Boundary, and Objectness terms kept strictly identical to V18.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def compute_v20_grounded_class_weights(
    class_pixel_counts: List[int],
    min_weight: float = 0.50,
    max_weight: float = 3.50,
    smoothing: float = 0.50,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Computes empirically grounded, conservatively capped class weights.
    Uses square-root smoothed inverse frequency relative to median class:
      w_c = clip((median_freq / freq_c)^smoothing, min_weight, max_weight)
    """
    counts = np.array(class_pixel_counts, dtype=np.float32)
    total = counts.sum()
    freqs = counts / max(total, 1.0)
    median_freq = float(np.median(freqs))

    raw_ratios = (median_freq / (freqs + 1e-8)) ** smoothing
    capped = np.clip(raw_ratios, min_weight, max_weight)
    return torch.from_numpy(capped).float().to(device=device)


# -----------------------------------------------------------------------------
# CLASS-WEIGHTED LOVASZ-SOFTMAX LOSS
# -----------------------------------------------------------------------------
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of the Lovasz extension for sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-10)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def class_weighted_lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: list,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """
    Multi-class Class-Weighted Lovasz-Softmax loss on flattened valid pixels.
    Args:
        probas: [P, C] class probabilities on valid pixels
        labels: [P] ground truth class IDs in {0..C-1}
        classes: list of class indices to evaluate
        class_weights: [C] tensor of class weights
    """
    if probas.numel() == 0:
        return probas.sum() * 0.0, {}

    C = probas.size(1)
    losses = []
    active_weights = []
    per_class_lovasz = {}

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

        w_c = class_weights[c] if class_weights is not None else torch.tensor(1.0, device=probas.device)
        active_weights.append(w_c)
        per_class_lovasz[c] = float(loss_c.item())

    if not losses:
        return torch.tensor(0.0, device=probas.device, requires_grad=True), {}

    loss_stack = torch.stack(losses)
    weight_stack = torch.stack(active_weights)
    weighted_mean = (loss_stack * weight_stack).sum() / (weight_stack.sum() + 1e-8)
    return weighted_mean, per_class_lovasz


class ClassWeightedLovaszSoftmaxLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, float]]:
        B, C, H, W = logits.shape
        probas = F.softmax(logits, dim=1)
        probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.view(-1)

        valid_mask = (target_flat != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True), {}

        probas_valid = probas_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        return class_weighted_lovasz_softmax_flat(
            probas_valid, target_valid, list(range(self.num_classes)), self.class_weights
        )


# -----------------------------------------------------------------------------
# SOFT DICE LOSS (Identical to V18)
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

        mask_expanded = valid.unsqueeze(1).float()
        probas = probas * mask_expanded
        target_one_hot = target_one_hot * mask_expanded

        dims = (0, 2, 3)
        intersection = torch.sum(probas * target_one_hot, dims)
        cardinality = torch.sum(probas + target_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice)


# -----------------------------------------------------------------------------
# BOUNDARY BAND LOSS (Identical to V18)
# -----------------------------------------------------------------------------
class BoundaryBandLoss(nn.Module):
    def __init__(self, band_width: int = 15):
        super().__init__()
        self.band_width = band_width
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, boundary_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        valid = (target_mask != IGNORE_INDEX).float()
        with torch.no_grad():
            classes = target_mask.clamp(0, 4).unsqueeze(1).float()
            dilated = F.max_pool2d(classes, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool2d(-classes, kernel_size=3, stride=1, padding=1)
            gt_boundary = ((dilated - eroded) > 0).float() * valid.unsqueeze(1)
        return self.bce(boundary_logits, gt_boundary)


# -----------------------------------------------------------------------------
# COMPOSITE V20 SEGMENTATION LOSS
# -----------------------------------------------------------------------------
class SegmentationV20Loss(nn.Module):
    """
    Composite V20 loss with Empirically Grounded Class-Weighted Lovasz & Cross-Entropy:
      0.40 * Capped WCE + 0.30 * CW-Lovasz + 0.20 * SoftDice + 0.05 * Boundary + 0.05 * Objectness
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
        ignore_index: int = IGNORE_INDEX
    ):
        super().__init__()
        self.num_classes = num_classes
        self.w_ce = w_ce
        self.w_lovasz = w_lovasz
        self.w_dice = w_dice
        self.w_bnd = w_bnd
        self.w_obj = w_obj
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.loss_ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        self.loss_lovasz = ClassWeightedLovaszSoftmaxLoss(
            num_classes=num_classes, class_weights=self.class_weights, ignore_index=ignore_index
        )
        self.loss_dice = SoftDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.loss_boundary = BoundaryBandLoss(band_width=15)
        self.loss_objectness = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_mask: torch.Tensor,
        aux_objectness_gt: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        logits = pred_dict["logits"]
        aux_bnd = pred_dict["aux_boundary"]
        aux_obj = pred_dict["aux_objectness"]

        has_valid_main = (target_mask != self.ignore_index).any()
        if has_valid_main:
            l_ce = self.loss_ce(logits, target_mask)
            l_lovasz, per_class_lovasz = self.loss_lovasz(logits, target_mask)
            l_dice = self.loss_dice(logits, target_mask)
        else:
            l_ce = torch.tensor(0.0, device=logits.device, requires_grad=True)
            l_lovasz = torch.tensor(0.0, device=logits.device, requires_grad=True)
            l_dice = torch.tensor(0.0, device=logits.device, requires_grad=True)
            per_class_lovasz = {}

        l_bnd = self.loss_boundary(aux_bnd, target_mask)

        if aux_objectness_gt is None:
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
            "l_obj": l_obj,
            "per_class_lovasz": per_class_lovasz
        }
