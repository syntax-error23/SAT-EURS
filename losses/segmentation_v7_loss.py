"""
Segmentation V7 Loss: Direct Competition Metric Objective
Stage 3 of SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Objective:
Directly maximizes Q_seg = mIoU * Macro_F1 by combining:
1. Class-Weighted Cross-Entropy (0.30)
2. Present-Class Lovasz-Softmax (0.35) — direct surrogate for mIoU
3. Batch-Level Soft Macro-F1 (0.25) — direct surrogate for Macro F1
4. Boundary-Band BCE Loss (0.10) — concentrates gradients on the 15px boundary strip
"""

import math
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.v4_losses import PresentClassLovaszSoftmaxLoss

class BatchSoftMacroF1Loss(nn.Module):
    """
    Batch-Level Differentiable Macro F1 Loss.
    Optimizes Macro F1 directly over classes present in the mini-batch ground truth.
    """
    def __init__(self, num_classes: int = 5, eps: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)  # [B, C, H, W]
        b, c, h, w = probs.shape

        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, c)  # [N, C]
        targets_flat = targets.reshape(-1)                     # [N]

        f1_scores = []
        for cls_id in range(self.num_classes):
            gt_c = (targets_flat == cls_id).float()
            if gt_c.sum() == 0:
                continue

            pred_c = probs_flat[:, cls_id]
            tp = (pred_c * gt_c).sum()
            fp = (pred_c * (1.0 - gt_c)).sum()
            fn = ((1.0 - pred_c) * gt_c).sum()

            soft_f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
            f1_scores.append(soft_f1)

        if not f1_scores:
            return (logits.float() * 0.0).sum()

        mean_f1 = torch.stack(f1_scores).mean()
        return 1.0 - mean_f1

class BoundaryBandLoss(nn.Module):
    """
    Class-balanced boundary loss targeting the 15px transition zone identified in Stage 1 audit.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def extract_boundaries(self, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = targets.unsqueeze(1).float()  # [B, 1, H, W]
            pad = self.kernel_size // 2
            m_pad = F.pad(m, (pad, pad, pad, pad), mode="replicate")
            dilated = F.max_pool2d(m_pad, kernel_size=self.kernel_size, stride=1)
            eroded = -F.max_pool2d(-m_pad, kernel_size=self.kernel_size, stride=1)
            boundary = (dilated != eroded).float()
            # 1-step dilation to cover ~5-15px transition zone
            boundary_band = F.max_pool2d(F.pad(boundary, (2, 2, 2, 2), mode="replicate"), kernel_size=5, stride=1)
            return boundary_band

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device_type = logits.device.type if logits.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits_f = logits.float()
            b_gt = self.extract_boundaries(targets)  # [B, 1, H, W]
            probs = F.softmax(logits_f, dim=1)
            # Compute spatial gradient magnitude of predicted probabilities as predicted boundary
            gx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
            gy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
            gx_pad = F.pad(gx, (0, 1, 0, 0))
            gy_pad = F.pad(gy, (0, 0, 0, 1))
            pred_b = torch.clamp((gx_pad + gy_pad).sum(dim=1, keepdim=True), 1e-6, 1.0 - 1e-6)

            pos_mask = (b_gt > 0.5)
            neg_mask = ~pos_mask
            loss_pos = F.binary_cross_entropy(pred_b[pos_mask], b_gt[pos_mask]) if pos_mask.any() else torch.tensor(0.0, device=logits.device)
            loss_neg = F.binary_cross_entropy(pred_b[neg_mask], b_gt[neg_mask]) if neg_mask.any() else torch.tensor(0.0, device=logits.device)
            return 0.7 * loss_pos + 0.3 * loss_neg

class SegmentationV7Loss(nn.Module):
    """
    Unified Direct Q_seg Metric Loss Suite:
    0.30 WCE + 0.35 Lovasz + 0.25 SoftMacroF1 + 0.10 Boundary
    """
    def __init__(self, num_classes: int = 5, class_weights: torch.Tensor = None):
        super().__init__()
        if class_weights is None:
            class_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00])
        self.register_buffer("class_weights", class_weights)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        self.lovasz = PresentClassLovaszSoftmaxLoss(num_classes=num_classes)
        self.soft_f1 = BatchSoftMacroF1Loss(num_classes=num_classes)
        self.boundary = BoundaryBandLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        logits_f = logits.float()
        l_ce = self.ce(logits_f, targets)
        l_lov = self.lovasz(logits_f, targets)
        l_f1 = self.soft_f1(logits_f, targets)
        l_bnd = self.boundary(logits_f, targets)

        total_loss = 0.30 * l_ce + 0.35 * l_lov + 0.25 * l_f1 + 0.10 * l_bnd

        loss_dict = {
            "loss_ce": round(l_ce.item(), 4),
            "loss_lovasz": round(l_lov.item(), 4),
            "loss_soft_f1": round(l_f1.item(), 4),
            "loss_boundary": round(l_bnd.item(), 4),
            "total_loss": round(total_loss.item(), 4)
        }
        return total_loss, loss_dict
