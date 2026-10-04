"""
Segmentation Metrics Evaluator
Accumulates global 5x5 confusion matrix across the evaluation set,
computing per-class IoU, per-class F1, mIoU, macro F1, and Q_seg.
"""

from typing import Dict, Any, List
import numpy as np
import torch


class SegmentationEvaluator:
    """
    Evaluator accumulating global confusion matrix across evaluation set.
    """
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred_logits_or_mask: torch.Tensor, target_mask: torch.Tensor):
        """
        Args:
            pred_logits_or_mask: [B, C, H, W] logits or [B, H, W] integer predictions.
            target_mask: [B, H, W] ground truth class labels.
        """
        if pred_logits_or_mask.ndim == 4:
            pred = pred_logits_or_mask.argmax(dim=1)
        else:
            pred = pred_logits_or_mask

        pred_np = pred.detach().cpu().numpy().flatten()
        target_np = target_mask.detach().cpu().numpy().flatten()

        mask = (target_np >= 0) & (target_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * target_np[mask].astype(int) + pred_np[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist

    def compute(self) -> Dict[str, Any]:
        """Compute metrics from the accumulated confusion matrix."""
        cm = self.confusion_matrix
        eps = 1e-10

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Per-class IoU
        denominator_iou = tp + fp + fn
        per_class_iou = np.where(denominator_iou > 0, tp / (denominator_iou + eps), 1.0)

        # Per-class F1
        denominator_f1 = 2 * tp + fp + fn
        per_class_f1 = np.where(denominator_f1 > 0, (2 * tp) / (denominator_f1 + eps), 1.0)

        miou = float(np.mean(per_class_iou))
        macro_f1 = float(np.mean(per_class_f1))

        # Coverage Penalty (Provisional): penalizes classes present in GT that received 0 TP
        gt_present = cm.sum(axis=1) > 0
        missed_classes = np.sum((gt_present) & (tp == 0))
        coverage_penalty = 0.0 if missed_classes > 0 else 1.0

        q_seg = macro_f1 * miou * coverage_penalty

        return {
            'mIoU': round(miou, 4),
            'macro_F1': round(macro_f1, 4),
            'per_class_IoU': [round(float(v), 4) for v in per_class_iou],
            'per_class_F1': [round(float(v), 4) for v in per_class_f1],
            'coverage_penalty': coverage_penalty,
            'Q_seg': round(q_seg, 4),
            'confusion_matrix': cm.tolist()
        }
