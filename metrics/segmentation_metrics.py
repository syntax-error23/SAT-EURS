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

        # Per-class Precision and Recall
        denom_prec = tp + fp
        per_class_precision = np.where(denom_prec > 0, tp / (denom_prec + eps), 1.0)
        denom_rec = tp + fn
        per_class_recall = np.where(denom_rec > 0, tp / (denom_rec + eps), 1.0)

        miou = float(np.mean(per_class_iou))
        macro_f1 = float(np.mean(per_class_f1))

        # Official MaCVi Coverage Penalty: drives Q_seg to 0 if any present GT class is ignored (TP == 0)
        gt_present = cm.sum(axis=1) > 0
        if gt_present.sum() > 0:
            soft_coverage_penalty = float(np.sum((gt_present) & (tp > 0)) / gt_present.sum())
            hard_coverage_penalty = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0
        else:
            soft_coverage_penalty = 1.0
            hard_coverage_penalty = 1.0

        official_q_seg = macro_f1 * miou
        q_seg = official_q_seg * hard_coverage_penalty

        return {
            'mIoU': round(miou, 4),
            'macro_F1': round(macro_f1, 4),
            'per_class_IoU': [round(float(v), 4) for v in per_class_iou],
            'per_class_F1': [round(float(v), 4) for v in per_class_f1],
            'per_class_precision': [round(float(v), 4) for v in per_class_precision],
            'per_class_recall': [round(float(v), 4) for v in per_class_recall],
            'coverage_penalty': round(hard_coverage_penalty, 4),
            'soft_coverage_penalty': round(soft_coverage_penalty, 4),
            'official_Q_seg': round(official_q_seg, 4),
            'Q_seg': round(q_seg, 4),
            'confusion_matrix': cm.tolist()
        }

