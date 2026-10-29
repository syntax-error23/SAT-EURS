"""
Evaluation Metrics Module for SAT-EURS.

Supports:
1. Segmentation Metrics:
   - Confusion matrix accumulation [5, 5]
   - Per-class IoU (Classes 0, 1, 2, 3, 4)
   - Explicit exposure of rare-class IoU: Class 3 (Elasmobranch) and Class 4 (Other)
   - Macro IoU (mean across 5 classes)
   - Pixel Accuracy
   - Per-class F1 and Macro F1
   - Padding exclusion via optional valid_mask
2. Restoration Metrics:
   - PSNR (dB) and Normalized PSNR
   - SSIM across RGB channels
   - Sharpness Ratio (Laplacian variance ratio)
   - Padding exclusion via optional valid_mask

Accepts:
   - Raw logits [B, 5, H, W] or predicted masks [B, H, W]
   - Integer target masks [B, H, W]
   - Normalized RGB images [B, 3, H, W] in [0.0, 1.0]
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    num_classes: int = 5
) -> Dict[str, Any]:
    """
    Computes segmentation metrics for a single sample or batch.

    Args:
        pred: [B, num_classes, H, W] raw logits OR [B, H, W] integer predictions.
        target: [B, H, W] ground truth integer class labels.
        valid_mask: Optional [B, H, W] or [H, W] boolean mask (True for content, False for padding).
        num_classes: Number of semantic classes (default: 5).

    Returns:
        Dict containing pixel_accuracy, macro_iou, per_class_iou, iou_class_0..4, macro_f1.
    """
    if pred.ndim == 4:
        pred_labels = pred.argmax(dim=1)
    elif pred.ndim == 3 or pred.ndim == 2:
        pred_labels = pred
    else:
        raise ValueError(f"pred must have 2, 3, or 4 dimensions, got {pred.shape}")

    pred_np = pred_labels.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    # Filter out-of-bounds labels and apply valid mask to exclude padding if provided
    valid = (target_np >= 0) & (target_np < num_classes) & (pred_np >= 0) & (pred_np < num_classes)
    if valid_mask is not None:
        vm_np = valid_mask.detach().cpu().numpy().astype(bool)
        if vm_np.ndim == 2 and pred_np.ndim == 3:
            vm_np = np.broadcast_to(vm_np, pred_np.shape)
        valid = valid & vm_np

    p_flat = pred_np[valid].astype(int)
    t_flat = target_np[valid].astype(int)

    if len(t_flat) == 0:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        pixel_acc = 0.0
    else:
        cm = np.bincount(
            num_classes * t_flat + p_flat,
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        pixel_acc = float(np.sum(p_flat == t_flat) / len(t_flat))

    eps = 1e-10
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    per_class_iou = [
        float(tp[c] / (denom_iou[c] + eps)) if denom_iou[c] > 0 else 1.0
        for c in range(num_classes)
    ]

    denom_f1 = 2 * tp + fp + fn
    per_class_f1 = [
        float((2 * tp[c]) / (denom_f1[c] + eps)) if denom_f1[c] > 0 else 1.0
        for c in range(num_classes)
    ]

    macro_iou = float(np.mean(per_class_iou))
    macro_f1 = float(np.mean(per_class_f1))

    return {
        "pixel_accuracy": round(pixel_acc, 6),
        "macro_iou": round(macro_iou, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class_iou": [round(v, 6) for v in per_class_iou],
        "per_class_f1": [round(v, 6) for v in per_class_f1],
        "iou_class_0": round(per_class_iou[0], 6),
        "iou_class_1": round(per_class_iou[1], 6),
        "iou_class_2": round(per_class_iou[2], 6),
        "iou_class_3": round(per_class_iou[3], 6),  # Explicit rare Class 3
        "iou_class_4": round(per_class_iou[4], 6),  # Explicit rare Class 4
        "confusion_matrix": cm
    }


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Computes PSNR and normalized PSNR on [H, W, C] in range [0, 1].
    """
    p = np.clip(pred, 0.0, 1.0)
    t = np.clip(target, 0.0, 1.0)

    if valid_mask is not None:
        vm = valid_mask.astype(bool)
        if vm.ndim == 2:
            vm = vm[:, :, None]
        p_valid = p[np.broadcast_to(vm, p.shape)]
        t_valid = t[np.broadcast_to(vm, t.shape)]
        if len(p_valid) == 0:
            return 0.0, 0.0
        mse = float(np.mean((p_valid - t_valid) ** 2))
    else:
        mse = float(np.mean((p - t) ** 2))

    if mse < 1e-10:
        psnr = 100.0
        psnr_norm = 1.0
    else:
        psnr = 10.0 * math.log10(1.0 / mse)
        # Normalized window [15.0, 35.0] dB
        psnr_norm = min(1.0, max(0.0, (psnr - 15.0) / (35.0 - 15.0)))

    return psnr, psnr_norm


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> float:
    """
    Computes SSIM across RGB channels.
    If valid_mask is provided, unpadded content region is sliced out.
    """
    p = np.clip(pred, 0.0, 1.0)
    t = np.clip(target, 0.0, 1.0)

    if valid_mask is not None:
        y_idx, x_idx = np.where(valid_mask.astype(bool))
        if len(y_idx) > 0:
            ymin, ymax = y_idx.min(), y_idx.max() + 1
            xmin, xmax = x_idx.min(), x_idx.max() + 1
            p = p[ymin:ymax, xmin:xmax]
            t = t[ymin:ymax, xmin:xmax]

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    channels_ssim = []
    num_ch = p.shape[2] if p.ndim == 3 else 1
    for c in range(num_ch):
        p_c = p[:, :, c] if p.ndim == 3 else p
        t_c = t[:, :, c] if t.ndim == 3 else t

        mu_p = cv2.GaussianBlur(p_c, (11, 11), 1.5)
        mu_t = cv2.GaussianBlur(t_c, (11, 11), 1.5)

        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_p_sq = cv2.GaussianBlur(p_c * p_c, (11, 11), 1.5) - mu_p_sq
        sigma_t_sq = cv2.GaussianBlur(t_c * t_c, (11, 11), 1.5) - mu_t_sq
        sigma_pt = cv2.GaussianBlur(p_c * t_c, (11, 11), 1.5) - mu_pt

        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
            (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2) + 1e-10
        )
        channels_ssim.append(float(ssim_map.mean()))

    return float(np.mean(channels_ssim))


def compute_sharpness_ratio(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> float:
    """Computes ratio of Laplacian variance between restored and clean images."""
    p = np.clip(pred, 0.0, 1.0)
    t = np.clip(target, 0.0, 1.0)

    if valid_mask is not None:
        y_idx, x_idx = np.where(valid_mask.astype(bool))
        if len(y_idx) > 0:
            ymin, ymax = y_idx.min(), y_idx.max() + 1
            xmin, xmax = x_idx.min(), x_idx.max() + 1
            p = p[ymin:ymax, xmin:xmax]
            t = t[ymin:ymax, xmin:xmax]

    if p.ndim == 3 and p.shape[2] == 3:
        p_gray = cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        t_gray = cv2.cvtColor((t * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        p_gray = (p * 255).astype(np.uint8)
        t_gray = (t * 255).astype(np.uint8)

    lap_p = cv2.Laplacian(p_gray, cv2.CV_64F)
    lap_t = cv2.Laplacian(t_gray, cv2.CV_64F)

    var_p = float(lap_p.var())
    var_t = float(lap_t.var())

    eps = 1e-7
    if var_t < eps:
        return 1.0 if var_p < eps else 0.0

    ratio = var_p / var_t
    return float(min(1.0, max(0.0, ratio)))


def compute_restoration_metrics(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Computes restoration metrics for a batched [B, 3, H, W] tensor.
    """
    pred_np = pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 3]
    target_np = target_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)
    vm_np = valid_mask.detach().cpu().numpy() if valid_mask is not None else None

    b = pred_np.shape[0]
    psnr_list, psnr_norm_list, ssim_list, sharp_list = [], [], [], []

    for i in range(b):
        p_i = pred_np[i]
        t_i = target_np[i]
        vm_i = vm_np[i] if (vm_np is not None and vm_np.ndim == 3) else vm_np

        psnr_val, psnr_norm_val = compute_psnr(p_i, t_i, vm_i)
        ssim_val = compute_ssim(p_i, t_i, vm_i)
        sharp_val = compute_sharpness_ratio(p_i, t_i, vm_i)

        psnr_list.append(psnr_val)
        psnr_norm_list.append(psnr_norm_val)
        ssim_list.append(ssim_val)
        sharp_list.append(sharp_val)

    mean_psnr = float(np.mean(psnr_list))
    mean_psnr_norm = float(np.mean(psnr_norm_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_sharp = float(np.mean(sharp_list))

    q_rest = 0.45 * mean_psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp

    return {
        "psnr": round(mean_psnr, 2),
        "psnr_norm": round(mean_psnr_norm, 4),
        "ssim": round(mean_ssim, 4),
        "sharpness_ratio": round(mean_sharp, 4),
        "q_rest": round(q_rest, 4)
    }


class MetricAccumulator:
    """
    Running accumulator for evaluation loops across the full validation dataset.
    Accumulates confusion matrices and restoration scores.
    """
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.psnr_list: List[float] = []
        self.psnr_norm_list: List[float] = []
        self.ssim_list: List[float] = []
        self.sharpness_list: List[float] = []

    def update(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        pred_logits_or_mask: torch.Tensor,
        target_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ):
        # 1. Update Segmentation
        seg_res = compute_segmentation_metrics(
            pred=pred_logits_or_mask,
            target=target_mask,
            valid_mask=valid_mask,
            num_classes=self.num_classes
        )
        self.confusion_matrix += seg_res["confusion_matrix"]

        # 2. Update Restoration per image
        pred_np = pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, 3]
        target_np = target_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)
        vm_np = valid_mask.detach().cpu().numpy() if valid_mask is not None else None

        b = pred_np.shape[0]
        for i in range(b):
            p_i = pred_np[i]
            t_i = target_np[i]
            vm_i = vm_np[i] if (vm_np is not None and vm_np.ndim == 3) else vm_np

            psnr_val, psnr_norm_val = compute_psnr(p_i, t_i, vm_i)
            ssim_val = compute_ssim(p_i, t_i, vm_i)
            sharp_val = compute_sharpness_ratio(p_i, t_i, vm_i)

            self.psnr_list.append(psnr_val)
            self.psnr_norm_list.append(psnr_norm_val)
            self.ssim_list.append(ssim_val)
            self.sharpness_list.append(sharp_val)

    def compute(self) -> Dict[str, Any]:
        """Calculates dataset-level metrics."""
        cm = self.confusion_matrix
        eps = 1e-10

        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        total_valid = cm.sum()
        pixel_accuracy = float(tp.sum() / total_valid) if total_valid > 0 else 0.0

        denom_iou = tp + fp + fn
        per_class_iou = [
            float(tp[c] / (denom_iou[c] + eps)) if denom_iou[c] > 0 else 1.0
            for c in range(self.num_classes)
        ]
        macro_iou = float(np.mean(per_class_iou))

        denom_f1 = 2 * tp + fp + fn
        per_class_f1 = [
            float((2 * tp[c]) / (denom_f1[c] + eps)) if denom_f1[c] > 0 else 1.0
            for c in range(self.num_classes)
        ]
        macro_f1 = float(np.mean(per_class_f1))

        # Restoration dataset means
        mean_psnr = float(np.mean(self.psnr_list)) if self.psnr_list else 0.0
        mean_psnr_norm = float(np.mean(self.psnr_norm_list)) if self.psnr_norm_list else 0.0
        mean_ssim = float(np.mean(self.ssim_list)) if self.ssim_list else 0.0
        mean_sharp = float(np.mean(self.sharpness_list)) if self.sharpness_list else 0.0
        q_rest = 0.45 * mean_psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp

        # Q_seg with soft coverage ratio (proportional detection rather than all-or-nothing kill switch)
        gt_present = cm.sum(axis=1) > 0
        present_count = int(np.sum(gt_present))
        detected_count = int(np.sum((gt_present) & (tp > 0)))
        coverage_ratio = float(detected_count / present_count) if present_count > 0 else 1.0
        q_seg = macro_f1 * macro_iou * coverage_ratio

        return {
            # Core restoration
            "psnr": round(mean_psnr, 2),
            "psnr_norm": round(mean_psnr_norm, 4),
            "ssim": round(mean_ssim, 4),
            "sharpness_ratio": round(mean_sharp, 4),
            "q_rest": round(q_rest, 4),

            # Core segmentation
            "pixel_accuracy": round(pixel_accuracy, 6),
            "macro_iou": round(macro_iou, 6),
            "macro_f1": round(macro_f1, 6),
            "coverage_penalty": round(coverage_ratio, 4),
            "coverage_ratio": round(coverage_ratio, 4),
            "q_seg": round(q_seg, 4),

            # Explicit per-class IoU breakdown
            "iou_class_0": round(per_class_iou[0], 6),
            "iou_class_1": round(per_class_iou[1], 6),
            "iou_class_2": round(per_class_iou[2], 6),
            "iou_class_3": round(per_class_iou[3], 6),  # Rare Class 3 (Elasmobranch)
            "iou_class_4": round(per_class_iou[4], 6),  # Rare Class 4 (Other)
            "per_class_iou": [round(v, 6) for v in per_class_iou],
            "per_class_f1": [round(v, 6) for v in per_class_f1],
            "num_evaluated_samples": len(self.psnr_list)
        }
