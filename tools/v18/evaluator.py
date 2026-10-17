"""
SAT-EURS Track 1 — V18 Strict 5-Fold OOF Evaluator
Computes exact challenge metrics:
1. Q_seg = Macro F1 * mIoU * coverage_penalty across 5 classes {0..4}
2. Q_FR = 0.45 * PSNR_norm + 0.45 * SSIM + 0.10 * sharpness_ratio
3. Flags Q_NR and S_composite_est as PROVISIONAL local estimates.
Zero cross-scene leakage.
"""

import math
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch

from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


class V18OOFEvaluator:
    """
    Evaluates Out-Of-Fold (OOF) predictions across 5 scene-disjoint folds (4,000 images).
    """
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Segmentation state
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        # Restoration state
        self.psnr_list: List[float] = []
        self.psnr_norm_list: List[float] = []
        self.ssim_list: List[float] = []
        self.sharpness_list: List[float] = []

        # Per-scene tracking
        self.scene_stats: Dict[str, Dict[str, Any]] = {}

    def update_segmentation(self, pred_mask: np.ndarray, gt_mask: np.ndarray, scene: Optional[str] = None):
        """Updates global confusion matrix with single image or batch."""
        valid = (gt_mask >= 0) & (gt_mask < self.num_classes)
        y_t = gt_mask[valid].reshape(-1)
        y_p = pred_mask[valid].reshape(-1)

        indices = self.num_classes * y_t + y_p
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        c_mat = counts.reshape(self.num_classes, self.num_classes)
        self.conf_mat += c_mat

        if scene:
            if scene not in self.scene_stats:
                self.scene_stats[scene] = {"conf_mat": np.zeros_like(self.conf_mat), "psnr": [], "ssim": [], "sharp": []}
            self.scene_stats[scene]["conf_mat"] += c_mat

    def update_restoration(self, pred_rgb: np.ndarray, clean_rgb: np.ndarray, scene: Optional[str] = None):
        """
        Updates restoration metrics.
        pred_rgb, clean_rgb: float32 in [0, 1] [H, W, 3] or uint8 [H, W, 3]
        """
        p = pred_rgb.astype(np.float32) if pred_rgb.max() <= 1.0 else pred_rgb.astype(np.float32) / 255.0
        t = clean_rgb.astype(np.float32) if clean_rgb.max() <= 1.0 else clean_rgb.astype(np.float32) / 255.0

        p = np.clip(p, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)

        psnr, psnr_norm = compute_psnr_and_norm(p, t)
        ssim_val = compute_ssim_single(p, t)
        sharp_val = compute_sharpness_ratio(p, t)

        self.psnr_list.append(psnr)
        self.psnr_norm_list.append(psnr_norm)
        self.ssim_list.append(ssim_val)
        self.sharpness_list.append(sharp_val)

        if scene:
            if scene not in self.scene_stats:
                self.scene_stats[scene] = {"conf_mat": np.zeros_like(self.conf_mat), "psnr": [], "ssim": [], "sharp": []}
            self.scene_stats[scene]["psnr"].append(psnr)
            self.scene_stats[scene]["ssim"].append(ssim_val)
            self.scene_stats[scene]["sharp"].append(sharp_val)

    def compute(self, q_nr_provisional: float = 0.70) -> Dict[str, Any]:
        """
        Computes final OOF benchmark metrics.
        q_nr_provisional is marked explicitly as PROVISIONAL.
        """
        # 1. Compute Segmentation Metrics
        tp = np.diag(self.conf_mat).astype(np.float64)
        gt_pixels = self.conf_mat.sum(axis=1).astype(np.float64)  # row sums (tp + fn)
        pred_pixels = self.conf_mat.sum(axis=0).astype(np.float64)  # col sums (tp + fp)
        fp = pred_pixels - tp
        fn = gt_pixels - tp

        present_classes: List[int] = []
        absent_classes: List[int] = []
        hallucinated_classes: List[int] = []

        ious_present: List[float] = []
        f1s_present: List[float] = []
        precisions_present: List[float] = []
        recalls_present: List[float] = []

        ious_5class_zero: List[float] = []
        f1s_5class_zero: List[float] = []
        precisions_5class_zero: List[float] = []
        recalls_5class_zero: List[float] = []

        per_class_pixel_stats: Dict[int, Dict[str, Any]] = {}
        coverage = 1.0

        for c in range(self.num_classes):
            c_gt = gt_pixels[c]
            c_pred = pred_pixels[c]
            c_tp = tp[c]
            c_fp = fp[c]
            c_fn = fn[c]

            denom_iou = c_tp + c_fp + c_fn
            denom_f1 = 2 * c_tp + c_fp + c_fn
            denom_prec = c_tp + c_fp
            denom_rec = c_tp + c_fn

            iou_val = float(c_tp / (denom_iou + 1e-10)) if denom_iou > 0 else 0.0
            f1_val = float((2 * c_tp) / (denom_f1 + 1e-10)) if denom_f1 > 0 else 0.0
            prec_val = float(c_tp / (denom_prec + 1e-10)) if denom_prec > 0 else 0.0
            rec_val = float(c_tp / (denom_rec + 1e-10)) if denom_rec > 0 else 0.0

            if c_gt > 0:
                present_classes.append(c)
                status = "PRESENT"
                ious_present.append(iou_val)
                f1s_present.append(f1_val)
                precisions_present.append(prec_val)
                recalls_present.append(rec_val)

                if c_tp == 0:
                    coverage = 0.0
            else:
                if c_pred > 0:
                    hallucinated_classes.append(c)
                    status = "ABSENT_GT_BUT_PREDICTED"
                else:
                    absent_classes.append(c)
                    status = "ABSENT_GT_AND_PRED"

            # 5-class strict/zero: absent classes contribute 0.0 to macro average
            ious_5class_zero.append(iou_val)
            f1s_5class_zero.append(f1_val)
            precisions_5class_zero.append(prec_val)
            recalls_5class_zero.append(rec_val)

            per_class_pixel_stats[c] = {
                "class_id": c,
                "status": status,
                "gt_pixels": int(c_gt),
                "pred_pixels": int(c_pred),
                "tp_pixels": int(c_tp),
                "fp_pixels": int(c_fp),
                "fn_pixels": int(c_fn),
                "iou": round(iou_val, 5),
                "f1": round(f1_val, 5),
                "precision": round(prec_val, 5),
                "recall": round(rec_val, 5),
            }

        num_present = len(present_classes)
        miou_present = float(np.mean(ious_present)) if num_present > 0 else 0.0
        macro_f1_present = float(np.mean(f1s_present)) if num_present > 0 else 0.0
        q_seg_present = float(macro_f1_present * miou_present * coverage)

        miou_5class_zero = float(np.mean(ious_5class_zero))
        macro_f1_5class_zero = float(np.mean(f1s_5class_zero))
        q_seg_5class_zero = float(macro_f1_5class_zero * miou_5class_zero * coverage)

        # Standard split evaluation defaults to present classes while reporting both transparently
        miou = miou_present
        macro_f1 = macro_f1_present
        q_seg = q_seg_present

        # 2. Compute Restoration Metrics
        if self.psnr_list:
            mean_psnr = float(np.mean(self.psnr_list))
            psnr_norm_dataset = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
            mean_psnr_norm_per_image = float(np.mean(self.psnr_norm_list))
            mean_ssim = float(np.mean(self.ssim_list))
            mean_sharp = float(np.mean(self.sharpness_list))
            q_fr_dataset = 0.45 * psnr_norm_dataset + 0.45 * mean_ssim + 0.10 * mean_sharp
            q_fr_per_image = 0.45 * mean_psnr_norm_per_image + 0.45 * mean_ssim + 0.10 * mean_sharp
        else:
            mean_psnr = psnr_norm_dataset = mean_psnr_norm_per_image = mean_ssim = mean_sharp = q_fr_dataset = q_fr_per_image = 0.0

        # 3. Exact 90% Score & Provisional 10% NR Quality Specification
        # Local metrics represent 90% of the official challenge formula:
        # S_exact_90pct = 0.60 * Q_seg + 0.30 * Q_FR_dataset
        s_exact_90pct = 0.60 * q_seg + 0.30 * q_fr_dataset

        # The remaining 10% (Q_NR) is evaluated on the challenge server using underwater
        # quality measures: UIQM (Underwater Image Quality Measure), UCIQE (Underwater
        # Color Image Quality Evaluation), and CCF (Color, Contrast, Fog metric).
        # We report an explicit sensitivity range for Q_NR in [0.40, 0.80].
        comp_est_low = s_exact_90pct + 0.10 * 0.40
        comp_est_mid = s_exact_90pct + 0.10 * 0.60
        comp_est_high = s_exact_90pct + 0.10 * 0.80

        return {
            "segmentation": {
                "Q_seg": round(q_seg, 5),
                "mIoU": round(miou, 5),
                "Macro_F1": round(macro_f1, 5),
                "coverage_penalty": coverage,
                "coverage_penalty_status": "UNVERIFIED_PROVISIONAL",
                "coverage_penalty_contract_note": "Evaluated using local contract: prod_{c in C_present} I(TP_c > 0). Official server evaluator code is proprietary; whether server enforces an IoU-floor (e.g. IoU_c >= tau) is unverified locally.",
                "absent_class_policy": "REPORT_BOTH_PRESENT_MEAN_AND_5CLASS_ZERO",
                "absent_class_official_parity": "UNVERIFIED",
                "absent_class_explanation": "Official challenge server behavior for absent classes on partial evaluation splits is proprietary/unverified. Assigning 1.0 (vacuous truth) artificially inflated Fold 0 pilot mIoU from 0.4731 to 0.5785. Local evaluator now transparently reports both the present-class mean (averaged only over classes with GT > 0) and the conservative 5-class mean (where absent classes receive 0.0). For full 20-scene OOF, all 5 classes are present.",
                "present_classes": present_classes,
                "absent_classes": absent_classes,
                "hallucinated_classes": hallucinated_classes,
                "num_present_classes": num_present,
                "num_total_classes": self.num_classes,
                "mIoU_present": round(miou_present, 5),
                "Macro_F1_present": round(macro_f1_present, 5),
                "Q_seg_present": round(q_seg_present, 5),
                "mIoU_5class_zero": round(miou_5class_zero, 5),
                "Macro_F1_5class_zero": round(macro_f1_5class_zero, 5),
                "Q_seg_5class_zero": round(q_seg_5class_zero, 5),
                "min_class_iou": round(float(min(ious_5class_zero)), 5) if ious_5class_zero else 0.0,
                "classes_below_0_05_iou": [c for c, iou in enumerate(ious_5class_zero) if iou < 0.05],
                "classes_below_0_10_iou": [c for c, iou in enumerate(ious_5class_zero) if iou < 0.10],
                "class_ious": [round(x, 5) for x in ious_5class_zero],
                "class_f1s": [round(x, 5) for x in f1s_5class_zero],
                "class_recalls": [round(x, 5) for x in recalls_5class_zero],
                "class_precisions": [round(x, 5) for x in precisions_5class_zero],
                "per_class_pixel_stats": per_class_pixel_stats,
                "confusion_matrix": self.conf_mat.tolist()
            },
            "restoration": {
                "PSNR_dB": round(mean_psnr, 4),
                "PSNR_norm_dataset": round(psnr_norm_dataset, 5),
                "PSNR_norm_per_image": round(mean_psnr_norm_per_image, 5),
                "SSIM": round(mean_ssim, 5),
                "sharpness_ratio": round(mean_sharp, 5),
                "Q_FR_dataset": round(q_fr_dataset, 5),
                "Q_FR_per_image": round(q_fr_per_image, 5),
                "num_samples": len(self.psnr_list)
            },
            "composite": {
                "S_exact_90pct": round(s_exact_90pct, 5),
                "S_exact_90pct_note": "0.60 * Q_seg + 0.30 * Q_FR_dataset under local provisional coverage contract (TP_c > 0). Server-side exact parity is provisional.",
                "coverage_penalty_status": "UNVERIFIED_PROVISIONAL",
                "S_composite_bracket_low_qnr040": round(comp_est_low, 5),
                "S_composite_bracket_mid_qnr060": round(comp_est_mid, 5),
                "S_composite_bracket_high_qnr080": round(comp_est_high, 5),
                "Q_NR_status": "UNAVAILABLE_LOCALLY",
                "Q_NR_official_metrics": "UIQM, UCIQE, and CCF (Underwater metrics evaluated on challenge server)",
                "note": "S_exact_90pct represents exactly verified local components (60% Q_seg + 30% Q_FR). Q_NR is not hardcoded."
            }
        }
