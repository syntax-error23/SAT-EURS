"""
Restoration Metrics Evaluator
Calculates per-image metrics (PSNR_norm, SSIM, Sharpness Ratio) and computes
the arithmetic mean across the evaluation set to yield Q_rest.
"""

from typing import Dict, Any, List
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_psnr_and_norm(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """
    Computes PSNR and normalized PSNR based on official Track 2 15-35 dB window:
        PSNR_norm = clip((PSNR - 15.0) / (35.0 - 15.0), 0.0, 1.0)
    """
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        psnr = 100.0
        psnr_norm = 1.0
    else:
        psnr = 10.0 * math.log10(1.0 / mse)
        # Official MaCVi 15-35 dB normalization window
        psnr_norm = min(1.0, max(0.0, (psnr - 15.0) / (35.0 - 15.0)))
    return psnr, psnr_norm


def compute_sharpness_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes ratio of Laplacian variance between restored and clean images.
    Includes zero-denominator protection guardrail.
    """
    # Convert to grayscale if 3-channel
    if pred.ndim == 3 and pred.shape[2] == 3:
        p_gray = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        t_gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        p_gray = (pred * 255).astype(np.uint8)
        t_gray = (target * 255).astype(np.uint8)

    lap_p = cv2.Laplacian(p_gray, cv2.CV_64F)
    lap_t = cv2.Laplacian(t_gray, cv2.CV_64F)

    var_p = float(lap_p.var())
    var_t = float(lap_t.var())

    # Zero-denominator protection policy
    eps = 1e-7
    if var_t < eps:
        return 1.0 if var_p < eps else 0.0

    ratio = var_p / var_t
    return float(min(1.0, max(0.0, ratio)))


def compute_ssim_single(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculates mean SSIM across RGB channels."""
    # Convert to float32 [H, W, C]
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    ssim_channels = []
    for c in range(pred.shape[2]):
        p_c = pred[:, :, c]
        t_c = target[:, :, c]

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
        ssim_channels.append(ssim_map.mean())

    return float(np.mean(ssim_channels))


class RestorationEvaluator:
    """
    Evaluator accumulating per-image restoration metrics and reporting dataset mean.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.psnr_list: List[float] = []
        self.psnr_norm_list: List[float] = []
        self.ssim_list: List[float] = []
        self.sharpness_list: List[float] = []

    def update(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor):
        """
        Args:
            pred_rgb: [B, 3, H, W] in [0, 1]
            target_rgb: [B, 3, H, W] in [0, 1]
        """
        pred_np = pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1) # [B, H, W, 3]
        target_np = target_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)

        b = pred_np.shape[0]
        for i in range(b):
            p = np.clip(pred_np[i], 0.0, 1.0)
            t = np.clip(target_np[i], 0.0, 1.0)

            psnr, psnr_norm = compute_psnr_and_norm(p, t)
            ssim_val = compute_ssim_single(p, t)
            sharp_ratio = compute_sharpness_ratio(p, t)

            self.psnr_list.append(psnr)
            self.psnr_norm_list.append(psnr_norm)
            self.ssim_list.append(ssim_val)
            self.sharpness_list.append(sharp_ratio)

    def compute(self) -> Dict[str, Any]:
        """Compute arithmetic mean across all evaluation images."""
        if not self.psnr_list:
            return {'PSNR': 0.0, 'PSNR_norm': 0.0, 'SSIM': 0.0, 'sharpness_ratio': 0.0, 'Q_rest': 0.0}

        mean_psnr = float(np.mean(self.psnr_list))
        mean_psnr_norm = float(np.mean(self.psnr_norm_list))
        mean_ssim = float(np.mean(self.ssim_list))
        mean_sharp = float(np.mean(self.sharpness_list))

        q_rest = 0.45 * mean_psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp

        return {
            'PSNR_dB': round(mean_psnr, 2),
            'PSNR_norm': round(mean_psnr_norm, 4),
            'SSIM': round(mean_ssim, 4),
            'sharpness_ratio': round(mean_sharp, 4),
            'Q_rest': round(q_rest, 4),
            'num_samples': len(self.psnr_list)
        }
