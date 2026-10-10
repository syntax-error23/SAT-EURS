"""
Loss Configuration & Square-Root Smoothed Class Weights
Computes class weights alpha_c strictly from the training split:
  w_c = 1 / sqrt(max(f_c, eps))
  alpha_c = w_c / ((1/C) * sum_j(w_j)) -> mean(alpha) = 1.0
Authoritative Multi-task objective:
  L_total = 1.0 * L_seg + 0.5 * L_rest
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
from typing import Dict, Any, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn

from data.dataset import EURSDataset, find_dataset_subdirs
from losses.multi_task_loss import MultiTaskLoss

# Alias for canonical EURS naming
EURSMultiTaskLoss = MultiTaskLoss


def compute_training_class_weights(
    data_dir: str = "data",
    manifest_path: str = "splits/manifest_full.json",
    num_classes: int = 5,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes class weights normalized to mean 1.0 strictly from the training split.
    """
    _, masks_dir, _ = find_dataset_subdirs(data_dir)
    if masks_dir is None:
        masks_dir = os.path.join(data_dir, 'masks')

    assert os.path.exists(masks_dir), f"Masks dir not found (checked 'masks' and 'gt_masks' under {data_dir}): {masks_dir}"
    assert os.path.exists(manifest_path), f"Manifest not found: {manifest_path}"

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    train_scenes = manifest.get("train_scenes", [])
    assert len(train_scenes) > 0, "No train scenes found in manifest!"

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    for s in train_scenes:
        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            candidate = os.path.join(masks_dir, f"{s}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break
        assert mask_path is not None, f"Mask for train scene {s} not found in {masks_dir}"

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim > 2 and mask.shape[2] > 1:
            mask = mask[:, :, 0]

        total_pixels += mask.size
        for c in range(num_classes):
            pixel_counts[c] += np.sum(mask == c)

    assert total_pixels > 0, "Zero pixels counted in train masks!"

    # Compute frequency strictly from train split
    f_c = pixel_counts / float(total_pixels)

    # Square-root smoothed inverse frequency with rare-class boost to prevent background dominance
    rare_boost = np.array([1.0, 1.0, 1.5, 2.0, 2.0], dtype=np.float64)
    w_c = (1.0 / np.sqrt(np.maximum(f_c, eps))) * rare_boost

    # Normalize so that (1/C) * sum(alpha) = 1.0 -> mean(alpha) = 1.0
    alpha = w_c / np.mean(w_c)

    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)

    # Invariant assertions
    assert abs(alpha_tensor.mean().item() - 1.0) < 1e-6, f"Class weight mean is not 1.0: {alpha_tensor.mean().item()}"
    assert torch.isfinite(alpha_tensor).all(), "Class weights contain NaN or Inf!"
    assert (alpha_tensor > 0).all(), "Class weights must all be strictly positive!"

    return alpha_tensor


def create_authoritative_criterion(
    alpha: torch.Tensor,
    weight_seg: float = 1.0,
    weight_rest: float = 0.2
) -> MultiTaskLoss:
    """
    Returns the authoritative MultiTaskLoss instance:
      L_total = weight_seg * L_seg + weight_rest * L_rest
      Default: 1.0 * L_seg + 0.2 * L_rest
    """
    return MultiTaskLoss(
        weight_seg=weight_seg,
        weight_rest=weight_rest,
        alpha_seg=alpha,
        num_classes=len(alpha)
    )
