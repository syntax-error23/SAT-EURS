"""
Evaluation Pipeline Harness for SAT-EURS.

Evaluates a model checkpoint on the locked validation split using the specified
preprocessing configuration, computing structured metrics:
  - PSNR, SSIM, Sharpness Ratio
  - Pixel Accuracy, Macro IoU
  - Explicit IoU for Classes 0, 1, 2, 3, 4
"""

import os
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from data.dataset import EURSDataset
from data.transforms import build_preprocessing_transform, unletterbox
from evaluation.metrics import MetricAccumulator
from evaluation.experiment_config import PreprocessingConfig


def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    is_letterbox: bool = False
) -> Dict[str, Any]:
    """
    Evaluates model across val_loader, accumulating metrics.

    Args:
        model: Model producing (restored_rgb, seg_logits).
        val_loader: DataLoader returning sample dicts.
        device: 'cpu' or 'cuda'.
        is_letterbox: If True, padding is tracked and excluded from metric accumulation.

    Returns:
        Structured dictionary of evaluation metrics.
    """
    model.eval()
    model.to(device)

    accumulator = MetricAccumulator(num_classes=5)

    with torch.no_grad():
        for batch in val_loader:
            deg = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            mask = batch["mask"].to(device)

            restored_rgb, seg_logits = model(deg)

            valid_mask = batch.get("valid_mask", None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)

            accumulator.update(
                pred_rgb=restored_rgb,
                target_rgb=clean,
                pred_logits_or_mask=seg_logits,
                target_mask=mask,
                valid_mask=valid_mask
            )

    return accumulator.compute()


def run_validation_evaluation(
    model: torch.nn.Module,
    dataset_root: str,
    splits_file: str,
    preprocessing: PreprocessingConfig,
    batch_size: int = 1,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Top-level evaluation pipeline loading the locked validation split and computing metrics.
    """
    transform = build_preprocessing_transform(
        resolution=preprocessing.resolution,
        mode=preprocessing.mode
    )

    val_dataset = EURSDataset(
        root_dir=dataset_root,
        split="val",
        tier="tier1",
        splits_file=splits_file,
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    is_letterbox = (preprocessing.mode == "letterbox")
    results = evaluate_model(model, val_loader, device=device, is_letterbox=is_letterbox)
    return results
