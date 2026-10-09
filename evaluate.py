"""
Root Evaluation Suite for EURS-Net V2.
Harmonized Evaluation Engine:
- Provides `evaluate(model, val_loader, device)` for train.py (returns dict with lowercase keys + nested structure).
- Provides `run_evaluation(...)` and CLI entrypoint for standalone evaluation runs.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from data.transforms import build_preprocessing_transform
from evaluation.metrics import MetricAccumulator
from evaluation.pipeline import evaluate_model
from utils.paths import get_dataset_root, get_splits_path, DATA_DIR, FROZEN_MANIFEST_PATH
from metrics import (
    SegmentationEvaluator,
    RestorationEvaluator,
    NonReferenceQualityEvaluator,
    compute_composite_score
)


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = 'cpu',
    max_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Standard evaluation function called by train.py and Phase 1/2 test harnesses.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()
    nr_eval = NonReferenceQualityEvaluator()

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader, 1):
            if isinstance(batch, (tuple, list)):
                deg, clean, mask, meta = batch
            else:
                deg = batch['degraded']
                clean = batch['clean']
                mask = batch['mask']

            deg = deg.to(dev)
            clean = clean.to(dev)
            mask = mask.to(dev)

            restored_rgb, seg_logits = model(deg)

            seg_eval.update(seg_logits, mask)
            rest_eval.update(restored_rgb, clean)
            nr_eval.update(restored_rgb)

            if max_batches is not None and b_idx >= max_batches:
                break

    seg_res = seg_eval.compute()
    rest_res = rest_eval.compute()
    nr_res = nr_eval.compute()

    comp_res = compute_composite_score(
        q_seg=seg_res['Q_seg'],
        q_rest=rest_res['Q_rest'],
        q_nr=nr_res['Q_NR']
    )

    print("\n" + "=" * 65)
    print("           EURS-NET V2 VALIDATION EVALUATION REPORT")
    print("=" * 65)
    print(f"  Segmentation Metrics (Dataset Confusion Matrix):")
    print(f"    - mIoU:              {seg_res['mIoU']:.4f}")
    print(f"    - Macro F1:          {seg_res['macro_F1']:.4f}")
    print(f"    - Per-class IoU:     {seg_res['per_class_IoU']}")
    print(f"    - Per-class F1:      {seg_res['per_class_F1']}")
    print(f"    - Coverage Penalty:  {seg_res['coverage_penalty']}")
    print(f"    - Q_seg:             {seg_res['Q_seg']:.4f}")
    print("  -------------------------------------------------------------")
    print(f"  Restoration Metrics (Per-Image Mean across {rest_res['num_samples']} samples):")
    print(f"    - PSNR:              {rest_res['PSNR_dB']:.2f} dB")
    print(f"    - PSNR_norm:         {rest_res['PSNR_norm']:.4f}")
    print(f"    - SSIM:              {rest_res['SSIM']:.4f}")
    print(f"    - Sharpness Ratio:   {rest_res['sharpness_ratio']:.4f}")
    print(f"    - Q_rest:            {rest_res['Q_rest']:.4f}")
    print("  -------------------------------------------------------------")
    print(f"  Non-Reference Metric (Q_NR):")
    print(f"    - Q_NR (Provisional): {nr_res['Q_NR']:.4f} (Status: {nr_res['status']})")
    print("  -------------------------------------------------------------")
    print(f"  COMPOSITE SCORE (60% Seg | 30% Rest | 10% NR):")
    print(f"    -> Overall Composite: {comp_res['Composite_Score']:.4f} (Status: {comp_res['status']})")
    print("=" * 65 + "\n")

    return {
        'segmentation': seg_res,
        'restoration': rest_res,
        'non_reference': nr_res,
        'composite': comp_res,
        # Flat lowercase aliases for evaluation/pipeline parity
        'psnr': rest_res['PSNR_dB'],
        'psnr_norm': rest_res['PSNR_norm'],
        'ssim': rest_res['SSIM'],
        'sharpness_ratio': rest_res['sharpness_ratio'],
        'q_rest': rest_res['Q_rest'],
        'macro_iou': seg_res['mIoU'],
        'macro_f1': seg_res['macro_F1'],
        'coverage_penalty': seg_res['coverage_penalty'],
        'q_seg': seg_res['Q_seg'],
        'composite_score': comp_res['Composite_Score'],
    }


def run_evaluation(
    checkpoint_path: Optional[str] = None,
    resolution: int = 1024,
    mode: str = "direct",
    dataset_root: Optional[str] = None,
    splits_file: Optional[str] = None,
    batch_size: int = 4,
    device: str = "cpu",
    save_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates EURS-Net V2 on the locked Tier-1 validation set via CLI or script.
    """
    root_dir = dataset_root or get_dataset_root()
    splits = splits_file or get_splits_path()
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    print("=================================================================")
    print("EURS-Net V2 Validation Split Evaluation")
    print("=================================================================")
    print(f"Dataset Root:    {root_dir}")
    print(f"Splits Manifest: {splits}")
    print(f"Resolution:      {resolution}x{resolution}")
    print(f"Mode:            {mode}")
    print(f"Device:          {dev}")
    print(f"Batch Size:      {batch_size}")
    print("=================================================================")

    # 1. Instantiate Model
    model = EURSNetV2(num_classes=5).to(dev)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint weights from: {checkpoint_path}")
    else:
        print("Evaluating initialized model (no checkpoint specified).")

    model.eval()

    # 2. Build Dataset & DataLoader
    transform = build_preprocessing_transform(resolution=resolution, mode=mode)
    val_dataset = EURSDataset(
        root_dir=root_dir,
        split="val",
        tier="tier1",
        splits_file=splits,
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Evaluating {len(val_dataset)} validation samples across {len(val_loader)} batches...")

    # 3. Execute Evaluation
    is_letterbox = (mode.lower().strip() == "letterbox")
    results = evaluate_model(model, val_loader, device=str(dev), is_letterbox=is_letterbox)

    # 4. Display Report
    print("\n======================= EVALUATION RESULTS =======================")
    print("Restoration Metrics:")
    print(f"  PSNR:             {results['psnr']} dB (Normalized: {results['psnr_norm']})")
    print(f"  SSIM:             {results['ssim']}")
    print(f"  Sharpness Ratio:  {results['sharpness_ratio']}")
    print(f"  Q_rest:           {results['q_rest']}")

    print("\nSegmentation Metrics:")
    print(f"  Pixel Accuracy:   {results['pixel_accuracy'] * 100:.2f}%")
    print(f"  Macro IoU:        {results['macro_iou'] * 100:.2f}%")
    print(f"  Macro F1:         {results['macro_f1'] * 100:.2f}%")
    print(f"  Coverage Ratio:   {results['coverage_penalty'] * 100:.2f}%")
    print(f"  Q_seg:            {results['q_seg']}")

    print("\nPer-Class IoU Breakdown:")
    print(f"  Class 0 (Background):   {results['iou_class_0'] * 100:.2f}%")
    print(f"  Class 1 (Coral):        {results['iou_class_1'] * 100:.2f}%")
    print(f"  Class 2 (Fish):         {results['iou_class_2'] * 100:.2f}%")
    print(f"  Class 3 (Elasmobranch): {results['iou_class_3'] * 100:.2f}% (Rare foreground)")
    print(f"  Class 4 (Other):        {results['iou_class_4'] * 100:.2f}% (Rare foreground)")

    composite = round(0.6 * results['q_seg'] + 0.3 * results['q_rest'], 4)
    print(f"\nComposite Score (0.6 * Q_seg + 0.3 * Q_rest): {composite}")
    print("=================================================================")

    results["composite_score"] = composite

    if save_json:
        os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {save_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EURS-Net V2 on validation set")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt/.pth)")
    parser.add_argument("--resolution", type=int, default=1024, help="Input resolution (e.g. 1024 or 768)")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "letterbox"], help="Preprocessing mode")
    parser.add_argument("--dataset-root", type=str, default=None, help="EURS dataset root directory")
    parser.add_argument("--splits-file", type=str, default=None, help="Path to splits.json")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to evaluate on ('cuda' or 'cpu')")
    parser.add_argument("--save-json", type=str, default=None, help="Path to save evaluation JSON")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        mode=args.mode,
        dataset_root=args.dataset_root,
        splits_file=args.splits_file,
        batch_size=args.batch_size,
        device=args.device,
        save_json=args.save_json
    )


if __name__ == "__main__":
    main()
