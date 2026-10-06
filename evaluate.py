"""
Standalone Validation Evaluator for EURS-Net V2
Computes dataset-level mIoU, per-class IoU, F1, PSNR_norm, SSIM, Sharpness Ratio, Q_seg, Q_rest, and Composite Score.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from models import EURSNetV2
from data.dataset import EURSDataset
from metrics import (
    SegmentationEvaluator,
    RestorationEvaluator,
    NonReferenceQualityEvaluator,
    compute_composite_score
)


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = 'cpu'
) -> dict:
    model.eval()
    model.to(device)

    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()
    nr_eval = NonReferenceQualityEvaluator()

    with torch.no_grad():
        for deg, clean, mask, meta in val_loader:
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            restored_rgb, seg_logits = model(deg)

            # Update metrics
            seg_eval.update(seg_logits, mask)
            rest_eval.update(restored_rgb, clean)
            nr_eval.update(restored_rgb)

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
        'composite': comp_res
    }


if __name__ == '__main__':
    from utils.paths import DATA_DIR, FROZEN_MANIFEST_PATH
    parser = argparse.ArgumentParser(description="Evaluate EURS-Net V2 on validation set")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--manifest', type=str, default=str(FROZEN_MANIFEST_PATH))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model = EURSNetV2(num_classes=5)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    val_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='val',
        manifest_path=args.manifest
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    evaluate(model, val_loader, device=args.device)
