"""
Pilot Comparison: Bounded Residual vs. Sigmoid Residual Parameterization
Runs a controlled comparison to measure:
- Loss trajectory & numerical stability
- Aggregate & per-scene clipping rates
- PSNR, SSIM, mIoU on validation set
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('experiments/exp_eurs_v3_litejoint'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import EURSDataset
from data.transforms import SynchronizedTransform
from model_v3 import EURSNetV3LiteJoint
from losses_v3 import MultiTaskLossV3, get_v3_class_weights
from sampler_v3 import ClassAwareSceneSampler
from metrics import SegmentationEvaluator, RestorationEvaluator, compute_composite_score


def evaluate_pilot_model(model, val_loader, device='cuda'):
    model.eval()
    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()

    scene_clipping = {}  # scene -> [clipping_counts, total_pixels]
    total_clipped = 0
    total_pixels = 0

    with torch.no_grad():
        for deg, clean, mask, meta in val_loader:
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                restored, seg = model(deg)

            restored_f32 = restored.float()
            seg_eval.update(seg.float(), mask)
            rest_eval.update(restored_f32, clean)

            # Clipping check
            b = restored_f32.shape[0]
            for i in range(b):
                sc = meta['scene_id'][i] if isinstance(meta['scene_id'], list) else meta['scene_id']
                img = restored_f32[i]
                clipped = ((img <= 0.0001) | (img >= 0.9999)).sum().item()
                px = img.numel()
                total_clipped += clipped
                total_pixels += px

                if sc not in scene_clipping:
                    scene_clipping[sc] = [0, 0]
                scene_clipping[sc][0] += clipped
                scene_clipping[sc][1] += px

    seg_res = seg_eval.compute()
    rest_res = rest_eval.compute()
    comp_res = compute_composite_score(q_seg=seg_res['Q_seg'], q_rest=rest_res['Q_rest'], q_nr=0.5)

    agg_clip_pct = 100.0 * total_clipped / total_pixels
    per_scene_clip = {sc: 100.0 * c[0] / c[1] for sc, c in scene_clipping.items()}

    return {
        'PSNR_dB': rest_res['PSNR_dB'],
        'PSNR_norm': rest_res['PSNR_norm'],
        'SSIM': rest_res['SSIM'],
        'Q_rest': rest_res['Q_rest'],
        'mIoU': seg_res['mIoU'],
        'macro_F1': seg_res['macro_F1'],
        'per_class_IoU': [round(x, 4) for x in seg_res['per_class_IoU']],
        'Q_seg': seg_res['Q_seg'],
        'Composite_Score': comp_res['Composite_Score'],
        'agg_clipping_pct': round(agg_clip_pct, 4),
        'per_scene_clipping_pct': {k: round(v, 4) for k, v in per_scene_clip.items()}
    }


def run_pilot(mode: str, num_steps: int = 150):
    print("=" * 65)
    print(f"  RUNNING PILOT: Mode = {mode.upper()} ({num_steps} steps)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    # Model
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode=mode, bound_scale=0.20).to(device)

    # Class weights & Criterion
    weights = get_v3_class_weights().to(device)
    criterion = MultiTaskLossV3(num_classes=5, alpha_seg=weights, weight_seg=1.0, weight_rest=0.5).to(device)

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 5e-5},
        {'params': model.fwb.parameters(), 'lr': 5e-5},
        {'params': model.restoration_decoder.parameters(), 'lr': 1e-4},
        {'params': model.segmentation_decoder.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    # Datasets
    tx = SynchronizedTransform(target_size=(1024, 1024))
    train_dataset = EURSDataset(data_dir="data", split="train", manifest_path="splits/manifest_full.json", transform=tx, target_size=(1024, 1024))
    val_dataset = EURSDataset(data_dir="data", split="val", manifest_path="splits/manifest_full.json", transform=None, target_size=(1024, 1024))

    sampler = ClassAwareSceneSampler(train_dataset, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model.train()
    step = 0
    t0 = time.time()
    loss_history = []

    for deg, clean, mask, meta in train_loader:
        step += 1
        deg = deg.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            restored, seg = model(deg)
            loss, l_dict = criterion(restored, seg, clean, mask)

        assert torch.isfinite(loss), f"Loss exploded to NaN/Inf at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        loss_history.append(loss.item())

        if step % 25 == 0 or step == num_steps:
            print(f"  Step {step:3d}/{num_steps:3d} | Loss: {loss.item():.4f} (Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f})")

        if step >= num_steps:
            break

    train_time = time.time() - t0
    print(f"  Pilot training completed in {train_time:.1f}s. Evaluating on full validation set (800 images)...")

    metrics = evaluate_pilot_model(model, val_loader, device=device)
    metrics['train_time_sec'] = round(train_time, 2)
    metrics['final_train_loss'] = round(loss_history[-1], 4)
    print(f"  Evaluation Results for {mode.upper()}:")
    print(f"    PSNR: {metrics['PSNR_dB']} dB | SSIM: {metrics['SSIM']} | Q_rest: {metrics['Q_rest']}")
    print(f"    mIoU: {metrics['mIoU']} | Macro F1: {metrics['macro_F1']} | Q_seg: {metrics['Q_seg']}")
    print(f"    Shark IoU: {metrics['per_class_IoU'][3]} | Other IoU: {metrics['per_class_IoU'][4]}")
    print(f"    Aggregate Clipping: {metrics['agg_clipping_pct']}% (Guardrail: < 5%)")
    print(f"    Per-Scene Clipping: {metrics['per_scene_clipping_pct']}")
    print(f"    Composite Score: {metrics['Composite_Score']:.4f}")
    return metrics


def main():
    results = {}
    print("Starting Pilot A: Bounded Residual Mode...")
    results['bounded'] = run_pilot('bounded', num_steps=150)

    print("\nStarting Pilot B: Sigmoid Residual Mode...")
    results['sigmoid'] = run_pilot('sigmoid', num_steps=150)

    out_path = "experiments/exp_eurs_v3_litejoint/pilot_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPilot comparison results saved to {out_path}")


if __name__ == '__main__':
    main()
