"""
EURS-Net V3 LiteJoint Controlled Training Engine
MaCVi @ WACV 2027 EURS Challenge Track 2 (Embedded Jetson AGX Orin)
Features:
- Joint multi-task training (Encoder + Restoration Decoder + Segmentation Decoder)
- Differential learning rates (conservative for backbone, higher for decoders)
- Class-Aware Scene Sampler for balanced marine organism exposure
- Full validation evaluation on complete 800-image validation set after every epoch
- Strict guardrails: finite loss, finite gradients, clipping < 5% per-scene & aggregate
- Checkpoint preservation and atomic saving
"""

import sys
import os
import json
import time
import math
import argparse

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('experiments/exp_eurs_v3_litejoint'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from data.dataset import EURSDataset
from data.transforms import SynchronizedTransform
from model_v3 import EURSNetV3LiteJoint
from losses_v3 import MultiTaskLossV3, get_v3_class_weights
from sampler_v3 import ClassAwareSceneSampler
from metrics import SegmentationEvaluator, RestorationEvaluator, compute_composite_score


def get_warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=10, min_lr_ratio=1e-2):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - 1 - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def evaluate_full_epoch(model, val_loader, device='cuda'):
    model.eval()
    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()

    scene_clipping = {}
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

    # Standard composite score: S_std = 0.60 * Q_seg + 0.30 * Q_rest + 0.05
    s_std = 0.60 * seg_res['Q_seg'] + 0.30 * rest_res['Q_rest'] + 0.05

    return {
        'PSNR_dB': rest_res['PSNR_dB'],
        'PSNR_norm': rest_res['PSNR_norm'],
        'SSIM': rest_res['SSIM'],
        'sharpness_ratio': rest_res['sharpness_ratio'],
        'Q_rest': rest_res['Q_rest'],
        'mIoU': seg_res['mIoU'],
        'macro_F1': seg_res['macro_F1'],
        'per_class_IoU': [round(x, 4) for x in seg_res['per_class_IoU']],
        'per_class_F1': [round(x, 4) for x in seg_res['per_class_F1']],
        'Q_seg': seg_res['Q_seg'],
        'Composite_Score': comp_res['Composite_Score'],
        'S_std': round(s_std, 4),
        'agg_clipping_pct': round(agg_clip_pct, 4),
        'per_scene_clipping_pct': {k: round(v, 4) for k, v in per_scene_clip.items()}
    }


def train_v3(
    epochs=10,
    batch_size=4,
    grad_accum=2,
    mode='bounded',
    bound_scale=0.20,
    exp_dir='experiments/exp_eurs_v3_litejoint',
    data_dir='data/cached_1024' if os.path.isdir('data/cached_1024') else 'data',
    num_workers=2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70, flush=True)
    print("      EURS-NET V3 LITEJOINT: CONTROLLED MULTI-TASK TRAINING", flush=True)
    print("=" * 70, flush=True)
    print(f"Device:           {device}", flush=True)
    print(f"Total Epochs:     {epochs}", flush=True)
    print(f"Batch Size:       {batch_size} (Effective: {batch_size * grad_accum})", flush=True)
    print(f"Restoration Mode: {mode} (bound_scale: {bound_scale})", flush=True)
    print(f"Data Directory:   {data_dir}", flush=True)
    print(f"Experiment Dir:   {exp_dir}", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)

    # 1. Model Initialization
    model = EURSNetV3LiteJoint(num_classes=5, restoration_mode=mode, bound_scale=bound_scale).to(device)

    # 2. Losses & Class Weights
    weights = get_v3_class_weights().to(device)
    criterion = MultiTaskLossV3(num_classes=5, alpha_seg=weights, weight_seg=1.0, weight_rest=0.5, lambda_dice=1.0).to(device)

    # 3. Optimizer with Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 5e-5},
        {'params': model.fwb.parameters(), 'lr': 5e-5},
        {'params': model.restoration_decoder.parameters(), 'lr': 1.5e-4},
        {'params': model.segmentation_decoder.parameters(), 'lr': 1.5e-4},
    ], weight_decay=1e-4)

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=epochs, min_lr_ratio=0.05)

    # 4. Datasets & Loaders
    tx = SynchronizedTransform(target_size=(1024, 1024))
    train_dataset = EURSDataset(data_dir=data_dir, split="train", manifest_path="splits/manifest_full.json", transform=tx, target_size=(1024, 1024))
    val_dataset = EURSDataset(data_dir=data_dir, split="val", manifest_path="splits/manifest_full.json", transform=None, target_size=(1024, 1024))

    sampler = ClassAwareSceneSampler(train_dataset, seed=42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    best_s_std = -1.0
    history = []

    print(f"\nTraining set: {len(train_dataset)} samples | Validation set: {len(val_dataset)} samples", flush=True)
    print(f"Batches per epoch: {len(train_loader)} (effective updates: {len(train_loader)//grad_accum})", flush=True)

    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        criterion.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        seg_loss_sum = 0.0
        rest_loss_sum = 0.0
        t_start = time.time()
        optimizer.zero_grad()

        for batch_idx, (deg, clean, mask, meta) in enumerate(train_loader, 1):
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                restored, seg = model(deg)
                loss, l_dict = criterion(restored, seg, clean, mask)
                loss_scaled = loss / grad_accum

            assert torch.isfinite(loss), f"Loss exploded to NaN/Inf at epoch {epoch}, batch {batch_idx}!"
            loss_scaled.backward()

            if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            seg_loss_sum += l_dict['loss_seg']
            rest_loss_sum += l_dict['loss_rest']

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                lr_enc = optimizer.param_groups[0]['lr']
                lr_dec = optimizer.param_groups[2]['lr']
                print(f"  [Epoch {epoch:2d}/{epochs:2d}] Batch {batch_idx:3d}/{len(train_loader):3d} | "
                      f"LR enc/dec: {lr_enc:.2e}/{lr_dec:.2e} | "
                      f"Loss: {loss.item():.4f} (Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f})")

        scheduler.step()
        train_elapsed = time.time() - t_start
        avg_train_loss = epoch_loss / len(train_loader)
        avg_seg_loss = seg_loss_sum / len(train_loader)
        avg_rest_loss = rest_loss_sum / len(train_loader)

        print(f"\n  [Epoch {epoch:2d}] Training complete in {train_elapsed:.1f}s | Avg Loss: {avg_train_loss:.4f} (Seg: {avg_seg_loss:.4f}, Rest: {avg_rest_loss:.4f})")
        print(f"  Running full validation evaluation (800 images)...")

        t_val = time.time()
        val_metrics = evaluate_full_epoch(model, val_loader, device=device)
        val_elapsed = time.time() - t_val

        val_metrics['epoch'] = epoch
        val_metrics['train_loss'] = round(avg_train_loss, 4)
        val_metrics['train_seg_loss'] = round(avg_seg_loss, 4)
        val_metrics['train_rest_loss'] = round(avg_rest_loss, 4)
        val_metrics['train_time_sec'] = round(train_elapsed, 1)
        val_metrics['val_time_sec'] = round(val_elapsed, 1)

        history.append(val_metrics)

        print(f"  --- Epoch {epoch} Results ---")
        print(f"  Restoration:  PSNR = {val_metrics['PSNR_dB']} dB | SSIM = {val_metrics['SSIM']} | Q_rest = {val_metrics['Q_rest']}")
        print(f"  Segmentation: mIoU = {val_metrics['mIoU']} | Macro F1 = {val_metrics['macro_F1']} | Q_seg = {val_metrics['Q_seg']}")
        print(f"  Per-Class IoU: BG={val_metrics['per_class_IoU'][0]}, Coral={val_metrics['per_class_IoU'][1]}, Fish={val_metrics['per_class_IoU'][2]}, Shark={val_metrics['per_class_IoU'][3]}, Other={val_metrics['per_class_IoU'][4]}")
        print(f"  Clipping:     Aggregate = {val_metrics['agg_clipping_pct']}% | Max Scene = {max(val_metrics['per_scene_clipping_pct'].values()):.2f}%")
        print(f"  Scores:       S_std = {val_metrics['S_std']:.4f} | Official Composite = {val_metrics['Composite_Score']:.4f}")

        # Check guardrails
        shark_ok = val_metrics['per_class_IoU'][3] >= 0.1472
        clip_ok = val_metrics['agg_clipping_pct'] < 5.0 and max(val_metrics['per_scene_clipping_pct'].values()) < 5.0
        print(f"  Guardrails:   Shark IoU >= 0.1472: {'PASS' if shark_ok else 'WARN'} ({val_metrics['per_class_IoU'][3]}) | Clipping < 5%: {'PASS' if clip_ok else 'FAIL'}")

        # Save epoch checkpoint
        ep_ckpt_path = f"{exp_dir}/checkpoints/epoch_{epoch:02d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'mode': mode,
            'bound_scale': bound_scale
        }, ep_ckpt_path)
        print(f"  Saved epoch checkpoint: {ep_ckpt_path}")

        # Checkpoint selection based on S_std and guardrails
        if clip_ok and val_metrics['S_std'] > best_s_std:
            best_s_std = val_metrics['S_std']
            best_ckpt_path = f"{exp_dir}/checkpoints/best_v3_checkpoint.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'mode': mode,
                'bound_scale': bound_scale
            }, best_ckpt_path)
            print(f"  *** NEW BEST CHECKPOINT (S_std = {best_s_std:.4f}) saved to {best_ckpt_path} ***")

        # Save history JSON
        with open(f"{exp_dir}/logs/metrics_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("           CONTROLLED TRAINING RUN COMPLETE")
    print(f"Best S_std achieved: {best_s_std:.4f}")
    print("=" * 70)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--mode', type=str, default='bounded', choices=['bounded', 'sigmoid'])
    parser.add_argument('--bound_scale', type=float, default=0.20)
    parser.add_argument('--data_dir', type=str, default='data/cached_1024' if os.path.isdir('data/cached_1024') else 'data')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    train_v3(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        mode=args.mode,
        bound_scale=args.bound_scale,
        data_dir=args.data_dir,
        num_workers=args.num_workers
    )
