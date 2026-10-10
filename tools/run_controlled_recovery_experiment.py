"""
Controlled Restoration-Recovery Experiment Runner
Tests the bounded residual formulation:
  delta = 0.20 * torch.tanh(self.head(x))
  restored = torch.clamp(base + upsample(delta), 0.0, 1.0)
Preserves best segmentation checkpoint weights, resets restoration head, and runs fine-tuning.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.paths import FROZEN_MANIFEST_PATH, CHECKPOINTS_DIR
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from data.samplers import SceneBalancedSampler
from losses.loss_config import compute_training_class_weights, create_authoritative_criterion
from metrics import SegmentationEvaluator, RestorationEvaluator, NonReferenceQualityEvaluator, compute_composite_score


def run_controlled_experiment(epochs: int = 5, lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("      CONTROLLED RESTORATION-RECOVERY EXPERIMENT")
    print(f"      Device: {device} | Epochs: {epochs} | LR: {lr}")
    print("=" * 70)

    exp_dir = os.path.join(CHECKPOINTS_DIR, "exp_bounded_recovery")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Instantiate model with bounded residual
    model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)

    # 2. Load best segmentation weights
    src_ckpt_path = "checkpoints/run2_50ep/forensic_backup/best_model_preserved.pt"
    assert os.path.exists(src_ckpt_path), f"Source checkpoint not found: {src_ckpt_path}"
    src_ckpt = torch.load(src_ckpt_path, map_location='cpu', weights_only=False)
    src_sd = src_ckpt['model_state_dict']

    model_sd = model.state_dict()
    loaded_keys, skipped_keys = [], []
    for k, v in src_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            # We skip the corrupted head weights so they re-initialize to zeros
            if 'restoration_decoder.head' in k:
                skipped_keys.append(k)
            else:
                model_sd[k] = v
                loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(model_sd)

    # Explicitly ensure head is zero-initialized
    nn.init.zeros_(model.restoration_decoder.head.weight)
    nn.init.zeros_(model.restoration_decoder.head.bias)

    print(f"\n[WEIGHT AUDIT] Loaded {len(loaded_keys)} tensors from {src_ckpt_path}")
    print(f"[WEIGHT AUDIT] Preserved: FWB, Backbone, Segmentation Decoder, Restoration Skips")
    print(f"[WEIGHT AUDIT] Re-initialized to zeros: {skipped_keys}")

    # 3. Data Loaders
    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    train_ds = EURSDataset(data_dir, split='train', manifest_path=FROZEN_MANIFEST_PATH)
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)

    train_sampler = SceneBalancedSampler(train_ds, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=4, sampler=train_sampler,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'), persistent_workers=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'), persistent_workers=(device.type == 'cuda')
    )

    # 4. Criterion & Optimizer
    alpha = compute_training_class_weights(data_dir, FROZEN_MANIFEST_PATH, num_classes=5).to(device)
    loss_fn = create_authoritative_criterion(alpha=alpha, weight_seg=1.0, weight_rest=0.5).to(device)

    # Optimizer state: fresh AdamW without historical corrupted momentum
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Function to evaluate all metrics
    def evaluate_model(epoch_num: int):
        model.eval()
        seg_eval = SegmentationEvaluator(num_classes=5)
        rest_eval = RestorationEvaluator()
        nr_eval = NonReferenceQualityEvaluator()

        delta_mins, delta_maxs, delta_means, delta_stds = [], [], [], []
        sat_fractions = []

        with torch.no_grad():
            for deg, clean, mask, meta in val_loader:
                deg = deg.to(device)
                clean = clean.to(device)
                mask = mask.to(device)

                # Trace intermediate delta
                wb_x = model.fwb(deg)
                stem, f256, f128, f64, f32 = model.backbone(wb_x)
                x128 = model.restoration_decoder.proj_f128(f128)
                import torch.nn.functional as F
                x256 = F.interpolate(x128, scale_factor=2.0, mode='bilinear', align_corners=False) + model.restoration_decoder.proj_f256(f256)
                x256 = model.restoration_decoder.dw_conv256(x256)
                x512 = F.interpolate(x256, scale_factor=2.0, mode='bilinear', align_corners=False) + model.restoration_decoder.proj_stem(stem)
                x512 = model.restoration_decoder.dw_conv512(x512)
                delta_512 = 0.20 * torch.tanh(model.restoration_decoder.head(x512))
                delta_1024 = F.interpolate(delta_512, size=deg.shape[2:], mode='bilinear', align_corners=False)
                restored = torch.clamp(wb_x + delta_1024, 0.0, 1.0)

                seg_logits = model.segmentation_decoder(f32, f64, f128, f256, stem)

                # Tensor statistics
                delta_mins.append(delta_1024.min().item())
                delta_maxs.append(delta_1024.max().item())
                delta_means.append(delta_1024.mean().item())
                delta_stds.append(delta_1024.std().item())

                # Saturation percentage
                sat = ((restored == 0.0) | (restored == 1.0)).float().mean().item()
                sat_fractions.append(sat)

                # Metrics update
                seg_eval.update(seg_logits, mask)
                rest_eval.update(restored, clean)
                nr_eval.update(restored)

        seg_res = seg_eval.compute()
        rest_res = rest_eval.compute()
        nr_res = nr_eval.compute()
        comp_res = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], nr_res['Q_NR'])

        return {
            'epoch': epoch_num,
            'composite': comp_res['Composite_Score'],
            'q_seg': seg_res['Q_seg'],
            'miou': seg_res['mIoU'],
            'macro_f1': seg_res['macro_F1'],
            'per_class_iou': seg_res['per_class_IoU'],
            'per_class_f1': seg_res['per_class_F1'],
            'coverage_penalty': seg_res['coverage_penalty'],
            'psnr': rest_res['PSNR_dB'],
            'psnr_norm': rest_res['PSNR_norm'],
            'ssim': rest_res['SSIM'],
            'sharpness': rest_res['sharpness_ratio'],
            'q_rest': rest_res['Q_rest'],
            'q_nr_surrogate': nr_res['Q_NR'],
            'delta_min': float(min(delta_mins)),
            'delta_max': float(max(delta_maxs)),
            'delta_mean': float(sum(delta_means)/len(delta_means)),
            'delta_std': float(sum(delta_stds)/len(delta_stds)),
            'saturation_pct': float(sum(sat_fractions)/len(sat_fractions)) * 100
        }

    # 5. Evaluate Baseline at Epoch 0
    print("\n--- Evaluating Initial Step 0 (Untrained Bounded Head) ---")
    ep0_res = evaluate_model(0)
    print(f"  Initial PSNR: {ep0_res['psnr']:.2f} dB | SSIM: {ep0_res['ssim']:.4f} | Q_rest: {ep0_res['q_rest']:.4f}")
    print(f"  Initial mIoU: {ep0_res['miou']:.4f} | F1: {ep0_res['macro_f1']:.4f} | Q_seg: {ep0_res['q_seg']:.4f}")
    print(f"  Initial Saturated Pixels: {ep0_res['saturation_pct']:.4f}% | Delta: [{ep0_res['delta_min']:.4f}, {ep0_res['delta_max']:.4f}]")
    print(f"  Initial Composite: {ep0_res['composite']:.4f}")

    history = [ep0_res]
    grad_accum = 4
    clip_grad = 1.0

    # 6. Run Short Controlled Fine-Tuning
    for epoch in range(1, epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_rest = 0.0
        head_grad_norms = []
        total_grad_norms = []

        for b_idx, (deg, clean, mask, meta) in enumerate(train_loader, 1):
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            restored, seg_logits = model(deg)
            loss, l_dict = loss_fn(restored, seg_logits, clean, mask)
            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            epoch_loss_total += loss.item()
            epoch_loss_seg += l_dict['loss_seg']
            epoch_loss_rest += l_dict['loss_rest']

            if b_idx % grad_accum == 0 or b_idx == len(train_loader):
                # Measure grad norms before clip
                hw_norm = model.restoration_decoder.head.weight.grad.norm().item()
                head_grad_norms.append(hw_norm)
                tot_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad).item()
                total_grad_norms.append(tot_norm)

                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # Validation at end of epoch
        ep_res = evaluate_model(epoch)
        ep_res['loss_total'] = epoch_loss_total / len(train_loader)
        ep_res['loss_seg'] = epoch_loss_seg / len(train_loader)
        ep_res['loss_rest'] = epoch_loss_rest / len(train_loader)
        ep_res['head_grad_norm'] = sum(head_grad_norms) / len(head_grad_norms)
        ep_res['total_grad_norm'] = sum(total_grad_norms) / len(total_grad_norms)
        history.append(ep_res)

        print(f"\n[EPOCH {epoch}/{epochs}] Loss: Total={ep_res['loss_total']:.4f} (Seg={ep_res['loss_seg']:.4f}, Rest={ep_res['loss_rest']:.4f})")
        print(f"  Head Grad Norm: {ep_res['head_grad_norm']:.4f} | Total Grad Norm: {ep_res['total_grad_norm']:.4f}")
        print(f"  Restoration: PSNR = {ep_res['psnr']:.2f} dB | SSIM = {ep_res['ssim']:.4f} | Q_rest = {ep_res['q_rest']:.4f}")
        print(f"  Residual: Delta range = [{ep_res['delta_min']:.4f}, {ep_res['delta_max']:.4f}] | Saturated = {ep_res['saturation_pct']:.4f}%")
        print(f"  Segmentation: mIoU = {ep_res['miou']:.4f} | Macro F1 = {ep_res['macro_f1']:.4f} | Q_seg = {ep_res['q_seg']:.4f}")
        print(f"  COMPOSITE SCORE: {ep_res['composite']:.4f}")

    # Save final experimental checkpoint
    exp_ckpt_path = os.path.join(exp_dir, "bounded_recovery_ep5.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'extra_state': {'val_metrics': history[-1]}
    }, exp_ckpt_path)
    print(f"\nSaved experiment checkpoint to: {exp_ckpt_path}")

    # Save JSON report
    with open(os.path.join(exp_dir, "experiment_report.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == '__main__':
    run_controlled_experiment(epochs=3, lr=1e-4)
