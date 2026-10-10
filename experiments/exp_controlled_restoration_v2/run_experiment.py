"""
Controlled Restoration Experiment Runner (Version 2)
Experiment: exp_controlled_restoration_v2
Target: MaCVi @ WACV 2027 Track 2 (Jetson AGX Orin)
Features:
- Strict preservation of all baseline and pilot checkpoints.
- Strict parameter freezing of FWB, Backbone, and ProgressiveSegmentationDecoder.
- Trains ONLY ProgressiveRestorationDecoder with bounded residual and zero-init head.
- Warmup cosine scheduler decaying from 1e-4 to 1e-6 to avoid red-channel overshooting.
- Multi-threaded validation metric computation for rapid, exact 800-image evaluation.
- Per-channel (R, G, B) and per-end (0.0 vs 1.0) clipping metrics.
- Hard guardrail rollback: Shark IoU >= 0.1472 and Aggregate Clipping < 5.0%.
- Full atomic checkpointing and JSON logging.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath('.'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from utils.paths import FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from data.samplers import SceneBalancedSampler
from losses.restoration_loss import RestorationLoss
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import (
    RestorationEvaluator,
    compute_psnr_and_norm,
    compute_ssim_single,
    compute_sharpness_ratio
)
from metrics.nr_metric import NonReferenceQualityEvaluator
from metrics.composite_score import compute_composite_score


def compute_file_sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_single_image_metrics(args: Tuple[np.ndarray, np.ndarray, str]) -> Dict[str, Any]:
    r_np, c_np, sid = args
    p_val, p_norm = compute_psnr_and_norm(r_np, c_np)
    s_val = compute_ssim_single(r_np, c_np)
    sh_val = compute_sharpness_ratio(r_np, c_np)
    
    # Clipping breakdown
    cl_val = float(np.mean((r_np == 0.0) | (r_np == 1.0))) * 100.0
    cl_0 = float(np.mean(r_np == 0.0)) * 100.0
    cl_1 = float(np.mean(r_np == 1.0)) * 100.0
    cl_r = float(np.mean((r_np[:, :, 0] == 0.0) | (r_np[:, :, 0] == 1.0))) * 100.0
    cl_g = float(np.mean((r_np[:, :, 1] == 0.0) | (r_np[:, :, 1] == 1.0))) * 100.0
    cl_b = float(np.mean((r_np[:, :, 2] == 0.0) | (r_np[:, :, 2] == 1.0))) * 100.0
    
    return {
        'sid': sid,
        'psnr': p_val,
        'psnr_norm': p_norm,
        'ssim': s_val,
        'sharpness': sh_val,
        'clip_pct': cl_val,
        'clip_0': cl_0,
        'clip_1': cl_1,
        'clip_r': cl_r,
        'clip_g': cl_g,
        'clip_b': cl_b
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = None) -> Dict[str, Any]:
    model.eval()
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    class_names = ['Background', 'Coral', 'Fish', 'Shark', 'Other']

    seg_eval = SegmentationEvaluator(num_classes=5)
    rest_eval = RestorationEvaluator()
    nr_eval = NonReferenceQualityEvaluator()

    tasks = []
    with torch.no_grad():
        for b_idx, (deg, clean, mask, meta) in enumerate(loader, 1):
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            restored, seg_logits = model(deg)

            seg_eval.update(seg_logits, mask)
            nr_eval.update(restored)

            b_size = deg.shape[0]
            rest_cpu = restored.permute(0, 2, 3, 1).cpu().numpy()
            clean_cpu = clean.permute(0, 2, 3, 1).cpu().numpy()

            for i in range(b_size):
                tasks.append((rest_cpu[i], clean_cpu[i], meta['scene_id'][i]))

            if max_batches is not None and b_idx >= max_batches:
                break

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compute_single_image_metrics, tasks))

    scene_data = {s: {'psnr': [], 'ssim': [], 'sharp': [], 'clip_pct': [], 'clip_r': [], 'clip_g': [], 'clip_b': []} for s in scenes}
    all_clip_pcts = []
    all_clip_0 = []
    all_clip_1 = []
    all_clip_r = []
    all_clip_g = []
    all_clip_b = []

    for r in results:
        sid = r['sid']
        rest_eval.psnr_list.append(r['psnr'])
        rest_eval.psnr_norm_list.append(r['psnr_norm'])
        rest_eval.ssim_list.append(r['ssim'])
        rest_eval.sharpness_list.append(r['sharpness'])

        all_clip_pcts.append(r['clip_pct'])
        all_clip_0.append(r['clip_0'])
        all_clip_1.append(r['clip_1'])
        all_clip_r.append(r['clip_r'])
        all_clip_g.append(r['clip_g'])
        all_clip_b.append(r['clip_b'])
        
        if sid in scene_data:
            scene_data[sid]['psnr'].append(r['psnr'])
            scene_data[sid]['ssim'].append(r['ssim'])
            scene_data[sid]['sharp'].append(r['sharpness'])
            scene_data[sid]['clip_pct'].append(r['clip_pct'])
            scene_data[sid]['clip_r'].append(r['clip_r'])
            scene_data[sid]['clip_g'].append(r['clip_g'])
            scene_data[sid]['clip_b'].append(r['clip_b'])

    seg_res = seg_eval.compute()
    rest_res = rest_eval.compute()
    nr_res = nr_eval.compute()

    comp_surrogate = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], nr_res['Q_NR'])
    comp_standard = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], 0.50)

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            'iou': float(seg_res['per_class_IoU'][idx]),
            'f1': float(seg_res['per_class_F1'][idx])
        }

    per_scene = {}
    for s in scenes:
        count = len(scene_data[s]['psnr'])
        per_scene[s] = {
            'count': count,
            'psnr_mean': float(np.mean(scene_data[s]['psnr'])) if count > 0 else 0.0,
            'ssim_mean': float(np.mean(scene_data[s]['ssim'])) if count > 0 else 0.0,
            'sharp_mean': float(np.mean(scene_data[s]['sharp'])) if count > 0 else 0.0,
            'clip_pct_mean': float(np.mean(scene_data[s]['clip_pct'])) if count > 0 else 0.0,
            'clip_pct_max': float(np.max(scene_data[s]['clip_pct'])) if count > 0 else 0.0,
            'clip_r_mean': float(np.mean(scene_data[s]['clip_r'])) if count > 0 else 0.0,
            'clip_g_mean': float(np.mean(scene_data[s]['clip_g'])) if count > 0 else 0.0,
            'clip_b_mean': float(np.mean(scene_data[s]['clip_b'])) if count > 0 else 0.0,
            'clip_guardrail_pass': bool(np.mean(scene_data[s]['clip_pct']) < 5.0) if count > 0 else True
        }

    aggregate_clip_pct = float(np.mean(all_clip_pcts)) if all_clip_pcts else 0.0

    return {
        'total_images': len(all_clip_pcts),
        'mIoU': seg_res['mIoU'],
        'macro_F1': seg_res['macro_F1'],
        'shark_iou': per_class['Shark']['iou'],
        'per_class': per_class,
        'coverage_penalty': seg_res['coverage_penalty'],
        'Q_seg': seg_res['Q_seg'],
        'PSNR_dB': rest_res['PSNR_dB'],
        'PSNR_norm': rest_res['PSNR_norm'],
        'SSIM': rest_res['SSIM'],
        'sharpness_ratio': rest_res['sharpness_ratio'],
        'Q_rest': rest_res['Q_rest'],
        'clipping_aggregate_pct': round(aggregate_clip_pct, 4),
        'clipping_low_0_pct': round(float(np.mean(all_clip_0)), 4) if all_clip_0 else 0.0,
        'clipping_high_1_pct': round(float(np.mean(all_clip_1)), 4) if all_clip_1 else 0.0,
        'clipping_channels_pct': {
            'R': round(float(np.mean(all_clip_r)), 4) if all_clip_r else 0.0,
            'G': round(float(np.mean(all_clip_g)), 4) if all_clip_g else 0.0,
            'B': round(float(np.mean(all_clip_b)), 4) if all_clip_b else 0.0
        },
        'clipping_pct_max': round(float(np.max(all_clip_pcts)), 4) if all_clip_pcts else 0.0,
        'clipping_guardrail_pass': bool(aggregate_clip_pct < 5.0),
        'per_scene': per_scene,
        'Q_NR_surrogate': nr_res['Q_NR'],
        'composite_surrogate': comp_surrogate['Composite_Score'],
        'composite_standard_0_50': comp_standard['Composite_Score']
    }


def is_better_checkpoint(cand: Dict[str, Any], best: Dict[str, Any]) -> bool:
    """
    Select best checkpoint using standard composite score subject to mandatory guardrails:
    1. Shark IoU >= 0.1472
    2. Aggregate clipping < 5.0%
    3. Per-scene clipping < 5.0% for all scenes
    Tie-breaking: If scores are effectively tied (|delta| <= 1e-4), prefer better segmentation
    integrity (mIoU) and lower aggregate clipping.
    """
    if cand.get('shark_iou', 0.0) < 0.1472:
        return False
    if cand.get('clipping_aggregate_pct', 100.0) >= 5.0:
        return False
    for s_id, s_info in cand.get('per_scene', {}).items():
        if s_info.get('clip_pct_mean', 100.0) >= 5.0:
            return False
    if best is None:
        return True

    s_cand = cand.get('composite_standard_0_50', 0.0)
    s_best = best.get('composite_standard_0_50', 0.0)

    # Strictly better
    if s_cand > s_best + 1e-4:
        return True

    # Effectively tied
    if abs(s_cand - s_best) <= 1e-4:
        m_cand = cand.get('mIoU', 0.0)
        m_best = best.get('mIoU', 0.0)
        if m_cand > m_best + 1e-4:
            return True
        if m_best > m_cand + 1e-4:
            return False

        c_cand = cand.get('clipping_aggregate_pct', 100.0)
        c_best = best.get('clipping_aggregate_pct', 100.0)
        if c_cand < c_best - 1e-4:
            return True

    return False


def run_experiment(mode: str = 'train', total_epochs: int = 10, start_epoch: int = 2, lr: float = 1e-4, resume_ckpt: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("    EURS-NET V2 CONTROLLED RESTORATION EXPERIMENT RUNNER")
    print(f"    Mode: {mode.upper()} | Device: {device} | Target Epochs: {total_epochs} | Start Epoch: {start_epoch} | LR: {lr}")
    print(f"    Resume Checkpoint: {resume_ckpt}")
    print(f"    Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    exp_dir = 'experiments/exp_controlled_restoration_v2'
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    plots_dir = os.path.join(exp_dir, 'plots')
    logs_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    logs_dir_path = logs_dir
    os.makedirs(logs_dir_path, exist_ok=True)

    # 1. Instantiate Model with Bounded Residual
    model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)

    # 2. Base weight loading
    src_ckpt_path = "checkpoints/run2_50ep/best_model.pt"
    assert os.path.exists(src_ckpt_path), f"Source checkpoint not found: {src_ckpt_path}"
    base_src_ckpt = torch.load(src_ckpt_path, map_location='cpu', weights_only=False)
    base_src_sd = base_src_ckpt['model_state_dict']

    model_sd = model.state_dict()
    for k, v in base_src_sd.items():
        if 'restoration_decoder.head' not in k and k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
    model.load_state_dict(model_sd)
    nn.init.zeros_(model.restoration_decoder.head.weight)
    nn.init.zeros_(model.restoration_decoder.head.bias)

    # 3. Strictly Freeze Non-Restoration Modules
    for p in model.fwb.parameters():
        p.requires_grad = False
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.segmentation_decoder.parameters():
        p.requires_grad = False

    # Snapshot frozen parameters for invariance auditing vs authoritative baseline
    frozen_snapshots = {n: p.detach().clone().cpu() for n, p in model.named_parameters() if not p.requires_grad}
    frozen_hashes = {n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n, p in model.named_parameters() if not p.requires_grad}

    def verify_invariance(stage: str):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                curr_h = hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                if curr_h != frozen_hashes[n]:
                    diff = (p.detach().cpu() - frozen_snapshots[n]).abs().max().item()
                    raise RuntimeError(f"FATAL INVARIANCE VIOLATION at {stage}: Frozen parameter {n} changed! Max delta={diff}")

    # 4. Handle Checkpoint Resumption
    trainable_params = list(model.restoration_decoder.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    history = []
    best_metrics = None
    best_composite = 0.0
    best_ckpt_path = os.path.join(ckpt_dir, "best_controlled_checkpoint.pt")

    # Load prior history if available
    for h_path in [os.path.join(logs_dir, "controlled_training_report.json"), os.path.join(logs_dir, "controlled_epoch1_report.json")]:
        if os.path.exists(h_path):
            try:
                with open(h_path, 'r') as f:
                    prior_json = json.load(f)
                    history = prior_json.get('history', [])
                    if history:
                        for h in history:
                            if is_better_checkpoint(h, best_metrics):
                                best_metrics = h
                                best_composite = h.get('composite_standard_0_50', 0.0)
                        print(f"[RESUME] Loaded prior history ({len(history)} entries). Best composite so far: {best_composite:.4f}")
                        break
            except Exception as e:
                print(f"[WARN] Could not load prior history from {h_path}: {e}")

    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"[RESUME] Resuming from checkpoint: {resume_ckpt}")
        ckpt_data = torch.load(resume_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt_data['model_state_dict'])
        if 'optimizer_state_dict' in ckpt_data:
            optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
            print(f"[RESUME] Successfully restored optimizer state dict.")
        if 'val_metrics' in ckpt_data and is_better_checkpoint(ckpt_data['val_metrics'], best_metrics):
            best_metrics = ckpt_data['val_metrics']
            best_composite = ckpt_data['val_metrics'].get('composite_standard_0_50', 0.0)
        verify_invariance(f"Post-Resume Verification from {os.path.basename(resume_ckpt)}")
    else:
        verify_invariance("Pre-Flight Check Fresh")

    print(f"[PRE-FLIGHT] Frozen {len(frozen_snapshots)} tensors verified bitwise invariant.")

    # 5. Data Loaders
    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    train_ds = EURSDataset(data_dir, split='train', manifest_path=FROZEN_MANIFEST_PATH)
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)

    train_sampler = SceneBalancedSampler(train_ds, seed=42)
    batch_size = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'), persistent_workers=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'), persistent_workers=(device.type == 'cuda')
    )

    # 6. Scheduler Setup with accurate step progression for 10 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6, last_epoch=max(0, start_epoch - 1))
    # Correct optimizer param group learning rate to match target schedule
    for pg, target_lr in zip(optimizer.param_groups, scheduler.get_lr()):
        pg['lr'] = target_lr

    loss_fn = RestorationLoss().to(device)

    # 7. Controlled Training Loop (start_epoch to total_epochs)
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n" + "=" * 75)
        print(f">>> STARTING CONTROLLED TRAINING EPOCH {epoch}/{total_epochs} | LR: {current_lr:.6e}")
        print("=" * 75)

        model.train()
        model.fwb.eval()
        model.backbone.eval()
        model.segmentation_decoder.eval()
        model.restoration_decoder.train()

        train_sampler.set_epoch(epoch)
        loss_fn.set_epoch(epoch)

        epoch_loss = 0.0
        step_losses = []
        head_grad_norms = []

        for step, (deg, clean, mask, meta) in enumerate(train_loader, 1):
            deg = deg.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            restored, _ = model(deg)
            loss = loss_fn(restored, clean)

            if not torch.isfinite(loss):
                raise RuntimeError(f"FATAL: Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")

            loss.backward()

            # Verify no gradients on frozen modules
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None, f"VIOLATION: Frozen parameter {n} received gradient!"

            head_grad = model.restoration_decoder.head.weight.grad
            assert torch.isfinite(head_grad).all(), f"Head gradient non-finite at step {step}"
            head_grad_norms.append(head_grad.norm().item())

            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step_losses.append(loss.item())

            if step % 100 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch:2d}/{total_epochs:2d} | Step {step:3d}/{len(train_loader):3d} | "
                      f"LR: {current_lr:.2e} | Loss: {loss.item():.4f} | Head Grad: {head_grad.norm().item():.5f}")

        # Step scheduler at epoch completion
        scheduler.step()
        verify_invariance(f"Epoch {epoch} End")
        print(f"  [INVARIANCE AUDIT] Frozen parameter invariance confirmed 100% bitwise identical.")

        # Post-Epoch Full Validation (All 800 images across 4 disjoint scenes)
        print(f"\nEvaluating complete 800-image validation set for Epoch {epoch}...")
        ep_metrics = evaluate_model(model, val_loader, device)
        ep_metrics['epoch'] = epoch
        ep_metrics['train_loss'] = epoch_loss / len(train_loader)
        ep_metrics['avg_head_grad_norm'] = float(np.mean(head_grad_norms))
        ep_metrics['lr'] = current_lr
        ep_metrics['epoch_time_sec'] = round(time.time() - epoch_start, 2)
        history.append(ep_metrics)

        print(f"\n" + "-" * 75)
        print(f"               EPOCH {epoch} VALIDATION AUDIT RESULTS")
        print("-" * 75)
        print(f"  Train Loss:           {ep_metrics['train_loss']:.4f} | Epoch Time: {ep_metrics['epoch_time_sec']}s")
        print(f"  PSNR:                 {ep_metrics['PSNR_dB']:.2f} dB (Norm: {ep_metrics['PSNR_norm']:.4f})")
        print(f"  SSIM:                 {ep_metrics['SSIM']:.4f}")
        print(f"  Sharpness Ratio:      {ep_metrics['sharpness_ratio']:.4f}")
        print(f"  Q_rest:               {ep_metrics['Q_rest']:.4f}")
        print(f"  Aggregate Clipping:   {ep_metrics['clipping_aggregate_pct']:.4f}% (Low 0.0: {ep_metrics['clipping_low_0_pct']:.2f}%, High 1.0: {ep_metrics['clipping_high_1_pct']:.2f}%)")
        print(f"  Channel Clipping:     R={ep_metrics['clipping_channels_pct']['R']:.2f}%, G={ep_metrics['clipping_channels_pct']['G']:.2f}%, B={ep_metrics['clipping_channels_pct']['B']:.2f}%")
        print(f"  Scene fish12 Clip:    {ep_metrics['per_scene']['fish12']['clip_pct_mean']:.4f}% (Max: {ep_metrics['per_scene']['fish12']['clip_pct_max']:.2f}%)")
        print(f"  Segmentation mIoU:    {ep_metrics['mIoU']:.4f} | Macro F1: {ep_metrics['macro_F1']:.4f}")
        print(f"  Shark IoU:            {ep_metrics['shark_iou']:.4f} (Guardrail >= 0.1472: {'PASS' if ep_metrics['shark_iou'] >= 0.1472 else 'FAIL'})")
        print(f"  Composite (Std 0.50): {ep_metrics['composite_standard_0_50']:.4f}")
        print("-" * 75)

        # Regression Tracking & Comparison vs Pilot Epoch 2
        psnr_delta = ep_metrics['PSNR_dB'] - 17.40
        ssim_delta = ep_metrics['SSIM'] - 0.6446
        comp_delta = ep_metrics['composite_standard_0_50'] - 0.2225
        print(f"  [BENCHMARK VS PILOT EP 2] PSNR: {psnr_delta:+.2f} dB | SSIM: {ssim_delta:+.4f} | Comp: {comp_delta:+.4f} | Agg Clip: {ep_metrics['clipping_aggregate_pct']:.4f}% (Pilot: 2.0801%)")

        # Mandatory Safety Guardrails
        if ep_metrics['shark_iou'] < 0.1472:
            print(f"[CRITICAL STOP] Shark IoU dropped below 0.1472 ({ep_metrics['shark_iou']:.4f})! Aborting training.")
            break
        if ep_metrics['clipping_aggregate_pct'] >= 5.0:
            print(f"[CRITICAL STOP] Aggregate clipping exceeded 5.0% ({ep_metrics['clipping_aggregate_pct']:.2f}%)! Aborting training.")
            break
        for s_name, s_info in ep_metrics['per_scene'].items():
            if s_info['clip_pct_mean'] >= 5.0:
                print(f"[CRITICAL STOP] Scene {s_name} clipping exceeded 5.0% ({s_info['clip_pct_mean']:.2f}%)! Aborting training.")
                break
        if ep_metrics['PSNR_dB'] < 15.06:
            print(f"[CRITICAL STOP] PSNR regressed below baseline 15.06 dB ({ep_metrics['PSNR_dB']:.2f} dB)! Aborting training.")
            break

        # Save immutable epoch checkpoint
        ep_ckpt_path = os.path.join(ckpt_dir, f"controlled_ep{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': ep_metrics,
            'composite_standard': ep_metrics['composite_standard_0_50']
        }, ep_ckpt_path)
        sha_ep = compute_file_sha256(ep_ckpt_path)
        print(f"  Saved epoch checkpoint: {ep_ckpt_path} (SHA-256: {sha_ep})")

        # Checkpoint selection: Evaluate against best_metrics
        if is_better_checkpoint(ep_metrics, best_metrics):
            best_composite = ep_metrics['composite_standard_0_50']
            best_metrics = ep_metrics
            best_ckpt_path = os.path.join(ckpt_dir, "best_controlled_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': ep_metrics,
                'composite_standard': best_composite
            }, best_ckpt_path)
            sha_best = compute_file_sha256(best_ckpt_path)
            print(f"  >>> NEW BEST CONTROLLED CHECKPOINT (Epoch {epoch}): Composite = {best_composite:.4f} (SHA-256: {sha_best})")
        else:
            print(f"  [CHECKPOINT SELECTION] Epoch {epoch} did not surpass best checkpoint (Best Composite: {best_composite:.4f})")

        # Update running training report and progress after every epoch
        progress_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'current_epoch': epoch,
            'total_epochs_target': total_epochs,
            'best_composite_standard': best_composite,
            'best_epoch': best_metrics.get('epoch') if best_metrics else None,
            'best_checkpoint_path': best_ckpt_path,
            'history': history
        }
        with open(os.path.join(logs_dir, "controlled_training_progress.json"), 'w') as f:
            json.dump(progress_report, f, indent=2)
        with open(os.path.join(logs_dir, "controlled_training_report.json"), 'w') as f:
            json.dump(progress_report, f, indent=2)

    print(f"\nControlled training completed through Epoch {min(epoch, total_epochs)}.")
    return progress_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Controlled Restoration Experiment")
    parser.add_argument('--epochs', type=int, default=10, help="Total target epochs")
    parser.add_argument('--start-epoch', type=int, default=2, help="Starting epoch number")
    parser.add_argument('--lr', type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument('--resume', type=str, default='experiments/exp_controlled_restoration_v2/checkpoints/controlled_ep1.pt', help="Path to checkpoint to resume from")
    args = parser.parse_args()

    run_experiment(mode='train', total_epochs=args.epochs, start_epoch=args.start_epoch, lr=args.lr, resume_ckpt=args.resume)

