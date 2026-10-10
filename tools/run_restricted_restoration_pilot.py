"""
Restricted Restoration-Only Pilot Experiment Runner for EURS-Net V2
Enforces strict parameter freezing on SharedBackbone, SegmentationDecoder, and FWB.
Optimizes ONLY ProgressiveRestorationDecoder and Bounded Head.
Guarantees Shark IoU >= 0.1472 and checks frozen weight bitwise invariance after every step and epoch.
"""

import os
import sys
import time
import json
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from utils.paths import FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from data.samplers import SceneBalancedSampler
from losses.restoration_loss import RestorationLoss
from metrics import SegmentationEvaluator, RestorationEvaluator, NonReferenceQualityEvaluator, compute_composite_score
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single

def hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

def run_pilot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("    EURS-NET V2 RESTRICTED RESTORATION-ONLY PILOT EXPERIMENT")
    print(f"    Device: {device} | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    exp_dir = 'experiments/exp_pilot_restricted_restoration'
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Instantiate model with verified bounded residual
    model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)

    # 2. Load authoritative preserved segmentation checkpoint
    src_ckpt_path = "checkpoints/run2_50ep/forensic_backup/best_model_preserved.pt"
    assert os.path.exists(src_ckpt_path), f"Checkpoint not found: {src_ckpt_path}"
    src_ckpt = torch.load(src_ckpt_path, map_location='cpu', weights_only=False)
    src_sd = src_ckpt['model_state_dict']

    model_sd = model.state_dict()
    for k, v in src_sd.items():
        if 'restoration_decoder.head' not in k and k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
    model.load_state_dict(model_sd)

    # Zero-initialize the bounded head
    nn.init.zeros_(model.restoration_decoder.head.weight)
    nn.init.zeros_(model.restoration_decoder.head.bias)

    # 3. Strictly Freeze Non-Restoration Modules
    for p in model.fwb.parameters():
        p.requires_grad = False
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.segmentation_decoder.parameters():
        p.requires_grad = False

    # 4. Programmatic Assertions on Parameter Ownership
    trainable_params = []
    frozen_param_names = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert n.startswith('restoration_decoder.'), f"VIOLATION: Trainable parameter outside restoration decoder: {n}"
            trainable_params.append(p)
        else:
            frozen_param_names.append(n)

    print(f"[PREFLIGHT] Trainable parameter tensors: {len(trainable_params)} (Restoration Decoder only)")
    print(f"[PREFLIGHT] Frozen parameter tensors:    {len(frozen_param_names)} (FWB, Backbone, Seg Decoder)")

    # 5. Capture Frozen Parameter Hashes and Clones for Invariance Auditing
    frozen_reference_hashes = {}
    frozen_reference_tensors = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            frozen_reference_hashes[n] = hash_tensor(p)
            frozen_reference_tensors[n] = p.detach().clone().cpu()

    def verify_frozen_invariance(stage: str):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                curr_hash = hash_tensor(p)
                if curr_hash != frozen_reference_hashes[name]:
                    diff = (p.detach().cpu() - frozen_reference_tensors[name]).abs().max().item()
                    raise RuntimeError(f"HARD STOP VIOLATION at {stage}: Frozen parameter {name} changed! Max delta = {diff}")

    # Verify initial invariance
    verify_frozen_invariance("Pre-flight")
    print("[PREFLIGHT] Frozen parameter reference snapshots captured and verified: PASSED")

    # 6. Prepare Datasets & Loaders
    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    train_ds = EURSDataset(data_dir, split='train', manifest_path=FROZEN_MANIFEST_PATH)
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)

    train_sampler = SceneBalancedSampler(train_ds, seed=42)
    # Use smallest practical batch size: 2
    batch_size = 2
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'), persistent_workers=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=(device.type == 'cuda'), persistent_workers=(device.type == 'cuda')
    )

    # 7. Optimizer & Restoration Loss (Numerically Stabilized)
    optimizer = optim.AdamW(model.restoration_decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = RestorationLoss().to(device)

    # 8. Full Validation Evaluator
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    class_names = ['Background', 'Coral', 'Fish', 'Shark', 'Other']

    def evaluate_pilot_checkpoint(epoch_label: str):
        model.eval()
        seg_eval = SegmentationEvaluator(num_classes=5)
        rest_eval = RestorationEvaluator()
        nr_eval = NonReferenceQualityEvaluator()

        scene_metrics = {s: {'psnr': [], 'ssim': [], 'clip': []} for s in scenes}
        clipping_fractions = []
        delta_mins, delta_maxs, delta_means = [], [], []

        with torch.no_grad():
            for deg, clean, mask, meta in val_loader:
                deg = deg.to(device)
                clean = clean.to(device)
                mask = mask.to(device)

                restored, seg_logits = model(deg)

                # Check numerical sanity of outputs
                assert torch.isfinite(restored).all(), f"Non-finite restored RGB in validation at {epoch_label}!"
                assert torch.isfinite(seg_logits).all(), f"Non-finite seg logits in validation at {epoch_label}!"

                seg_eval.update(seg_logits, mask)
                rest_eval.update(restored, clean)
                nr_eval.update(restored)

                clip_batch = ((restored == 0.0) | (restored == 1.0)).float().mean().item()
                clipping_fractions.append(clip_batch)

                # Per scene breakdown
                for idx in range(deg.shape[0]):
                    sid = meta['scene_id'][idx]
                    r_np = restored[idx].permute(1, 2, 0).cpu().numpy()
                    c_np = clean[idx].permute(1, 2, 0).cpu().numpy()
                    p_val, _ = compute_psnr_and_norm(r_np, c_np)
                    s_val = compute_ssim_single(r_np, c_np)
                    cl_val = float(np.mean((r_np == 0.0) | (r_np == 1.0))) * 100
                    if sid in scene_metrics:
                        scene_metrics[sid]['psnr'].append(p_val)
                        scene_metrics[sid]['ssim'].append(s_val)
                        scene_metrics[sid]['clip'].append(cl_val)

        seg_res = seg_eval.compute()
        rest_res = rest_eval.compute()
        nr_res = nr_eval.compute()
        comp_res = compute_composite_score(seg_res['Q_seg'], rest_res['Q_rest'], nr_res['Q_NR'])

        per_scene_summary = {}
        for s in scenes:
            per_scene_summary[s] = {
                'psnr_mean': float(np.mean(scene_metrics[s]['psnr'])) if scene_metrics[s]['psnr'] else 0.0,
                'ssim_mean': float(np.mean(scene_metrics[s]['ssim'])) if scene_metrics[s]['ssim'] else 0.0,
                'clip_pct_mean': float(np.mean(scene_metrics[s]['clip'])) if scene_metrics[s]['clip'] else 0.0
            }

        per_class_summary = {}
        for idx, cname in enumerate(class_names):
            per_class_summary[cname] = {
                'iou': float(seg_res['per_class_IoU'][idx]),
                'f1': float(seg_res['per_class_F1'][idx])
            }

        shark_iou = per_class_summary['Shark']['iou']

        return {
            'epoch_label': epoch_label,
            'composite_score': float(comp_res['Composite_Score']),
            'q_seg': float(seg_res['Q_seg']),
            'mIoU': float(seg_res['mIoU']),
            'macro_f1': float(seg_res['macro_F1']),
            'shark_iou': shark_iou,
            'per_class': per_class_summary,
            'q_rest': float(rest_res['Q_rest']),
            'psnr': float(rest_res['PSNR_dB']),
            'psnr_norm': float(rest_res['PSNR_norm']),
            'ssim': float(rest_res['SSIM']),
            'sharpness_ratio': float(rest_res['sharpness_ratio']),
            'q_nr_surrogate': float(nr_res['Q_NR']),
            'clipping_pct': float(np.mean(clipping_fractions)) * 100,
            'per_scene': per_scene_summary
        }

    # 9. Initial Step 0 Evaluation
    print("\n--- Evaluating Epoch 0 Baseline on Authoritative Contract ---")
    ep0_metrics = evaluate_pilot_checkpoint("Epoch 0 (Baseline)")
    print(f"  Epoch 0: PSNR = {ep0_metrics['psnr']:.2f} dB | SSIM = {ep0_metrics['ssim']:.4f} | Clipping = {ep0_metrics['clipping_pct']:.4f}%")
    print(f"  Epoch 0: mIoU = {ep0_metrics['mIoU']:.4f} | Shark IoU = {ep0_metrics['shark_iou']:.4f} | Macro F1 = {ep0_metrics['macro_f1']:.4f}")
    print(f"  Epoch 0: Composite = {ep0_metrics['composite_score']:.4f}")

    assert ep0_metrics['shark_iou'] >= 0.1470, f"Baseline Shark IoU failed check: {ep0_metrics['shark_iou']}"

    history = [ep0_metrics]
    best_ckpt_path = None
    best_composite = ep0_metrics['composite_score']

    # 10. Run Controlled 3-Epoch Pilot
    epochs = 3
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n" + "=" * 60)
        print(f">>> STARTING PILOT EPOCH {epoch}/{epochs}")
        print("=" * 60)

        model.train()
        # Keep FWB, Backbone, and SegDecoder explicitly in eval mode
        model.fwb.eval()
        model.backbone.eval()
        model.segmentation_decoder.eval()
        model.restoration_decoder.train()

        train_sampler.set_epoch(epoch)
        loss_fn.set_epoch(epoch)

        epoch_loss = 0.0
        head_grad_norms = []
        dec_grad_norms = []

        for step, (deg, clean, mask, meta) in enumerate(train_loader, 1):
            deg = deg.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            restored, _ = model(deg)

            loss = loss_fn(restored, clean)

            if not torch.isfinite(loss):
                raise RuntimeError(f"HARD STOP: Non-finite loss detected at epoch {epoch}, step {step}: {loss.item()}")

            loss.backward()

            # Programmatic assertion: Frozen parameters must NOT have gradients
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None, f"VIOLATION: Frozen parameter {n} received a gradient!"

            # Check gradients finite on trainable parameters
            hw_g = model.restoration_decoder.head.weight.grad
            assert torch.isfinite(hw_g).all(), f"HARD STOP: Non-finite head weight gradient at epoch {epoch}, step {step}!"
            head_grad_norms.append(hw_g.norm().item())

            other_norms = [p.grad.norm().item() for n, p in model.restoration_decoder.named_parameters() if p.grad is not None and 'head' not in n]
            if other_norms:
                dec_grad_norms.append(float(np.mean(other_norms)))

            nn.utils.clip_grad_norm_(model.restoration_decoder.parameters(), max_norm=1.0)
            optimizer.step()

            # Immediate invariance check after step
            verify_frozen_invariance(f"Step {step}")

            epoch_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch} | Step {step:3d}/{len(train_loader)} | Loss: {loss.item():.4f} | Head Grad: {hw_g.norm().item():.5f}")

        # Post-Epoch Frozen Invariance Check
        verify_frozen_invariance(f"Epoch {epoch} End")
        print(f"  [AUDIT] Frozen Parameter Invariance Check Epoch {epoch}: PASSED (100% BITWISE INVARIANT)")

        # Evaluate Validation Contract
        ep_metrics = evaluate_pilot_checkpoint(f"Epoch {epoch}")
        ep_metrics['train_loss'] = epoch_loss / len(train_loader)
        ep_metrics['avg_head_grad_norm'] = float(np.mean(head_grad_norms))
        ep_metrics['avg_dec_grad_norm'] = float(np.mean(dec_grad_norms))
        history.append(ep_metrics)

        print(f"\n[VALIDATION EPOCH {epoch} RESULTS]")
        print(f"  Train Loss: {ep_metrics['train_loss']:.4f} | Head Grad: {ep_metrics['avg_head_grad_norm']:.5f}")
        print(f"  Restoration: PSNR = {ep_metrics['psnr']:.2f} dB | SSIM = {ep_metrics['ssim']:.4f} | Clipping = {ep_metrics['clipping_pct']:.4f}%")
        print(f"  Segmentation: mIoU = {ep_metrics['mIoU']:.4f} | Shark IoU = {ep_metrics['shark_iou']:.4f} | Macro F1 = {ep_metrics['macro_f1']:.4f}")
        print(f"  Composite Score: {ep_metrics['composite_score']:.4f}")

        # HARD ROLLBACK CONDITION CHECK
        if ep_metrics['shark_iou'] < 0.1472:
            print(f"\n[CRITICAL HARD STOP] Shark IoU dropped to {ep_metrics['shark_iou']:.4f} (< 0.1472)! TERMINATING EXPERIMENT.")
            break
        if ep_metrics['clipping_pct'] > 5.0:
            print(f"\n[CRITICAL HARD STOP] Excessive clipping detected: {ep_metrics['clipping_pct']:.2f}%! TERMINATING EXPERIMENT.")
            break

        # Save Epoch Checkpoint
        ckpt_filename = f"pilot_restricted_ep{epoch}.pt"
        ckpt_path = os.path.join(exp_dir, ckpt_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': ep_metrics,
            'composite_score': ep_metrics['composite_score']
        }, ckpt_path)
        print(f"  Saved epoch checkpoint: {ckpt_path}")

        if ep_metrics['composite_score'] > best_composite and ep_metrics['shark_iou'] >= 0.1472:
            best_composite = ep_metrics['composite_score']
            best_ckpt_path = os.path.join(exp_dir, "best_pilot_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': ep_metrics,
                'composite_score': ep_metrics['composite_score']
            }, best_ckpt_path)
            print(f"  >>> NEW BEST PILOT CHECKPOINT SAVED: Composite = {best_composite:.4f}")

    runtime = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"    PILOT EXPERIMENT EXECUTION COMPLETED (Total Runtime: {runtime:.1f}s)")
    print("=" * 80)

    # Save complete JSON log
    report_path = os.path.join(exp_dir, "pilot_restricted_report.json")
    with open(report_path, "w") as f:
        json.dump({
            'history': history,
            'total_runtime_seconds': runtime,
            'best_checkpoint': best_ckpt_path,
            'best_composite': best_composite
        }, f, indent=2)

    print(f"Complete pilot experiment report saved to: {report_path}")
    return history

if __name__ == '__main__':
    run_pilot()
