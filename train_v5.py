"""
EURS-Net V5 Training Engine:
  - Generator: `EURSNetV5` with Encoder Self-Attention & Bidirectional Cross-Attention
    (`EncoderSelfAndCrossAttentionV5`, `MidLevelCrossAttentionV5`), Unbounded Per-Cell 3x4
    Cross-Channel Affine Color Restorer (`PerCellAffineRestorationDecoderV5`), and
    LR-ASPP Global Context Segmentation (`DualBranchSegmentationDecoderV5`).
  - Discriminator: `PatchGANDiscriminator` (70x70 Markovian Binary Patch Classifier, spectral-normed).
  - Statistical Distribution Alignment:
    * `compute_dataset_and_class_statistics()` (Overall & Per-Class Mean/Variance + Pairwise Distances)
    * `DistributionAlignedCopyPasteAugmenter` (augments low-distribution classes 2, 3, 4 while aligning
      class mean/variance toward overall distribution and physically re-degrading pasted clean patches
      so `degraded` and `clean` remain 100% paired for joint restoration + segmentation training).
    * `MeanVarianceDistributionLoss` (minimizes overall and per-class mean & variance distances).
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, Any, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from utils.paths import DATA_DIR, FROZEN_MANIFEST_PATH
from models.eurs_net_v5 import EURSNetV5
from models.patch_discriminator import (
    PatchGANDiscriminator,
    PatchGANLoss,
    MeanVarianceDistributionLoss,
)
from data.dataset import EURSDataset
from data.transforms import SynchronizedTransform
from data.class_stats_transform import (
    compute_dataset_and_class_statistics,
    DistributionAlignedCopyPasteAugmenter,
)
from data.samplers import SceneBalancedSampler, StandardSampler
from losses.v4_losses import V4MultiTaskLoss
from train_v4 import (
    verify_protected_checkpoints,
    evaluate_v4_model,
)
from utils.checkpointing import save_checkpoint


def build_v5_optimizers(
    model: EURSNetV5,
    discriminator: PatchGANDiscriminator,
    base_lr: float = 3e-4,
    d_lr: float = 2e-4,
    weight_decay: float = 1e-4,
) -> Tuple[optim.Optimizer, optim.Optimizer]:
    """
    Construct AdamW optimizers for Generator `EURSNetV5` and `PatchGANDiscriminator`:
      - Group 0 (pretrained backbone & imagenet_encoder): 0.45 * base_lr
      - Group 1 (warm-started V4 decoders):               0.75 * base_lr
      - Group 2 (new V5 attention, 3x4 cell affine, LR-ASPP): 1.25 * base_lr
    """
    v5_new_prefixes = (
        "encoder_attention.",
        "mid_cross_attention.",
        "restoration_decoder.global_ctx_proj",
        "restoration_decoder.cell_affine_32",
        "restoration_decoder.cell_scale",
        "restoration_decoder.affine_head_256",
        "restoration_decoder.fullres_refiner",
        "segmentation_decoder.lraspp_32",
        "segmentation_decoder.context_64",
        "segmentation_decoder.context_gate",
        "segmentation_decoder.rest_adapter_256",
        "segmentation_decoder.rest_gate",
        "segmentation_decoder.mb_lat_",
        "segmentation_decoder.mb_gate_",
    )

    enc_params, dec_params, v5_new_params, v5_out_head_params = [], [], [], []
    for name, param in model.named_parameters():
        if name in (
            "restoration_decoder.cell_affine_32.6.weight",
            "restoration_decoder.cell_affine_32.6.bias",
            "restoration_decoder.affine_head_256.6.weight",
            "restoration_decoder.affine_head_256.6.bias",
            "restoration_decoder.fullres_refiner.6.weight",
            "restoration_decoder.fullres_refiner.6.bias",
        ):
            v5_out_head_params.append(param)
        elif name.startswith(v5_new_prefixes):
            v5_new_params.append(param)
        elif name.startswith(("backbone.", "fwb.", "imagenet_encoder.")):
            enc_params.append(param)
        else:
            dec_params.append(param)

    opt_g = optim.AdamW(
        [
            {"name": "encoders", "params": enc_params, "lr": base_lr * 0.25, "weight_decay": weight_decay},
            {"name": "decoders", "params": dec_params, "lr": base_lr * 0.40, "weight_decay": weight_decay},
            {"name": "v5_new", "params": v5_new_params, "lr": base_lr * 1.00, "weight_decay": weight_decay},
            {"name": "v5_out_heads", "params": v5_out_head_params, "lr": base_lr * 0.05, "weight_decay": weight_decay},
        ],
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    opt_d = optim.AdamW(
        discriminator.parameters(),
        lr=d_lr,
        betas=(0.5, 0.999),
        weight_decay=1e-4,
    )
    return opt_g, opt_d


def set_v5_phase_state(model: EURSNetV5, epoch: int, rest_warmup_epochs: int) -> str:
    """
    Configure parameter requires_grad and BatchNorm running stats protection:
    - `fwb` and the V2 restoration core (`proj_f128`, `proj_f256`, `dw_conv256`, `proj_stem`,
      `dw_conv512`, `head`) are kept anchored so base restoration never regresses, while
      `cell_affine_32` (32x32 3x4 cross-channel affine grid), `global_ctx_proj`,
      `affine_head_256`, and `fullres_refiner` learn high-PSNR color & detail restoration.
    - All pre-calibrated BatchNorm2d running_mean / running_var buffers are kept in .eval()
      mode so 4-image mini-batches never corrupt running statistics.
    """
    v2_rest_core = (
        "fwb.",
        "backbone.",
        "restoration_decoder.proj_f128",
        "restoration_decoder.proj_f256",
        "restoration_decoder.dw_conv256",
        "restoration_decoder.proj_stem",
        "restoration_decoder.dw_conv512",
        "restoration_decoder.head",
    )
    for name, param in model.named_parameters():
        if name.startswith(v2_rest_core):
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.fwb.eval()
    model.backbone.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if epoch <= rest_warmup_epochs:
        return f"PHASE_1_RESTORATION_AND_ATTENTION_BOOST (epoch {epoch}/{rest_warmup_epochs})"
    return f"PHASE_2_JOINT_GAN_ATTENTIVE_FULL (epoch {epoch})"


def get_cosine_scheduler(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, min_ratio: float = 0.05):
    def lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        prog = float(ep - warmup_epochs) / float(max(1, total_epochs - 1 - warmup_epochs))
        prog = min(max(prog, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)


def train_v5(args: argparse.Namespace) -> Dict[str, Any]:
    verify_protected_checkpoints()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Ensure statistical class audit JSON exists
    stats_json = "audits/class_distribution_stats.json"
    if not os.path.exists(stats_json):
        compute_dataset_and_class_statistics(data_dir=args.data_dir, manifest_path=args.manifest, save_report_path=stats_json)

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    use_amp = args.use_amp and device.type == 'cuda'
    target_amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and (target_amp_dtype == torch.float16)

    print("=" * 80)
    print("  EURS-NET V5: SELF/CROSS-ATTENTION + 3x4 AFFINE + PATCHGAN + CLASS STATS ENGINE")
    print("=" * 80)
    print(f"  Device:               {device} (AMP: {use_amp}, dtype: {target_amp_dtype})")
    print(f"  Resolution:           {args.target_size}x{args.target_size} (Val: 1024x1024)")
    print(f"  Batch Size:           {args.batch_size} (Grad Accum: {args.grad_accum} -> Eff: {args.batch_size * args.grad_accum})")
    print(f"  Epochs:               {args.epochs} (Phase 1 Rest/Attn Boost: {args.rest_warmup_epochs})")
    print(f"  Train All 20 Scenes:  {args.train_all_scenes}")
    print(f"  Dist-Aligned Paste:   {args.paste_prob} (Low-dist classes 2, 3, 4 with physical RSUIGM pairing)")
    print(f"  Warm-Start Ckpt:      {args.warmstart}")
    print("=" * 80)

    # 1. Distribution-Aligned Rare-Class Copy-Paste & Synchronized Spatial Transforms
    cp_aug = DistributionAlignedCopyPasteAugmenter(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        paste_prob=args.paste_prob,
        max_pastes_per_image=3,
        base_seed=args.seed,
        include_all_scenes=args.train_all_scenes,
    )
    tx = SynchronizedTransform(
        target_size=(args.target_size, args.target_size),
        scale_range=(0.60, 1.0),
        rot90_prob=0.25,
        base_seed=args.seed,
        copy_paste_augmenter=cp_aug,
    )

    train_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='all' if args.train_all_scenes else 'train',
        manifest_path=args.manifest,
        transform=tx,
        target_size=(args.target_size, args.target_size),
        base_seed=args.seed,
    )
    val_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='val',
        manifest_path=args.manifest,
        transform=None,
        target_size=(1024, 1024),
        base_seed=args.seed,
    )

    train_sampler = SceneBalancedSampler(train_dataset, seed=args.seed)
    eff_workers = 0 if device.type == 'cpu' else args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=eff_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(eff_workers > 0),
    )
    if args.max_val_batches is not None and len(val_dataset) >= 800:
        strat_indices = [0, 50, 200, 250, 400, 450, 600, 650]
        val_loader = DataLoader(
            torch.utils.data.Subset(val_dataset, strat_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=eff_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(eff_workers > 0),
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=eff_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(eff_workers > 0),
        )

    # 2. Initialize Generator EURSNetV5 & PatchGANDiscriminator
    model = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=True).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6, base_channels=48).to(device)

    if args.warmstart and os.path.exists(args.warmstart):
        ws_info = model.load_v4_or_v2_weights(args.warmstart)
        print(f"[WARM-START] Loaded {ws_info['transferred_tensors']}/{ws_info['total_v5_tensors']} tensors from {args.warmstart}")

    opt_g, opt_d = build_v5_optimizers(model, discriminator, base_lr=args.lr, d_lr=args.d_lr, weight_decay=args.weight_decay)
    sched_g = get_cosine_scheduler(opt_g, warmup_epochs=1, total_epochs=args.epochs, min_ratio=0.05)
    sched_d = get_cosine_scheduler(opt_d, warmup_epochs=1, total_epochs=args.epochs, min_ratio=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # 3. Losses: Multi-Task V4 + Mean/Variance Distribution Loss + PatchGAN Binary Classifier Loss
    # Boost alpha for Class 3 (Shark/Ray) and Class 4 (Other/Starfish) based on class stats audit
    class_alpha = torch.tensor([0.45, 0.65, 1.45, 2.50, 2.65], dtype=torch.float32, device=device)
    criterion_mt = V4MultiTaskLoss(
        num_classes=5,
        alpha=class_alpha,
        weight_seg=args.weight_seg,
        weight_rest=args.weight_rest,
        lambda_dice=1.0,
        lambda_lovasz=0.35,
        lovasz_start_epoch=1,
        lambda_boundary=0.15,
        weight_charb=1.5,
        weight_ssim=0.6,
        weight_color=0.35,
        weight_lap=0.20,
    ).to(device)
    criterion_dist = MeanVarianceDistributionLoss(num_classes=5).to(device)
    criterion_gan = PatchGANLoss(adv_weight=args.adv_weight, fm_weight=args.fm_weight).to(device)

    history = []
    best_score = -1.0
    best_metrics = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        discriminator.train()
        phase_str = set_v5_phase_state(model, epoch, args.rest_warmup_epochs)
        train_dataset.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        criterion_mt.set_epoch(epoch)

        # In Phase 1, emphasize Restoration + Mean/Variance Distribution + PatchGAN to rapidly push PSNR > 21+ dB
        rest_mult = 2.0 if epoch <= args.rest_warmup_epochs else 1.25
        dist_weight = 0.60 if epoch <= args.rest_warmup_epochs else 0.40
        criterion_mt.weight_rest = args.weight_rest * rest_mult

        print(f"\n--- Epoch {epoch}/{args.epochs} | {phase_str} ---")
        running = {
            'g_total': 0.0, 'seg': 0.0, 'rest': 0.0, 'dist': 0.0,
            'g_gan': 0.0, 'd_loss': 0.0, 'd_acc': 0.0, 'train_psnr': 0.0,
        }
        n_batches = 0
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)

        for b_idx, (deg, clean, mask, _) in enumerate(train_loader, 1):
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # A. Forward Generator EURSNetV5
            with torch.amp.autocast(device_type=device.type, dtype=target_amp_dtype, enabled=use_amp):
                restored_rgb, seg_logits, boundary_logits = model.forward_with_aux(deg)

            # B. Update PatchGANDiscriminator (Binary Classifier on Real Clean vs Fake Restored)
            d_loss, d_info = criterion_gan.compute_d_loss(discriminator, deg, clean, restored_rgb.detach())
            d_loss_scaled = d_loss / float(args.grad_accum)
            if use_scaler:
                scaler.scale(d_loss_scaled).backward()
            else:
                d_loss_scaled.backward()

            # C. Compute Generator Losses (Strict FP32)
            mt_loss, mt_info = criterion_mt(restored_rgb, seg_logits, clean, mask, boundary_logits=boundary_logits)
            dist_loss, dist_info = criterion_dist(restored_rgb, clean, mask)
            g_gan_loss, gan_info = criterion_gan.compute_g_loss(discriminator, deg, clean, restored_rgb)

            # Direct MSE/PSNR booster term to drive PSNR toward 25+ dB
            mse_val = F.mse_loss(restored_rgb.float(), clean.float())
            psnr_surrogate_loss = 2.5 * mse_val

            g_total = (
                mt_loss
                + rest_mult * psnr_surrogate_loss
                + dist_weight * dist_loss
                + g_gan_loss
            )
            g_scaled = g_total / float(args.grad_accum)
            if use_scaler:
                scaler.scale(g_scaled).backward()
            else:
                g_scaled.backward()

            if (b_idx % args.grad_accum == 0) or (b_idx == len(train_loader)):
                if use_scaler:
                    scaler.unscale_(opt_d)
                    scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=2.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                if use_scaler:
                    scaler.step(opt_d)
                    scaler.step(opt_g)
                    scaler.update()
                else:
                    opt_d.step()
                    opt_g.step()
                opt_d.zero_grad(set_to_none=True)
                opt_g.zero_grad(set_to_none=True)
                global_step += 1

            with torch.no_grad():
                batch_psnr = float((-10.0 * torch.log10(mse_val.clamp(min=1e-8))).item())

            running['g_total'] += float(g_total.detach().item())
            running['seg'] += mt_info['loss_seg']
            running['rest'] += mt_info['loss_rest']
            running['dist'] += dist_info['loss_dist_total']
            running['g_gan'] += gan_info['g_gan_total']
            running['d_loss'] += d_info['d_loss']
            running['d_acc'] += d_info['d_acc_total']
            running['train_psnr'] += batch_psnr
            n_batches += 1

            if b_idx % 200 == 0 or b_idx == len(train_loader):
                print(
                    f"  [Batch {b_idx:4d}/{len(train_loader)}] "
                    f"G={running['g_total']/n_batches:.3f} | Seg={running['seg']/n_batches:.3f} | "
                    f"Rest={running['rest']/n_batches:.3f} | Dist={running['dist']/n_batches:.3f} | "
                    f"D_loss={running['d_loss']/n_batches:.3f} (D_acc={running['d_acc']/n_batches*100:.1f}%) | "
                    f"TrainPSNR={running['train_psnr']/n_batches:.2f} dB"
                )

            if args.max_train_batches is not None and b_idx >= args.max_train_batches:
                break

        sched_g.step()
        sched_d.step()

        # Validation on all 800 validation images at 1024x1024
        val_t0 = time.time()
        val_metrics = evaluate_v4_model(model, val_loader, device=device, max_batches=args.max_val_batches)
        val_dt = time.time() - val_t0
        ep_dt = time.time() - ep_t0

        comp = val_metrics['composite_official_surrogate']
        print(
            f"  [VAL Epoch {epoch}] Composite={comp:.4f} | Q_seg={val_metrics['Q_seg']:.4f} "
            f"(mIoU={val_metrics['mIoU']:.4f}, F1={val_metrics['macro_F1']:.4f}, "
            f"Shark={val_metrics['shark_iou']:.4f}, Other={val_metrics['other_iou']:.4f}) | "
            f"Q_rest={val_metrics['Q_rest']:.4f} (PSNR={val_metrics['PSNR_dB']:.2f} dB, "
            f"SSIM={val_metrics['SSIM']:.4f}, Sharp={val_metrics['sharpness_ratio']:.4f}, "
            f"Clip={val_metrics['clipping_aggregate_pct']:.2f}%) | Time={ep_dt:.1f}s (val {val_dt:.1f}s)"
        )

        rec = {
            'epoch': epoch,
            'train': {k: v / max(1, n_batches) for k, v in running.items()},
            'val': val_metrics,
        }
        history.append(rec)
        with open(os.path.join(args.log_dir, "v5_training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save latest checkpoint (including discriminator state)
        latest_path = os.path.join(args.checkpoint_dir, "latest_v5_checkpoint.pt")
        save_checkpoint(
            filepath=latest_path,
            model=model,
            optimizer=opt_g,
            scheduler=sched_g,
            scaler=scaler if use_scaler else None,
            epoch=epoch,
            global_step=global_step,
            best_score=max(best_score, comp),
            extra_state={
                "val_metrics": val_metrics,
                "discriminator_state_dict": discriminator.state_dict(),
            },
        )

        if val_metrics['coverage_penalty'] == 1.0 and comp > best_score:
            best_score = comp
            best_metrics = val_metrics
            best_path = os.path.join(args.checkpoint_dir, "best_v5_checkpoint.pt")
            save_checkpoint(
                filepath=best_path,
                model=model,
                optimizer=opt_g,
                scheduler=sched_g,
                scaler=scaler if use_scaler else None,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                extra_state={
                    "val_metrics": val_metrics,
                    "discriminator_state_dict": discriminator.state_dict(),
                },
            )
            print(f"  >>> [NEW BEST V5 CHECKPOINT] Composite={best_score:.4f} (PSNR={val_metrics['PSNR_dB']:.2f} dB, mIoU={val_metrics['mIoU']:.4f}) saved to {best_path}")

    return {"best_score": best_score, "best_metrics": best_metrics, "history": history}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EURS-Net V5 Attentive GAN + Class-Stats Training Engine")
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--manifest", type=str, default=FROZEN_MANIFEST_PATH)
    p.add_argument("--warmstart", type=str, default="checkpoints/v4_full20/best_v4_checkpoint.pt")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/v5_gan_attn")
    p.add_argument("--log_dir", type=str, default="logs/v5_gan_attn")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--rest_warmup_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--target_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3.5e-4)
    p.add_argument("--d_lr", type=float, default=2.0e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--weight_seg", type=float, default=1.0)
    p.add_argument("--weight_rest", type=float, default=1.2)
    p.add_argument("--adv_weight", type=float, default=0.05)
    p.add_argument("--fm_weight", type=float, default=0.25)
    p.add_argument("--paste_prob", type=float, default=0.65)
    p.add_argument("--train_all_scenes", action="store_true", default=True)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_v5(args)
