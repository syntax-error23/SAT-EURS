"""
Production Training Script for EURS-Net V2 (Track 1 & Track 2 Training Engine)
Features:
- Config-driven (configs/baseline_v1.yaml) or CLI overrides.
- Blackwell RTX 5090 optimized (native bfloat16 AMP support without GradScaler skips).
- Correct scaling sequence when fp16 is active: scale -> backward -> unscale_ -> clip -> step -> update.
- Gradient accumulation support (--grad_accum).
- Tunable DataLoader num_workers with persistent_workers and pin_memory.
- Samplers: StandardSampler & SceneBalancedSampler with fixed epoch step semantics.
- Square-root smoothed class weights (mean 1.0) computed strictly from training split.
- Authoritative multi-task loss objective: 1.0 * L_seg + 0.5 * L_rest.
- Comprehensive atomic checkpointing (utils/checkpointing.py) with full state restoration.
- Validation evaluation via evaluate.py.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import math
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from utils.paths import DATA_DIR, FROZEN_MANIFEST_PATH, CHECKPOINTS_DIR, CONFIGS_DIR
from models import EURSNetV2
from data.dataset import EURSDataset
from data.transforms import SynchronizedTransform
from data.samplers import StandardSampler, SceneBalancedSampler
from losses.loss_config import compute_training_class_weights, create_authoritative_criterion
from utils.checkpointing import save_checkpoint, load_checkpoint
from evaluate import evaluate


def get_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int = 5,
    total_epochs: int = 50,
    min_lr_ratio: float = 1e-3
):
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - 1 - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def train_model(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Precision configuration: Blackwell RTX 5090 natively supports bfloat16
    use_amp = args.use_amp and device.type == 'cuda'
    if use_amp:
        if args.amp_dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
            target_amp_dtype = torch.bfloat16
            use_scaler = False  # GradScaler is neither needed nor recommended for bfloat16
        else:
            target_amp_dtype = torch.float16
            use_scaler = True
    else:
        target_amp_dtype = torch.float32
        use_scaler = False

    # Safety guardrail: Multi-process DataLoader workers on Windows CPU can cause severe RAM exhaustion.
    # Automatically clamp workers to 0 on CPU to ensure 100% stable execution on developer laptops.
    eff_workers = 0 if device.type == 'cpu' else args.num_workers
    eff_persistent = False if eff_workers == 0 else (args.num_workers > 0)
    eff_pin_memory = (device.type == 'cuda')

    # Safety guardrail: CPU dry-run automatically uses 256x256 to prevent CPU freeze / memory thrashing on laptop
    if device.type == 'cpu' and args.dry_run and not getattr(args, '_target_size_explicit', False):
        args.target_size = 256

    print("=" * 65)
    print("                  EURS-NET V2 TRAINING ENGINE")
    print("=" * 65)
    print(f"  Target Device:        {device}")
    print(f"  Batch Size:           {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"  Grad Accumulation:    {args.grad_accum}")
    print(f"  DataLoader Workers:   {eff_workers} (configured: {args.num_workers})")
    print(f"  Sampler:              {args.sampler}")
    print(f"  Base Learning Rate:   {args.lr}")
    print(f"  Total Epochs:         {args.epochs}")
    print(f"  Warmup Epochs:        {args.warmup_epochs}")
    print(f"  Target Resolution:    {args.target_size}x{args.target_size}")
    print(f"  Mixed Precision AMP:  {use_amp} (dtype: {target_amp_dtype}, GradScaler: {use_scaler})")
    print(f"  Data Directory:       {args.data_dir}")
    print(f"  Manifest:             {args.manifest}")
    print("=" * 65 + "\n")

    # 1. Dataset & Loaders
    tx = SynchronizedTransform(target_size=(args.target_size, args.target_size)) if args.augment else None

    train_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='train',
        manifest_path=args.manifest,
        transform=tx,
        target_size=(args.target_size, args.target_size)
    )
    val_dataset = EURSDataset(
        data_dir=args.data_dir,
        split='val',
        manifest_path=args.manifest,
        transform=None,
        target_size=(args.target_size, args.target_size)
    )

    # Instantiate chosen sampler
    if args.sampler == 'scene_balanced':
        train_sampler = SceneBalancedSampler(train_dataset, seed=args.seed)
    else:
        train_sampler = StandardSampler(train_dataset, shuffle=True, seed=args.seed)

    # Safety guardrail: Multi-process DataLoader workers on Windows CPU can cause severe RAM exhaustion.
    # Automatically clamp workers to 0 on CPU to ensure 100% stable execution on developer laptops.
    eff_workers = 0 if device.type == 'cpu' else args.num_workers
    eff_persistent = False if eff_workers == 0 else (args.num_workers > 0)
    eff_pin_memory = (device.type == 'cuda')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=eff_workers,
        pin_memory=eff_pin_memory,
        persistent_workers=eff_persistent
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=eff_workers,
        pin_memory=eff_pin_memory,
        persistent_workers=eff_persistent
    )

    # 2. Model, Loss, Optimizer
    model = EURSNetV2(num_classes=5).to(device)

    # Compute training class weights (strictly training split)
    alpha = compute_training_class_weights(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        num_classes=5
    ).to(device)

    # Authoritative multi-task criterion: 1.0 * L_seg + 0.5 * L_rest
    loss_fn = create_authoritative_criterion(alpha=alpha, weight_seg=1.0, weight_rest=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_ratio=args.min_lr / args.lr
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(use_scaler and device.type == 'cuda'))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_score = -1.0
    start_epoch = 1
    global_step = 0

    # Resume from checkpoint if requested
    if args.resume and os.path.exists(args.resume):
        ckpt_meta = load_checkpoint(
            filepath=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_scaler else None,
            map_location=str(device)
        )
        start_epoch = ckpt_meta.get("epoch", 0) + 1
        global_step = ckpt_meta.get("global_step", 0)
        best_score = ckpt_meta.get("best_score", -1.0)
        print(f"[RESUME] Resumed from {args.resume} at epoch {start_epoch}, step {global_step}, best score {best_score:.4f}")

    # 3. Training Loop
    epochs_to_run = 1 if args.dry_run else args.epochs

    for epoch in range(start_epoch, epochs_to_run + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        optimizer.zero_grad()

        for batch_idx, (deg, clean, mask, meta) in enumerate(train_loader, 1):
            global_step += 1
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=target_amp_dtype, enabled=use_amp):
                restored_rgb, seg_logits = model(deg)
                if restored_rgb.shape[2:] != (args.target_size, args.target_size):
                    restored_rgb = nn.functional.interpolate(restored_rgb, size=(args.target_size, args.target_size), mode='bilinear', align_corners=False)
                    seg_logits = nn.functional.interpolate(seg_logits, size=(args.target_size, args.target_size), mode='bilinear', align_corners=False)

                loss, l_dict = loss_fn(restored_rgb, seg_logits, clean, mask)
                loss_scaled = loss / args.grad_accum

            assert torch.isfinite(loss), f"Loss diverged to NaN/Inf at epoch {epoch}, batch {batch_idx}!"

            if use_scaler:
                scaler.scale(loss_scaled).backward()
                if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_scaled.backward()
                if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader):
                print(f"  [Epoch {epoch:2d}/{epochs_to_run:2d}] Batch {batch_idx:2d}/{len(train_loader):2d} | "
                      f"LR: {current_lr:.2e} | Loss: {loss.item():.4f} (Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f})")

            if args.dry_run and batch_idx >= 2:
                if batch_idx % args.grad_accum != 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                        optimizer.step()
                    optimizer.zero_grad()
                print(f"  [DRY RUN] Completed {batch_idx} batches smoke test successfully.")
                break

        scheduler.step()

        # Validation & Checkpoint Selection
        if epoch % args.val_interval == 0 or epoch == epochs_to_run:
            print(f"\n  Running validation evaluation for Epoch {epoch}...")
            val_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                device=device.type,
                max_batches=1 if args.dry_run else None
            )
            comp_score = val_metrics['composite']['Composite_Score']

            # Save latest checkpoint atomically
            latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
            save_checkpoint(
                filepath=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                extra_state={"val_metrics": val_metrics}
            )

            if comp_score > best_score:
                best_score = comp_score
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                save_checkpoint(
                    filepath=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    epoch=epoch,
                    global_step=global_step,
                    best_score=best_score,
                    extra_state={"val_metrics": val_metrics}
                )
                print(f"  [CHECKPOINT] New best composite score {best_score:.4f} achieved! Saved {best_path}\n")

    print("[TRAINING] Finished execution successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train EURS-Net V2")
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--manifest', type=str, default=str(FROZEN_MANIFEST_PATH))
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument('--sampler', type=str, default='standard', choices=['standard', 'scene_balanced'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--target_size', type=int, default=1024)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load YAML config if specified (explicit CLI arguments override YAML settings)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        cli_args = {a.split('=')[0][2:] for a in sys.argv[1:] if a.startswith('--')}
        if 'training' in cfg:
            if 'epochs' not in cli_args:
                args.epochs = cfg['training'].get('total_epochs', args.epochs)
            if 'warmup_epochs' not in cli_args:
                args.warmup_epochs = cfg['training'].get('warmup_epochs', args.warmup_epochs)
            if 'lr' not in cli_args:
                args.lr = cfg['training'].get('base_lr', args.lr)
            if 'min_lr' not in cli_args:
                args.min_lr = cfg['training'].get('min_lr', args.min_lr)
            if 'batch_size' not in cli_args:
                args.batch_size = cfg['training'].get('batch_size', args.batch_size)
            if 'grad_accum' not in cli_args:
                args.grad_accum = cfg['training'].get('grad_accum_steps', args.grad_accum)
            if 'seed' not in cli_args:
                args.seed = cfg['training'].get('seed', args.seed)
        if 'precision' in cfg:
            if 'use_amp' not in cli_args:
                args.use_amp = cfg['precision'].get('use_amp', args.use_amp)
            if 'amp_dtype' not in cli_args:
                args.amp_dtype = cfg['precision'].get('amp_dtype', args.amp_dtype)
        if 'dataloader' in cfg:
            if 'num_workers' not in cli_args:
                args.num_workers = cfg['dataloader'].get('num_workers', args.num_workers)
            if 'sampler' not in cli_args:
                args.sampler = cfg['dataloader'].get('sampler', args.sampler)
            if 'target_size' not in cli_args:
                ts = cfg['dataloader'].get('target_size', [args.target_size, args.target_size])
                args.target_size = ts[0] if isinstance(ts, list) else ts
        if 'paths' in cfg:
            if 'data_dir' not in cli_args:
                args.data_dir = cfg['paths'].get('cache_dir', cfg['paths'].get('data_dir', args.data_dir))
            if 'manifest' not in cli_args:
                args.manifest = cfg['paths'].get('manifest_path', args.manifest)
            if 'checkpoint_dir' not in cli_args:
                args.checkpoint_dir = cfg['paths'].get('checkpoint_dir', args.checkpoint_dir)
        print(f"[CONFIG] Loaded base configuration from: {args.config}")

    args._target_size_explicit = ('target_size' in cli_args) if 'cli_args' in locals() else any(a.startswith('--target_size') for a in sys.argv)
    train_model(args)
