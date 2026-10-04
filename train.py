"""
Production Training Script for EURS-Net V2 (Track 1 & Track 2 Training Engine)
Features:
- Mixed precision AMP (torch.amp.autocast) with GradScaler.
- Correct scaling sequence: scale(loss) -> backward -> unscale_ -> clip_grad_norm_ -> step -> update.
- Gradient accumulation support (--grad_accum).
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

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
    print("=" * 65)
    print("                  EURS-NET V2 TRAINING ENGINE")
    print("=" * 65)
    print(f"  Target Device:        {device}")
    print(f"  Batch Size:           {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"  Grad Accumulation:    {args.grad_accum}")
    print(f"  Sampler:              {args.sampler}")
    print(f"  Base Learning Rate:   {args.lr}")
    print(f"  Total Epochs:         {args.epochs}")
    print(f"  Warmup Epochs:        {args.warmup_epochs}")
    print(f"  Target Resolution:    {args.target_size}x{args.target_size}")
    print(f"  Mixed Precision AMP:  {args.use_amp and device.type == 'cuda'}")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
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

    use_cuda_amp = args.use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

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
            scaler=scaler,
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

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda_amp):
                restored_rgb, seg_logits = model(deg)
                if restored_rgb.shape[2:] != (args.target_size, args.target_size):
                    restored_rgb = nn.functional.interpolate(restored_rgb, size=(args.target_size, args.target_size), mode='bilinear', align_corners=False)
                    seg_logits = nn.functional.interpolate(seg_logits, size=(args.target_size, args.target_size), mode='bilinear', align_corners=False)

                loss, l_dict = loss_fn(restored_rgb, seg_logits, clean, mask)
                loss_scaled = loss / args.grad_accum

            assert torch.isfinite(loss), f"Loss diverged to NaN/Inf at epoch {epoch}, batch {batch_idx}!"

            scaler.scale(loss_scaled).backward()

            if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader):
                print(f"  [Epoch {epoch:2d}/{epochs_to_run:2d}] Batch {batch_idx:2d}/{len(train_loader):2d} | "
                      f"LR: {current_lr:.2e} | Loss: {loss.item():.4f} (Seg: {l_dict['loss_seg']:.4f}, Rest: {l_dict['loss_rest']:.4f})")

        scheduler.step()

        # Validation & Checkpoint Selection
        if epoch % args.val_interval == 0 or epoch == epochs_to_run:
            print(f"\n  Running validation evaluation for Epoch {epoch}...")
            val_metrics = evaluate(model, val_loader, device=device.type)
            comp_score = val_metrics['composite']['Composite_Score']

            # Save latest checkpoint atomically
            latest_path = os.path.join(args.checkpoint_dir, 'latest.pt')
            save_checkpoint(
                filepath=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
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
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_score=best_score,
                    extra_state={"val_metrics": val_metrics}
                )
                print(f"  [CHECKPOINT] New best composite score {best_score:.4f} achieved! Saved {best_path}\n")

    print("[TRAINING] Finished execution successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train EURS-Net V2")
    parser.add_argument('--data_dir', type=str, default='d:/MACVI/data/mini_dev')
    parser.add_argument('--manifest', type=str, default='d:/MACVI/splits/manifest_full.json')
    parser.add_argument('--checkpoint_dir', type=str, default='d:/MACVI/checkpoints')
    parser.add_argument('--sampler', type=str, default='standard', choices=['standard', 'scene_balanced'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--dry_run', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    train_model(args)
