"""
Stage 3: High-Resolution DPT Segmentation Training (DINOv2HighResDPT)
SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Key Features:
1. Model: DINOv2HighResDPT (Early ViT blocks 0..15 frozen; blocks 16..23 fine-tuned at lr=1.5e-5).
2. Detail Stem: Parallel Stride-2 and Stride-4 RGB Detail Stem + Depthwise-Separable Top-Down Fusion.
3. Warm-Start: Loaded strictly from #18713 checkpoints with zero-initialized detail head.
4. Loss: Direct Competition Metric Loss (0.30 WCE + 0.35 Lovasz + 0.25 SoftMacroF1 + 0.10 Boundary).
5. High-Resolution: Trained at 700x700 (50x50 tokens) on RTX 5090.
6. Augmentation: Exact-(wt, depth) matched Copy-Paste on 16 training scenes.
"""

import os
import os.path as osp
import sys
import json
import time
import glob
import random
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from models.dinov2_highres_dpt import DINOv2HighResDPT
from losses.segmentation_v7_loss import SegmentationV7Loss
from metrics.segmentation_metrics import SegmentationEvaluator
from data.controlled_copy_paste import ControlledCopyPasteAugmenter
from tools.train_track1_screening import ControlledTrack1Dataset, apply_rsuigm_batch_gpu

def build_dpt_from_checkpoint(ckpt_path: str, device: torch.device) -> DINOv2HighResDPT:
    model = DINOv2HighResDPT(num_classes=5, freeze_blocks_up_to=15, pretrained=False).to(device)
    if osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded V6 checkpoint from {ckpt_path} (new high-res layers: {len(incompatible.missing_keys)})")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return model

def train_dpt_fold(
    fold_idx: int,
    epochs: int = 5,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    backbone_lr: float = 1.5e-5,
    decoder_lr: float = 5e-5,
    stem_lr: float = 1.5e-4,
    crop_size: int = 700,
    save_root: str = "checkpoints/v7_dpt"
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"DINOV2_HIGHRES_DPT_F{fold_idx}"
    print("\n" + "=" * 78, flush=True)
    print(f"TRAINING STAGE 3 DINOv2-HighRes-DPT: {run_id}", flush=True)
    print(f"  Fold: {fold_idx} | Resolution: {crop_size}x{crop_size} | Epochs: {epochs} | Batch: {batch_size*grad_accum_steps}", flush=True)
    print(f"  LRs: Backbone(16..23)={backbone_lr}, Decoder={decoder_lr}, Stem={stem_lr}", flush=True)
    print("=" * 78, flush=True)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    fold_meta = manifest["folds"][f"fold_{fold_idx}"]
    train_scenes = fold_meta["train_scenes"]
    val_scenes = fold_meta["val_scenes"]

    # Load 16-scene RSUIGM table
    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold_idx}.pt"
    rsuigm_table = torch.load(tbl_path, map_location=device, weights_only=False)

    # Copy-paste augmenter
    cp_augmenter = ControlledCopyPasteAugmenter(
        train_scenes=train_scenes,
        data_root="data/cached_1024",
        paste_prob=0.50,
        max_pastes=2,
        base_seed=42
    )

    train_ds = ControlledTrack1Dataset(
        scenes=train_scenes,
        crop_size=crop_size,
        is_train=True,
        use_rsuigm=True,
        rsuigm_table=rsuigm_table,
        copy_paste_augmenter=cp_augmenter,
        base_seed=42
    )
    val_ds = ControlledTrack1Dataset(
        scenes=val_scenes,
        crop_size=crop_size,
        is_train=False,
        use_rsuigm=True,
        rsuigm_table=rsuigm_table,
        base_seed=42
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    # Load from #18713 checkpoint
    init_ckpt = f"checkpoints/track1_screening/DINOV2_CP_F{fold_idx}/best_model.pt"
    model = build_dpt_from_checkpoint(init_ckpt, device=device)

    # Differential parameter groups
    trainable_backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    param_groups = [
        {"params": trainable_backbone, "lr": backbone_lr, "weight_decay": 1e-4},
        {"params": model.proj.parameters(), "lr": decoder_lr, "weight_decay": 1e-4},
        {"params": model.fusion.parameters(), "lr": decoder_lr, "weight_decay": 1e-4},
        {"params": model.stem.parameters(), "lr": stem_lr, "weight_decay": 1e-4},
        {"params": model.fuse_stage4.parameters(), "lr": stem_lr, "weight_decay": 1e-4},
        {"params": model.fuse_stage2.parameters(), "lr": stem_lr, "weight_decay": 1e-4},
        {"params": model.detail_head.parameters(), "lr": stem_lr, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SegmentationV7Loss(num_classes=5).to(device)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    save_dir = osp.join(save_root, run_id)
    os.makedirs(save_dir, exist_ok=True)

    best_q_seg = -1.0
    best_metrics = {}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        t_ep = time.time()

        optimizer.zero_grad()
        for step, (x_raw, y_mask, wts, depths) in enumerate(train_loader):
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            x_rest = apply_rsuigm_batch_gpu(x_raw, wts, depths, rsuigm_table)
            x_norm = (x_rest - mean_im) / std_im

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)
                loss, _ = criterion(logits, y_mask)
                loss_scaled = loss / grad_accum_steps

            loss_scaled.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        # Full Validation on all 800 images
        model.eval()
        evaluator = SegmentationEvaluator(num_classes=5)
        with torch.no_grad():
            for x_raw, y_mask, wts, depths in val_loader:
                x_raw = x_raw.to(device, non_blocking=True)
                y_mask = y_mask.to(device, non_blocking=True)
                x_rest = apply_rsuigm_batch_gpu(x_raw, wts, depths, rsuigm_table)
                x_norm = (x_rest - mean_im) / std_im

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x_norm)
                evaluator.update(logits.float(), y_mask)

        val_res = evaluator.compute()
        val_q_seg = val_res["Q_seg"]
        print(f"  Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val mIoU: {val_res['mIoU']:.4f} | F1: {val_res['macro_F1']:.4f} | Q_seg: {val_q_seg:.4f} | "
              f"Rare IoU: [{val_res['per_class_IoU'][3]:.4f}, {val_res['per_class_IoU'][4]:.4f}]", flush=True)

        if val_q_seg > best_q_seg:
            best_q_seg = val_q_seg
            best_metrics = val_res
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_res,
                "fold": fold_idx
            }, osp.join(save_dir, "best_model.pt"))
            print(f"  -> Saved new best DPT model (Q_seg: {best_q_seg:.4f})", flush=True)

    print(f"\nFold {fold_idx} DPT training completed in {time.time() - t0:.1f}s. Best Q_seg: {best_q_seg:.4f}", flush=True)
    return {"fold": fold_idx, "best_q_seg": best_q_seg, "best_metrics": best_metrics}

def run_stage3_pilot():
    print("STARTING STAGE 3 DPT HIGH-RES TRAINING PILOT (FOLDS 0 & 1)...", flush=True)
    res_f0 = train_dpt_fold(fold_idx=0, epochs=5)
    res_f1 = train_dpt_fold(fold_idx=1, epochs=5)
    return res_f0, res_f1

if __name__ == '__main__':
    run_stage3_pilot()
