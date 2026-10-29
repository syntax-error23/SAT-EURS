"""
Stage 4: Complementary ConvNeXt-Base FPN Hierarchical Training
SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Key Features:
1. Model: ConvNeXt-Base FPN (Hierarchical multi-scale pyramid at Stride 4, 8, 16, 32).
2. Overfitting Prevention: Stem and Stage 0 strictly frozen.
3. Orthogonal Representations: Convolutional inductive bias completely breaks the ViT patch-14 error correlation.
4. Loss: SegmentationV7Loss (0.30 WCE + 0.35 Lovasz + 0.25 SoftMacroF1 + 0.10 Boundary).
5. Augmentation: Exact-(wt, depth) matched Copy-Paste on 16 training scenes.
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

from models.convnext_fpn import build_convnext_fpn
from losses.segmentation_v7_loss import SegmentationV7Loss
from metrics.segmentation_metrics import SegmentationEvaluator
from data.controlled_copy_paste import ControlledCopyPasteAugmenter
from tools.train_track1_screening import ControlledTrack1Dataset, apply_rsuigm_batch_gpu

def train_convnext_fold(
    fold_idx: int,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 1e-4,
    crop_size: int = 640,
    save_root: str = "checkpoints/v7_convnext"
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"CONVNEXT_F{fold_idx}"
    print("\n" + "=" * 78, flush=True)
    print(f"TRAINING STAGE 4 CONVNEXT-FPN: {run_id}", flush=True)
    print(f"  Fold: {fold_idx} | Resolution: {crop_size}x{crop_size} | Epochs: {epochs} | Batch: {batch_size*grad_accum_steps}", flush=True)
    print(f"  LR: {lr} | Stage 0 Frozen: True", flush=True)
    print("=" * 78, flush=True)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    fold_meta = manifest["folds"][f"fold_{fold_idx}"]
    train_scenes = fold_meta["train_scenes"]
    val_scenes = fold_meta["val_scenes"]

    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold_idx}.pt"
    rsuigm_table = torch.load(tbl_path, map_location=device, weights_only=False)

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    model = build_convnext_fpn(num_classes=5, freeze_stage0=True).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
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
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        # Validation on held-out images
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
            print(f"  -> Saved new best ConvNeXt model (Q_seg: {best_q_seg:.4f})", flush=True)

    print(f"\nFold {fold_idx} ConvNeXt training completed in {time.time() - t0:.1f}s. Best Q_seg: {best_q_seg:.4f}", flush=True)
    return {"fold": fold_idx, "best_q_seg": best_q_seg, "best_metrics": best_metrics}

def run_stage4_pilot():
    print("STARTING STAGE 4 CONVNEXT-FPN TRAINING PILOT (FOLDS 0 & 1)...", flush=True)
    res_f0 = train_convnext_fold(fold_idx=0, epochs=5)
    res_f1 = train_convnext_fold(fold_idx=1, epochs=5)
    return res_f0, res_f1

if __name__ == '__main__':
    run_stage4_pilot()
