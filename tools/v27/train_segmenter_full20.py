"""
SAT-EURS Track 1 — V27 Full-20-Scene Segmenter Training Pipeline
Integrates:
1. All 20 EURS scenes (4,000 images total) to eliminate rare-class blindness (ray1, sharkFish, seahorse1, starfish2, turtles1)
2. V27 Small-Object-Aware crop sampling (Class 2 fish, Class 3 shark/ray, Class 4 macro-fauna)
3. Multi-scale zoom jitter (384..512 crop size resized to 512x512)
4. SWA (Stochastic Weight Averaging) over final epochs
5. Immediate unbuffered file & console logging
"""

import os
import os.path as osp
import sys
import time
import json
import random
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, ".")

from data.v27.dataset import V27SmallObjectAwareDataset
from data.v18.dataset import V18SceneBalancedSampler
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter, EURSAPEXV18Segmenter
from losses.v18.segmentation_losses import SegmentationV18Loss, compute_capped_class_weights


ALL_20_SCENES = [
    'fish1', 'fish11', 'fish12', 'fish13', 'fish14', 'fish15', 'fish17', 'fish18',
    'fish2', 'fish20', 'fish21', 'fish4', 'fish5', 'fish7', 'fish9',
    'ray1', 'seahorse1', 'sharkFish', 'starfish2', 'turtles1'
]


def log_msg(msg: str, log_file=None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def train_segmenter_full20(
    epochs: int = 10,
    swa_start_epoch: int = 7,
    batch_size: int = 8,
    accum_steps: int = 1,
    base_lr: float = 8e-5,
    crop_size: int = 512,
    warmstart_ckpt: Optional[str] = None,
    refiner_ckpt: str = "checkpoints/v18/fold_0/refiner_best.pt",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    out_dir: str = "checkpoints/v27",
    log_path: str = "reports/v27/train_full20.log",
    device_id: int = 0
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    log_msg("=" * 85, log_file)
    log_msg("SAT-EURS V27 — FULL-20-SCENE SMALL-OBJECT-AWARE SEGMENTER TRAINING", log_file)
    log_msg(f"Device: {device} ({torch.cuda.get_device_name(device)})", log_file)
    log_msg(f"All 20 Scenes: {len(ALL_20_SCENES)} scenes (4,000 images total)", log_file)
    log_msg(f"Hyperparameters: epochs={epochs}, swa_start={swa_start_epoch}, batch_size={batch_size}, lr={base_lr}", log_file)
    log_msg("=" * 85, log_file)

    # 1. Setup Frozen Restorer for input features
    log_msg(f"\n[1/5] Loading Restoration Refiner from {refiner_ckpt}...", log_file)
    rsuigm_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(refiner_ckpt, map_location="cpu")
    ref_weights = ref_ckpt["state_dict"] if "state_dict" in ref_ckpt else ref_ckpt
    refiner.load_state_dict(ref_weights)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False

    # 2. Build EURSAPEXV18Segmenter & Warm Start
    log_msg(f"\n[2/5] Building EURSAPEXV18Segmenter...", log_file)
    segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, neck_dim=256).to(device)

    # Candidate warmstart checkpoints
    if warmstart_ckpt is None:
        for candidate in [
            "checkpoints/v27/segmenter_v27_fold0_best.pt",
            "checkpoints/v26/segmenter_full20_best.pt",
            "checkpoints/v18/fold_0/segmenter_best.pt"
        ]:
            if osp.exists(candidate):
                warmstart_ckpt = candidate
                break

    if warmstart_ckpt and osp.exists(warmstart_ckpt):
        ws_obj = torch.load(warmstart_ckpt, map_location="cpu")
        ws_weights = ws_obj["state_dict"] if "state_dict" in ws_obj else ws_obj
        segmenter.load_state_dict(ws_weights, strict=True)
        log_msg(f"  Successfully warm-started model weights from {warmstart_ckpt}!", log_file)
    else:
        log_msg(f"  WARNING: No warm-start checkpoint found! Initializing from scratch.", log_file)

    # 3. Setup Dataset with Small-Object-Aware sampling across all 20 scenes
    log_msg(f"\n[3/5] Initializing V27 Small-Object-Aware Full-20-Scene Dataset (4,000 images)...", log_file)
    train_ds = V27SmallObjectAwareDataset(
        scenes=ALL_20_SCENES,
        target_size=crop_size,
        is_train=True,
        object_crop_prob=0.70,
        min_crop_ratio=0.75,
        base_seed=42
    )
    sampler = V18SceneBalancedSampler(train_ds, batch_size=batch_size, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    log_msg(f"  Dataset ready: {len(train_ds)} samples -> {len(train_loader)} batches/epoch.", log_file)

    # 4. Setup Loss & Optimizer
    log_msg(f"\n[4/5] Initializing Loss & Optimizer...", log_file)
    class_pixel_counts = [1000000, 50000, 30000, 5000, 2000]
    class_weights = compute_capped_class_weights(class_pixel_counts, min_weight=0.50, max_weight=4.00, device=device)
    loss_fn = SegmentationV18Loss(
        num_classes=5,
        class_weights=class_weights,
        w_ce=0.40,
        w_lovasz=0.30,
        w_dice=0.20,
        w_bnd=0.05,
        w_obj=0.05,
        ignore_index=255
    ).to(device)

    param_groups = [
        {"params": [p for p in segmenter.vit_backbone.parameters() if p.requires_grad], "lr": base_lr * 0.10},
        {"params": [p for p in segmenter.conv_backbone.parameters() if p.requires_grad], "lr": base_lr * 0.20},
        {"params": list(segmenter.detail_stem.parameters()) + list(segmenter.cond_encoder.parameters()) + list(segmenter.neck.parameters()), "lr": base_lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # SWA Model
    swa_model = AveragedModel(segmenter)

    # 5. Training Loop
    log_msg(f"\n[5/5] Beginning Training ({epochs} epochs, SWA starting at epoch {swa_start_epoch})...", log_file)
    best_loss = 1e9
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        segmenter.train()
        running_loss = 0.0
        step_count = 0

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            deg = batch["deg_rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            wt = batch["wt"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)

            with torch.no_grad():
                out_ref = refiner(deg, wt, depth)
                restored = out_ref["refined_image"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seg_out = segmenter(restored, deg, wt, depth, unpadded_size=(crop_size, crop_size))
                l_dict = loss_fn(seg_out, mask)
                loss = l_dict["loss"] / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(segmenter.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += l_dict["loss"].item()
            step_count += 1

            if (step_count) % 100 == 0 or step_count == len(train_loader):
                cur_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
                log_msg(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step_count:03d}/{len(train_loader)}] "
                    f"Loss: {running_loss/step_count:.4f} (CE: {l_dict['l_ce']:.4f}, Lov: {l_dict['l_lovasz']:.4f}, Dice: {l_dict['l_dice']:.4f}) | VRAM: {cur_vram:.2f} GB",
                    log_file
                )

        scheduler.step()
        mean_loss = running_loss / max(1, step_count)
        elapsed = time.time() - t0

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(segmenter)
            log_msg(f"--> Epoch {epoch:02d} complete in {elapsed:.1f}s | Mean Loss: {mean_loss:.4f} [SWA updated] | LR: {scheduler.get_last_lr()[0]:.6f}", log_file)
        else:
            log_msg(f"--> Epoch {epoch:02d} complete in {elapsed:.1f}s | Mean Loss: {mean_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}", log_file)

        # Save latest
        latest_path = osp.join(out_dir, "segmenter_full20_latest.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": segmenter.state_dict(),
            "mean_loss": mean_loss,
            "scenes": ALL_20_SCENES
        }, latest_path)

        # Save best
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = osp.join(out_dir, "segmenter_full20_best.pt")
            torch.save({
                "epoch": epoch,
                "state_dict": segmenter.state_dict(),
                "best_loss": best_loss,
                "scenes": ALL_20_SCENES
            }, best_path)
            log_msg(f"  *** New best Full-20 Segmenter checkpoint saved to {best_path} (Loss: {best_loss:.4f}) ***", log_file)

    # Save SWA Model
    swa_path = osp.join(out_dir, "segmenter_full20_swa.pt")
    torch.save({
        "epoch": epochs,
        "state_dict": swa_model.module.state_dict(),
        "scenes": ALL_20_SCENES,
        "is_swa": True
    }, swa_path)
    log_msg(f"  *** Final SWA checkpoint saved to {swa_path} ***", log_file)

    log_msg(f"\nTraining completed in {(time.time() - t_start)/60:.2f} minutes! Best loss: {best_loss:.4f}", log_file)
    log_file.close()
    return swa_path


if __name__ == "__main__":
    train_segmenter_full20(epochs=8, swa_start_epoch=5, batch_size=8, accum_steps=1, base_lr=8e-5)
