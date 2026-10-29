"""
SAT-EURS Track 1 — V26 Full-20-Scene Segmenter Training Pipeline
Trains EURSAPEXV18Segmenter on all 20 EURS clean scenes and their 200 degradation variants
(4,000 images total) using the RTX 5090.
Key features:
- Balanced scene sampling to prevent rare-class dilution (Class 3 shark, Class 4 macro-fauna)
- Warm-starts from proven V18 dual-encoder weights
- Strict single-image protocol compliance
- Cosine LR schedule with differential backbone learning rates
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

sys.path.insert(0, ".")

from data.v18.dataset import V18EURSInvarianceDataset, V18SceneBalancedSampler
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter, EURSAPEXV18Segmenter
from losses.v18.segmentation_losses import SegmentationV18Loss, compute_capped_class_weights


ALL_20_SCENES = [
    'fish1', 'fish11', 'fish12', 'fish13', 'fish14', 'fish15', 'fish17', 'fish18',
    'fish2', 'fish20', 'fish21', 'fish4', 'fish5', 'fish7', 'fish9',
    'ray1', 'seahorse1', 'sharkFish', 'starfish2', 'turtles1'
]


def train_segmenter_full20(
    epochs: int = 10,
    batch_size: int = 8,
    accum_steps: int = 1,
    base_lr: float = 1e-4,
    crop_size: int = 512,
    warmstart_ckpt: str = "checkpoints/v18/fold_0/segmenter_best.pt",
    refiner_ckpt: str = "checkpoints/v18/fold_0/refiner_best.pt",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    out_dir: str = "checkpoints/v26",
    device_id: int = 0
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print("SAT-EURS V26 — FULL-20-SCENE SEGMENTER TRAINING PIPELINE", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"All 20 Scenes: {len(ALL_20_SCENES)} scenes (4,000 images total)", flush=True)
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, accum_steps={accum_steps}, base_lr={base_lr}", flush=True)
    print("=" * 85, flush=True)

    # 1. Setup Frozen Restorer for input features
    print(f"\n[1/5] Loading Restoration Refiner from {refiner_ckpt}...", flush=True)
    rsuigm_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(refiner_ckpt, map_location="cpu")
    ref_weights = ref_ckpt["state_dict"] if "state_dict" in ref_ckpt else ref_ckpt
    refiner.load_state_dict(ref_weights)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False

    # 2. Build EURSAPEXV18Segmenter & Warm Start
    print(f"\n[2/5] Building EURSAPEXV18Segmenter and warm-starting from {warmstart_ckpt}...", flush=True)
    segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, neck_dim=256).to(device)

    if osp.exists(warmstart_ckpt):
        ws_obj = torch.load(warmstart_ckpt, map_location="cpu")
        ws_weights = ws_obj["state_dict"] if "state_dict" in ws_obj else ws_obj
        segmenter.load_state_dict(ws_weights, strict=True)
        print(f"  Successfully warm-started model weights from {warmstart_ckpt}!", flush=True)
    else:
        print(f"  WARNING: Warm-start checkpoint {warmstart_ckpt} not found! Initializing from scratch.", flush=True)

    # 3. Setup Dataset and Scene-Balanced Sampler across all 20 scenes
    print(f"\n[3/5] Initializing Full-20-Scene Dataset (4,000 images)...", flush=True)
    train_ds = V18EURSInvarianceDataset(scenes=ALL_20_SCENES, crop_size=crop_size, is_train=True, base_seed=42)
    sampler = V18SceneBalancedSampler(train_ds, batch_size=batch_size, seed=42)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    print(f"  Dataset ready: {len(train_ds)} samples -> {len(train_loader)} batches/epoch.", flush=True)

    # 4. Setup Loss & Optimizer
    print(f"\n[4/5] Initializing Loss & Optimizer...", flush=True)
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

    # 5. Training Loop
    print(f"\n[5/5] Beginning Training ({epochs} epochs)...", flush=True)
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
                print(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step_count:03d}/{len(train_loader)}] "
                    f"Loss: {running_loss/step_count:.4f} (CE: {l_dict['ce']:.4f}, Lov: {l_dict['lovasz']:.4f}, Dice: {l_dict['dice']:.4f}) | VRAM: {cur_vram:.2f} GB",
                    flush=True
                )

        scheduler.step()
        mean_loss = running_loss / max(1, step_count)
        elapsed = time.time() - t0
        print(f"--> Epoch {epoch:02d} complete in {elapsed:.1f}s | Mean Loss: {mean_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

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
            print(f"  *** New best Full-20 Segmenter checkpoint saved to {best_path} (Loss: {best_loss:.4f}) ***", flush=True)

    print(f"\nTraining completed in {(time.time() - t_start)/60:.2f} minutes! Best loss: {best_loss:.4f}", flush=True)
    return best_path


if __name__ == "__main__":
    train_segmenter_full20(epochs=10, batch_size=8, accum_steps=1, base_lr=1e-4)
