"""
SAT-EURS Track 1 — V29 Full-20 SegFormer-B5 Training Engine
Experiment: EXP-02-Full20 (Full-20 Scene Coralscapes SegFormer-B5 Fine-Tuning)
Target: Replicating Rank 5 Winner's Strategy (ENEA EURS S59 Full20)

Properties:
- All 20 scenes (4,000 degraded images)
- Backbone: SegFormer-B5 initialized from CoralscapesV2
- 5-Class EURS head
- Input: Physical-Inversion Restored RGB via 20-Scene Calibrated RSUIGM table
- Loss: Class-Weighted CE ([0.8, 1.0, 1.2, 2.0, 2.0]) + PresentClassLovaszSoftmaxLoss
- Saves to: checkpoints/v29/segformer_b5_full20_best.pt
"""

import os
import os.path as osp
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from tools.train_track1_screening import apply_rsuigm_batch_gpu
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_v29_segformer_b5 import EURSSegmentationDataset, evaluate_segformer_model


ALL_20_SCENES = [
    "fish1", "fish11", "fish12", "fish13", "fish14", "fish15",
    "fish17", "fish18", "fish2", "fish20", "fish21", "fish4",
    "fish5", "fish7", "fish9", "ray1", "seahorse1", "sharkFish",
    "starfish2", "turtles1"
]


def train_segformer_full20(
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 6e-5,
    save_dir: str = "checkpoints/v29"
) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("TRAINING V29 CORALSCAPES SEGFORMER-B5 (FULL-20 SCENES)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"  All 20 Scenes: {ALL_20_SCENES}")
    print("=" * 80)

    # 1. 20-Scene Table
    tbl_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    if not osp.exists(tbl_path):
        # Fallback to tbl19_16sc_fold0 if 20-scene not present
        tbl_path = "checkpoints/track1_screening/tbl19_16sc_fold0.pt"
    print(f"  Loading RSUIGM table from {tbl_path}...")
    table = torch.load(tbl_path, map_location=device, weights_only=False)

    # 2. Datasets: All 20 scenes for training (4,000 samples)
    # For validation monitor, use representative 4-scene monitor (Fold 0 scenes: fish1, fish11, fish12, seahorse1)
    train_ds = EURSSegmentationDataset(scenes=ALL_20_SCENES, crop_size=512, is_train=True, base_seed=1337)
    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    val_ds = EURSSegmentationDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=1337)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model: Load Coralscapes Pretrained SegFormer-B5
    model_id = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024"
    print(f"  Loading SegFormer-B5 from local cache: {model_id}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(device)

    # Normalization constants
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 4. Optimization & Losses
    ce_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    loss_ce = nn.CrossEntropyLoss(weight=ce_weights)
    loss_lovasz = PresentClassLovaszSoftmaxLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(save_dir, exist_ok=True)
    out_ckpt = osp.join(save_dir, "segformer_b5_full20_best.pt")

    t0 = time.time()
    best_loss = 999.0

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        t_ep = time.time()
        model.train()
        total_loss = 0.0

        for x_deg, y_gt, wts, depths in train_loader:
            x_deg = x_deg.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            with torch.no_grad():
                rest_rgb = apply_rsuigm_batch_gpu(x_deg, wts, depths, table)
                x_norm = (rest_rgb - im_mean) / im_std

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=x_norm)
                logits = outputs.logits
                logits = F.interpolate(logits, size=y_gt.shape[-2:], mode="bilinear", align_corners=False)

                l_c = loss_ce(logits, y_gt)
                l_v = loss_lovasz(logits, y_gt)
                loss = l_c + 0.50 * l_v

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        val_metrics = evaluate_segformer_model(model, val_loader, table, device, im_mean, im_std, max_eval=160)
        q_seg = val_metrics["Q_seg"]

        print(f"  Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val mIoU: {val_metrics['mIoU']} | F1: {val_metrics['macro_F1']} | Q_seg: {q_seg:.4f} | "
              f"IoU: {val_metrics['per_class_IoU']}", flush=True)

        # In Full-20 training, save when train loss reaches new low
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "val_metrics": val_metrics,
                "scenes": ALL_20_SCENES,
                "hyperparams": {"lr": lr, "batch_size": batch_size}
            }, out_ckpt)
            print(f"  -> Saved new best SegFormer-B5 Full-20 checkpoint (Train Loss: {train_loss:.4f}) to {out_ckpt}", flush=True)

    print("\n" + "=" * 80)
    print(f"Full-20 SegFormer-B5 Training Complete in {time.time() - t0:.1f}s!")
    print(f"  Best Train Loss: {best_loss:.4f}")
    print(f"  Checkpoint: {out_ckpt}")
    print("=" * 80)

    return {
        "best_loss": best_loss,
        "checkpoint": out_ckpt
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    train_segformer_full20(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
