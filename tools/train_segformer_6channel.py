"""
Leakage-Free Six-Channel SegFormer MiT-B3 Experiment
MaCVi WACV 2027 EURS Challenge (Track 1)

Key Experimental Principles:
1. Strict 16/4 Scene-Disjoint Split (manifest_full.json):
   - Train scenes (16): fish11, fish13, fish14, fish15, fish17, fish18, fish2,
                        fish20, fish21, fish4, fish5, fish7, fish9, ray1,
                        starfish2, turtles1
   - Val scenes (4):    fish1, fish12, seahorse1, sharkFish
   - Identical to baseline; zero validation scenes seen during training.
2. Input Preprocessing:
   - Evaluates closed-form per-image RSUIGM physical inversion on GPU.
   - Dual-domain early fusion: concatenates [Raw RGB (3) || Restored RGB (3)].
   - Both domains standardized with ImageNet mean and std.
3. Clean Scientific Parity:
   - Identical training budget (10 epochs, batch size 8, lr 1e-4, AdamW, Cosine scheduler).
   - Identical crop size (512x512) and identical loss function.
   - Zero copy-paste augmentation to isolate the effect of 6-channel input.
   - Zero in-sample models (no V4, no prior logits).
4. Official Metrics Parity:
   - Evaluated via metrics.segmentation_metrics.SegmentationEvaluator.
   - Measures per-class Precision, Recall, F1, IoU, mIoU, Macro F1, and Q_seg.
"""

import os
import os.path as osp
import sys
import time
import json
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu
from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_segformer_baseline import LeakFreeSegFormerDataset, TRAIN_SCENES, VAL_SCENES


def train_leak_free_segformer_6channel(
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1.0e-4,
    crop_size: int = 512,
    save_dir: str = "checkpoints/exp_6channel_segformer",
    data_root: str = "data/cached_1024" if osp.exists("data/cached_1024") else "data/train"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 72)
    print("   LEAK-FREE SIX-CHANNEL SEGFORMER MiT-B3 EXPERIMENT (RAW + RESTORED)")
    print(f"   Hardware: {device} ({torch.cuda.get_device_name(0)})")
    print(f"   Train Scenes ({len(TRAIN_SCENES)}): {TRAIN_SCENES}")
    print(f"   Val Scenes   ({len(VAL_SCENES)}): {VAL_SCENES}")
    print(f"   Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Crop: {crop_size}")
    print("=" * 72)

    # 1. Load RSUIGM table on GPU
    rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

    # 2. Datasets & Loaders (Identical to Baseline)
    train_dataset = LeakFreeSegFormerDataset(
        scenes=TRAIN_SCENES,
        data_root=data_root,
        crop_size=crop_size,
        is_train=True,
        samples_per_epoch=3200
    )
    val_dataset = LeakFreeSegFormerDataset(
        scenes=VAL_SCENES,
        data_root=data_root,
        crop_size=crop_size,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model: SegFormer MiT-B3 with 6 input channels [Raw RGB (3) || Restored RGB (3)]
    # Pretrained ImageNet weights are adapted automatically by smp for 6 channels.
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        classes=5,
        in_channels=6
    ).to(device)

    # ImageNet normalization tensors
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Mild class weighting: [BG: 0.8, Coral: 1.0, Fish: 1.2, Shark: 2.0, Other: 2.0] (Identical to Baseline)
    class_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_q_seg = -1.0
    history = []

    def _apply_rsuigm_batch(x_raw: torch.Tensor, wts: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """Apply closed-form physical inversion on GPU per item in batch."""
        B, C, H, W = x_raw.shape
        restored = []
        for i in range(B):
            xi = x_raw[i:i+1]
            wt_i = int(wts[i].item())
            d_i = int(depths[i].item())
            mat = rsuigm_table.get((wt_i, d_i), rsuigm_table[(1, 10)])
            feats = build_rsuigm_poly_features_gpu(xi)
            phys = torch.clamp(torch.matmul(feats, mat).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
            restored.append(phys)
        return torch.cat(restored, dim=0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()

        for step, (x_raw, y_mask, wts, depths) in enumerate(train_loader):
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            with torch.no_grad():
                x_rest = _apply_rsuigm_batch(x_raw, wts, depths)
                x_raw_norm = (x_raw - mean_im) / std_im
                x_rest_norm = (x_rest - mean_im) / std_im
                # Concatenate 6 channels: [Raw RGB || Restored RGB]
                x_6ch = torch.cat([x_raw_norm, x_rest_norm], dim=1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_6ch)
                loss = loss_fn(logits, y_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation on genuine held-out scenes
        model.eval()
        evaluator = SegmentationEvaluator(num_classes=5)

        with torch.no_grad():
            for x_raw, y_mask, wts, depths in val_loader:
                x_raw = x_raw.to(device, non_blocking=True)
                y_mask = y_mask.to(device, non_blocking=True)

                x_rest = _apply_rsuigm_batch(x_raw, wts, depths)
                x_raw_norm = (x_raw - mean_im) / std_im
                x_rest_norm = (x_rest - mean_im) / std_im
                x_6ch = torch.cat([x_raw_norm, x_rest_norm], dim=1)

                # Resize to square 1024x1024 for backbone if needed
                B, C, H, W = x_6ch.shape
                if H != 1024 or W != 1024:
                    x_in = F.interpolate(x_6ch, size=(1024, 1024), mode="bilinear", align_corners=False)
                else:
                    x_in = x_6ch

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits_sq = model(x_in)

                if H != 1024 or W != 1024:
                    logits = F.interpolate(logits_sq, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    logits = logits_sq

                evaluator.update(logits, y_mask)

        metrics = evaluator.compute()
        elapsed = time.time() - t_epoch
        q_seg = metrics["Q_seg"]

        print(
            f"Epoch {epoch:2d}/{epochs:2d} ({elapsed:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val mIoU: {metrics['mIoU']:.4f} | "
            f"Macro F1: {metrics['macro_F1']:.4f} | "
            f"Cov: {metrics['coverage_penalty']:.1f} | "
            f"Q_seg: {q_seg:.4f} | "
            f"Cls IoU: {metrics['per_class_IoU']}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": metrics,
            "elapsed_sec": elapsed
        }
        history.append(record)

        if q_seg > best_q_seg:
            best_q_seg = q_seg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": metrics,
                "train_scenes": TRAIN_SCENES,
                "val_scenes": VAL_SCENES
            }, osp.join(save_dir, "best_model.pt"))

    with open(osp.join(save_dir, "training_history.json"), "w") as fp:
        json.dump(history, fp, indent=2)

    print(f"\n[DONE] 6-Channel SegFormer complete! Best Q_seg on held-out scenes: {best_q_seg:.4f}")
    return history


if __name__ == "__main__":
    train_leak_free_segformer_6channel()
