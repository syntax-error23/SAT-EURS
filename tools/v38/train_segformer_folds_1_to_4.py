"""
SAT-EURS Track 1 — V38 Complete 5-Fold Coralscapes-V2 SegFormer-B5 Training (Folds 1..4)
Trains strictly scene-disjoint SegFormer-B5 models on Folds 1, 2, 3, and 4 using the exact
validated V29 Fold 0 recipe (4 epochs, lr=6e-5, batch_size=8, rare-class centered cropping,
table_candidate_B_fold_{k}.pt restoration).
Fold 0 already exists at checkpoints/v29/segformer_b5_fold_0_best.pt (peaked at Epoch 4).
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import os.path as osp
import sys
import time
import json
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

from tools.train_v29_segformer_b5 import EURSSegmentationDataset
from tools.train_track1_screening import apply_rsuigm_batch_gpu
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from tools.v38.run_oof_ablation_suite import compute_global_metrics
import numpy as np


def evaluate_present_oof(model, val_loader, table, device, im_mean, im_std, max_eval=200):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    count = 0
    with torch.no_grad():
        for i_deg, y_gt, wts, depths in val_loader:
            i_deg = i_deg.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            rest_rgb = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)
            x_norm = (rest_rgb - im_mean) / im_std
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=x_norm)
                logits = F.interpolate(outputs.logits.float(), size=y_gt.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)
            indices = 5 * y_gt.view(-1).cpu().numpy().astype(np.int64) + preds.view(-1).cpu().numpy().astype(np.int64)
            cm += np.bincount(indices, minlength=25).reshape(5, 5)
            count += preds.shape[0]
            if count >= max_eval:
                break
    return compute_global_metrics(cm)


def train_fold(fold: int, epochs: int = 4, batch_size: int = 8, lr: float = 6e-5, save_dir: str = "checkpoints/v38") -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print(f"[V38 SEGFORMER-B5] TRAINING FOLD {fold} ({epochs} epochs, batch_size={batch_size}, lr={lr})", flush=True)
    print("=" * 85, flush=True)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    f_data = manifest["folds"][f"fold_{fold}"]
    train_scenes = sorted(f_data["train_scenes"])
    val_scenes = sorted(f_data["val_scenes"])
    print(f"  Train Scenes ({len(train_scenes)}): {train_scenes}", flush=True)
    print(f"  Val Scenes   ({len(val_scenes)}): {val_scenes}", flush=True)

    tbl_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    table = torch.load(tbl_path, map_location=device, weights_only=False)

    train_ds = EURSSegmentationDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=42 + fold)
    val_ds = EURSSegmentationDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=42 + fold)
    val_ds.items = val_ds.items[::5]  # Stride 5 across all 4 validation scenes (160 images covering all 4 scenes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model_id = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024"
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=5,
        ignore_mismatched_sizes=True,
        local_files_only=True
    ).to(device)

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    ce_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    loss_ce = nn.CrossEntropyLoss(weight=ce_weights)
    loss_lovasz = PresentClassLovaszSoftmaxLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(save_dir, exist_ok=True)
    out_ckpt = osp.join(save_dir, f"segformer_b5_fold_{fold}_best.pt")

    best_q_seg = -1.0
    best_metrics = {}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        t_ep = time.time()

        for step, (i_deg, y_gt, wts, depths) in enumerate(train_loader):
            i_deg = i_deg.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            with torch.no_grad():
                rest_rgb = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)

            x_norm = (rest_rgb - im_mean) / im_std

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=x_norm)
                logits = F.interpolate(outputs.logits.float(), size=y_gt.shape[-2:], mode="bilinear", align_corners=False)
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

        val_metrics = evaluate_present_oof(model, val_loader, table, device, im_mean, im_std, max_eval=200)
        q_seg = val_metrics["Q_seg"]

        print(f"  [Fold {fold}] Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Loss: {train_loss:.4f} | "
              f"Val Q_seg(pres): {q_seg:.4f} | mIoU(pres): {val_metrics['mIoU']:.4f} | IoU: {val_metrics['class_ious']}", flush=True)

        if q_seg > best_q_seg:
            best_q_seg = q_seg
            best_metrics = val_metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "fold": fold,
                "train_scenes": train_scenes,
                "val_scenes": val_scenes,
                "hyperparams": {"lr": lr, "batch_size": batch_size, "epochs": epochs}
            }, out_ckpt)
            print(f"  -> Saved new best Fold {fold} SegFormer-B5 checkpoint (Q_seg={best_q_seg:.4f}) to {out_ckpt}", flush=True)

    print(f"[Fold {fold}] Complete in {time.time() - t0:.1f}s | Best Val Q_seg: {best_q_seg:.4f}", flush=True)
    del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
    torch.cuda.empty_cache()
    return {"fold": fold, "best_metrics": best_metrics, "checkpoint": out_ckpt}


if __name__ == "__main__":
    t_all = time.time()
    results = {}
    for f in [1, 2, 3, 4]:
        results[f] = train_fold(fold=f, epochs=4, batch_size=8, lr=6e-5, save_dir="checkpoints/v38")
    os.makedirs("reports/v38", exist_ok=True)
    with open("reports/v38/segformer_folds_1_to_4_train_report.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nALL 4 REMAINING SEGFORMER-B5 FOLDS TRAINED IN {time.time() - t_all:.1f}s!", flush=True)
