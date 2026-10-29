"""
Experiment 3.1: High-Resolution RGB Detail Stem + Lovasz-Softmax Fine-Tuning Pilot
Trains DINOv2HighResDecoder on Screening Folds 0 & 1.

Features:
1. Initialized strictly from existing SOTA DINOV2_CP_F{0,1} checkpoints.
2. Parallel Stride-2/Stride-4 High-Res Detail Stem + Zero-Initialized Residual Projection.
3. Loss: Damped Weighted Cross-Entropy + Present-Class Lovasz-Softmax (0.25 weight).
4. Augmentation: Leak-Free Exact-(wt, depth) Matched Copy-Paste on 16 train scenes.
5. Strict Validation: Full 800-image evaluation per fold at each epoch.
6. Multi-Scale TTA evaluation on Pooled Screening Folds (0-1) [1,600 images] to test Safeguard 4.
"""

import os
import os.path as osp
import sys
import json
import time
import random
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")

from models.dinov2_highres_decoder import DINOv2HighResDecoder
from metrics.segmentation_metrics import SegmentationEvaluator
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from data.controlled_copy_paste import ControlledCopyPasteAugmenter
from tools.train_track1_screening import ControlledTrack1Dataset, apply_rsuigm_batch_gpu

def build_highres_model_from_checkpoint(ckpt_path: str, device: torch.device) -> DINOv2HighResDecoder:
    """Builds DINOv2HighResDecoder and loads base weights from an existing checkpoint."""
    model = DINOv2HighResDecoder(num_classes=5, pretrained_backbone=False).to(device)
    if osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded base weights from {ckpt_path} (new stem layers: {len(incompatible.missing_keys)})")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return model

def train_fold_pilot(
    fold_idx: int,
    epochs: int = 4,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    backbone_lr: float = 1e-5,
    base_decoder_lr: float = 5e-5,
    stem_lr: float = 2e-4,
    crop_size: int = 518,
    save_root: str = "checkpoints/track1_screening",
    seed: int = 42
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"DINOV2_HIGHRES_F{fold_idx}"
    print("\n" + "=" * 78)
    print(f"TRAINING HIGH-RES PILOT: {run_id}")
    print(f"  Fold: {fold_idx} | Epochs: {epochs} | Batch: {batch_size}x{grad_accum_steps} = {batch_size*grad_accum_steps}")
    print(f"  LRs: Backbone={backbone_lr}, BaseDecoder={base_decoder_lr}, Stem={stem_lr}")
    print("=" * 78)

    t_start = time.time()

    # Load 5-fold manifest
    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)
    fold_meta = manifest["folds"][f"fold_{fold_idx}"]
    train_scenes = fold_meta["train_scenes"]
    val_scenes = fold_meta["val_scenes"]

    # Load fold-specific 16-scene RSUIGM table
    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold_idx}.pt"
    assert osp.exists(tbl_path), f"RSUIGM table not found: {tbl_path}"
    rsuigm_table = torch.load(tbl_path, map_location=device, weights_only=False)

    # Copy-paste augmenter
    cp_augmenter = ControlledCopyPasteAugmenter(
        train_scenes=train_scenes,
        data_root="data/cached_1024",
        paste_prob=0.50,
        max_pastes=2,
        base_seed=seed
    )

    train_ds = ControlledTrack1Dataset(
        scenes=train_scenes,
        crop_size=crop_size,
        is_train=True,
        use_rsuigm=True,
        rsuigm_table=rsuigm_table,
        copy_paste_augmenter=cp_augmenter,
        base_seed=seed
    )
    val_ds = ControlledTrack1Dataset(
        scenes=val_scenes,
        crop_size=crop_size,
        is_train=False,
        use_rsuigm=True,
        rsuigm_table=rsuigm_table,
        base_seed=seed
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Initialize model from existing DINOV2_CP checkpoint
    init_ckpt = f"checkpoints/track1_screening/DINOV2_CP_F{fold_idx}/best_model.pt"
    model = build_highres_model_from_checkpoint(init_ckpt, device=device)

    # Parameter groups with differential learning rates
    param_groups = [
        {"params": model.encoder.parameters(), "lr": backbone_lr, "weight_decay": 1e-4},
        {"params": model.proj.parameters(), "lr": base_decoder_lr, "weight_decay": 1e-4},
        {"params": model.fusion.parameters(), "lr": base_decoder_lr, "weight_decay": 1e-4},
        {"params": model.stem.parameters(), "lr": stem_lr, "weight_decay": 1e-4},
        {"params": model.fuse_stage4.parameters(), "lr": stem_lr, "weight_decay": 1e-4},
        {"params": model.fuse_stage2.parameters(), "lr": stem_lr, "weight_decay": 1e-4},
        {"params": model.zero_conv.parameters(), "lr": stem_lr, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Losses
    class_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    lovasz_loss_fn = PresentClassLovaszSoftmaxLoss(num_classes=5)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    save_dir = osp.join(save_root, run_id)
    os.makedirs(save_dir, exist_ok=True)
    best_q_seg = -1.0
    best_metrics = {}

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
                loss_ce = ce_loss_fn(logits.float(), y_mask)
                loss_lov = lovasz_loss_fn(logits.float(), y_mask)
                loss = loss_ce + 0.25 * loss_lov
                loss_scaled = loss / grad_accum_steps

            loss_scaled.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
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
        print(f"  Epoch {epoch}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val mIoU: {val_res['mIoU']:.4f} | F1: {val_res['macro_F1']:.4f} | Q_seg: {val_q_seg:.4f} | "
              f"Rare IoU: [{val_res['per_class_IoU'][3]:.4f}, {val_res['per_class_IoU'][4]:.4f}]")

        if val_q_seg > best_q_seg:
            best_q_seg = val_q_seg
            best_metrics = val_res
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_res,
                "fold": fold_idx
            }, osp.join(save_dir, "best_model.pt"))
            print(f"  -> Saved new best model (Q_seg: {best_q_seg:.4f})")

    print(f"\nFold {fold_idx} training completed in {time.time() - t_start:.1f}s. Best Q_seg: {best_q_seg:.4f}")
    return {"fold": fold_idx, "best_q_seg": best_q_seg, "best_metrics": best_metrics}

def evaluate_pooled_screening_tta():
    """Evaluates Pooled Screening Folds 0 & 1 with Multi-Scale TTA against locked baseline floors."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 78)
    print("EVALUATING POOLED SCREENING FOLDS (0-1) WITH MULTI-SCALE TTA")
    print("=" * 78)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    with open("reports/track1_research/v7_locked_baseline.json") as f:
        locked = json.load(f)
    base_floors = locked["pooled_screening_folds_0_1"]["hard_floors"]

    data_root = "data/cached_1024"
    masks_dir = osp.join(data_root, "masks")
    deg_dir = osp.join(data_root, "degraded")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    scales = [448, 518, 588]
    scale_weights = [0.20, 0.60, 0.20]

    evaluator = SegmentationEvaluator(num_classes=5)
    t0 = time.time()
    total_imgs = 0

    for f_idx in [0, 1]:
        val_scenes = manifest["folds"][f"fold_{f_idx}"]["val_scenes"]
        tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{f_idx}.pt"
        tbl = torch.load(tbl_path, map_location=device, weights_only=False)

        ckpt_path = f"checkpoints/track1_screening/DINOV2_HIGHRES_F{f_idx}/best_model.pt"
        print(f"\nEvaluating Fold {f_idx} held-out scenes {val_scenes} using {ckpt_path}...")
        model = DINOv2HighResDecoder(num_classes=5, pretrained_backbone=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for sc in val_scenes:
            gt_mask = cv2.imread(osp.join(masks_dir, f"{sc}.png"), cv2.IMREAD_GRAYSCALE)
            h_orig, w_orig = gt_mask.shape
            y_t = torch.from_numpy(gt_mask.astype(np.int64)).long().unsqueeze(0).to(device)

            sc_deg = osp.join(deg_dir, sc)
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])

            for df in deg_files:
                parts = osp.splitext(df)[0].split("_")
                wt = int(parts[1])
                dp = int(parts[2])

                bgr = cv2.imread(osp.join(sc_deg, df), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                x_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([dp], device=device)

                with torch.no_grad():
                    x_rest = apply_rsuigm_batch_gpu(x_t, wt_t, dp_t, tbl)
                    x_norm = (x_rest - mean_im) / std_im

                    p_accum = torch.zeros((1, 5, 518, 518), device=device)
                    for sc_sz, sc_w in zip(scales, scale_weights):
                        x_sc = F.interpolate(x_norm, size=(sc_sz, sc_sz), mode="bilinear", align_corners=False)
                        x_sc_f = torch.flip(x_sc, dims=[3])

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            l_orig = model(x_sc)
                            l_flip = model(x_sc_f)

                        p_orig = F.softmax(l_orig.float(), dim=1)
                        p_flip = torch.flip(F.softmax(l_flip.float(), dim=1), dims=[3])
                        p_fused = 0.5 * p_orig + 0.5 * p_flip

                        p_accum += sc_w * F.interpolate(p_fused, size=(518, 518), mode="bilinear", align_corners=False)

                    p_orig_size = F.interpolate(p_accum, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                evaluator.update(p_orig_size, y_t)
                total_imgs += 1

    print(f"\nEvaluated {total_imgs} images in {time.time() - t0:.1f}s.")
    res = evaluator.compute()

    print("\n" + "=" * 78)
    print("EXPERIMENT 3.1 POOLED SCREENING RESULTS (FOLDS 0-1, 1,600 IMAGES):")
    print(f"  Locked Floor:  mIoU >= {base_floors['mIoU']:.4f} | F1 >= {base_floors['macro_F1']:.4f} | Q_seg >= {base_floors['Q_seg']:.4f}")
    print(f"  Exp 3.1 Pilot: mIoU =  {res['mIoU']:.4f} | F1 =  {res['macro_F1']:.4f} | Q_seg =  {res['Q_seg']:.4f}")
    print("-" * 78)
    print(f"  Baseline Per-Class IoU: {base_floors['per_class_IoU']}")
    print(f"  Exp 3.1  Per-Class IoU: {res['per_class_IoU']}")
    print(f"  Baseline Per-Class F1:  {base_floors['per_class_F1']}")
    print(f"  Exp 3.1  Per-Class F1:  {res['per_class_F1']}")
    print("=" * 78)

    # Check promotion gates
    passed_q_seg = res["Q_seg"] >= base_floors["Q_seg"]
    passed_miou = res["mIoU"] >= base_floors["mIoU"]
    passed_f1 = res["macro_F1"] >= base_floors["macro_F1"]
    passed_classes = all(res["per_class_IoU"][i] >= base_floors["per_class_IoU"][i] - 1e-4 for i in range(5))

    verdict = passed_q_seg and passed_miou and passed_f1 and passed_classes
    print(f"\nPROMOTION VERDICT (SAFEGUARD 4): {'PASS (Scale to Folds 2-4)' if verdict else 'FAIL (Reject / Revise)'}")

    report = {
        "experiment": "3.1_highres_detail_stem_pilot",
        "evaluation_set": "pooled_screening_folds_0_1 (1,600 images)",
        "locked_floors": base_floors,
        "pilot_metrics": res,
        "verdict": "PASS" if verdict else "FAIL",
        "passed_q_seg": bool(passed_q_seg),
        "passed_miou": bool(passed_miou),
        "passed_f1": bool(passed_f1),
        "passed_per_class": bool(passed_classes)
    }

    with open("reports/track1_research/exp3_1_screening_results.json", "w") as fp:
        json.dump(report, fp, indent=2)

    return report

def run_experiment_3_1():
    print("STARTING EXPERIMENT 3.1 PILOT PIPELINE...")
    # Train Fold 0
    train_fold_pilot(fold_idx=0, epochs=4)
    # Train Fold 1
    train_fold_pilot(fold_idx=1, epochs=4)
    # Evaluate Pooled Screening Folds with Multi-Scale TTA
    evaluate_pooled_screening_tta()

if __name__ == '__main__':
    run_experiment_3_1()
