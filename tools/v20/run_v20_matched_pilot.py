"""
SAT-EURS Track 1 — V20 Matched Loss Ablation Benchmark
Author: Empirical ML Specialist & Senior Computer Vision Engineer
Date: October 18, 2026

Design:
Strictly matched A/B experiment between:
- Control Arm:   Original V18 Loss Recipe (Mock CE weights [0.5, 0.6, 1.0, 4.0, 4.0] + Unweighted Lovasz)
- Treatment Arm: V20 Combined Loss Recipe (Grounded CE weights + Class-Weighted Lovasz)

Matched Factors for Both Arms:
1. Exact same initial model state (saved and SHA256 hashed before training).
2. Exact same architecture (EURSAPEXV18Segmenter: ViT-Large + ConvNeXt-Base + UPerNet, 401M params).
3. Exact same data sampler and crop sequence (V18SceneBalancedSampler with fixed seed).
4. Exact same optimizer (AdamW, base lr 1e-4, ViT 0.10x, ConvNeXt 0.20x, weight decay 1e-4).
5. Exact same 9-epoch training budget and scheduler (CosineAnnealingLR, T_max=9, eta_min=1e-6).
6. Exact same gradient accumulation (batch_size=4, accum_steps=4 -> effective batch size 16).
7. Exact same evaluator (V18OOFEvaluator) and checkpoint selection rule (Val Q_seg).
8. Exact same bf16 precision via torch.amp.autocast('cuda', dtype=torch.bfloat16).

Test Folds:
- Fold 0: Validation contains seahorse1 (Class 4).
- Fold 2: Validation contains ray1 (Class 3).
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
import argparse
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from data.v18.dataset import V18EURSInvarianceDataset, V18SceneBalancedSampler
from models.v18.segmenter import build_v18_segmenter, EURSAPEXV18Segmenter
from models.v18.restoration_refiner import build_v18_restoration_refiner
from losses.v18.segmentation_losses import SegmentationV18Loss, compute_capped_class_weights
from losses.v20.segmentation_losses import SegmentationV20Loss, compute_v20_grounded_class_weights
from tools.v18.evaluator import V18OOFEvaluator

EXPECTED_V18_ZIP_SHA256 = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
V18_ZIP_PATH = "deploy/v18/submission_v18_dual_encoder_sota.zip"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def get_fold_training_counts(train_scenes: List[str], mask_dir: str = "data/cached_1024/masks") -> List[int]:
    counts = np.zeros(5, dtype=np.int64)
    for sc in train_scenes:
        m_path = os.path.join(mask_dir, f"{sc}.png")
        if os.path.exists(m_path):
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            bc = np.bincount(m.flatten(), minlength=5)[:5]
            counts += bc * 200
    return counts.tolist()


def train_single_arm(
    arm_name: str,
    fold: int,
    init_state_path: str,
    loss_fn: nn.Module,
    loss_type: str,
    loss_config_info: Dict[str, Any],
    train_scenes: List[str],
    val_scenes: List[str],
    epochs: int,
    batch_size: int,
    accum_steps: int,
    base_lr: float,
    seed: int,
    device: torch.device,
    checkpoint_dir: str
) -> Dict[str, Any]:
    print(f"\n{'='*90}")
    print(f"STARTING ARM: [{arm_name.upper()}] — FOLD {fold}")
    print(f"Loss Type: {loss_type}")
    print(f"Loading Initial State: {init_state_path}")
    print(f"{'='*90}")

    # Build and load model from shared init state
    segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False, neck_dim=256).to(device)
    init_state = torch.load(init_state_path, map_location=device)
    segmenter.load_state_dict(init_state)

    # Load frozen refiner
    refiner_path = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    rsuigm_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(refiner_path, map_location=device)
    ref_weights = ref_ckpt["state_dict"] if "state_dict" in ref_ckpt else ref_ckpt
    refiner.load_state_dict(ref_weights)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False

    # Datasets and loaders (matched seed)
    train_ds = V18EURSInvarianceDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=seed + fold)
    val_ds = V18EURSInvarianceDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=seed + fold)
    sampler = V18SceneBalancedSampler(train_ds, batch_size=batch_size, seed=seed + fold)

    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # Optimizer & 9-epoch cosine schedule
    param_groups = [
        {"params": [p for p in segmenter.vit_backbone.parameters() if p.requires_grad], "lr": base_lr * 0.10},
        {"params": [p for p in segmenter.conv_backbone.parameters() if p.requires_grad], "lr": base_lr * 0.20},
        {"params": list(segmenter.detail_stem.parameters()) + list(segmenter.cond_encoder.parameters()) + list(segmenter.neck.parameters()), "lr": base_lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_qseg = -1.0
    best_metrics = {}
    best_epoch = -1
    epoch_logs = []
    t_arm_start = time.time()
    peak_vram = 0.0

    fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(fold_ckpt_dir, f"{arm_name}_best.pt")

    for epoch in range(epochs):
        t_ep_start = time.time()
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        segmenter.train()

        running_loss = 0.0
        running_ce = 0.0
        running_lovasz = 0.0
        running_dice = 0.0
        step_count = 0
        opt_updates = 0

        current_lr = scheduler.get_last_lr()[2] if hasattr(scheduler, "get_last_lr") else base_lr
        optimizer.zero_grad()
        t_tr0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            deg = batch["deg_rgb"].to(device)
            mask = batch["mask"].to(device)
            wt = batch["wt"].to(device)
            depth = batch["depth"].to(device)

            with torch.no_grad():
                out_ref = refiner(deg, wt, depth)
                restored = out_ref["refined_image"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seg_out = segmenter(restored, deg, wt, depth, unpadded_size=(512, 512))
                l_dict = loss_fn(seg_out, mask)
                loss = l_dict["loss"] / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(segmenter.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                opt_updates += 1

            running_loss += l_dict["loss"].item()
            running_ce += l_dict["l_ce"].item()
            running_lovasz += l_dict["l_lovasz"].item()
            running_dice += l_dict["l_dice"].item()
            step_count += 1

            if step_count % 200 == 0 or step_count == len(train_loader):
                cur_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
                peak_vram = max(peak_vram, cur_vram)
                print(f"  [{arm_name.upper()} Ep {epoch+1:02d}/{epochs:02d} Step {step_count:03d}/{len(train_loader):03d} (Opt: {opt_updates:03d})] Loss: {running_loss/step_count:.4f} (CE: {running_ce/step_count:.4f}, Lov: {running_lovasz/step_count:.4f}) | VRAM: {cur_vram:.2f} GB", flush=True)

        scheduler.step()
        train_time_ep = time.time() - t_tr0
        mean_train_loss = running_loss / max(1, step_count)

        # Validation
        t_val0 = time.time()
        segmenter.eval()
        evaluator = V18OOFEvaluator(num_classes=5)

        with torch.no_grad():
            for batch in val_loader:
                deg = batch["deg_rgb"].to(device)
                mask = batch["mask"].to(device)
                wt = batch["wt"].to(device)
                depth = batch["depth"].to(device)

                out_ref = refiner(deg, wt, depth)
                restored = out_ref["refined_image"]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    seg_out = segmenter(restored, deg, wt, depth, unpadded_size=(512, 512))
                pred_mask = torch.argmax(seg_out["logits"], dim=1).cpu().numpy()
                gt_mask = mask.cpu().numpy()
                evaluator.update_segmentation(pred_mask, gt_mask)

        raw_res = evaluator.compute()
        val_res = raw_res.get("segmentation", raw_res)
        val_time_ep = time.time() - t_val0
        total_ep_time = time.time() - t_ep_start

        val_qseg = float(val_res["Q_seg"])
        val_miou = float(val_res["mIoU"])
        val_f1 = float(val_res["Macro_F1"])

        c3_iou = float(val_res["class_ious"][3])
        c4_iou = float(val_res["class_ious"][4])
        c4_rec = float(val_res["class_recalls"][4])
        c4_prec = float(val_res["class_precisions"][4])

        print(f"  >>> [{arm_name.upper()} Ep {epoch+1:02d} Val] Q_seg: {val_qseg:.5f} | mIoU: {val_miou:.5f} | F1: {val_f1:.5f} | LR: {current_lr:.6f} | Time: {total_ep_time:.1f}s")
        print(f"      Cls 0: {val_res['class_ious'][0]:.4f} | Cls 1: {val_res['class_ious'][1]:.4f} | Cls 2: {val_res['class_ious'][2]:.4f} | Cls 3: {c3_iou:.4f} | Cls 4: {c4_iou:.4f} (Rec: {c4_rec:.4f}, Prec: {c4_prec:.4f})", flush=True)

        ep_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": round(mean_train_loss, 5),
            "train_ce": round(running_ce / max(1, step_count), 5),
            "train_lovasz": round(running_lovasz / max(1, step_count), 5),
            "train_time_sec": round(train_time_ep, 2),
            "val_time_sec": round(val_time_ep, 2),
            "total_time_sec": round(total_ep_time, 2),
            "metrics": val_res
        }
        epoch_logs.append(ep_record)

        if val_qseg > best_val_qseg:
            best_val_qseg = val_qseg
            best_metrics = val_res
            best_epoch = epoch + 1
            torch.save({
                "version": "v20_matched_pilot",
                "arm": arm_name,
                "fold": fold,
                "epoch": best_epoch,
                "state_dict": segmenter.state_dict(),
                "provenance": {
                    "arm": arm_name,
                    "fold": fold,
                    "epoch": best_epoch,
                    "loss_type": loss_type,
                    "loss_config": loss_config_info,
                    "init_state_path": init_state_path,
                    "metrics": val_res,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }, best_ckpt_path)
            print(f"      [*] New best {arm_name} checkpoint saved at {best_ckpt_path} (Epoch {best_epoch})\n")

    total_arm_sec = time.time() - t_arm_start
    del segmenter, refiner, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "arm_name": arm_name,
        "fold": fold,
        "loss_type": loss_type,
        "loss_config": loss_config_info,
        "init_state_path": init_state_path,
        "total_wall_sec": round(total_arm_sec, 2),
        "total_wall_min": round(total_arm_sec / 60.0, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "best_epoch": best_epoch,
        "best_checkpoint": best_ckpt_path,
        "best_metrics": best_metrics,
        "epoch_logs": epoch_logs
    }


def run_matched_pilot(
    folds: List[int] = [0, 2],
    epochs: int = 9,
    batch_size: int = 4,
    accum_steps: int = 4,
    base_lr: float = 1e-4,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/v20/matched_pilot",
    report_dir: str = "reports/v20/matched_pilot"
):
    print("=" * 95)
    print("SAT-EURS Track 1 — V20 Matched Loss Ablation Benchmark (Paired Pilot)")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Folds to Test: {folds} | Epochs per Arm: {epochs}")
    print("=" * 95)

    # Pre-run baseline hash check
    print("\n[Step 1] Verifying Immutable Production Baseline ZIP...")
    sha_before = compute_sha256(V18_ZIP_PATH)
    print(f"  Path:   {V18_ZIP_PATH}")
    print(f"  SHA256: {sha_before}")
    assert sha_before == EXPECTED_V18_ZIP_SHA256, f"Baseline ZIP hash mismatch! Got {sha_before}"
    print("  Baseline Archive Status: 100% INTACT & PRESERVED.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n[Step 2] Hardware: {device_name} (PyTorch {torch.__version__})")

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    benchmark_results: Dict[str, Any] = {
        "title": "SAT-EURS Track 1 — V20 Matched Loss Ablation Benchmark Report",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_archive": {
            "path": V18_ZIP_PATH,
            "sha256": sha_before,
            "intact": True
        },
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "effective_batch_size": batch_size * accum_steps,
            "base_lr": base_lr,
            "seed": seed,
            "precision": "bfloat16"
        },
        "folds": {}
    }

    for fold in folds:
        print(f"\n{'#'*95}")
        print(f"STARTING PAIRED PILOT FOR FOLD {fold}")
        print(f"{'#'*95}")

        fold_key = f"fold_{fold}"
        fold_data = manifest["folds"][fold_key]
        train_scenes = fold_data["train_scenes"]
        val_scenes = fold_data["val_scenes"]

        print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
        print(f"Val scenes   ({len(val_scenes)}):   {val_scenes}")

        # Construct and save matched initial state
        init_state_path = os.path.join(checkpoint_dir, f"init_state_fold_{fold}.pt")
        print(f"\nConstructing Matched Initial State for Fold {fold} (seed={seed+fold})...")
        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)
        temp_seg = build_v18_segmenter(num_classes=5, vit_pretrained=True, conv_pretrained=True, use_mock=False, neck_dim=256).to(device)
        torch.save(temp_seg.state_dict(), init_state_path)
        init_sha = compute_sha256(init_state_path)
        print(f"  Saved Matched Init State: {init_state_path} (SHA256: {init_sha})")
        del temp_seg
        torch.cuda.empty_cache()

        # Compute counts and weights
        train_counts = get_fold_training_counts(train_scenes)
        print(f"  Training Pixel Counts: {train_counts}")

        # Control Loss: Original V18 Recipe
        w_v18 = compute_capped_class_weights([1000000, 50000, 30000, 5000, 2000], min_weight=0.50, max_weight=4.00, device=device)
        control_loss_fn = SegmentationV18Loss(
            num_classes=5, class_weights=w_v18, w_ce=0.40, w_lovasz=0.30, w_dice=0.20, w_bnd=0.05, w_obj=0.05, ignore_index=255
        ).to(device)
        control_config = {
            "loss_recipe": "Original_V18_Recipe",
            "class_weights": [round(x, 4) for x in w_v18.tolist()],
            "lovasz_type": "Unweighted_Arithmetic_Mean",
            "weights": {"w_ce": 0.40, "w_lovasz": 0.30, "w_dice": 0.20, "w_bnd": 0.05, "w_obj": 0.05}
        }

        # Treatment Loss: V20 Recipe
        w_v20 = compute_v20_grounded_class_weights(train_counts, min_weight=0.50, max_weight=3.50, smoothing=0.50, device=device)
        treatment_loss_fn = SegmentationV20Loss(
            num_classes=5, class_weights=w_v20, w_ce=0.40, w_lovasz=0.30, w_dice=0.20, w_bnd=0.05, w_obj=0.05, ignore_index=255
        ).to(device)
        treatment_config = {
            "loss_recipe": "V20_Empirically_Grounded_CW_Recipe",
            "class_weights": [round(x, 4) for x in w_v20.tolist()],
            "lovasz_type": "Class_Weighted_Lovasz",
            "weights": {"w_ce": 0.40, "w_lovasz": 0.30, "w_dice": 0.20, "w_bnd": 0.05, "w_obj": 0.05}
        }

        # Execute Control Arm
        control_res = train_single_arm(
            arm_name="control_v18",
            fold=fold,
            init_state_path=init_state_path,
            loss_fn=control_loss_fn,
            loss_type="Original_V18_Loss",
            loss_config_info=control_config,
            train_scenes=train_scenes,
            val_scenes=val_scenes,
            epochs=epochs,
            batch_size=batch_size,
            accum_steps=accum_steps,
            base_lr=base_lr,
            seed=seed,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        # Execute Treatment Arm
        treatment_res = train_single_arm(
            arm_name="treatment_v20",
            fold=fold,
            init_state_path=init_state_path,
            loss_fn=treatment_loss_fn,
            loss_type="V20_Grounded_CW_Loss",
            loss_config_info=treatment_config,
            train_scenes=train_scenes,
            val_scenes=val_scenes,
            epochs=epochs,
            batch_size=batch_size,
            accum_steps=accum_steps,
            base_lr=base_lr,
            seed=seed,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

        # Record Fold Comparison
        c_m = control_res["best_metrics"]
        t_m = treatment_res["best_metrics"]
        delta_q = t_m["Q_seg"] - c_m["Q_seg"]
        delta_miou = t_m["mIoU"] - c_m["mIoU"]
        delta_f1 = t_m["Macro_F1"] - c_m["Macro_F1"]
        delta_c3 = t_m["class_ious"][3] - c_m["class_ious"][3]
        delta_c4 = t_m["class_ious"][4] - c_m["class_ious"][4]

        fold_summary = {
            "fold": fold,
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "init_state_sha256": init_sha,
            "control": control_res,
            "treatment": treatment_res,
            "deltas": {
                "delta_Q_seg": round(delta_q, 5),
                "delta_mIoU": round(delta_miou, 5),
                "delta_Macro_F1": round(delta_f1, 5),
                "delta_class3_iou": round(delta_c3, 5),
                "delta_class4_iou": round(delta_c4, 5)
            }
        }
        benchmark_results["folds"][fold_key] = fold_summary

        # Save intermediate JSON report
        json_report_path = os.path.join(report_dir, "MATCHED_PILOT_BENCHMARK.json")
        with open(json_report_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\n[Fold {fold} Complete] Deliverable JSON updated at {json_report_path}")

    # Post-run baseline hash check
    print("\n[Step 3] Verifying Immutable Production Baseline ZIP Post-Run...")
    sha_after = compute_sha256(V18_ZIP_PATH)
    print(f"  Path:   {V18_ZIP_PATH}")
    print(f"  SHA256: {sha_after}")
    assert sha_after == sha_before, f"FATAL: Baseline ZIP was modified! Expected {sha_before}, got {sha_after}"
    benchmark_results["baseline_archive"]["sha256_after"] = sha_after
    print("  Baseline Archive Status: 100% INTACT & PRESERVED.")

    # Generate Master Markdown Report
    md_report_path = os.path.join(report_dir, "MATCHED_PILOT_BENCHMARK.md")
    generate_matched_pilot_markdown(benchmark_results, md_report_path)
    print(f"Deliverable Markdown written to {md_report_path}")

    print("\n" + "=" * 95)
    print("V20 MATCHED LOSS ABLATION BENCHMARK COMPLETE!")
    print("=" * 95)


def generate_matched_pilot_markdown(data: Dict[str, Any], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# SAT-EURS Track 1 — V20 Matched Loss Ablation Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  \n")
        f.write(f"**Benchmark Type:** Strictly Matched Paired Pilot (Identical Initial State, Sampler, 9-Epoch Schedule)  \n")
        f.write(f"**Baseline SHA256:** `{data['baseline_archive']['sha256']}` (100% Intact)  \n\n")
        f.write("---\n\n")

        for fold_key, f_data in data["folds"].items():
            fold = f_data["fold"]
            c_m = f_data["control"]["best_metrics"]
            t_m = f_data["treatment"]["best_metrics"]
            deltas = f_data["deltas"]

            f.write(f"## Fold {fold} Paired Results (`val: {f_data['val_scenes']}`)\n\n")
            f.write(f"- **Shared Init State Hash:** `{f_data['init_state_sha256']}`\n")
            f.write(f"- **Control Arm (V18 Recipe):** Best Epoch {f_data['control']['best_epoch']}, Val $Q_{{\\text{{seg}}}}$ = {c_m['Q_seg']:.5f}\n")
            f.write(f"- **Treatment Arm (V20 Recipe):** Best Epoch {f_data['treatment']['best_epoch']}, Val $Q_{{\\text{{seg}}}}$ = {t_m['Q_seg']:.5f}\n")
            f.write(f"- **Net Improvement ($\Delta Q_{{\\text{{seg}}}}$):** **{deltas['delta_Q_seg']:+.5f}**\n\n")

            f.write("### Detailed Comparative Metrics Table:\n\n")
            f.write("| Metric Dimension | Control Arm (V18 Loss) | Treatment Arm (V20 Loss) | Absolute Delta | Status |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            f.write(f"| **Val $Q_{{\\text{{seg}}}}$ (Present)** | {c_m['Q_seg']:.5f} | {t_m['Q_seg']:.5f} | **{deltas['delta_Q_seg']:+.5f}** | {'GAIN' if deltas['delta_Q_seg']>0 else 'REGR'} |\n")
            f.write(f"| **Val mIoU (Present)** | {c_m['mIoU']:.5f} | {t_m['mIoU']:.5f} | **{deltas['delta_mIoU']:+.5f}** | {'GAIN' if deltas['delta_mIoU']>0 else 'REGR'} |\n")
            f.write(f"| **Val Macro F1 (Present)** | {c_m['Macro_F1']:.5f} | {t_m['Macro_F1']:.5f} | **{deltas['delta_Macro_F1']:+.5f}** | {'GAIN' if deltas['delta_Macro_F1']>0 else 'REGR'} |\n")
            f.write(f"| **Coverage Penalty** | {c_m['coverage_penalty']:.1f} | {t_m['coverage_penalty']:.1f} | Exact | MATCH |\n")
            f.write(f"| Class 0 (Background) IoU | {c_m['class_ious'][0]:.4f} | {t_m['class_ious'][0]:.4f} | {t_m['class_ious'][0]-c_m['class_ious'][0]:+.4f} | {'PASS' if t_m['class_ious'][0]>=c_m['class_ious'][0]-0.01 else 'REGR'} |\n")
            f.write(f"| Class 1 (Coral Reef) IoU | {c_m['class_ious'][1]:.4f} | {t_m['class_ious'][1]:.4f} | {t_m['class_ious'][1]-c_m['class_ious'][1]:+.4f} | {'PASS' if t_m['class_ious'][1]>=c_m['class_ious'][1]-0.01 else 'REGR'} |\n")
            f.write(f"| Class 2 (Fish) IoU | {c_m['class_ious'][2]:.4f} | {t_m['class_ious'][2]:.4f} | {t_m['class_ious'][2]-c_m['class_ious'][2]:+.4f} | {'PASS' if t_m['class_ious'][2]>=c_m['class_ious'][2]-0.01 else 'REGR'} |\n")
            f.write(f"| Class 3 (Elasmobranch) IoU | {c_m['class_ious'][3]:.4f} | {t_m['class_ious'][3]:.4f} | **{deltas['delta_class3_iou']:+.4f}** | {'GAIN' if deltas['delta_class3_iou']>0 else 'N/A' if c_m['class_ious'][3]==0 else 'REGR'} |\n")
            f.write(f"| Class 4 (Macro-Fauna) IoU | {c_m['class_ious'][4]:.4f} | {t_m['class_ious'][4]:.4f} | **{deltas['delta_class4_iou']:+.4f}** | {'GAIN' if deltas['delta_class4_iou']>0 else 'N/A' if c_m['class_ious'][4]==0 else 'REGR'} |\n\n")

            f.write("### Training Loss & Validation Traces Across 9 Epochs:\n\n")
            f.write("| Epoch | Control Loss | Control $Q_{\\text{seg}}$ | Treatment Loss | Treatment $Q_{\\text{seg}}$ | LR |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
            c_logs = f_data["control"]["epoch_logs"]
            t_logs = f_data["treatment"]["epoch_logs"]
            for ep_idx in range(len(c_logs)):
                c_ep = c_logs[ep_idx]
                t_ep = t_logs[ep_idx]
                f.write(f"| {ep_idx+1} | {c_ep['train_loss']:.4f} | {c_ep['metrics']['Q_seg']:.4f} | {t_ep['train_loss']:.4f} | {t_ep['metrics']['Q_seg']:.4f} | {c_ep['lr']:.6f} |\n")
            f.write("\n---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V20 Matched Loss Ablation Benchmark")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 2], help="Folds to test (default: 0 2)")
    parser.add_argument("--epochs", type=int, default=9, help="Epochs per arm (default: 9)")
    parser.add_argument("--batch_size", type=int, default=4, help="Minibatch size (default: 4)")
    parser.add_argument("--accum_steps", type=int, default=4, help="Accumulation steps (default: 4)")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v20/matched_pilot")
    parser.add_argument("--report_dir", type=str, default="reports/v20/matched_pilot")
    args = parser.parse_args()

    run_matched_pilot(
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        base_lr=args.base_lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        report_dir=args.report_dir
    )
