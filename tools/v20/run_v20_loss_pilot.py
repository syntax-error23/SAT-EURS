"""
SAT-EURS Track 1 — V20 Controlled Rare-Class Loss Pilot Execution Script
Author: Empirical ML Specialist & Senior Computer Vision Engineer
Date: October 18, 2026

Experiment: EXP-V20-PILOT
Objective: Screen whether empirically grounded, capped class-weighted Lovasz & CE loss
           improves rare-class (Classes 3 & 4) segmentation without regressing Classes 0, 1, 2.
Split: Fold 0 (16 train scenes, 4 held-out val scenes: fish1, fish11, fish12, seahorse1)
Reference Baselines:
- V18 Pilot (1 Epoch): Class 4 IoU = 0.0, Q_seg = 0.0
- V18 Full Best (Epoch 6): Class 4 IoU = 0.28066, Recall = 0.29802, Q_seg = 0.35079
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
from models.v18.segmenter import build_v18_segmenter
from models.v18.restoration_refiner import build_v18_restoration_refiner
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


def load_v18_checkpoint_provenance(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("provenance", {})


def run_v20_pilot(
    fold: int = 0,
    epochs: int = 2,
    batch_size: int = 4,
    accum_steps: int = 4,
    base_lr: float = 1e-4,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/v20/pilot",
    report_dir: str = "reports/v20/pilot"
):
    print("=" * 80)
    print("SAT-EURS Track 1 — V20 Controlled Rare-Class Loss Pilot")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    # 1. Baseline Archive Integrity Check (Preflight)
    print("\n[Step 1/6] Verifying V18 Baseline Archive Integrity (Pre-Run)...")
    if not os.path.exists(V18_ZIP_PATH):
        raise FileNotFoundError(f"Missing immutable baseline ZIP: {V18_ZIP_PATH}")
    sha_before = compute_sha256(V18_ZIP_PATH)
    print(f"  V18 Baseline Archive: {V18_ZIP_PATH}")
    print(f"  SHA256 (Before):      {sha_before}")
    if sha_before != EXPECTED_V18_ZIP_SHA256:
        raise RuntimeError(f"FATAL: V18 ZIP hash mismatch! Expected {EXPECTED_V18_ZIP_SHA256}, got {sha_before}")
    print("  Baseline Archive Status: 100% INTACT & VERIFIED.")

    # 2. Hardware and Device Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n[Step 2/6] Hardware Environment: {device_name} (PyTorch {torch.__version__})")
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)

    # 3. Load Split Manifest
    manifest_path = "splits/manifest_5fold.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    fold_key = f"fold_{fold}"
    fold_data = manifest["folds"][fold_key]
    train_scenes = fold_data["train_scenes"]
    val_scenes = fold_data["val_scenes"]
    print(f"\n[Step 3/6] Split Configuration (Fold {fold}):")
    print(f"  Train Scenes ({len(train_scenes)}): {train_scenes}")
    print(f"  Val Scenes ({len(val_scenes)}):   {val_scenes}")

    # Compute exact empirical class distribution on 16 training scenes
    mask_dir = "data/cached_1024/masks"
    train_counts = np.zeros(5, dtype=np.int64)
    for sc in train_scenes:
        m_path = os.path.join(mask_dir, f"{sc}.png")
        if os.path.exists(m_path):
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            bc = np.bincount(m.flatten(), minlength=5)[:5]
            train_counts += bc * 200

    print(f"  Empirical Training Pixel Counts: {train_counts.tolist()}")
    class_weights = compute_v20_grounded_class_weights(
        train_counts.tolist(), min_weight=0.50, max_weight=3.50, smoothing=0.50, device=device
    )
    print(f"  Derived V20 Class Weights:       {class_weights.tolist()}")

    # 4. Model Setup
    print("\n[Step 4/6] Initializing Models & Frozen V18 Refiner...")
    refiner_path = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    if not os.path.exists(refiner_path):
        raise FileNotFoundError(f"Missing V18 refiner checkpoint: {refiner_path}")
    table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    rsuigm_table = torch.load(table_path, map_location=device)

    refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(refiner_path, map_location=device)
    ref_weights = ref_ckpt["state_dict"] if "state_dict" in ref_ckpt else ref_ckpt
    refiner.load_state_dict(ref_weights)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False
    print("  Frozen V18 Refiner loaded successfully.")

    segmenter = build_v18_segmenter(
        num_classes=5,
        vit_pretrained=True,
        conv_pretrained=True,
        use_mock=False,
        neck_dim=256
    ).to(device)
    print("  DINOv3 Dual-Encoder Segmenter initialized with pretrained weights.")

    loss_fn = SegmentationV20Loss(
        num_classes=5,
        class_weights=class_weights,
        w_ce=0.40,
        w_lovasz=0.30,
        w_dice=0.20,
        w_bnd=0.05,
        w_obj=0.05,
        ignore_index=255
    ).to(device)
    print("  V20 Class-Weighted Lovasz & Grounded CE Loss initialized.")

    param_groups = [
        {"params": [p for p in segmenter.vit_backbone.parameters() if p.requires_grad], "lr": base_lr * 0.10},
        {"params": [p for p in segmenter.conv_backbone.parameters() if p.requires_grad], "lr": base_lr * 0.20},
        {"params": list(segmenter.detail_stem.parameters()) + list(segmenter.cond_encoder.parameters()) + list(segmenter.neck.parameters()), "lr": base_lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. Datasets & Loaders
    train_ds = V18EURSInvarianceDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=seed + fold)
    val_ds = V18EURSInvarianceDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=seed + fold)
    sampler = V18SceneBalancedSampler(train_ds, batch_size=batch_size, seed=seed + fold)
    num_workers = 4 if device.type == "cuda" else 0
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    print(f"  Train: {len(train_ds)} samples ({len(train_loader)} minibatches, {len(train_loader)//accum_steps} optimizer updates/epoch)")
    print(f"  Val:   {len(val_ds)} samples ({len(val_loader)} minibatches)")

    # 6. Training & Evaluation Loop
    print("\n[Step 5/6] Executing Controlled V20 Pilot Training Loop...")
    os.makedirs(os.path.join(checkpoint_dir, f"fold_{fold}"), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    epoch_logs = []
    best_val_qseg = -1.0
    best_metrics = {}
    best_epoch = -1
    t_train_start = time.time()
    peak_vram_global = 0.0

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

            if step_count % 100 == 0 or step_count == len(train_loader):
                cur_vram = torch.cuda.memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
                peak_vram_global = max(peak_vram_global, cur_vram)
                print(f"  [Ep {epoch+1:02d}/{epochs:02d}] Step {step_count:03d}/{len(train_loader):03d} (Opt: {opt_updates:03d}) | Loss: {running_loss/step_count:.4f} (CE: {running_ce/step_count:.4f}, Lov: {running_lovasz/step_count:.4f}) | VRAM: {cur_vram:.2f} GB", flush=True)

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
        c4_iou = float(val_res["class_ious"][4])
        c4_rec = float(val_res["class_recalls"][4])
        c4_prec = float(val_res["class_precisions"][4])

        print(f"\n  >>> [Epoch {epoch+1:02d} Validation] Total Time: {total_ep_time:.1f}s (Train: {train_time_ep:.1f}s, Val: {val_time_ep:.1f}s)")
        print(f"      Q_seg: {val_qseg:.5f} | mIoU: {val_miou:.5f} | F1: {val_f1:.5f} | Cov: {val_res['coverage_penalty']}")
        print(f"      Class 0 (Bg):    IoU={val_res['class_ious'][0]:.4f}, Rec={val_res['class_recalls'][0]:.4f}")
        print(f"      Class 1 (Coral): IoU={val_res['class_ious'][1]:.4f}, Rec={val_res['class_recalls'][1]:.4f}")
        print(f"      Class 2 (Fish):  IoU={val_res['class_ious'][2]:.4f}, Rec={val_res['class_recalls'][2]:.4f}")
        print(f"      Class 4 (Fauna): IoU={c4_iou:.4f}, Rec={c4_rec:.4f}, Prec={c4_prec:.4f}\n", flush=True)

        ep_record = {
            "epoch": epoch + 1,
            "train_loss": round(mean_train_loss, 5),
            "train_time_sec": round(train_time_ep, 2),
            "val_time_sec": round(val_time_ep, 2),
            "total_time_sec": round(total_ep_time, 2),
            "metrics": val_res
        }
        epoch_logs.append(ep_record)

        # Checkpoint saving on best Val Q_seg (or best Class 4 IoU if coverage is 0)
        is_best = False
        if val_qseg > best_val_qseg:
            best_val_qseg = val_qseg
            is_best = True
        elif best_val_qseg <= 0.0 and c4_iou > best_metrics.get("class_ious", [0, 0, 0, 0, 0])[4]:
            is_best = True

        if is_best:
            best_metrics = val_res
            best_epoch = epoch + 1
            ckpt_save_path = os.path.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
            torch.save({
                "version": "v20_pilot",
                "arch_name": "EURSAPEXV18Segmenter",
                "fold": fold,
                "train_scenes": train_scenes,
                "val_scenes": val_scenes,
                "seed": seed + fold,
                "epoch": best_epoch,
                "state_dict": segmenter.state_dict(),
                "provenance": {
                    "version": "v20_pilot",
                    "loss_type": "EmpiricalClassWeightedLovaszAndCE",
                    "class_weights": class_weights.tolist(),
                    "fold": fold,
                    "epoch": best_epoch,
                    "metrics": val_res,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }, ckpt_save_path)
            print(f"      [*] Saved new best checkpoint at {ckpt_save_path} (Epoch {best_epoch})\n")

    total_training_wall_sec = time.time() - t_train_start

    # 7. Post-Run Baseline Archive Integrity Verification
    print("\n[Step 6/6] Verifying V18 Baseline Archive Integrity (Post-Run)...")
    sha_after = compute_sha256(V18_ZIP_PATH)
    print(f"  V18 Baseline Archive: {V18_ZIP_PATH}")
    print(f"  SHA256 (After):       {sha_after}")
    assert sha_after == sha_before, "FATAL: V18 Baseline archive was modified during execution!"
    print("  Baseline Archive Status: 100% INTACT & VERIFIED.")

    # 8. Compare against V18 Baselines
    v18_pilot_ckpt = "checkpoints/v18/pilot/fold_0/segmenter_best.pt"
    v18_full_ckpt = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
    v18_pilot_prov = load_v18_checkpoint_provenance(v18_pilot_ckpt)
    v18_full_prov = load_v18_checkpoint_provenance(v18_full_ckpt)

    pilot_report = {
        "title": "SAT-EURS Track 1 — V20 Controlled Rare-Class Loss Pilot Report",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "experiment_id": "EXP-V20-PILOT-FOLD0",
        "fold": fold,
        "seed": seed + fold,
        "epochs_trained": epochs,
        "total_wall_sec": round(total_training_wall_sec, 2),
        "total_wall_min": round(total_training_wall_sec / 60.0, 2),
        "peak_vram_gb": round(peak_vram_global, 2),
        "baseline_archive": {
            "path": V18_ZIP_PATH,
            "sha256": sha_after,
            "intact": True
        },
        "loss_configuration": {
            "type": "EmpiricallyGroundedClassWeightedLovaszAndCE",
            "class_weights": [round(x, 4) for x in class_weights.tolist()],
            "component_weights": {
                "w_ce": 0.40,
                "w_lovasz": 0.30,
                "w_dice": 0.20,
                "w_boundary": 0.05,
                "w_objectness": 0.05
            }
        },
        "epoch_logs": epoch_logs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "comparison_vs_v18": {
            "v18_epoch1_pilot": v18_pilot_prov.get("metrics", {}),
            "v18_epoch6_best": v18_full_prov.get("metrics", {})
        }
    }

    json_report_path = os.path.join(report_dir, "PILOT_BENCHMARK_REPORT.json")
    with open(json_report_path, "w") as f:
        json.dump(pilot_report, f, indent=2)
    print(f"\nDeliverable JSON written to {json_report_path}")

    # Generate Markdown Report
    v18_ep1_c4 = v18_pilot_prov.get("metrics", {}).get("class_ious", [0, 0, 0, 0, 0])[4] if v18_pilot_prov else 0.0
    v18_ep6_c4 = v18_full_prov.get("metrics", {}).get("class_ious", [0, 0, 0, 0, 0])[4] if v18_full_prov else 0.28066
    v20_best_c4 = best_metrics.get("class_ious", [0, 0, 0, 0, 0])[4]

    md_report_path = os.path.join(report_dir, "PILOT_BENCHMARK_REPORT.md")
    with open(md_report_path, "w") as f:
        f.write("# SAT-EURS Track 1 — V20 Controlled Rare-Class Loss Pilot Report\n\n")
        f.write(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  \n")
        f.write(f"**Hardware:** {device_name}  \n")
        f.write(f"**Evaluation Fold:** Fold {fold} (`fish1`, `fish11`, `fish12`, `seahorse1`)  \n")
        f.write(f"**Training Duration:** {total_training_wall_sec/60.0:.2f} minutes ({epochs} epochs)  \n")
        f.write(f"**Peak VRAM:** {peak_vram_global:.2f} GB  \n")
        f.write(f"**Verified V18 Archive SHA256:** `{sha_after}` (100% Intact)  \n\n")
        f.write("---\n\n")
        f.write("## 1. Executive Summary & Screening Decision\n\n")
        f.write(f"- **V20 Best Class-4 IoU:** **{v20_best_c4:.5f}** (Best Epoch: {best_epoch})\n")
        f.write(f"- **V18 Matched Epoch 1 Pilot Class-4 IoU:** {v18_ep1_c4:.5f}\n")
        f.write(f"- **V18 Full Epoch 6 Best Class-4 IoU:** {v18_ep6_c4:.5f}\n\n")
        f.write("### Epoch-by-Epoch Progress Table:\n\n")
        f.write("| Epoch | Train Loss | Val Q_seg | Val mIoU | Val Macro F1 | Cls 0 IoU | Cls 1 IoU | Cls 2 IoU | Cls 4 IoU | Cls 4 Recall | Cls 4 Prec |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for ep in epoch_logs:
            m = ep["metrics"]
            f.write(f"| **{ep['epoch']}** | {ep['train_loss']:.4f} | {m['Q_seg']:.5f} | {m['mIoU']:.5f} | {m['Macro_F1']:.5f} | {m['class_ious'][0]:.4f} | {m['class_ious'][1]:.4f} | {m['class_ious'][2]:.4f} | **{m['class_ious'][4]:.4f}** | {m['class_recalls'][4]:.4f} | {m['class_precisions'][4]:.4f} |\n")
        f.write("\n---\n")

    print(f"Deliverable Markdown written to {md_report_path}")
    print("\n" + "=" * 80)
    print("V20 PILOT RUN COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V20 Controlled Loss Pilot")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (default: 0)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train (default: 2)")
    parser.add_argument("--batch_size", type=int, default=4, help="Minibatch size (default: 4)")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v20/pilot")
    parser.add_argument("--report_dir", type=str, default="reports/v20/pilot")
    args = parser.parse_args()

    run_v20_pilot(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        base_lr=args.base_lr,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        report_dir=args.report_dir
    )
