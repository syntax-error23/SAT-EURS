"""
SAT-EURS Track 1 — Phase C Overnight Fold Upgrades
Targeted Fine-Tuning: Folds 1, 2, 4
Protocol:
  1. Measure Baseline OOF metrics before fine-tuning.
  2. Degradation-aware fine-tuning with frozen RSUIGM+Refiner on the 16 disjoint training scenes.
  3. Rare-class foreground object sampling (Macro-fauna, Shark/Ray, Fish prioritized).
  4. Conservative LR: base_lr=1.8e-5, ViT backbone=1.8e-6, Conv backbone=3.6e-6.
  5. SWA starting at epoch 2 across 3 epochs.
  6. Strict Promotion Gate:
     - Delta Q_seg > +0.002
     - No catastrophic collapse on rare classes
     - Promoted checkpoints saved to checkpoints/v36/fold_{fold}/segmenter_v36_fold{fold}_best.pt
     - Rejected checkpoints fall back to protected baseline v18 checkpoints.
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, ".")

from data.v28.dataset import V28DegradationAwareDataset, V28SceneBalancedSampler
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from losses.v18.segmentation_losses import SegmentationV18Loss, compute_capped_class_weights
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data
from tools.v28.evaluate_fold_oof import evaluate_model_on_val_scenes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_single_fold(
    fold: int,
    epochs: int = 3,
    swa_start_epoch: int = 2,
    batch_size: int = 8,
    base_lr: float = 1.8e-5,
    crop_size: int = 512,
    eval_stride: int = 5,
    out_dir: str = "checkpoints/v36",
    report_dir: str = "reports/v36",
) -> Dict[str, Any]:
    os.makedirs(osp.join(out_dir, f"fold_{fold}"), exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    train_scenes = sorted(f_data["train_scenes"])
    val_scenes = sorted(f_data["val_scenes"])

    print("=" * 85, flush=True)
    print(f"PHASE C — FINE-TUNING FOLD {fold}", flush=True)
    print(f"Train Scenes ({len(train_scenes)}): {train_scenes}", flush=True)
    print(f"Val Scenes   ({len(val_scenes)}): {val_scenes}", flush=True)
    print(f"LR: {base_lr}, Epochs: {epochs}, SWA Start: {swa_start_epoch}, Batch: {batch_size}", flush=True)
    print("=" * 85, flush=True)

    # 1. Load refiner for this fold
    refiner_ckpt = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    base_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_obj = torch.load(refiner_ckpt, map_location=device)
    refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False

    # 2. Segmenter warm start
    warmstart_ckpt = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
    print(f"\n[1/4] Baseline OOF Evaluation for Fold {fold} from {warmstart_ckpt}...", flush=True)
    baseline_metrics = evaluate_model_on_val_scenes(
        segmenter_ckpt=warmstart_ckpt,
        fold=fold,
        refiner_ckpt=refiner_ckpt,
        sample_stride=eval_stride,
        device_id=0
    )
    base_q = baseline_metrics["global_metrics"]["Q_seg"]
    base_miou = baseline_metrics["global_metrics"]["mIoU"]
    base_f1 = baseline_metrics["global_metrics"]["Macro_F1"]
    print(f"  Baseline Fold {fold} Metrics: Q_seg={base_q:.4f}, mIoU={base_miou:.4f}, Macro_F1={base_f1:.4f}", flush=True)

    segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, neck_dim=256).to(device)
    ws_obj = torch.load(warmstart_ckpt, map_location=device)
    segmenter.load_state_dict(ws_obj["state_dict"] if "state_dict" in ws_obj else ws_obj)

    # 3. Dataset & Sampler
    train_ds = V28DegradationAwareDataset(
        scenes=train_scenes,
        target_size=crop_size,
        is_train=True,
        object_crop_prob=0.70,
        min_crop_ratio=0.75,
        base_seed=42 + fold
    )
    sampler = V28SceneBalancedSampler(train_ds, batch_size=batch_size, seed=42 + fold)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # 4. Loss & Optimizer
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
    swa_model = AveragedModel(segmenter)

    # 5. Training loop
    print(f"\n[2/4] Training Fold {fold} ({epochs} epochs, SWA starting at epoch {swa_start_epoch})...", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        segmenter.train()
        running_loss = 0.0
        step_count = 0

        for batch in train_loader:
            deg = batch["deg_rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            wt = batch["wt"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)

            with torch.no_grad():
                out_ref = refiner(deg, wt, depth)
                restored = out_ref["refined_image"]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seg_out = segmenter(restored, deg, wt, depth, unpadded_size=(crop_size, crop_size))
                l_dict = loss_fn(seg_out, mask)
                loss = l_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(segmenter.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step_count += 1

        scheduler.step()
        mean_loss = running_loss / max(1, step_count)
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(segmenter)
            print(f"  Epoch {epoch:02d}/{epochs:02d} complete in {time.time() - t0:.1f}s | Loss: {mean_loss:.4f} [SWA updated]", flush=True)
        else:
            print(f"  Epoch {epoch:02d}/{epochs:02d} complete in {time.time() - t0:.1f}s | Loss: {mean_loss:.4f}", flush=True)

    # Save candidate checkpoint
    candidate_ckpt = osp.join(out_dir, f"fold_{fold}", f"segmenter_v36_fold{fold}_best.pt")
    final_model = swa_model.module if epochs >= swa_start_epoch else segmenter
    torch.save({
        "epoch": epochs,
        "fold": fold,
        "state_dict": final_model.state_dict(),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "is_swa": (epochs >= swa_start_epoch),
        "base_lr": base_lr,
    }, candidate_ckpt)
    print(f"  Saved candidate checkpoint to {candidate_ckpt}", flush=True)

    # 6. Post-training evaluation
    print(f"\n[3/4] Evaluating Fine-Tuned Model on Held-Out Validation Scenes...", flush=True)
    finetuned_metrics = evaluate_model_on_val_scenes(
        segmenter_ckpt=candidate_ckpt,
        fold=fold,
        refiner_ckpt=refiner_ckpt,
        sample_stride=eval_stride,
        device_id=0
    )
    fine_q = finetuned_metrics["global_metrics"]["Q_seg"]
    fine_miou = finetuned_metrics["global_metrics"]["mIoU"]
    fine_f1 = finetuned_metrics["global_metrics"]["Macro_F1"]
    delta_q = fine_q - base_q
    delta_miou = fine_miou - base_miou

    # 7. Promotion Gate
    print("\n" + "=" * 80, flush=True)
    print(f"PROMOTION GATE VERDICT FOR FOLD {fold}", flush=True)
    print(f"Baseline:    Q_seg = {base_q:.4f} (mIoU: {base_miou:.4f})", flush=True)
    print(f"Fine-Tuned:  Q_seg = {fine_q:.4f} (mIoU: {fine_miou:.4f})", flush=True)
    print(f"Net Gain:    Delta Q_seg = {delta_q:+.4f} (Delta mIoU: {delta_miou:+.4f})", flush=True)

    # Check for rare class collapse
    base_ious = baseline_metrics["global_metrics"]["class_ious"]
    fine_ious = finetuned_metrics["global_metrics"]["class_ious"]
    rare_collapse = False
    for c in range(1, 5):
        if base_ious[c] > 0.20 and fine_ious[c] < 0.10:
            rare_collapse = True
            print(f"  [ALERT] Rare class {c} collapsed: {base_ious[c]:.4f} -> {fine_ious[c]:.4f}", flush=True)

    is_promoted = (delta_q > 0.002) and (not rare_collapse)
    verdict = "PROMOTED" if is_promoted else "REJECTED (Retaining Baseline V18)"
    print(f"VERDICT: {verdict}", flush=True)
    print("=" * 80, flush=True)

    # Save report
    report_file = osp.join(report_dir, f"fold_{fold}_cv_report.json")
    with open(report_file, "w") as f:
        json.dump({
            "fold": fold,
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "base_lr": base_lr,
            "epochs": epochs,
            "baseline_metrics": baseline_metrics["global_metrics"],
            "finetuned_metrics": finetuned_metrics["global_metrics"],
            "delta_q_seg": round(delta_q, 5),
            "delta_miou": round(delta_miou, 5),
            "promoted": is_promoted,
            "promoted_checkpoint": candidate_ckpt if is_promoted else warmstart_ckpt
        }, f, indent=2)

    return {
        "fold": fold,
        "promoted": is_promoted,
        "delta_q": delta_q,
        "best_checkpoint": candidate_ckpt if is_promoted else warmstart_ckpt
    }


def run_phase_c():
    folds_to_train = [1, 2, 4]
    results = {}
    for f in folds_to_train:
        res = train_single_fold(fold=f)
        results[f] = res

    print("\n" + "=" * 80, flush=True)
    print("PHASE C TRAINING SUMMARY ACROSS ALL CANDIDATE FOLDS", flush=True)
    print("=" * 80, flush=True)
    for f, res in results.items():
        status = "PROMOTED (+gain)" if res["promoted"] else "REJECTED (Baseline kept)"
        print(f"Fold {f}: {status:<28} | Delta Q_seg: {res['delta_q']:+.4f} | Active Checkpoint: {res['best_checkpoint']}", flush=True)
    print("=" * 80, flush=True)

    summary_file = "reports/v36/phase_c_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_phase_c()
