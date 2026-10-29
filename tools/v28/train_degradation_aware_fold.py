"""
SAT-EURS Track 1 — V28 Degradation-Aware Cross-Validation Training Pipeline
Trains a segmenter fold model on its exact 16 training scenes using paired degraded images.
Zero cross-scene leakage. Evaluates on held-out 4 validation scenes.
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


def train_fold(
    fold: int = 0,
    epochs: int = 4,
    swa_start_epoch: int = 3,
    batch_size: int = 8,
    base_lr: float = 4e-5,
    crop_size: int = 512,
    smoke_test_steps: Optional[int] = None,
    eval_stride: int = 5,  # Stride 5 = 40 variants/scene, 1 = 200 variants/scene
    device_id: int = 0,
    out_dir: str = "checkpoints/v28",
    report_dir: str = "reports/v28/phase2"
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    train_scenes = sorted(f_data["train_scenes"])
    val_scenes = sorted(f_data["val_scenes"])

    assert len(set(train_scenes).intersection(set(val_scenes))) == 0, f"LEAKAGE DETECTED in Fold {fold}!"

    print("=" * 85, flush=True)
    print(f"SAT-EURS V28 — DEGRADATION-AWARE TRAINING: FOLD {fold}", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"Train Scenes ({len(train_scenes)}): {train_scenes}", flush=True)
    print(f"Val Scenes   ({len(val_scenes)}): {val_scenes}", flush=True)
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={base_lr}", flush=True)
    print("=" * 85, flush=True)

    # 1. Load frozen refiner for this fold
    refiner_ckpt = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    base_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_obj = torch.load(refiner_ckpt, map_location="cpu")
    refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False

    # 2. Build Segmenter & Warm Start from verified fold baseline
    segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, neck_dim=256).to(device)
    if fold == 0:
        warmstart_ckpt = "checkpoints/v27/segmenter_v27_fold0_best.pt"
    else:
        warmstart_ckpt = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"

    print(f"\n[1/5] Warm-starting model weights from {warmstart_ckpt}...", flush=True)
    ws_obj = torch.load(warmstart_ckpt, map_location="cpu")
    ws_weights = ws_obj["state_dict"] if "state_dict" in ws_obj else ws_obj
    segmenter.load_state_dict(ws_weights, strict=True)

    # 3. First, measure Baseline OOF performance before fine-tuning
    print(f"\n[2/5] Evaluating Baseline OOF Performance before fine-tuning...", flush=True)
    baseline_metrics = evaluate_model_on_val_scenes(
        segmenter_ckpt=warmstart_ckpt,
        fold=fold,
        refiner_ckpt=refiner_ckpt,
        sample_stride=eval_stride,
        device_id=device_id
    )

    # 4. Setup Dataset with Scene-Balanced Sampler
    print(f"\n[3/5] Initializing V28 Degradation-Aware Dataset ({len(train_scenes)} scenes)...", flush=True)
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

    # 5. Setup Loss & Optimizer
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

    # 6. Training Loop
    print(f"\n[4/5] Training Fold {fold} ({epochs} epochs, SWA starting at epoch {swa_start_epoch})...", flush=True)
    t_start = time.time()
    best_loss = 1e9
    training_steps_executed = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        segmenter.train()
        running_loss = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(train_loader):
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

            running_loss += l_dict["loss"].item()
            step_count += 1
            training_steps_executed += 1

            if smoke_test_steps and training_steps_executed >= smoke_test_steps:
                print(f"  [SMOKE TEST] Reached limit of {smoke_test_steps} steps. Breaking loop.", flush=True)
                break

            if step_count % 50 == 0 or step_count == len(train_loader):
                cur_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
                print(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step_count:03d}/{len(train_loader)}] "
                    f"Loss: {running_loss/step_count:.4f} (CE: {l_dict['l_ce']:.4f}, Lov: {l_dict['l_lovasz']:.4f}) | VRAM: {cur_vram:.2f} GB",
                    flush=True
                )

        scheduler.step()
        mean_loss = running_loss / max(1, step_count)
        elapsed = time.time() - t0

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(segmenter)
            print(f"--> Epoch {epoch:02d} complete in {elapsed:.1f}s | Mean Loss: {mean_loss:.4f} [SWA updated]", flush=True)
        else:
            print(f"--> Epoch {epoch:02d} complete in {elapsed:.1f}s | Mean Loss: {mean_loss:.4f}", flush=True)

        if smoke_test_steps and training_steps_executed >= smoke_test_steps:
            break

    # Save fine-tuned checkpoint
    if epochs >= swa_start_epoch and (not smoke_test_steps):
        final_model = swa_model.module
        is_swa = True
    else:
        final_model = segmenter
        is_swa = False

    best_ckpt_path = osp.join(out_dir, f"segmenter_v28_fold{fold}_best.pt")
    torch.save({
        "epoch": epochs,
        "fold": fold,
        "state_dict": final_model.state_dict(),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "is_swa": is_swa,
        "base_lr": base_lr,
    }, best_ckpt_path)
    print(f"Saved Fold {fold} checkpoint to {best_ckpt_path}", flush=True)

    # 7. Post-Training Out-Of-Fold Evaluation
    print(f"\n[5/5] Evaluating Fine-Tuned Model on Held-Out Validation Scenes...", flush=True)
    finetuned_metrics = evaluate_model_on_val_scenes(
        segmenter_ckpt=best_ckpt_path,
        fold=fold,
        refiner_ckpt=refiner_ckpt,
        sample_stride=eval_stride,
        device_id=device_id
    )

    # Compare Baseline vs Fine-tuned
    base_q = baseline_metrics["global_metrics"]["Q_seg"]
    fine_q = finetuned_metrics["global_metrics"]["Q_seg"]
    delta_q = fine_q - base_q

    base_miou = baseline_metrics["global_metrics"]["mIoU"]
    fine_miou = finetuned_metrics["global_metrics"]["mIoU"]
    delta_miou = fine_miou - base_miou

    print("\n" + "=" * 80, flush=True)
    print(f"FOLD {fold} CROSS-VALIDATION PROMOTION VERDICT", flush=True)
    print("=" * 80, flush=True)
    print(f"Baseline Q_seg:    {base_q:.4f} (mIoU: {base_miou:.4f})", flush=True)
    print(f"Fine-Tuned Q_seg:   {fine_q:.4f} (mIoU: {fine_miou:.4f})", flush=True)
    print(f"Net Delta Q_seg:   {delta_q:+.4f} (Delta mIoU: {delta_miou:+.4f})", flush=True)
    if delta_q > 0.005:
        verdict = "PROMOTED (+0.005 gain threshold met)"
    elif delta_q >= 0:
        verdict = "MARGINAL (Positive delta, proceed with caution)"
    else:
        verdict = "REJECTED (Negative delta on held-out scenes)"
    print(f"Verdict: {verdict}", flush=True)
    print("=" * 80, flush=True)

    result_summary = {
        "fold": fold,
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "checkpoint_path": best_ckpt_path,
        "baseline_checkpoint": warmstart_ckpt,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "delta_Q_seg": round(float(delta_q), 5),
        "delta_mIoU": round(float(delta_miou), 5),
        "promotion_verdict": verdict,
        "training_time_s": round(time.time() - t_start, 1),
    }

    report_file = osp.join(report_dir, f"fold_{fold}_cv_report.json")
    with open(report_file, "w") as f:
        json.dump(result_summary, f, indent=2)
    print(f"Saved fold report to {report_file}", flush=True)

    return result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V28 Fold Training Pipeline")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--swa_start", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--smoke_test_steps", type=int, default=None)
    parser.add_argument("--eval_stride", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    train_fold(
        fold=args.fold,
        epochs=args.epochs,
        swa_start_epoch=args.swa_start,
        batch_size=args.batch_size,
        base_lr=args.lr,
        smoke_test_steps=args.smoke_test_steps,
        eval_stride=args.eval_stride,
        device_id=args.device_id
    )
