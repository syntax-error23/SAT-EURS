"""
Phase 3: Matched Architecture Experiment on S_folds (Folds 0 & 1)
Evaluates Control (M_base) vs Candidate (M_hr: DINOv2-L + HighResRefiner)
under strict matched-control conditions (WCE loss, identical seed, bfloat16 AMP).
"""

import os
import os.path as osp
import sys
import time
import json
import csv
import random
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from tools.train_track1_screening import ControlledTrack1Dataset, build_dinov2_model, apply_rsuigm_batch_gpu
from models.dinov2_highres_decoder import DINOv2HighResDecoder
from metrics.segmentation_metrics import SegmentationEvaluator


def compute_boundary_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 5, kernel_size: int = 3) -> float:
    """
    Computes Boundary mIoU within a boundary band of width kernel_size.
    Args:
        pred: [B, H, W] int tensor
        target: [B, H, W] int tensor
    """
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    target_np = target.detach().cpu().numpy().astype(np.uint8)
    b, h, w = pred_np.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    class_bious = []

    for c in range(1, num_classes): # Foreground classes 1..4
        gt_c = (target_np == c).astype(np.uint8)
        pred_c = (pred_np == c).astype(np.uint8)

        total_intersection = 0
        total_union = 0

        for i in range(b):
            gt_i = gt_c[i]
            pred_i = pred_c[i]

            if gt_i.sum() == 0 and pred_i.sum() == 0:
                continue

            # Boundary band = dilation - erosion
            gt_bound = cv2.dilate(gt_i, kernel) - cv2.erode(gt_i, kernel)
            pred_bound = cv2.dilate(pred_i, kernel) - cv2.erode(pred_i, kernel)

            # Intersection of boundary regions
            gt_b_mask = (gt_bound > 0) & (gt_i > 0)
            pred_b_mask = (pred_bound > 0) & (pred_i > 0)

            inter = np.logical_and(gt_b_mask, pred_b_mask).sum()
            union = np.logical_or(gt_b_mask, pred_b_mask).sum()

            total_intersection += inter
            total_union += union

        if total_union > 0:
            class_bious.append(total_intersection / (total_union + 1e-10))

    return float(np.mean(class_bious)) if len(class_bious) > 0 else 0.0


def evaluate_screening_fold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    rsuigm_table: Dict,
    mean_im: torch.Tensor,
    std_im: torch.Tensor
) -> Tuple[Dict[str, Any], float]:
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=5)
    boundary_ious = []

    with torch.no_grad():
        for x_raw, y_mask, wts, depths in val_loader:
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            x_in = apply_rsuigm_batch_gpu(x_raw, wts, depths, rsuigm_table)
            x_norm = (x_in - mean_im) / std_im

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)

            evaluator.update(logits, y_mask)

            # Evaluate boundary iou for this batch
            pred_mask = logits.argmax(dim=1)
            b_iou = compute_boundary_iou(pred_mask, y_mask, num_classes=5, kernel_size=3)
            boundary_ious.append(b_iou)

    metrics = evaluator.compute()
    avg_boundary_iou = float(np.mean(boundary_ious)) if boundary_ious else 0.0
    return metrics, avg_boundary_iou


def train_single_arm_fold(
    arm_name: str,
    model_type: str, # "base" or "highres"
    fold_idx: int,
    manifest: Dict[str, Any],
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    seed: int = 42,
    save_dir: str = "checkpoints/track1_screening"
) -> Tuple[Dict[str, Any], float, str]:
    # Set deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_key = f"fold_{fold_idx}"
    train_scenes = manifest["folds"][fold_key]["train_scenes"]
    val_scenes = manifest["folds"][fold_key]["val_scenes"]

    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold_idx}.pt"
    rsuigm_table = torch.load(tbl_path, map_location=device)

    print("\n" + "=" * 80)
    print(f"TRAINING PHASE 3: Arm={arm_name} | Fold={fold_idx} | Model={model_type}")
    print(f"  Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"  Val scenes   ({len(val_scenes)}): {val_scenes}")
    print(f"  Restoration: {tbl_path}")
    print("=" * 80)

    out_ckpt_dir = osp.join(save_dir, f"V15_{arm_name.upper()}_F{fold_idx}")
    os.makedirs(out_ckpt_dir, exist_ok=True)
    best_ckpt_path = osp.join(out_ckpt_dir, "best_model.pt")

    if osp.exists(best_ckpt_path):
        try:
            saved_data = torch.load(best_ckpt_path, map_location="cpu")
            if "metrics" in saved_data and "Q_seg" in saved_data["metrics"]:
                print(f"[RESUME] Found existing completed checkpoint at {best_ckpt_path} with Q_seg = {saved_data['metrics']['Q_seg']:.4f}. Skipping training.", flush=True)
                return saved_data["metrics"], saved_data.get("boundary_iou", 0.0), best_ckpt_path
        except Exception as e:
            print(f"[RESUME] Checkpoint {best_ckpt_path} unreadable ({e}). Re-training.", flush=True)

    # Datasets
    train_ds = ControlledTrack1Dataset(
        scenes=train_scenes,
        crop_size=518,
        is_train=True,
        use_rsuigm=True,
        rsuigm_table=rsuigm_table,
        copy_paste_augmenter=None,
        base_seed=seed
    )
    val_ds = ControlledTrack1Dataset(
        scenes=val_scenes,
        crop_size=518,
        is_train=False,
        use_rsuigm=True,
        rsuigm_table=rsuigm_table,
        base_seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Build model (Branch B: fresh pretrained DINOv2-L)
    if model_type == "base":
        model = build_dinov2_model(num_classes=5).to(device)
        param_groups = [
            {"params": model.encoder.parameters(), "lr": 3e-5},
            {"params": model.proj.parameters(), "lr": 1e-4},
            {"params": model.fusion.parameters(), "lr": 1e-4},
        ]
    elif model_type == "highres":
        model = DINOv2HighResDecoder(num_classes=5, pretrained_backbone=True).to(device)
        param_groups = [
            {"params": model.encoder.parameters(), "lr": 3e-5},
            {"params": model.proj.parameters(), "lr": 1e-4},
            {"params": model.fusion.parameters(), "lr": 1e-4},
            {"params": model.stem.parameters(), "lr": 1e-4},
            {"params": model.fuse_stage4.parameters(), "lr": 1e-4},
            {"params": model.fuse_stage2.parameters(), "lr": 1e-4},
            {"params": model.zero_conv.parameters(), "lr": 1e-4},
        ]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    class_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    best_qseg = -1.0
    best_metrics = {}
    best_boundary_iou = 0.0

    total_steps = 0
    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_ep_start = time.time()

        optimizer.zero_grad()

        for step_idx, (x_raw, y_mask, wts, depths) in enumerate(train_loader):
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            x_in = apply_rsuigm_batch_gpu(x_raw, wts, depths, rsuigm_table)
            x_norm = (x_in - mean_im) / std_im

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)
                loss = ce_loss_fn(logits, y_mask)
                loss = loss / grad_accum_steps

            loss.backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1

            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1

        scheduler.step()
        ep_time = time.time() - t_ep_start
        avg_loss = epoch_loss / max(1, n_batches)

        # Validation at epoch end
        val_metrics, val_boundary_iou = evaluate_screening_fold(
            model, val_loader, device, rsuigm_table, mean_im, std_im
        )

        qseg = val_metrics["Q_seg"]
        miou = val_metrics["mIoU"]
        f1 = val_metrics["macro_F1"]
        c3_f1 = val_metrics["per_class_F1"][3]
        c4_f1 = val_metrics["per_class_F1"][4]

        print(f"  Epoch {epoch}/{epochs} ({ep_time:.1f}s) | Train Loss: {avg_loss:.4f} | Val Q_seg: {qseg:.4f} (mIoU: {miou:.4f}, F1: {f1:.4f}, C3_F1: {c3_f1:.4f}, C4_F1: {c4_f1:.4f}, Bound_mIoU: {val_boundary_iou:.4f})", flush=True)

        if qseg > best_qseg:
            best_qseg = qseg
            best_metrics = val_metrics
            best_boundary_iou = val_boundary_iou
            torch.save({
                "epoch": epoch,
                "arm": arm_name,
                "fold": fold_idx,
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "metrics": val_metrics,
                "boundary_iou": val_boundary_iou
            }, best_ckpt_path)

    total_time = time.time() - t_train_start
    print(f"Finished {arm_name} Fold {fold_idx} in {total_time:.1f}s. Best Q_seg = {best_qseg:.4f} saved to {best_ckpt_path}", flush=True)
    return best_metrics, best_boundary_iou, best_ckpt_path


def run_phase3_experiment():
    print("=" * 80)
    print("PHASE 3: MATCHED ARCHITECTURE EXPERIMENT ON SCREENING FOLDS (0 & 1)")
    print("=" * 80)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    ledger_path = "reports/track1_research/v15/experiment_ledger.csv"
    results = {}

    # Define the 2 matched arms across Folds 0 & 1
    experiments = [
        ("Control_M_base", "base", 0),
        ("Control_M_base", "base", 1),
        ("Candidate_M_hr", "highres", 0),
        ("Candidate_M_hr", "highres", 1),
    ]

    for arm_name, m_type, f_idx in experiments:
        exp_id = f"EXP3_{arm_name}_F{f_idx}"
        metrics, bound_iou, ckpt_path = train_single_arm_fold(
            arm_name=arm_name,
            model_type=m_type,
            fold_idx=f_idx,
            manifest=manifest,
            epochs=5,
            batch_size=4,
            grad_accum_steps=2,
            seed=42
        )
        results[(arm_name, f_idx)] = (metrics, bound_iou, ckpt_path)

        # Record in ledger
        row = {
            "experiment_id": exp_id,
            "phase": "Phase 3 Matched Architecture",
            "arm": arm_name,
            "fold": f_idx,
            "architecture": "DINOv2-L Base" if m_type == "base" else "DINOv2-L + HighResRefiner",
            "loss": "WCE [0.8, 1.0, 1.2, 2.0, 2.0]",
            "epochs": 5,
            "steps": 2000,
            "input_size": "518x518",
            "best_val_qseg": f"{metrics['Q_seg']:.4f}",
            "val_miou": f"{metrics['mIoU']:.4f}",
            "val_macro_f1": f"{metrics['macro_F1']:.4f}",
            "val_cov_penalty": f"{metrics['coverage_penalty']:.4f}",
            "val_f1_c3": f"{metrics['per_class_F1'][3]:.4f}",
            "val_f1_c4": f"{metrics['per_class_F1'][4]:.4f}",
            "boundary_miou": f"{bound_iou:.4f}",
            "promotion_status": "PENDING_POOLING",
            "checkpoint_path": ckpt_path,
            "notes": "Matched control training completed."
        }
        already_recorded = False
        if osp.exists(ledger_path):
            with open(ledger_path, "r") as f_in:
                reader = csv.DictReader(f_in)
                for r in reader:
                    if r.get("experiment_id") == exp_id:
                        already_recorded = True
                        break
        if not already_recorded:
            with open(ledger_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "experiment_id", "phase", "arm", "fold", "architecture",
                    "loss", "epochs", "steps", "input_size", "best_val_qseg",
                    "val_miou", "val_macro_f1", "val_cov_penalty", "val_f1_c3",
                    "val_f1_c4", "boundary_miou", "promotion_status", "checkpoint_path", "notes"
                ])
                writer.writerow(row)

    # Compute Pooled Metrics for Control vs Candidate on S_folds (0-1)
    print("\n" + "=" * 80)
    print("PHASE 3: POOLED EVALUATION ACROSS SCREENING FOLDS 0 & 1 (1,600 IMAGES)")
    print("=" * 80)

    for arm in ["Control_M_base", "Candidate_M_hr"]:
        cm_pooled = np.array(results[(arm, 0)][0]["confusion_matrix"]) + np.array(results[(arm, 1)][0]["confusion_matrix"])
        tp = np.diag(cm_pooled).astype(float)
        fp = cm_pooled.sum(axis=0) - tp
        fn = cm_pooled.sum(axis=1) - tp
        eps = 1e-10

        iou = tp / (tp + fp + fn + eps)
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        miou = float(np.mean(iou))
        macro_f1 = float(np.mean(f1))
        gt_present = cm_pooled.sum(axis=1) > 0
        cov_penalty = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0
        qseg = miou * macro_f1 * cov_penalty
        avg_bound = (results[(arm, 0)][1] + results[(arm, 1)][1]) / 2.0

        print(f"\nArm: {arm}")
        print(f"  Pooled Q_seg:         {qseg:.4f}")
        print(f"  Pooled mIoU:          {miou:.4f}")
        print(f"  Pooled Macro F1:      {macro_f1:.4f}")
        print(f"  Pooled Coverage Pen:  {cov_penalty:.4f}")
        print(f"  Per-Class IoU:        {[round(float(v), 4) for v in iou]}")
        print(f"  Per-Class F1:         {[round(float(v), 4) for v in f1]}")
        print(f"  Rare C3 (Elas) F1:    {f1[3]:.4f}")
        print(f"  Rare C4 (Other) F1:   {f1[4]:.4f}")
        print(f"  Mean Boundary mIoU:   {avg_bound:.4f}")

        row = {
            "experiment_id": f"EXP3_{arm}_POOLED",
            "phase": "Phase 3 Matched Architecture",
            "arm": arm,
            "fold": "0+1 Pooled",
            "architecture": "DINOv2-L Base" if arm == "Control_M_base" else "DINOv2-L + HighResRefiner",
            "loss": "WCE [0.8, 1.0, 1.2, 2.0, 2.0]",
            "epochs": 5,
            "steps": 2000,
            "input_size": "518x518",
            "best_val_qseg": f"{qseg:.4f}",
            "val_miou": f"{miou:.4f}",
            "val_macro_f1": f"{macro_f1:.4f}",
            "val_cov_penalty": f"{cov_penalty:.4f}",
            "val_f1_c3": f"{f1[3]:.4f}",
            "val_f1_c4": f"{f1[4]:.4f}",
            "boundary_miou": f"{avg_bound:.4f}",
            "promotion_status": "EVALUATED",
            "checkpoint_path": "checkpoints/track1_screening",
            "notes": "Full 1600 image pooled screening metric"
        }
        with open(ledger_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment_id", "phase", "arm", "fold", "architecture",
                "loss", "epochs", "steps", "input_size", "best_val_qseg",
                "val_miou", "val_macro_f1", "val_cov_penalty", "val_f1_c3",
                "val_f1_c4", "boundary_miou", "promotion_status", "checkpoint_path", "notes"
            ])
            writer.writerow(row)

    print("\nPhase 3 execution completed successfully.")

if __name__ == "__main__":
    run_phase3_experiment()
