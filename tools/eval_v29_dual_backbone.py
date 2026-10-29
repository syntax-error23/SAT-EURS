"""
SAT-EURS Track 1 — V29 Dual-Backbone Ensembling & Anti-Trap Calibration Evaluator
Experiments: EXP-03 (Dual-Backbone Ensembling) & EXP-04 (Anti-Trap Presence Gating)

Evaluates on Fold 0 held-out validation scenes ('fish1', 'fish11', 'fish12', 'seahorse1'):
1. Baseline: DINOv2-Large (V27 Fold 0) alone
2. SegFormer-B5 (Coralscapes Fine-Tuned) alone
3. Dual-Backbone Linear Blends (w * P_SegFormer + (1-w) * P_DINOv2)
4. Anti-Trap Presence Gating on rare classes (Macrofauna & Elasmobranchs)
"""

import os
import os.path as osp
import sys
import time
import json
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

from models.v18.segmenter import build_v18_segmenter
from tools.train_track1_screening import apply_rsuigm_batch_gpu
from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_v29_segformer_b5 import EURSSegmentationDataset


def evaluate_dual_backbone(
    segformer_ckpt: str = "checkpoints/v29/segformer_b5_fold_0_best.pt",
    dinov2_ckpt: str = "checkpoints/v27/segmenter_v27_fold0_best.pt",
    fold: int = 0,
    batch_size: int = 8,
    device_id: int = 0
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SAT-EURS V29 DUAL-BACKBONE HELD-OUT EVALUATION & CALIBRATION BENCHMARK")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"SegFormer Checkpoint: {segformer_ckpt}")
    print(f"DINOv2 Checkpoint:    {dinov2_ckpt}")
    print("=" * 85)

    # 1. Manifest & Table
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    f_data = manifest["folds"][f"fold_{fold}"]
    val_scenes = sorted(f_data["val_scenes"])
    print(f"Held-Out Validation Scenes ({len(val_scenes)}): {val_scenes}")

    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold}.pt"
    table = torch.load(tbl_path, map_location=device, weights_only=False)

    val_ds = EURSSegmentationDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=42 + fold)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Load DINOv2 model
    print("Loading DINOv2 Segmenter...")
    dino_model = build_v18_segmenter(num_classes=5).to(device)
    dino_data = torch.load(dinov2_ckpt, map_location=device, weights_only=False)
    if "state_dict" in dino_data:
        dino_model.load_state_dict(dino_data["state_dict"])
    elif "model_state_dict" in dino_data:
        dino_model.load_state_dict(dino_data["model_state_dict"])
    else:
        dino_model.load_state_dict(dino_data)
    dino_model.eval()

    # 3. Load SegFormer-B5 model
    print("Loading SegFormer-B5 Segmenter...")
    model_id = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024"
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(device)
    seg_data = torch.load(segformer_ckpt, map_location=device, weights_only=False)
    seg_model.load_state_dict(seg_data["model_state_dict"])
    seg_model.eval()

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Collect all predictions in memory for fast sweep
    print(f"Extracting validation probabilities across {len(val_ds)} samples...")
    all_dino_probs = []
    all_seg_probs = []
    all_gts = []

    t0 = time.time()
    with torch.no_grad():
        for x_deg, y_gt, wt, depth in val_loader:
            x_deg = x_deg.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)
            wt = wt.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            # Restored input via RSUIGM table
            x_rest = apply_rsuigm_batch_gpu(x_deg, wt, depth, table)

            # DINOv2 Forward
            dino_out = dino_model(x_rest, x_deg, wt, depth)
            dino_prob = F.softmax(dino_out["logits"], dim=1)

            # SegFormer-B5 Forward
            x_norm = (x_rest - im_mean) / im_std
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seg_out = seg_model(pixel_values=x_norm)
                seg_logits = F.interpolate(seg_out.logits.float(), size=y_gt.shape[-2:], mode="bilinear", align_corners=False)
                seg_prob = F.softmax(seg_logits, dim=1)

            all_dino_probs.append(dino_prob.cpu())
            all_seg_probs.append(seg_prob.cpu())
            all_gts.append(y_gt.cpu())

    print(f"Extraction completed in {time.time() - t0:.1f}s.")
    all_dino_probs = torch.cat(all_dino_probs, dim=0)
    all_seg_probs = torch.cat(all_seg_probs, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    N = all_gts.shape[0]

    # Benchmark function
    def eval_probs(probs: torch.Tensor, name: str) -> Dict[str, Any]:
        evaluator = SegmentationEvaluator(num_classes=5)
        # Process in chunks of 16 to save memory
        for i in range(0, N, 16):
            p_chunk = probs[i:i+16].to(device)
            gt_chunk = all_gts[i:i+16].to(device)
            pred_chunk = torch.argmax(p_chunk, dim=1)
            evaluator.update(pred_chunk, gt_chunk)
        metrics = evaluator.compute()
        q_seg = metrics["Q_seg"]
        miou = metrics["mIoU"]
        f1 = metrics["macro_F1"]
        ious = [round(x, 4) for x in metrics["per_class_IoU"]]
        print(f"  {name:30s} | Q_seg: {q_seg:.4f} | mIoU: {miou:.4f} | F1: {f1:.4f} | IoUs: {ious}")
        return metrics

    print("\n" + "=" * 85)
    print("1. SINGLE MODEL BASELINES")
    print("=" * 85)
    m_dino = eval_probs(all_dino_probs, "DINOv2 (V27 Fold 0)")
    m_seg = eval_probs(all_seg_probs, "SegFormer-B5 (Coralscapes)")

    print("\n" + "=" * 85)
    print("2. DUAL-BACKBONE LINEAR BLEND SWEEP")
    print("=" * 85)
    best_blend_w = 0.0
    best_blend_q = 0.0
    for w_seg in [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        fused = w_seg * all_seg_probs + (1.0 - w_seg) * all_dino_probs
        m = eval_probs(fused, f"Blend (w_seg={w_seg:.2f})")
        if m["Q_seg"] > best_blend_q:
            best_blend_q = m["Q_seg"]
            best_blend_w = w_seg

    print(f"\n-> Optimal Blend Weight: w_seg = {best_blend_w:.2f} (Q_seg: {best_blend_q:.4f})")

    print("\n" + "=" * 85)
    print("3. ANTI-TRAP PRESENCE GATING ON OPTIMAL BLEND")
    print("=" * 85)
    best_fused = best_blend_w * all_seg_probs + (1.0 - best_blend_w) * all_dino_probs

    best_gate_tau = 0.0
    best_gate_q = best_blend_q
    # Test presence gating for rare classes 3 and 4:
    # If maximum probability of class across image is less than tau, suppress class probability to 0
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        gated = best_fused.clone()
        for c in [3, 4]:
            # Max prob per image: shape (N, 1, 1)
            max_c = gated[:, c, :, :].amax(dim=(-2, -1), keepdim=True)
            suppress_mask = (max_c < tau).float()
            gated[:, c, :, :] = gated[:, c, :, :] * (1.0 - suppress_mask)
        m = eval_probs(gated, f"Gating (tau={tau:.2f})")
        if m["Q_seg"] > best_gate_q:
            best_gate_q = m["Q_seg"]
            best_gate_tau = tau

    print(f"\n-> Optimal Presence Gating: tau = {best_gate_tau:.2f} (Q_seg: {best_gate_q:.4f})")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segformer_ckpt", type=str, default="checkpoints/v29/segformer_b5_fold_0_best.pt")
    parser.add_argument("--dinov2_ckpt", type=str, default="checkpoints/v27/segmenter_v27_fold0_best.pt")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluate_dual_backbone(
        segformer_ckpt=args.segformer_ckpt,
        dinov2_ckpt=args.dinov2_ckpt,
        fold=args.fold,
        batch_size=args.batch_size
    )
