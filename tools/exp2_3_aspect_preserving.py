"""
Experiment 2.3: Aspect-Preserving Inference vs. Square Inference
Step 1: Compare single-scale square (518x518) vs. single-scale aspect-preserving
(long side 518, short side rounded to nearest multiple of 14) on Pooled Screening Folds (0-1) [1,600 images].

Safeguard 4 Hard Floors to beat (from reports/track1_research/v7_locked_baseline.json):
- Screening Folds (0-1) Q_seg: >= 0.3838
- Screening Folds (0-1) mIoU:  >= 0.5461
- Screening Folds (0-1) Macro F1: >= 0.7028
- Per-Class IoU: [0.6184, 0.5884, 0.5666, 0.3915, 0.5658]
- Per-Class F1:  [0.7642, 0.7409, 0.7233, 0.5627, 0.7227]
"""

import os
import os.path as osp
import sys
import json
import time
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu

def compute_aspect_preserving_dims(w: int, h: int, max_dim: int = 518, patch_size: int = 14) -> Tuple[int, int]:
    """Computes (new_w, new_h) where max(new_w, new_h) = max_dim and both are multiples of 14."""
    if w >= h:
        new_w = max_dim
        scale = max_dim / float(w)
        raw_h = h * scale
        new_h = max(patch_size, int(round(raw_h / patch_size)) * patch_size)
    else:
        new_h = max_dim
        scale = max_dim / float(h)
        raw_w = w * scale
        new_w = max(patch_size, int(round(raw_w / patch_size)) * patch_size)
    return new_w, new_h

def run_step1_aspect_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("EXPERIMENT 2.3 STEP 1: SINGLE-SCALE SQUARE VS. ASPECT-PRESERVING (SCREENING 0-1)")
    print("=" * 76)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    with open("reports/track1_research/v7_locked_baseline.json") as f:
        locked = json.load(f)
    base_floors = locked["pooled_screening_folds_0_1"]["hard_floors"]

    data_root = "data/cached_1024"
    clean_dir = osp.join(data_root, "clean")
    masks_dir = osp.join(data_root, "masks")
    deg_dir = osp.join(data_root, "degraded")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    eval_square = SegmentationEvaluator(num_classes=5)
    eval_aspect = SegmentationEvaluator(num_classes=5)

    t0 = time.time()
    total_imgs = 0

    for f_idx in [0, 1]:
        val_scenes = manifest["folds"][f"fold_{f_idx}"]["val_scenes"]
        tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{f_idx}.pt"
        tbl = torch.load(tbl_path, map_location=device, weights_only=False)

        ckpt_path = f"checkpoints/track1_screening/DINOV2_CP_F{f_idx}/best_model.pt"
        print(f"\nEvaluating Fold {f_idx} on held-out scenes {val_scenes} using {ckpt_path}...")
        model = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for sc in val_scenes:
            gt_mask = cv2.imread(osp.join(masks_dir, f"{sc}.png"), cv2.IMREAD_GRAYSCALE)
            h_orig, w_orig = gt_mask.shape
            y_t = torch.from_numpy(gt_mask.astype(np.int64)).long().unsqueeze(0).to(device)

            sc_deg = osp.join(deg_dir, sc)
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])

            # Compute aspect-preserving dimensions for this scene
            asp_w, asp_h = compute_aspect_preserving_dims(w_orig, h_orig, max_dim=518, patch_size=14)

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

                    # 1. Square Inference (518x518) + HFlip
                    x_sq = F.interpolate(x_norm, size=(518, 518), mode="bilinear", align_corners=False)
                    x_sq_f = torch.flip(x_sq, dims=[3])
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        l_sq = model(x_sq)
                        l_sq_f = model(x_sq_f)
                    p_sq = 0.5 * F.softmax(l_sq.float(), dim=1) + 0.5 * torch.flip(F.softmax(l_sq_f.float(), dim=1), dims=[3])
                    p_sq_orig = F.interpolate(p_sq, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                    # 2. Aspect-Preserving Inference (asp_h, asp_w) + HFlip
                    x_asp = F.interpolate(x_norm, size=(asp_h, asp_w), mode="bilinear", align_corners=False)
                    x_asp_f = torch.flip(x_asp, dims=[3])
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        l_asp = model(x_asp)
                        l_asp_f = model(x_asp_f)
                    p_asp = 0.5 * F.softmax(l_asp.float(), dim=1) + 0.5 * torch.flip(F.softmax(l_asp_f.float(), dim=1), dims=[3])
                    p_asp_orig = F.interpolate(p_asp, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                eval_square.update(p_sq_orig, y_t)
                eval_aspect.update(p_asp_orig, y_t)
                total_imgs += 1

    print(f"\nEvaluated {total_imgs} images in {time.time() - t0:.1f}s.")
    res_sq = eval_square.compute()
    res_asp = eval_aspect.compute()

    print("\n" + "=" * 76)
    print("STEP 1 COMPARISON (POOLED SCREENING FOLDS 0-1, 1,600 IMAGES):")
    print(f"  Locked Floor:    mIoU >= {base_floors['mIoU']:.4f} | F1 >= {base_floors['macro_F1']:.4f} | Q_seg >= {base_floors['Q_seg']:.4f}")
    print(f"  Square (518):    mIoU =  {res_sq['mIoU']:.4f} | F1 =  {res_sq['macro_F1']:.4f} | Q_seg =  {res_sq['Q_seg']:.4f}")
    print(f"  Aspect-Preserve: mIoU =  {res_asp['mIoU']:.4f} | F1 =  {res_asp['macro_F1']:.4f} | Q_seg =  {res_asp['Q_seg']:.4f}")
    print("-" * 76)
    print(f"  Square Per-Class IoU: {res_sq['per_class_IoU']}")
    print(f"  Aspect Per-Class IoU: {res_asp['per_class_IoU']}")
    delta_q = res_asp['Q_seg'] - res_sq['Q_seg']
    print(f"  Delta Q_seg: {delta_q:+.4f} ({'PASS STEP 1' if delta_q >= 0 else 'FAIL STEP 1'})")
    print("=" * 76)

    report = {
        "experiment": "2.3_step1_aspect_vs_square",
        "evaluation_set": "pooled_screening_folds_0_1 (1,600 images)",
        "locked_floors": base_floors,
        "square_metrics": res_sq,
        "aspect_metrics": res_asp,
        "delta_q_seg": round(delta_q, 4),
        "step1_passed": bool(delta_q >= 0)
    }
    with open("reports/track1_research/exp2_3_step1_results.json", "w") as fp:
        json.dump(report, fp, indent=2)

    return report

if __name__ == '__main__':
    run_step1_aspect_comparison()
