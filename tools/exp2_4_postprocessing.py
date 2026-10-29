"""
Experiment 2.4: Isolated Post-Processing Ablations
Evaluates on Pooled Screening Folds (0-1) [1,600 images]:
1. Control: Baseline Multi-Scale TTA (locked floor: Q_seg = 0.3838, mIoU = 0.5461, F1 = 0.7028)
2. Option A: RGB-Guided Probability Filtering (PyTorch GPU Guided Filter)
3. Option B: Core-Anchored Boundary Hysteresis on Rare Classes (3 & 4)
4. Option C: Combined Guided Filter + Hysteresis

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
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu

def fast_guided_filter_gpu(guidance: torch.Tensor, prob: torch.Tensor, r: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """
    PyTorch GPU Implementation of He et al. Guided Filter.
    guidance: [1, 1, H, W] in [0, 1]
    prob: [1, C, H, W] in [0, 1]
    """
    k = 2 * r + 1
    mean_I = F.avg_pool2d(guidance, k, stride=1, padding=r)
    mean_P = F.avg_pool2d(prob, k, stride=1, padding=r)
    corr_I = F.avg_pool2d(guidance * guidance, k, stride=1, padding=r)
    corr_IP = F.avg_pool2d(guidance * prob, k, stride=1, padding=r)

    var_I = corr_I - mean_I * mean_I
    cov_IP = corr_IP - mean_I * mean_P

    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I

    mean_a = F.avg_pool2d(a, k, stride=1, padding=r)
    mean_b = F.avg_pool2d(b, k, stride=1, padding=r)

    q = mean_a * guidance + mean_b
    q = torch.clamp(q, 0.0, 1.0)
    # Re-normalize across classes
    q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
    return q

def apply_boundary_hysteresis(
    pred_probs_np: np.ndarray,
    tau_high: float = 0.45,
    tau_low: float = 0.20,
    min_area: int = 25
) -> np.ndarray:
    """
    Core-anchored boundary hysteresis on rare classes (3 & 4).
    pred_probs_np: [5, H, W]
    """
    c, h, w = pred_probs_np.shape
    calibrated_probs = pred_probs_np.copy()

    for cls_id in (3, 4):
        p_cls = pred_probs_np[cls_id]
        cand_mask = (p_cls > tau_low).astype(np.uint8)
        seed_mask = (p_cls > tau_high).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand_mask, connectivity=8)

        valid_mask = np.zeros((h, w), dtype=bool)
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            comp = (labels == lab)
            # Check if component contains at least one core seed pixel
            if np.any(comp & (seed_mask > 0)):
                valid_mask |= comp

        # Suppress candidate pixels that are not anchored to a core seed
        suppress_mask = (cand_mask > 0) & (~valid_mask)
        calibrated_probs[cls_id, suppress_mask] = 0.0

    return calibrated_probs.argmax(axis=0).astype(np.uint8)

def run_experiment_2_4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("EXPERIMENT 2.4: ISOLATED POST-PROCESSING ABLATIONS (SCREENING FOLDS 0-1)")
    print("=" * 76)

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

    eval_control = SegmentationEvaluator(num_classes=5)
    eval_optA_guided = SegmentationEvaluator(num_classes=5)
    eval_optB_hyst = SegmentationEvaluator(num_classes=5)
    eval_optC_combo = SegmentationEvaluator(num_classes=5)

    t0 = time.time()
    total_imgs = 0

    for f_idx in [0, 1]:
        val_scenes = manifest["folds"][f"fold_{f_idx}"]["val_scenes"]
        tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{f_idx}.pt"
        tbl = torch.load(tbl_path, map_location=device, weights_only=False)

        ckpt_path = f"checkpoints/track1_screening/DINOV2_CP_F{f_idx}/best_model.pt"
        print(f"\nEvaluating Fold {f_idx} held-out scenes {val_scenes} using {ckpt_path}...")
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

                    # Control: Raw interpolation
                    p_orig_size = F.interpolate(p_accum, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                    # Option A: RGB-Guided Filter
                    # Guidance is luminance of restored image
                    rest_gray = 0.299 * x_rest[:, 0:1] + 0.587 * x_rest[:, 1:2] + 0.114 * x_rest[:, 2:3]
                    p_guided = fast_guided_filter_gpu(rest_gray, p_orig_size, r=4, eps=1e-3)

                # Option B: Hysteresis on Control
                p_ctrl_np = p_orig_size.squeeze(0).cpu().numpy()
                mask_hyst_b = apply_boundary_hysteresis(p_ctrl_np, tau_high=0.45, tau_low=0.20, min_area=25)
                mask_hyst_b_t = torch.from_numpy(mask_hyst_b).unsqueeze(0).to(device)

                # Option C: Hysteresis on Guided Filter
                p_guided_np = p_guided.squeeze(0).cpu().numpy()
                mask_combo_c = apply_boundary_hysteresis(p_guided_np, tau_high=0.45, tau_low=0.20, min_area=25)
                mask_combo_c_t = torch.from_numpy(mask_combo_c).unsqueeze(0).to(device)

                # Update evaluators
                eval_control.update(p_orig_size, y_t)
                eval_optA_guided.update(p_guided, y_t)
                eval_optB_hyst.update(mask_hyst_b_t, y_t)
                eval_optC_combo.update(mask_combo_c_t, y_t)

                total_imgs += 1

    print(f"\nEvaluated {total_imgs} images across 4 post-processing modes in {time.time() - t0:.1f}s.")

    res_ctrl = eval_control.compute()
    res_optA = eval_optA_guided.compute()
    res_optB = eval_optB_hyst.compute()
    res_optC = eval_optC_combo.compute()

    print("\n" + "=" * 76)
    print("EXPERIMENT 2.4 SCREENING RESULTS (POOLED FOLDS 0-1, 1,600 IMAGES):")
    print(f"  Locked Floor:        mIoU >= {base_floors['mIoU']:.4f} | F1 >= {base_floors['macro_F1']:.4f} | Q_seg >= {base_floors['Q_seg']:.4f}")
    print(f"  Control (Raw TTA):   mIoU =  {res_ctrl['mIoU']:.4f} | F1 =  {res_ctrl['macro_F1']:.4f} | Q_seg =  {res_ctrl['Q_seg']:.4f}")
    print(f"  Opt A (Guided Filt): mIoU =  {res_optA['mIoU']:.4f} | F1 =  {res_optA['macro_F1']:.4f} | Q_seg =  {res_optA['Q_seg']:.4f}")
    print(f"  Opt B (Hysteresis):  mIoU =  {res_optB['mIoU']:.4f} | F1 =  {res_optB['macro_F1']:.4f} | Q_seg =  {res_optB['Q_seg']:.4f}")
    print(f"  Opt C (Combo):       mIoU =  {res_optC['mIoU']:.4f} | F1 =  {res_optC['macro_F1']:.4f} | Q_seg =  {res_optC['Q_seg']:.4f}")
    print("-" * 76)
    print(f"  Control Per-Class IoU: {res_ctrl['per_class_IoU']}")
    print(f"  Opt A   Per-Class IoU: {res_optA['per_class_IoU']}")
    print(f"  Opt B   Per-Class IoU: {res_optB['per_class_IoU']}")
    print(f"  Opt C   Per-Class IoU: {res_optC['per_class_IoU']}")
    print("=" * 76)

    report = {
        "experiment": "2.4_isolated_postprocessing_ablations",
        "evaluation_set": "pooled_screening_folds_0_1 (1,600 images)",
        "locked_floors": base_floors,
        "control_metrics": res_ctrl,
        "optA_guided_metrics": res_optA,
        "optB_hysteresis_metrics": res_optB,
        "optC_combo_metrics": res_optC,
    }
    with open("reports/track1_research/exp2_4_screening_results.json", "w") as fp:
        json.dump(report, fp, indent=2)

    return report

if __name__ == '__main__':
    run_experiment_2_4()
