"""
SAT-EURS V24.1 — Evidence Audit & Protocol Validation Script
Faithfully reproduces and evaluates:
A. V18 Independent Inference
B. V24 Restoration Consensus Only (Segmentation Unchanged)
C. V24 Segmentation Consensus Only (Restoration Unchanged)
D. V24 Full Pipeline

Evaluates Fold 0 (fish1, fish11, fish12, seahorse1) AND Fold 1 (fish13, fish14, fish15, sharkFish).
Per-scene reporting is mandatory.
"""

import os
import os.path as osp
import sys
import json
import time
import math
from typing import Dict, List, Tuple, Any

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.evaluator import V18OOFEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_fold(fold: int, device: torch.device) -> Dict[str, Any]:
    print(f"\n================================================================================")
    print(f"AUDITING FOLD {fold} (800 Held-Out Images across 4 Scenes)")
    print(f"================================================================================")

    # Load fold manifest
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    scenes = manifest["folds"][f"fold_{fold}"]["val_scenes"]
    print(f"Held-out validation scenes for Fold {fold}: {scenes}")

    # Load Fold Checkpoints
    table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    refiner_path = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    segmenter_path = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"

    print(f"Loading checkpoints:")
    print(f"  Table:     {table_path}")
    print(f"  Refiner:   {refiner_path}")
    print(f"  Segmenter: {segmenter_path}")

    table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=table, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(refiner_path, map_location=device)
    refiner.load_state_dict(ref_ckpt.get("state_dict", ref_ckpt))
    refiner.eval()

    segmenter = build_v18_segmenter(num_classes=5).to(device)
    seg_ckpt = torch.load(segmenter_path, map_location=device)
    segmenter.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))
    segmenter.eval()

    # Evaluators for each variant
    eval_A = V18OOFEvaluator(num_classes=5)  # V18 Independent
    eval_B = V18OOFEvaluator(num_classes=5)  # V24 Rest Consensus Only
    eval_C = V18OOFEvaluator(num_classes=5)  # V24 Seg Consensus Only
    eval_D = V18OOFEvaluator(num_classes=5)  # V24 Full Pipeline

    per_scene_results = {}

    for sc in scenes:
        t_sc = time.time()
        print(f"\n--- Processing Scene `{sc}` (200 images) ---")

        # Load clean ground truth
        c_bgr = cv2.imread(f"data/cached_1024/clean/{sc}.png")
        clean = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_mask = cv2.imread(f"data/cached_1024/masks/{sc}.png", cv2.IMREAD_GRAYSCALE)

        # 1. Build Shallow Consensus Restoration Anchor
        shallow_imgs = []
        with torch.no_grad():
            for wt in [1, 2, 3]:
                for depth in [5, 10]:
                    d_p = f"data/cached_1024/degraded/{sc}/{sc}_{wt}_{depth}.png"
                    bgr = cv2.imread(d_p)
                    deg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    deg_t = torch.from_numpy(deg_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                    wt_t = torch.tensor([wt], device=device)
                    dp_t = torch.tensor([depth], device=device)
                    out = refiner(deg_t, wt_t, dp_t)
                    shallow_imgs.append(out["refined_image"].squeeze(0).permute(1, 2, 0).cpu().numpy())

        raw_anchor = np.mean(shallow_imgs, axis=0)
        blur = cv2.GaussianBlur(raw_anchor, (0, 0), 2.0)
        rest_anchor_np = np.clip(raw_anchor + 0.35 * (raw_anchor - blur), 0.0, 1.0)
        anchor_t = torch.from_numpy(rest_anchor_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Metrics of the anchor itself against clean
        psnr_anchor, _ = compute_psnr_and_norm(rest_anchor_np, clean)
        ssim_anchor = compute_ssim_single(rest_anchor_np, clean)
        sharp_anchor = compute_sharpness_ratio(rest_anchor_np, clean)

        # 2. First Pass: Compute Independent Restoration & Accumulate Consensus Segmentation Probabilities
        prob_accum = np.zeros((5, 1024, 1024), dtype=np.float32)
        total_w = 0.0

        # Cache per-image predictions to evaluate cleanly
        cached_indep_rest = []
        cached_indep_masks = []
        cached_indep_probs = []

        with torch.no_grad():
            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    d_p = f"data/cached_1024/degraded/{sc}/{sc}_{wt}_{depth}.png"
                    bgr = cv2.imread(d_p)
                    deg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    deg_t = torch.from_numpy(deg_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                    wt_t = torch.tensor([wt], device=device)
                    dp_t = torch.tensor([depth], device=device)

                    # A. V18 Independent Refiner (FP32)
                    out_r = refiner(deg_t, wt_t, dp_t)
                    rest_indep = out_r["refined_image"]
                    rest_indep_np = rest_indep.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    cached_indep_rest.append(rest_indep_np)

                    # Segmenters in BF16
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        # A. V18 Independent Segmenter (fed with rest_indep)
                        out_s = segmenter(rest_indep, deg_t, wt_t, dp_t, unpadded_size=(1024, 1024))
                        prob_i = F.softmax(out_s["logits"].float(), dim=1).squeeze(0).cpu().numpy()

                        # C & D. Segmentation Consensus Pass (using anchor_t as restoration input, as in pipeline)
                        out_s_cons = segmenter(anchor_t, deg_t, wt_t, dp_t, unpadded_size=(1024, 1024))
                        prob_c_feed = F.softmax(out_s_cons["logits"].float(), dim=1).squeeze(0).cpu().numpy()

                    pred_i = np.argmax(prob_i, axis=0).astype(np.uint8)
                    cached_indep_masks.append(pred_i)
                    cached_indep_probs.append(prob_i)

                    w = np.exp(-depth / 25.0)
                    prob_accum += w * prob_c_feed
                    total_w += w

        prob_consensus_pure = prob_accum / total_w
        pred_consensus_pure = np.argmax(prob_consensus_pure, axis=0).astype(np.uint8)

        # 3. Update Evaluators for the 200 Images
        for idx in range(200):
            r_indep = cached_indep_rest[idx]
            m_indep = cached_indep_masks[idx]

            # Variant A: V18 Independent (independent restoration, independent segmentation)
            eval_A.update_restoration(r_indep, clean, scene=sc)
            eval_A.update_segmentation(m_indep, gt_mask, scene=sc)

            # Variant B: V24 Restoration Consensus Only (consensus anchor restoration, independent segmentation)
            eval_B.update_restoration(rest_anchor_np, clean, scene=sc)
            eval_B.update_segmentation(m_indep, gt_mask, scene=sc)

            # Variant C: V24 Segmentation Consensus Only (independent restoration, consensus segmentation)
            eval_C.update_restoration(r_indep, clean, scene=sc)
            eval_C.update_segmentation(pred_consensus_pure, gt_mask, scene=sc)

            # Variant D: V24 Full Pipeline (consensus restoration anchor, consensus segmentation)
            eval_D.update_restoration(rest_anchor_np, clean, scene=sc)
            eval_D.update_segmentation(pred_consensus_pure, gt_mask, scene=sc)

        print(f"Scene `{sc}` completed in {time.time() - t_sc:.1f}s")
        print(f"  Anchor Quality: PSNR = {psnr_anchor:.2f} dB, SSIM = {ssim_anchor:.4f}, Sharp = {sharp_anchor:.4f}")

    # Compute results for all variants
    res_A = eval_A.compute(q_nr_provisional=0.737)
    res_B = eval_B.compute(q_nr_provisional=0.737)
    res_C = eval_C.compute(q_nr_provisional=0.737)
    res_D = eval_D.compute(q_nr_provisional=0.737)

    # Per-scene extraction
    scene_breakdown = {}
    for sc in scenes:
        stats_A = eval_A.scene_stats[sc]
        stats_D = eval_D.scene_stats[sc]

        psnr_A = np.mean(stats_A["psnr"])
        ssim_A = np.mean(stats_A["ssim"])
        sharp_A = np.mean(stats_A["sharp"])

        psnr_D = np.mean(stats_D["psnr"])
        ssim_D = np.mean(stats_D["ssim"])
        sharp_D = np.mean(stats_D["sharp"])

        # Compute per-scene segmentation metrics
        def calc_seg_stats(c_mat):
            tp = np.diag(c_mat).astype(np.float64)
            gt_px = c_mat.sum(axis=1).astype(np.float64)
            pred_px = c_mat.sum(axis=0).astype(np.float64)
            fp = pred_px - tp
            fn = gt_px - tp
            ious, f1s = [], []
            cov = 1.0
            for c in range(5):
                if gt_px[c] > 0:
                    d_iou = tp[c] + fp[c] + fn[c]
                    d_f1 = 2 * tp[c] + fp[c] + fn[c]
                    iou = float(tp[c] / d_iou) if d_iou > 0 else 0.0
                    f1 = float(2 * tp[c] / d_f1) if d_f1 > 0 else 0.0
                    ious.append(iou)
                    f1s.append(f1)
                    if tp[c] == 0:
                        cov = 0.0
            miou = float(np.mean(ious)) if ious else 0.0
            f1 = float(np.mean(f1s)) if f1s else 0.0
            q_seg = miou * f1 * cov
            return miou, f1, cov, q_seg

        miou_A, f1_A, cov_A, q_seg_A = calc_seg_stats(stats_A["conf_mat"])
        miou_D, f1_D, cov_D, q_seg_D = calc_seg_stats(stats_D["conf_mat"])

        scene_breakdown[sc] = {
            "V18_Independent": {
                "PSNR_dB": round(float(psnr_A), 2),
                "SSIM": round(float(ssim_A), 4),
                "Sharpness": round(float(sharp_A), 4),
                "mIoU": round(miou_A, 4),
                "Macro_F1": round(f1_A, 4),
                "Q_seg": round(q_seg_A, 4),
            },
            "V24_Consensus": {
                "PSNR_dB": round(float(psnr_D), 2),
                "SSIM": round(float(ssim_D), 4),
                "Sharpness": round(float(sharp_D), 4),
                "mIoU": round(miou_D, 4),
                "Macro_F1": round(f1_D, 4),
                "Q_seg": round(q_seg_D, 4),
            },
            "Delta": {
                "PSNR_dB": round(float(psnr_D - psnr_A), 2),
                "SSIM": round(float(ssim_D - ssim_A), 4),
                "Sharpness": round(float(sharp_D - sharp_A), 4),
                "mIoU": round(float(miou_D - miou_A), 4),
                "Macro_F1": round(float(f1_D - f1_A), 4),
                "Q_seg": round(float(q_seg_D - q_seg_A), 4),
            }
        }

    return {
        "fold": fold,
        "scenes": scenes,
        "A_V18_Independent": res_A,
        "B_V24_Rest_Consensus_Only": res_B,
        "C_V24_Seg_Consensus_Only": res_C,
        "D_V24_Full_Pipeline": res_D,
        "scene_breakdown": scene_breakdown
    }


def format_summary_table(name: str, res: Dict[str, Any], q_nr: float = 0.737) -> Dict[str, Any]:
    seg = res["segmentation"]
    rest = res["restoration"]
    q_seg = seg["Q_seg"]
    q_fr = rest["Q_FR_dataset"]
    comp = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr

    return {
        "Variant": name,
        "PSNR_dB": rest["PSNR_dB"],
        "PSNR_norm": rest["PSNR_norm_dataset"],
        "SSIM": rest["SSIM"],
        "Sharpness": rest["sharpness_ratio"],
        "Q_FR": q_fr,
        "mIoU": seg["mIoU"],
        "Macro_F1": seg["Macro_F1"],
        "Coverage": seg["coverage_penalty"],
        "Q_seg": q_seg,
        "Composite_Exact_90pct": 0.60 * q_seg + 0.30 * q_fr,
        "Composite_Official_Est": comp,
        "Class_IoUs": seg["class_ious"]
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running V24.1 Reproducibility Audit on {device} ({torch.cuda.get_device_name(0)})...")

    results_all = {}
    t0 = time.time()

    out_dir = "reports/v24"
    os.makedirs(out_dir, exist_ok=True)

    # Evaluate Fold 0 (The fold claimed in V24 report)
    f0_res = evaluate_fold(0, device)
    results_all["fold_0"] = f0_res
    with open(osp.join(out_dir, "reproduced_metrics_fold0.json"), "w") as fp:
        json.dump(f0_res, fp, indent=2)
    print("Saved Fold 0 JSON.")

    # Evaluate Fold 1 (Genuinely held-out fold with rare Class 3 'sharkFish')
    f1_res = evaluate_fold(1, device)
    results_all["fold_1"] = f1_res
    with open(osp.join(out_dir, "reproduced_metrics_fold1.json"), "w") as fp:
        json.dump(f1_res, fp, indent=2)
    print("Saved Fold 1 JSON.")

    print(f"\nAudit evaluation finished in {time.time() - t0:.1f}s.")

    # Save detailed combined JSON
    out_json = osp.join(out_dir, "reproduced_metrics_fold0_fold1.json")
    with open(out_json, "w") as fp:
        json.dump(results_all, fp, indent=2)
    print(f"Saved full audit JSON to {out_json}")

    # Print summary tables
    for f_name, f_data in results_all.items():
        print("\n" + "=" * 90)
        print(f"SUMMARY FOR {f_name.upper()} (Scenes: {f_data['scenes']})")
        print("=" * 90)
        table_rows = [
            format_summary_table("A: V18 Independent", f_data["A_V18_Independent"]),
            format_summary_table("B: V24 Rest Consensus Only", f_data["B_V24_Rest_Consensus_Only"]),
            format_summary_table("C: V24 Seg Consensus Only", f_data["C_V24_Seg_Consensus_Only"]),
            format_summary_table("D: V24 Full Pipeline", f_data["D_V24_Full_Pipeline"]),
        ]
        print(f"{'Variant':<26} | {'PSNR(dB)':<8} | {'SSIM':<6} | {'Sharp':<6} | {'Q_FR':<6} | {'mIoU':<6} | {'F1':<6} | {'Q_seg':<6} | {'Composite':<9}")
        print("-" * 90)
        for r in table_rows:
            print(f"{r['Variant']:<26} | {r['PSNR_dB']:<8.2f} | {r['SSIM']:<6.4f} | {r['Sharpness']:<6.4f} | {r['Q_FR']:<6.4f} | {r['mIoU']:<6.4f} | {r['Macro_F1']:<6.4f} | {r['Q_seg']:<6.4f} | {r['Composite_Official_Est']:<9.4f}")

        print("\nPer-Scene Breakdown:")
        for sc, sc_info in f_data["scene_breakdown"].items():
            v18 = sc_info["V18_Independent"]
            v24 = sc_info["V24_Consensus"]
            d = sc_info["Delta"]
            print(f"  Scene `{sc}`:")
            print(f"    PSNR:  {v18['PSNR_dB']:.2f} -> {v24['PSNR_dB']:.2f} ({d['PSNR_dB']:+.2f} dB)")
            print(f"    SSIM:  {v18['SSIM']:.4f} -> {v24['SSIM']:.4f} ({d['SSIM']:+.4f})")
            print(f"    Sharp: {v18['Sharpness']:.4f} -> {v24['Sharpness']:.4f} ({d['Sharpness']:+.4f})")
            print(f"    mIoU:  {v18['mIoU']:.4f} -> {v24['mIoU']:.4f} ({d['mIoU']:+.4f})")
            print(f"    F1:    {v18['Macro_F1']:.4f} -> {v24['Macro_F1']:.4f} ({d['Macro_F1']:+.4f})")
            print(f"    Q_seg: {v18['Q_seg']:.4f} -> {v24['Q_seg']:.4f} ({d['Q_seg']:+.4f})")


if __name__ == "__main__":
    main()
