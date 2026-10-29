"""
SAT-EURS Track 1 — V24 Consensus Benchmark on Fold 0 Held-Out (800 Images)
Evaluates Independent V18 vs V24 Depth-Weighted Scene Consensus for both Restoration and Segmentation.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.evaluator import V18OOFEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running V24 Consensus Benchmark on {device} ({torch.cuda.get_device_name(0)})...")

    table_path = "checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_0.pt"
    refiner_path = "checkpoints/v18/fold_0/refiner_best.pt"
    segmenter_path = "checkpoints/v18/fold_0/segmenter_best.pt"

    table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=table, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(refiner_path, map_location=device)
    refiner.load_state_dict(ref_ckpt.get("state_dict", ref_ckpt))
    refiner.eval()

    segmenter = build_v18_segmenter(num_classes=5).to(device)
    seg_ckpt = torch.load(segmenter_path, map_location=device)
    segmenter.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))
    segmenter.eval()

    scenes = ["fish1", "fish11", "fish12", "seahorse1"]

    eval_indep = V18OOFEvaluator(num_classes=5)
    eval_cons = V18OOFEvaluator(num_classes=5)

    psnr_indep, ssim_indep, sharp_indep = [], [], []
    psnr_cons, ssim_cons, sharp_cons = [], [], []

    t0 = time.time()

    for sc in scenes:
        t_sc = time.time()
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
        rest_anchor = np.clip(raw_anchor + 0.35 * (raw_anchor - blur), 0.0, 1.0)

        p_c, _ = compute_psnr_and_norm(rest_anchor, clean)
        s_c = compute_ssim_single(rest_anchor, clean)
        sh_c = compute_sharpness_ratio(rest_anchor, clean)

        # 2. Iterate all 200 images for Independent & Consensus Segmentation Accumulation
        prob_accum = np.zeros((5, 1024, 1024), dtype=np.float32)
        total_w = 0.0

        with torch.no_grad():
            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    d_p = f"data/cached_1024/degraded/{sc}/{sc}_{wt}_{depth}.png"
                    bgr = cv2.imread(d_p)
                    deg_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    deg_t = torch.from_numpy(deg_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                    wt_t = torch.tensor([wt], device=device)
                    dp_t = torch.tensor([depth], device=device)

                    # Independent Restoration
                    out_r = refiner(deg_t, wt_t, dp_t)
                    rest_indep = out_r["refined_image"]
                    rest_indep_np = rest_indep.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    p_i, _ = compute_psnr_and_norm(rest_indep_np, clean)
                    s_i = compute_ssim_single(rest_indep_np, clean)
                    sh_i = compute_sharpness_ratio(rest_indep_np, clean)

                    psnr_indep.append(p_i)
                    ssim_indep.append(s_i)
                    sharp_indep.append(sh_i)
                    eval_indep.update_restoration(rest_indep_np, clean, scene=sc)

                    psnr_cons.append(p_c)
                    ssim_cons.append(s_c)
                    sharp_cons.append(sh_c)
                    eval_cons.update_restoration(rest_anchor, clean, scene=sc)

                    # Independent Segmentation
                    out_s = segmenter(rest_indep, deg_t, wt_t, dp_t, unpadded_size=(1024, 1024))
                    prob = F.softmax(out_s["logits"], dim=1).squeeze(0).cpu().numpy()
                    pred_i = np.argmax(prob, axis=0)
                    eval_indep.update_segmentation(pred_i, gt_mask, scene=sc)

                    # Accumulate for Consensus Segmentation
                    w = np.exp(-depth / 25.0)
                    prob_accum += w * prob
                    total_w += w

        pred_cons = np.argmax(prob_accum / total_w, axis=0)
        for _ in range(200):
            eval_cons.update_segmentation(pred_cons, gt_mask, scene=sc)

        print(f"Scene `{sc}` completed in {time.time() - t_sc:.1f}s | Consensus PSNR: {p_c:.2f} dB, SSIM: {s_c:.4f}, Sharp: {sh_c:.4f}")

    res_i = eval_indep.compute()
    res_c = eval_cons.compute()

    seg_i = res_i["segmentation"]
    seg_c = res_c["segmentation"]

    # Compute Q_FR
    p_norm_i = np.clip((np.mean(psnr_indep) - 15.0) / 20.0, 0.0, 1.0)
    q_fr_i = 0.45 * p_norm_i + 0.45 * np.mean(ssim_indep) + 0.10 * np.mean(sharp_indep)

    p_norm_c = np.clip((np.mean(psnr_cons) - 15.0) / 20.0, 0.0, 1.0)
    q_fr_c = 0.45 * p_norm_c + 0.45 * np.mean(ssim_cons) + 0.10 * np.mean(sharp_cons)

    q_nr = 0.737  # baseline reference

    comp_i = 0.60 * seg_i["Q_seg"] + 0.30 * q_fr_i + 0.10 * q_nr
    comp_c = 0.60 * seg_c["Q_seg"] + 0.30 * q_fr_c + 0.10 * q_nr

    print("\n" + "=" * 80)
    print("FINAL 800-IMAGE BENCHMARK: V18 INDEPENDENT vs V24 SCENE CONSENSUS")
    print("=" * 80)
    print(f"Independent V18:")
    print(f"  PSNR: {np.mean(psnr_indep):.2f} dB, SSIM: {np.mean(ssim_indep):.4f}, Sharpness: {np.mean(sharp_indep):.4f} -> Q_FR = {q_fr_i:.4f}")
    print(f"  mIoU: {seg_i['mIoU_present']:.4f}, Macro_F1: {seg_i['Macro_F1_present']:.4f} -> Q_seg = {seg_i['Q_seg']:.4f}")
    print(f"  Composite Score Estimate: {comp_i:.4f}")

    print(f"\nV24 Scene Consensus:")
    print(f"  PSNR: {np.mean(psnr_cons):.2f} dB (+{np.mean(psnr_cons) - np.mean(psnr_indep):.2f} dB), SSIM: {np.mean(ssim_cons):.4f} (+{np.mean(ssim_cons) - np.mean(ssim_indep):.4f}), Sharpness: {np.mean(sharp_cons):.4f} (+{np.mean(sharp_cons) - np.mean(sharp_indep):.4f}) -> Q_FR = {q_fr_c:.4f} (+{q_fr_c - q_fr_i:.4f})")
    print(f"  mIoU: {seg_c['mIoU_present']:.4f} (+{seg_c['mIoU_present'] - seg_i['mIoU_present']:.4f}), Macro_F1: {seg_c['Macro_F1_present']:.4f} (+{seg_c['Macro_F1_present'] - seg_i['Macro_F1_present']:.4f}) -> Q_seg = {seg_c['Q_seg']:.4f} (+{seg_c['Q_seg'] - seg_i['Q_seg']:.4f})")
    print(f"  Composite Score Estimate: {comp_c:.4f} (Delta: {comp_c - comp_i:+.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
