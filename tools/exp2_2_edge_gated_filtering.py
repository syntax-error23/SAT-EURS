"""
Experiment 2.2: Edge-Gated Denoising & Unsharp Filtering on Pooled Screening Folds (0-1)
Sweeps a small pre-declared parameter grid:
- Config 1: Unsharp alpha=1.34, sigma=2.0 (pure unsharp boost)
- Config 2: Unsharp alpha=1.38, sigma=2.0 (pure unsharp boost)
- Config 3: Edge-gated: Bilateral (d=5, sc=15, ss=15) on flat + Unsharp (alpha=1.35) on edges
- Config 4: Edge-gated: Bilateral (d=5, sc=25, ss=25) on flat + Unsharp (alpha=1.38) on edges

Hard Floors to beat (from reports/track1_research/v7_locked_baseline.json):
- Screening Folds (0-1) Sharpness Floor: >= 0.5122
- Screening Folds (0-1) PSNR: >= 24.29 dB
- Screening Folds (0-1) SSIM: >= 0.7614
- Screening Folds (0-1) Q_FR: >= 0.6030
"""

import os
import os.path as osp
import sys
import json
import time
from typing import Dict, Tuple, List, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from metrics.restoration_metrics import RestorationEvaluator
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu
from tools.exp2_1_rsuigm_calibration import fit_16scene_rsuigm_table

def apply_edge_gated_filter(
    res_np: np.ndarray,
    unsharp_alpha: float = 1.35,
    unsharp_sigma: float = 2.0,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 20.0,
    bilateral_sigma_space: float = 20.0,
    edge_thresh: float = 0.15
) -> np.ndarray:
    """
    Edge-gated filter:
    1. Compute gradient magnitude map on luminance.
    2. Compute bilateral filtered base (denoises flat open water).
    3. Compute unsharp filtered base (sharpens coral, fish, boundaries).
    4. Soft-blend based on edge mask.
    """
    gray = cv2.cvtColor(res_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2) / 255.0
    edge_weight = np.clip((grad_mag - edge_thresh) / (0.35 - edge_thresh + 1e-6), 0.0, 1.0)[:, :, np.newaxis]

    # Denoised flat base
    if bilateral_d > 0:
        denoised = cv2.bilateralFilter(res_np, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    else:
        denoised = res_np

    # Sharpened edge base
    blur = cv2.GaussianBlur(res_np, (0, 0), unsharp_sigma)
    beta = 1.0 - unsharp_alpha
    sharpened = np.clip(cv2.addWeighted(res_np, unsharp_alpha, blur, beta, 0), 0, 255).astype(np.uint8)

    # Blend
    fused = (edge_weight * sharpened.astype(np.float32) + (1.0 - edge_weight) * denoised.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return fused

def run_experiment_2_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("EXPERIMENT 2.2: EDGE-GATED DENOISING & UNSHARP FILTERING ON SCREENING FOLDS (0-1)")
    print("=" * 76)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    with open("reports/track1_research/v7_locked_baseline.json") as f:
        locked = json.load(f)
    base_floors = locked["pooled_screening_folds_0_1"]["hard_floors"]
    base_rest = locked["pooled_screening_folds_0_1"]["restoration"]

    # 1. Fit or load 16-scene 19-term tables for Fold 0 and Fold 1
    tables = {}
    for f_idx in [0, 1]:
        tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{f_idx}.pt"
        if osp.exists(tbl_path):
            print(f"Loading cached 16-scene 19-term table for Fold {f_idx}...")
            tables[f_idx] = torch.load(tbl_path, map_location=device, weights_only=False)
        else:
            print(f"Fitting 16-scene 19-term table for Fold {f_idx}...")
            train_sc = manifest["folds"][f"fold_{f_idx}"]["train_scenes"]
            t = fit_16scene_rsuigm_table(train_sc, basis_type="19term", device=device)
            torch.save(t, tbl_path)
            tables[f_idx] = t

    configs = {
        "cfg1_unsharp_1.34": {"type": "pure", "alpha": 1.34, "sigma": 2.0},
        "cfg2_unsharp_1.38": {"type": "pure", "alpha": 1.38, "sigma": 2.0},
        "cfg3_edge_gated_1.35": {"type": "gated", "alpha": 1.35, "sigma": 2.0, "d": 5, "sc": 15.0, "ss": 15.0, "th": 0.12},
        "cfg4_edge_gated_1.38": {"type": "gated", "alpha": 1.38, "sigma": 2.0, "d": 5, "sc": 25.0, "ss": 25.0, "th": 0.12},
    }

    # Evaluate each config over all 1,600 images of screening folds 0 & 1
    evaluators = {k: RestorationEvaluator() for k in configs}

    data_root = "data/cached_1024"
    clean_dir = osp.join(data_root, "clean")
    deg_dir = osp.join(data_root, "degraded")

    t0 = time.time()
    total_imgs = 0

    for f_idx in [0, 1]:
        val_scenes = manifest["folds"][f"fold_{f_idx}"]["val_scenes"]
        tbl = tables[f_idx]

        for sc in val_scenes:
            c_bgr = cv2.imread(osp.join(clean_dir, f"{sc}.png"), cv2.IMREAD_COLOR)
            c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            c_t = torch.from_numpy(c_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            sc_deg = osp.join(deg_dir, sc)
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])

            for df in deg_files:
                parts = osp.splitext(df)[0].split("_")
                wt = int(parts[1])
                dp = int(parts[2])

                bgr = cv2.imread(osp.join(sc_deg, df), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                x_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

                W = tbl[(wt, dp)]
                with torch.no_grad():
                    feats = build_rsuigm_poly_features_gpu(x_t)
                    b, h_w, c = feats.shape
                    res_flat = torch.matmul(feats.reshape(-1, c), W).clamp_(0.0, 1.0)
                    res_1024 = res_flat.reshape(b, 1024, 1024, 3).permute(0, 3, 1, 2)

                res_np = (res_1024.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

                for cfg_name, p in configs.items():
                    if p["type"] == "pure":
                        blur = cv2.GaussianBlur(res_np, (0, 0), p["sigma"])
                        beta = 1.0 - p["alpha"]
                        out_u8 = np.clip(cv2.addWeighted(res_np, p["alpha"], blur, beta, 0), 0, 255).astype(np.uint8)
                    else:
                        out_u8 = apply_edge_gated_filter(
                            res_np,
                            unsharp_alpha=p["alpha"],
                            unsharp_sigma=p["sigma"],
                            bilateral_d=p["d"],
                            bilateral_sigma_color=p["sc"],
                            bilateral_sigma_space=p["ss"],
                            edge_thresh=p["th"]
                        )
                    out_t = torch.from_numpy(out_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                    evaluators[cfg_name].update(out_t, c_t)

                total_imgs += 1

    elapsed = time.time() - t0
    print(f"\nEvaluated {total_imgs} images across {len(configs)} configs in {elapsed:.1f}s.")

    results = {}
    print("\n" + "=" * 76)
    print("EXPERIMENT 2.2 SCREENING SUMMARY (POOLED FOLDS 0-1, 1,600 IMAGES):")
    print(f"  Hard Floors: PSNR >= {base_rest['PSNR_dB']:.2f} dB | SSIM >= {base_rest['SSIM']:.4f} | Sharp >= {base_floors['Sharpness_OOF_Base']:.4f} | Q_FR >= {base_rest['Q_rest']:.4f}")
    print("-" * 76)

    best_cfg = None
    best_qfr = -1.0

    for cfg_name, ev in evaluators.items():
        res = ev.compute()
        results[cfg_name] = res

        psnr_pass = res["PSNR_dB"] >= base_rest["PSNR_dB"]
        ssim_pass = res["SSIM"] >= base_rest["SSIM"]
        sharp_pass = res["sharpness_ratio"] >= base_floors["Sharpness_OOF_Base"]
        qfr_pass = res["Q_rest"] > base_rest["Q_rest"]

        all_pass = psnr_pass and ssim_pass and sharp_pass and qfr_pass

        status_str = "PASS (PROMOTABLE)" if all_pass else "FAIL_GUARDRAIL"
        print(f"  {cfg_name:24s}: PSNR={res['PSNR_dB']:.2f} dB | SSIM={res['SSIM']:.4f} | Sharp={res['sharpness_ratio']:.4f} | Q_FR={res['Q_rest']:.4f} -> {status_str}")

        if all_pass and res["Q_rest"] > best_qfr:
            best_qfr = res["Q_rest"]
            best_cfg = cfg_name

    print("-" * 76)
    print(f"Best Promotable Setting: {best_cfg} (Q_FR = {best_qfr:.4f})")
    print("=" * 76)

    # Save to JSON
    report = {
        "experiment": "2.2_edge_gated_denoising_and_unsharp_filtering",
        "evaluation_set": "pooled_screening_folds_0_1 (1,600 images)",
        "locked_baseline_screening": base_rest,
        "grid_results": results,
        "best_promotable_config": best_cfg
    }
    with open("reports/track1_research/exp2_2_screening_results.json", "w") as fp:
        json.dump(report, fp, indent=2)

    return report

if __name__ == '__main__':
    run_experiment_2_2()
