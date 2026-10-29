"""
Experiment 2.1: RSUIGM Restoration Calibration & Basis Ablation
Evaluates strictly on Pooled Screening Folds (0-1) [1,600 images]:
1. Control: Baseline 19-term fit (as in #18700 locked baseline)
2. Option 1: 16-Scene Equal-Weighted 19-Term Fit (all 16 permitted training scenes per fold)
3. Option 2: 28-Term Spatial-Context Basis (19-term + 9 symmetric local spatial-mean terms via 7x7 box filter)

Hard Floors to beat (from reports/track1_research/v7_locked_baseline.json):
- Screening Folds (0-1) PSNR: 24.29 dB
- Screening Folds (0-1) SSIM: 0.7614
- Screening Folds (0-1) Sharpness Floor: 0.5122
- Screening Folds (0-1) Q_FR: 0.6030
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

def compute_box_filter(img_bchw: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic reflect-padded kxk box filter on GPU."""
    pad = k // 2
    x_pad = F.pad(img_bchw, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size=k, stride=1, padding=0)

def build_28term_features_gpu(img_bchw: torch.Tensor, wt: int, depth: int) -> torch.Tensor:
    """
    28-Term Basis:
    19 standard terms (from build_rsuigm_poly_features_gpu) +
    9 spatial context terms:
      - 3 terms: Local 7x7 spatial mean (R_mean7, G_mean7, B_mean7)
      - 3 terms: Local 15x15 spatial mean (R_mean15, G_mean15, B_mean15)
      - 3 terms: High-frequency spatial detail (img - mean7)
    All 9 terms vary per-pixel across the spatial grid, ensuring non-collinearity.
    """
    b, c, h, w = img_bchw.shape
    feats_19 = build_rsuigm_poly_features_gpu(img_bchw) # [B, H*W, 19]

    mean7_bchw = compute_box_filter(img_bchw, 7) # [B, 3, H, W]
    mean15_bchw = compute_box_filter(img_bchw, 15) # [B, 3, H, W]
    diff7_bchw = img_bchw - mean7_bchw # [B, 3, H, W]

    mean7_flat = mean7_bchw.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    mean15_flat = mean15_bchw.permute(0, 2, 3, 1).reshape(b, h * w, 3)
    diff7_flat = diff7_bchw.permute(0, 2, 3, 1).reshape(b, h * w, 3)

    feats_28 = torch.cat([feats_19, mean7_flat, mean15_flat, diff7_flat], dim=-1) # [B, H*W, 28]
    return feats_28

def fit_16scene_rsuigm_table(
    train_scenes: List[str],
    basis_type: str = "19term",
    cached_dir: str = "data/cached_1024",
    device: torch.device = None
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Fits table on all 16 train_scenes with equal pixel representation per scene.
    Downsamples to 128x128 on GPU (16 * 128 * 128 = 262,144 pixels per condition).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_dir = osp.join(cached_dir, "clean")
    deg_dir = osp.join(cached_dir, "degraded")

    clean_list = []
    for s in train_scenes:
        c_bgr = cv2.imread(osp.join(clean_dir, f"{s}.png"), cv2.IMREAD_COLOR)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_list.append(torch.from_numpy(c_rgb).permute(2, 0, 1))

    clean_gpu = torch.stack(clean_list, dim=0).to(device) # [16, 3, 1024, 1024]
    clean_128 = F.avg_pool2d(clean_gpu, kernel_size=8, stride=8) # [16, 3, 128, 128]
    Y_target = clean_128.permute(0, 2, 3, 1).reshape(-1, 3) # [262144, 3]

    n_terms = 28 if basis_type == "28term" else 19
    reg_eye = 1e-2 * torch.eye(n_terms, device=device, dtype=torch.float32)
    table = {}

    for wt in range(1, 11):
        for depth in range(5, 105, 5):
            deg_list = []
            for s in train_scenes:
                p = osp.join(deg_dir, s, f"{s}_{wt}_{depth}.png")
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                deg_list.append(torch.from_numpy(rgb).permute(2, 0, 1))
            deg_gpu = torch.stack(deg_list, dim=0).to(device)
            deg_128 = F.avg_pool2d(deg_gpu, kernel_size=8, stride=8) # [16, 3, 128, 128]

            if basis_type == "28term":
                feats = build_28term_features_gpu(deg_128, wt, depth)
            else:
                feats = build_rsuigm_poly_features_gpu(deg_128)

            X_feats = feats.reshape(-1, n_terms)

            XtX = torch.matmul(X_feats.t(), X_feats) + reg_eye
            XtY = torch.matmul(X_feats.t(), Y_target)
            W = torch.linalg.solve(XtX, XtY)
            table[(wt, depth)] = W

    return table

def evaluate_table_screening(
    table: Dict[Tuple[int, int], torch.Tensor],
    fold_idx: int,
    manifest: Dict[str, Any],
    basis_type: str = "19term",
    unsharp_alpha: float = 1.30,
    unsharp_sigma: float = 2.0,
    data_root: str = "data/cached_1024",
    device: torch.device = None
) -> Tuple[RestorationEvaluator, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f_meta = manifest["folds"][f"fold_{fold_idx}"]
    val_scenes = f_meta["val_scenes"]

    clean_dir = osp.join(data_root, "clean")
    deg_dir = osp.join(data_root, "degraded")

    evaluator = RestorationEvaluator()
    t0 = time.time()

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

            W = table[(wt, dp)]
            with torch.no_grad():
                if basis_type == "28term":
                    feats = build_28term_features_gpu(x_t, wt, dp)
                else:
                    feats = build_rsuigm_poly_features_gpu(x_t)

                b, h_w, c = feats.shape
                res_flat = torch.matmul(feats.reshape(-1, c), W).clamp_(0.0, 1.0)
                res_1024 = res_flat.reshape(b, 1024, 1024, 3).permute(0, 3, 1, 2)

            res_np = (res_1024.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(res_np, (0, 0), unsharp_sigma)
            beta = 1.0 - unsharp_alpha
            sharp_np = np.clip(cv2.addWeighted(res_np, unsharp_alpha, blur, beta, 0), 0, 255).astype(np.float32) / 255.0
            sharp_t = torch.from_numpy(sharp_np).permute(2, 0, 1).unsqueeze(0).to(device)

            evaluator.update(sharp_t, c_t)

    return evaluator, time.time() - t0

def run_experiment_2_1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("EXPERIMENT 2.1: RSUIGM CALIBRATION & BASIS ABLATION ON SCREENING FOLDS 0 & 1")
    print("=" * 76)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    # 1. Option 1: 16-Scene Equal-Weighted 19-Term Fit
    print("\n--- [Option 1] 16-Scene Equal-Weighted 19-Term Fit ---")
    pooled_eval_19 = RestorationEvaluator()
    for f_idx in [0, 1]:
        train_sc = manifest["folds"][f"fold_{f_idx}"]["train_scenes"]
        print(f"  Fitting Fold {f_idx} 19-term table on {len(train_sc)} scenes...")
        t_start = time.time()
        tbl_19 = fit_16scene_rsuigm_table(train_sc, basis_type="19term", device=device)
        print(f"    Table fitted in {time.time() - t_start:.1f}s. Evaluating held-out scenes...")
        eval_f, elapsed = evaluate_table_screening(tbl_19, f_idx, manifest, basis_type="19term", device=device)
        res_f = eval_f.compute()
        print(f"    Fold {f_idx} metrics (800 imgs in {elapsed:.1f}s): PSNR={res_f['PSNR_dB']:.2f} dB | SSIM={res_f['SSIM']:.4f} | Sharp={res_f['sharpness_ratio']:.4f}")
        # Pool
        pooled_eval_19.psnr_list.extend(eval_f.psnr_list)
        pooled_eval_19.psnr_norm_list.extend(eval_f.psnr_norm_list)
        pooled_eval_19.ssim_list.extend(eval_f.ssim_list)
        pooled_eval_19.sharpness_list.extend(eval_f.sharpness_list)

    res_opt1_pooled = pooled_eval_19.compute()
    print("\nOption 1 Pooled Screening Folds (0-1) [1,600 images]:")
    print(f"  PSNR:      {res_opt1_pooled['PSNR_dB']:.2f} dB")
    print(f"  SSIM:      {res_opt1_pooled['SSIM']:.4f}")
    print(f"  Sharpness: {res_opt1_pooled['sharpness_ratio']:.4f}")
    print(f"  Q_FR:      {res_opt1_pooled['Q_rest']:.4f}")

    # 2. Option 2: 16-Scene 28-Term Spatial-Context Basis Fit
    print("\n--- [Option 2] 16-Scene 28-Term Spatial-Context Basis Fit ---")
    pooled_eval_28 = RestorationEvaluator()
    for f_idx in [0, 1]:
        train_sc = manifest["folds"][f"fold_{f_idx}"]["train_scenes"]
        print(f"  Fitting Fold {f_idx} 28-term table on {len(train_sc)} scenes...")
        t_start = time.time()
        tbl_28 = fit_16scene_rsuigm_table(train_sc, basis_type="28term", device=device)
        print(f"    Table fitted in {time.time() - t_start:.1f}s. Evaluating held-out scenes...")
        eval_f, elapsed = evaluate_table_screening(tbl_28, f_idx, manifest, basis_type="28term", device=device)
        res_f = eval_f.compute()
        print(f"    Fold {f_idx} metrics (800 imgs in {elapsed:.1f}s): PSNR={res_f['PSNR_dB']:.2f} dB | SSIM={res_f['SSIM']:.4f} | Sharp={res_f['sharpness_ratio']:.4f}")
        # Pool
        pooled_eval_28.psnr_list.extend(eval_f.psnr_list)
        pooled_eval_28.psnr_norm_list.extend(eval_f.psnr_norm_list)
        pooled_eval_28.ssim_list.extend(eval_f.ssim_list)
        pooled_eval_28.sharpness_list.extend(eval_f.sharpness_list)

    res_opt2_pooled = pooled_eval_28.compute()
    print("\nOption 2 Pooled Screening Folds (0-1) [1,600 images]:")
    print(f"  PSNR:      {res_opt2_pooled['PSNR_dB']:.2f} dB")
    print(f"  SSIM:      {res_opt2_pooled['SSIM']:.4f}")
    print(f"  Sharpness: {res_opt2_pooled['sharpness_ratio']:.4f}")
    print(f"  Q_FR:      {res_opt2_pooled['Q_rest']:.4f}")

    # Comparison against Locked Baseline Floors
    with open("reports/track1_research/v7_locked_baseline.json") as f:
        locked = json.load(f)
    base_floors = locked["pooled_screening_folds_0_1"]["hard_floors"]
    base_rest = locked["pooled_screening_folds_0_1"]["restoration"]

    print("\n" + "=" * 76)
    print("EXPERIMENT 2.1 VERIFICATION AGAINST LOCKED SCREENING FLOORS:")
    print(f"  Locked Baseline (Screening 0-1): PSNR={base_rest['PSNR_dB']:.2f} dB | SSIM={base_rest['SSIM']:.4f} | Sharp={base_floors['Sharpness_OOF_Base']:.4f} | Q_FR={base_rest['Q_rest']:.4f}")
    print(f"  Option 1 (16-Scene 19-Term):     PSNR={res_opt1_pooled['PSNR_dB']:.2f} dB | SSIM={res_opt1_pooled['SSIM']:.4f} | Sharp={res_opt1_pooled['sharpness_ratio']:.4f} | Q_FR={res_opt1_pooled['Q_rest']:.4f}")
    print(f"  Option 2 (16-Scene 28-Term):     PSNR={res_opt2_pooled['PSNR_dB']:.2f} dB | SSIM={res_opt2_pooled['SSIM']:.4f} | Sharp={res_opt2_pooled['sharpness_ratio']:.4f} | Q_FR={res_opt2_pooled['Q_rest']:.4f}")
    print("=" * 76)

    # Save results ledger
    report = {
        "experiment": "2.1_rsuigm_calibration_and_basis_ablation",
        "evaluation_set": "pooled_screening_folds_0_1 (1,600 images)",
        "locked_baseline_screening": base_rest,
        "option_1_16scene_19term": res_opt1_pooled,
        "option_2_16scene_28term": res_opt2_pooled,
        "deltas_vs_baseline": {
            "opt1_psnr_delta": round(res_opt1_pooled['PSNR_dB'] - base_rest['PSNR_dB'], 4),
            "opt1_ssim_delta": round(res_opt1_pooled['SSIM'] - base_rest['SSIM'], 4),
            "opt1_sharp_delta": round(res_opt1_pooled['sharpness_ratio'] - base_floors['Sharpness_OOF_Base'], 4),
            "opt1_qfr_delta": round(res_opt1_pooled['Q_rest'] - base_rest['Q_rest'], 4),
            "opt2_psnr_delta": round(res_opt2_pooled['PSNR_dB'] - base_rest['PSNR_dB'], 4),
            "opt2_ssim_delta": round(res_opt2_pooled['SSIM'] - base_rest['SSIM'], 4),
            "opt2_sharp_delta": round(res_opt2_pooled['sharpness_ratio'] - base_floors['Sharpness_OOF_Base'], 4),
            "opt2_qfr_delta": round(res_opt2_pooled['Q_rest'] - base_rest['Q_rest'], 4),
        }
    }
    with open("reports/track1_research/exp2_1_results.json", "w") as fp:
        json.dump(report, fp, indent=2)

    return report

if __name__ == '__main__':
    run_experiment_2_1()
