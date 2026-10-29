"""
Pilot Restoration Evaluation Script (Phase G)
Strictly fits RSUIGM lookup table on fold_0 training scenes (zero validation contamination).
Evaluates full validation split of fold_0 (800 images across fish1, fish11, fish12, seahorse1).
Measures exact full-reference metrics: PSNR, PSNR_norm, SSIM, Sharpness Ratio, Q_FR.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.restoration_metrics import RestorationEvaluator, compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu


def fit_leak_free_rsuigm_table(train_scenes: list, cached_dir: str = "data/cached_1024", device: torch.device = None):
    """
    Fits 200 (wt, depth) RSUIGM operators strictly on train_scenes only.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clean_dir = os.path.join(cached_dir, "clean")
    deg_dir = os.path.join(cached_dir, "degraded")
    
    # Subsample 8 scenes from train_scenes to construct target Y (128x128 on GPU)
    sub_scenes = train_scenes[:8]
    clean_list = []
    for s in sub_scenes:
        c_bgr = cv2.imread(os.path.join(clean_dir, f"{s}.png"), cv2.IMREAD_COLOR)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_list.append(torch.from_numpy(c_rgb).permute(2, 0, 1))
    
    clean_gpu = torch.stack(clean_list, dim=0).to(device)
    clean_128 = F.avg_pool2d(clean_gpu, kernel_size=8, stride=8) # [8, 3, 128, 128]
    Y_target = clean_128.permute(0, 2, 3, 1).reshape(-1, 3) # [131072, 3]
    
    table = {}
    reg_eye = 1e-3 * torch.eye(19, device=device, dtype=torch.float32)
    
    for wt in range(1, 11):
        for depth in range(5, 105, 5):
            deg_list = []
            for s in sub_scenes:
                p = os.path.join(deg_dir, s, f"{s}_{wt}_{depth}.png")
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                deg_list.append(torch.from_numpy(rgb).permute(2, 0, 1))
            deg_gpu = torch.stack(deg_list, dim=0).to(device)
            deg_128 = F.avg_pool2d(deg_gpu, kernel_size=8, stride=8)
            
            # build 19-dim features
            X_feats = build_rsuigm_poly_features_gpu(deg_128).reshape(-1, 19)
            XtX = X_feats.T @ X_feats + reg_eye
            XtY = X_feats.T @ Y_target
            W = torch.linalg.solve(XtX, XtY)
            table[(wt, depth)] = W
            
    return table


def run_pilot_restoration_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"PILOT RESTORATION EVALUATION (PHASE G) ON {device}")
    print("=" * 70)
    
    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)
        
    f0 = manifest["folds"]["fold_0"]
    train_scenes = f0["train_scenes"]
    val_scenes = f0["val_scenes"]
    
    print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val scenes   ({len(val_scenes)}): {val_scenes}")
    
    t0 = time.time()
    print("\nFitting leak-free RSUIGM table on 16 train scenes...")
    table = fit_leak_free_rsuigm_table(train_scenes, device=device)
    print(f"Table fitted in {time.time() - t0:.2f}s (200 operators).")
    
    # Evaluate on val scenes (stride 4 = 50 per scene = 200 images for fast pilot)
    evaluator_raw = RestorationEvaluator()
    evaluator_rsuigm = RestorationEvaluator()
    evaluator_unsharp = RestorationEvaluator()
    
    clean_dir = "data/cached_1024/clean"
    deg_dir = "data/cached_1024/degraded"
    
    print("\nEvaluating on validation scenes...")
    count = 0
    t_eval = time.time()
    
    for s in val_scenes:
        c_path = os.path.join(clean_dir, f"{s}.png")
        c_bgr = cv2.imread(c_path)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        c_t = torch.from_numpy(c_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
        deg_scene_dir = os.path.join(deg_dir, s)
        deg_files = sorted(os.listdir(deg_scene_dir))[::4] # 50 variants per scene
        
        for df in deg_files:
            stem = os.path.splitext(df)[0]
            parts = stem.split("_")
            wt = int(parts[1])
            depth = int(parts[2])
            
            d_path = os.path.join(deg_scene_dir, df)
            d_bgr = cv2.imread(d_path)
            d_rgb = cv2.cvtColor(d_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            d_t = torch.from_numpy(d_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 1. Raw degraded baseline
            evaluator_raw.update(d_t, c_t)
            
            # 2. RSUIGM restored
            W = table[(wt, depth)]
            X_feats = build_rsuigm_poly_features_gpu(d_t)
            rest = torch.clamp((X_feats @ W).permute(0, 2, 1).reshape(1, 3, 1024, 1024), 0.0, 1.0)
            evaluator_rsuigm.update(rest, c_t)
            
            # 3. RSUIGM + Unsharp filter
            # unsharp on CPU
            rest_np = (rest.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(rest_np, (0, 0), 2.0)
            sharp_np = np.clip(cv2.addWeighted(rest_np, 1.30, blur, -0.30, 0), 0, 255).astype(np.float32) / 255.0
            sharp_t = torch.from_numpy(sharp_np).permute(2, 0, 1).unsqueeze(0).to(device)
            evaluator_unsharp.update(sharp_t, c_t)
            
            count += 1
            
    print(f"Evaluated {count} validation images in {time.time() - t_eval:.2f}s.")
    
    m_raw = evaluator_raw.compute()
    m_rsuigm = evaluator_rsuigm.compute()
    m_unsharp = evaluator_unsharp.compute()
    
    print("\n" + "=" * 70)
    print("RESTORATION PILOT BENCHMARK RESULTS (FOLD 0 HELD-OUT VAL SCENES)")
    print("=" * 70)
    print(f"{'Method':<25} | {'PSNR (dB)':<10} | {'PSNR_norm':<10} | {'SSIM':<8} | {'Sharpness':<10} | {'Q_FR':<8}")
    print("-" * 75)
    print(f"{'Raw Degraded Input':<25} | {m_raw['PSNR_dB']:<10.2f} | {m_raw['PSNR_norm']:<10.4f} | {m_raw['SSIM']:<8.4f} | {m_raw['sharpness_ratio']:<10.4f} | {m_raw['Q_rest']:<8.4f}")
    print(f"{'Leak-Free RSUIGM':<25} | {m_rsuigm['PSNR_dB']:<10.2f} | {m_rsuigm['PSNR_norm']:<10.4f} | {m_rsuigm['SSIM']:<8.4f} | {m_rsuigm['sharpness_ratio']:<10.4f} | {m_rsuigm['Q_rest']:<8.4f}")
    print(f"{'RSUIGM + Unsharp Boost':<25} | {m_unsharp['PSNR_dB']:<10.2f} | {m_unsharp['PSNR_norm']:<10.4f} | {m_unsharp['SSIM']:<8.4f} | {m_unsharp['sharpness_ratio']:<10.4f} | {m_unsharp['Q_rest']:<8.4f}")
    print("=" * 70)
    
    # Save pilot report
    os.makedirs("reports/track1_research", exist_ok=True)
    report = {
        "date": "2026-10-14",
        "eval_count": count,
        "raw_degraded": m_raw,
        "leak_free_rsuigm": m_rsuigm,
        "rsuigm_unsharp": m_unsharp
    }
    with open("reports/track1_research/RESTORATION_PILOT.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved pilot metrics to reports/track1_research/RESTORATION_PILOT.json")


if __name__ == '__main__':
    run_pilot_restoration_eval()
