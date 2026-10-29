"""
Audit Script: The [:8] Restoration Calibration Question
Compares:
1. Table A: fit on f_meta["train_scenes"][:8] (used in DINOv2-SOTA, which yielded PSNR 25.44 dB on Dev)
2. Table B: fit on clean_scenes[:8] (used in #18700 MultiScale, which yielded PSNR 25.07 dB on Dev)
3. Table C: fit on all 16 train_scenes of the fold (equal weighted)
Evaluates on fold_0 held-out scenes (800 images across fish1, fish11, fish12, seahorse1).
"""

import os
import os.path as osp
import sys
import json
import time
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from metrics.restoration_metrics import RestorationEvaluator
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu
from tools.train_track1_screening import apply_rsuigm_batch_gpu

def fit_custom_rsuigm_table(scenes_to_fit: list, cached_dir: str = "data/cached_1024", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_dir = osp.join(cached_dir, "clean")
    deg_dir = osp.join(cached_dir, "degraded")

    clean_list = []
    for s in scenes_to_fit:
        c_bgr = cv2.imread(osp.join(clean_dir, f"{s}.png"), cv2.IMREAD_COLOR)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_list.append(torch.from_numpy(c_rgb).permute(2, 0, 1))

    clean_gpu = torch.stack(clean_list, dim=0).to(device)
    clean_128 = F.avg_pool2d(clean_gpu, kernel_size=8, stride=8)
    Y_target = clean_128.permute(0, 2, 3, 1).reshape(-1, 3)

    table = {}
    reg_eye = 1e-3 * torch.eye(19, device=device, dtype=torch.float32)

    for wt in range(1, 11):
        for depth in range(5, 105, 5):
            deg_list = []
            for s in scenes_to_fit:
                p = osp.join(deg_dir, s, f"{s}_{wt}_{depth}.png")
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                deg_list.append(torch.from_numpy(rgb).permute(2, 0, 1))
            deg_gpu = torch.stack(deg_list, dim=0).to(device)
            deg_128 = F.avg_pool2d(deg_gpu, kernel_size=8, stride=8)

            feats = build_rsuigm_poly_features_gpu(deg_128)
            X_feats = feats.reshape(-1, 19)

            XtX = torch.matmul(X_feats.t(), X_feats) + reg_eye
            XtY = torch.matmul(X_feats.t(), Y_target)
            W = torch.linalg.solve(XtX, XtY)
            table[(wt, depth)] = W

    return table

def evaluate_table_on_val_scenes(table, val_scenes, sample_stride=4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = RestorationEvaluator()
    cached_dir = "data/cached_1024"
    clean_dir = osp.join(cached_dir, "clean")
    deg_dir = osp.join(cached_dir, "degraded")

    for s in val_scenes:
        c_bgr = cv2.imread(osp.join(clean_dir, f"{s}.png"), cv2.IMREAD_COLOR)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        c_t = torch.from_numpy(c_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        sc_deg = osp.join(deg_dir, s)
        deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])[::sample_stride]

        for df in deg_files:
            parts = osp.splitext(df)[0].split("_")
            wt = int(parts[1])
            dp = int(parts[2])

            bgr = cv2.imread(osp.join(sc_deg, df), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([dp], device=device)

            with torch.no_grad():
                res = apply_rsuigm_batch_gpu(x_t, wt_t, dp_t, table)

            # Apply standard unsharp
            res_np = (res.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
            sharp_np = np.clip(cv2.addWeighted(res_np, 1.30, blur, -0.30, 0), 0, 255).astype(np.float32) / 255.0
            sharp_t = torch.from_numpy(sharp_np).permute(2, 0, 1).unsqueeze(0).to(device)

            evaluator.update(sharp_t, c_t)

    return evaluator.compute()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("AUDITING RSUIGM RESTORATION SCENE SELECTION")
    print("=" * 76)

    manifest = json.load(open("splits/manifest_5fold.json"))
    f0_train = manifest["folds"]["fold_0"]["train_scenes"]
    f0_val = manifest["folds"]["fold_0"]["val_scenes"]

    scenes_A = f0_train[:8]
    scenes_B = sorted(list(manifest["folds"]["fold_0"]["train_scenes"]))[:8]
    # In #18700, clean_scenes was glob.glob('data/cached_1024/clean/*.png')
    all_clean = sorted([osp.splitext(osp.basename(f))[0] for f in glob.glob("data/cached_1024/clean/*.png")])
    scenes_18700 = all_clean[:8]
    scenes_all16 = f0_train # all 16 train scenes

    print(f"Table A (DINOv2-SOTA F0[:8]): {scenes_A}")
    print(f"Table B (#18700 Glob[:8]):     {scenes_18700}")
    print(f"Table C (All 16 Train Scenes): {scenes_all16}")

    print("\nFitting Table A...")
    tbl_A = fit_custom_rsuigm_table(scenes_A, device=device)
    res_A = evaluate_table_on_val_scenes(tbl_A, f0_val, sample_stride=4, device=device)
    print(f"  Table A: PSNR={res_A['PSNR_dB']:.2f} dB | SSIM={res_A['SSIM']:.4f} | Sharp={res_A['sharpness_ratio']:.4f} | Q_FR={res_A['Q_rest']:.4f}")

    print("\nFitting Table B (#18700 selection)...")
    tbl_B = fit_custom_rsuigm_table(scenes_18700, device=device)
    res_B = evaluate_table_on_val_scenes(tbl_B, f0_val, sample_stride=4, device=device)
    print(f"  Table B: PSNR={res_B['PSNR_dB']:.2f} dB | SSIM={res_B['SSIM']:.4f} | Sharp={res_B['sharpness_ratio']:.4f} | Q_FR={res_B['Q_rest']:.4f}")

    print("\nFitting Table C (All 16 Train Scenes)...")
    tbl_C = fit_custom_rsuigm_table(scenes_all16, device=device)
    res_C = evaluate_table_on_val_scenes(tbl_C, f0_val, sample_stride=4, device=device)
    print(f"  Table C: PSNR={res_C['PSNR_dB']:.2f} dB | SSIM={res_C['SSIM']:.4f} | Sharp={res_C['sharpness_ratio']:.4f} | Q_FR={res_C['Q_rest']:.4f}")
    print("=" * 76)

if __name__ == "__main__":
    main()
