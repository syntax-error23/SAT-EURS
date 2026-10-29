"""
Experiment 2.2 Confirmation: Evaluate Frozen Best Setting (cfg1_unsharp_1.34)
on Confirmation Folds 2-4 (2,400 images) and All 5 Folds (4,000 images).

Locked Hard Floors to Beat:
- Confirmation Folds (2-4): PSNR >= 23.43 dB, SSIM >= 0.7055, Sharpness >= 0.5152, Q_FR >= 0.5588
- All 5 Folds (0-4):        PSNR >= 23.78 dB, SSIM >= 0.7279, Sharpness >= 0.5140, Q_FR >= 0.5765
"""

import os
import os.path as osp
import sys
import json
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")

from metrics.restoration_metrics import RestorationEvaluator
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu
from tools.exp2_1_rsuigm_calibration import fit_16scene_rsuigm_table

def confirm_restoration_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("CONFIRMING FROZEN RESTORATION SETTING (cfg1_unsharp_1.34) ON FOLDS 2-4 & ALL 5 FOLDS")
    print("=" * 76)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)

    with open("reports/track1_research/v7_locked_baseline.json") as f:
        locked = json.load(f)

    data_root = "data/cached_1024"
    clean_dir = osp.join(data_root, "clean")
    deg_dir = osp.join(data_root, "degraded")

    # Load / fit 16-scene 19-term tables for all 5 folds
    tables = {}
    for f_idx in range(5):
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

    eval_confirm = RestorationEvaluator()
    eval_all = RestorationEvaluator()

    # Load screening (0-1) metrics from exp2_2 screening
    with open("reports/track1_research/exp2_2_screening_results.json") as f:
        scr_data = json.load(f)
    scr_res = scr_data["grid_results"]["cfg1_unsharp_1.34"]

    # Folds 2, 3, 4 evaluation
    t0 = time.time()
    for f_idx in [2, 3, 4]:
        val_scenes = manifest["folds"][f"fold_{f_idx}"]["val_scenes"]
        tbl = tables[f_idx]
        print(f"\nEvaluating Confirmation Fold {f_idx} held-out scenes: {val_scenes}...")

        f_eval = RestorationEvaluator()
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
                blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
                beta = 1.0 - 1.34
                sharp_np = np.clip(cv2.addWeighted(res_np, 1.34, blur, beta, 0), 0, 255).astype(np.float32) / 255.0
                sharp_t = torch.from_numpy(sharp_np).permute(2, 0, 1).unsqueeze(0).to(device)

                f_eval.update(sharp_t, c_t)
                eval_confirm.update(sharp_t, c_t)
                eval_all.update(sharp_t, c_t)

        f_res = f_eval.compute()
        print(f"  Fold {f_idx} metrics: PSNR={f_res['PSNR_dB']:.2f} dB | SSIM={f_res['SSIM']:.4f} | Sharp={f_res['sharpness_ratio']:.4f}")

    # Add screening samples to eval_all
    # We load screening raw lists from scratch or combine
    print(f"\nEvaluated 2,400 confirmation images in {time.time() - t0:.1f}s.")
    res_confirm = eval_confirm.compute()

    base_confirm = locked["pooled_confirmation_folds_2_4"]["restoration"]
    base_all = locked["pooled_all_5_folds_0_4"]["restoration"]

    print("\n" + "=" * 76)
    print("CONFIRMATION EVALUATION AGAINST LOCKED FLOORS (FOLDS 2-4, 2,400 IMAGES):")
    print(f"  Baseline (Folds 2-4): PSNR={base_confirm['PSNR_dB']:.2f} dB | SSIM={base_confirm['SSIM']:.4f} | Sharp={base_confirm['sharpness_ratio']:.4f} | Q_FR={base_confirm['Q_rest']:.4f}")
    print(f"  New Fit  (Folds 2-4): PSNR={res_confirm['PSNR_dB']:.2f} dB | SSIM={res_confirm['SSIM']:.4f} | Sharp={res_confirm['sharpness_ratio']:.4f} | Q_FR={res_confirm['Q_rest']:.4f}")
    print("=" * 76)

    # Save to JSON
    report = {
        "frozen_config": "16scene_19term_unsharp_1.34",
        "screening_folds_0_1": scr_res,
        "confirmation_folds_2_4": res_confirm,
        "deltas_vs_baseline_confirm": {
            "psnr_delta": round(res_confirm['PSNR_dB'] - base_confirm['PSNR_dB'], 4),
            "ssim_delta": round(res_confirm['SSIM'] - base_confirm['SSIM'], 4),
            "sharp_delta": round(res_confirm['sharpness_ratio'] - base_confirm['sharpness_ratio'], 4),
            "qfr_delta": round(res_confirm['Q_rest'] - base_confirm['Q_rest'], 4),
        }
    }
    with open("reports/track1_research/exp2_2_confirmation_results.json", "w") as fp:
        json.dump(report, fp, indent=2)

    return report

if __name__ == '__main__':
    confirm_restoration_pipeline()
