"""
SAT-EURS Track 1 — EXP-14: Comprehensive Restoration Reconciliation
Measures the progression of restoration fidelity across the full pipeline:
  1. Raw Degraded Inputs
  2. Pure Physical Prior (Scale-Consistent RSUIGM Matrix)
  3. Neural Refiner (No Unsharp)
  4. Full Locked Restoration Pipeline (Neural Refiner + alpha=0.10, sigma=2.0)
Evaluated across all 20 EURS scenes against clean ground truth.
"""

import os
import os.path as osp
import sys
import time
import json
import cv2
import numpy as np
import torch

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data, V18CanonicalPreprocessor
from metrics.restoration_metrics import compute_psnr_and_norm
from scratch.fast_restoration_eval import precompute_clean_stats, fast_ssim, fast_sharpness


def run_restoration_reconciliation(sample_stride: int = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("SAT-EURS TRACK 1 — EXP-14: COMPREHENSIVE RESTORATION RECONCILIATION")
    print("=" * 95)

    manifest = validate_manifest_and_folds()
    base_table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    refiner_ckpts = {
        0: 'checkpoints/v18/fold_0/refiner_best.pt',
        1: 'checkpoints/v18/fold_1/refiner_best.pt',
        2: 'checkpoints/v18/fold_2/refiner_best.pt',
        3: 'checkpoints/v18/fold_3/refiner_best.pt',
        4: 'checkpoints/v18/fold_4/refiner_best.pt',
    }

    stages = ["1_raw_degraded", "2_physical_prior", "3_refiner_no_unsharp", "4_refiner_locked_a10"]
    metrics_acc = {s: {"psnr": [], "pnorm": [], "ssim": [], "sharp": []} for s in stages}

    t0 = time.time()
    total_imgs = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpts[fold], map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        print(f"Fold {fold} reconciling restoration on {val_scenes}...", flush=True)

        for sc in val_scenes:
            clean_path = f"data/cached_1024/clean/{sc}.png"
            clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gh, gw = clean_rgb.shape[:2]

            mu_t_list, sigma_t_sq_list, var_t = precompute_clean_stats(clean_rgb)

            deg_scene_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])[::sample_stride]

            for fname in deg_files:
                parts = osp.splitext(fname)[0].split("_")
                wt, depth = int(parts[1]), int(parts[2])

                bgr = cv2.imread(osp.join(deg_scene_dir, fname))
                prep = preprocessor.prepare_input(bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)

                with torch.inference_mode():
                    # Pure physical prior (base restored RGB)
                    base_rest = refiner.compute_base_restoration(raw_pad.float(), wt_t, dp_t, unsharp=False)
                    # Neural refiner
                    out_r = refiner(raw_pad.float(), wt_t, dp_t)
                    ref_pad = out_r["refined_image"]

                # Unpad all outputs
                raw_unpad = preprocessor.unpad_output(raw_pad.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                base_unpad = preprocessor.unpad_output(base_rest.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                ref_unpad = preprocessor.unpad_output(ref_pad.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)

                # Resize to clean dimensions if needed
                if raw_unpad.shape[:2] != (gh, gw):
                    raw_unpad = cv2.resize(raw_unpad, (gw, gh), interpolation=cv2.INTER_LINEAR)
                    base_unpad = cv2.resize(base_unpad, (gw, gh), interpolation=cv2.INTER_LINEAR)
                    ref_unpad = cv2.resize(ref_unpad, (gw, gh), interpolation=cv2.INTER_LINEAR)

                raw_unpad = np.clip(raw_unpad, 0.0, 1.0)
                base_unpad = np.clip(base_unpad, 0.0, 1.0)
                ref_unpad = np.clip(ref_unpad, 0.0, 1.0)

                # Stage 4: Unsharp masking alpha=0.10, sigma=2.0
                blur = cv2.GaussianBlur(ref_unpad, (0, 0), 2.0)
                locked_unpad = np.clip(ref_unpad + 0.10 * (ref_unpad - blur), 0.0, 1.0)

                stage_images = {
                    "1_raw_degraded": raw_unpad,
                    "2_physical_prior": base_unpad,
                    "3_refiner_no_unsharp": ref_unpad,
                    "4_refiner_locked_a10": locked_unpad,
                }

                for s_name, img in stage_images.items():
                    psnr, pnorm = compute_psnr_and_norm(img, clean_rgb)
                    ssim = fast_ssim(img, clean_rgb, mu_t_list, sigma_t_sq_list)
                    sharp = fast_sharpness(img, var_t)

                    metrics_acc[s_name]["psnr"].append(psnr)
                    metrics_acc[s_name]["pnorm"].append(pnorm)
                    metrics_acc[s_name]["ssim"].append(ssim)
                    metrics_acc[s_name]["sharp"].append(sharp)

                total_imgs += 1

    print(f"\nEvaluated {total_imgs} images across 20 scenes in {time.time() - t0:.1f}s.")

    summary = {}
    print("\n" + "=" * 105)
    print(f"{'Pipeline Stage':<25} | {'PSNR (dB)':<10} | {'PSNR_norm':<10} | {'SSIM':<8} | {'Sharpness':<10} | {'Q_FR':<8} | {'Pts (30%)':<10}")
    print("=" * 105)
    for s_name in stages:
        m_psnr = float(np.mean(metrics_acc[s_name]["psnr"]))
        m_pnorm = float(np.mean(metrics_acc[s_name]["pnorm"]))
        m_ssim = float(np.mean(metrics_acc[s_name]["ssim"]))
        m_sharp = float(np.mean(metrics_acc[s_name]["sharp"]))
        q_fr = float(0.45 * m_pnorm + 0.45 * m_ssim + 0.10 * m_sharp)
        pts = round(0.30 * q_fr, 5)

        summary[s_name] = {
            "psnr_db": round(m_psnr, 3),
            "psnr_norm": round(m_pnorm, 4),
            "ssim": round(m_ssim, 4),
            "sharpness": round(m_sharp, 4),
            "Q_FR": round(q_fr, 5),
            "composite_pts": pts
        }
        print(f"{s_name:<25} | {m_psnr:<10.3f} | {m_pnorm:<10.4f} | {m_ssim:<8.4f} | {m_sharp:<10.4f} | {q_fr:<8.4f} | {pts:<10.5f}")
    print("=" * 105)

    out_file = "experiments/overnight_20261024/predictions/exp14_restoration_reconcile.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_file}")

    return summary


if __name__ == "__main__":
    run_restoration_reconciliation(sample_stride=5)
