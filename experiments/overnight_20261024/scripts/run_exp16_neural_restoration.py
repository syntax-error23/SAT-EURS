"""
SAT-EURS Track 1 — EXP-16: Benchmark Existing Neural Restorers
Evaluates and compares:
  1. V18 Conditioned Restoration Refiner Ensemble (5 folds)
  2. V26 Deep Bilateral Restorer (checkpoints/v26/restorer_full20_best.pt)
Evaluated across all 20 EURS scenes against clean reference ground truth.
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
from models.v26.deep_bilateral_restorer import DeepBilateralRestorerV26
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data, V18CanonicalPreprocessor
from metrics.restoration_metrics import compute_psnr_and_norm
from scratch.fast_restoration_eval import precompute_clean_stats, fast_ssim, fast_sharpness


def run_neural_restorer_benchmark(sample_stride: int = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("SAT-EURS TRACK 1 — EXP-16: NEURAL RESTORATION ARCHITECTURE BENCHMARK")
    print("=" * 95)

    manifest = validate_manifest_and_folds()
    base_table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    # 1. Load V18 Refiners
    refiner_ckpts = {
        0: 'checkpoints/v18/fold_0/refiner_best.pt',
        1: 'checkpoints/v18/fold_1/refiner_best.pt',
        2: 'checkpoints/v18/fold_2/refiner_best.pt',
        3: 'checkpoints/v18/fold_3/refiner_best.pt',
        4: 'checkpoints/v18/fold_4/refiner_best.pt',
    }
    v18_refiners = {}
    for f in range(5):
        r = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpts[f], map_location=device)
        r.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        r.eval()
        v18_refiners[f] = r

    # 2. Load V26 Deep Bilateral Restorer
    v26_restorer = DeepBilateralRestorerV26().to(device)
    v26_restorer.register_rsuigm_table(base_table)
    v26_ckpt = torch.load("checkpoints/v26/restorer_full20_best.pt", map_location=device)
    v26_restorer.load_state_dict(v26_ckpt["state_dict"] if "state_dict" in v26_ckpt else v26_ckpt)
    v26_restorer.eval()

    models = ["v18_refiner_ensemble", "v26_deep_bilateral"]
    metrics_acc = {m: {"psnr": [], "pnorm": [], "ssim": [], "sharp": []} for m in models}

    t0 = time.time()
    total_imgs = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        print(f"Fold {fold} evaluating {val_scenes}...", flush=True)

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
                    # 1. V18 Refiner
                    out_v18 = v18_refiners[fold](raw_pad.float(), wt_t, dp_t)
                    pad_v18 = out_v18["refined_image"]

                    # 2. V26 Bilateral Restorer
                    out_v26 = v26_restorer(raw_pad.float(), wt_t, dp_t)
                    pad_v26 = out_v26["restored_image"]

                # Unpad and apply standard unsharp alpha=0.10, sigma=2.0
                unpad_v18 = preprocessor.unpad_output(pad_v18.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                unpad_v26 = preprocessor.unpad_output(pad_v26.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)

                if unpad_v18.shape[:2] != (gh, gw):
                    unpad_v18 = cv2.resize(unpad_v18, (gw, gh), interpolation=cv2.INTER_LINEAR)
                    unpad_v26 = cv2.resize(unpad_v26, (gw, gh), interpolation=cv2.INTER_LINEAR)

                unpad_v18 = np.clip(unpad_v18, 0.0, 1.0)
                unpad_v26 = np.clip(unpad_v26, 0.0, 1.0)

                blur_18 = cv2.GaussianBlur(unpad_v18, (0, 0), 2.0)
                final_18 = np.clip(unpad_v18 + 0.10 * (unpad_v18 - blur_18), 0.0, 1.0)

                blur_26 = cv2.GaussianBlur(unpad_v26, (0, 0), 2.0)
                final_26 = np.clip(unpad_v26 + 0.10 * (unpad_v26 - blur_26), 0.0, 1.0)

                # Compute metrics
                for m_name, img in [("v18_refiner_ensemble", final_18), ("v26_deep_bilateral", final_26)]:
                    psnr, pnorm = compute_psnr_and_norm(img, clean_rgb)
                    ssim = fast_ssim(img, clean_rgb, mu_t_list, sigma_t_sq_list)
                    sharp = fast_sharpness(img, var_t)

                    metrics_acc[m_name]["psnr"].append(psnr)
                    metrics_acc[m_name]["pnorm"].append(pnorm)
                    metrics_acc[m_name]["ssim"].append(ssim)
                    metrics_acc[m_name]["sharp"].append(sharp)

                total_imgs += 1

    print(f"\nEvaluated {total_imgs} images across 20 scenes in {time.time() - t0:.1f}s.")

    summary = {}
    print("\n" + "=" * 105)
    print(f"{'Restoration Architecture':<25} | {'PSNR (dB)':<10} | {'PSNR_norm':<10} | {'SSIM':<8} | {'Sharpness':<10} | {'Q_FR':<8} | {'Delta Q_FR'}")
    print("=" * 105)
    base_q = None
    for m in models:
        m_psnr = float(np.mean(metrics_acc[m]["psnr"]))
        m_pnorm = float(np.mean(metrics_acc[m]["pnorm"]))
        m_ssim = float(np.mean(metrics_acc[m]["ssim"]))
        m_sharp = float(np.mean(metrics_acc[m]["sharp"]))
        q_fr = float(0.45 * m_pnorm + 0.45 * m_ssim + 0.10 * m_sharp)

        if base_q is None:
            base_q = q_fr
        dq = q_fr - base_q

        summary[m] = {
            "psnr_db": round(m_psnr, 3),
            "psnr_norm": round(m_pnorm, 4),
            "ssim": round(m_ssim, 4),
            "sharpness": round(m_sharp, 4),
            "Q_FR": round(q_fr, 5),
            "delta_vs_v18": round(dq, 5)
        }
        print(f"{m:<25} | {m_psnr:<10.3f} | {m_pnorm:<10.4f} | {m_ssim:<8.4f} | {m_sharp:<10.4f} | {q_fr:<8.4f} | {dq:+.5f}")
    print("=" * 105)

    out_file = "experiments/overnight_20261024/predictions/exp16_neural_restoration.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_file}")

    return summary


if __name__ == "__main__":
    run_neural_restorer_benchmark(sample_stride=5)
