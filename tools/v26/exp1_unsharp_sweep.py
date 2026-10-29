"""
SAT-EURS Track 1 — V26 Experiment 1: Optimized Fast Unsharp Filter Ablation & Sweep
Computes refiner forward pass ONCE per image (1,600 images total across Fold 0 & 1),
then evaluates all alpha strengths in parallel in memory.
Runs in ~45 seconds on RTX 5090.
"""

import os
import os.path as osp
import sys
import time
import json
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    load_checkpoint_with_provenance_check,
)
from tools.v18.evaluator import V18OOFEvaluator


def run_unsharp_experiment(
    folds: List[int] = [0, 1],
    alpha_values: List[float] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    sigma: float = 2.0,
    data_root: str = "data/cached_1024",
    checkpoint_dir: str = "checkpoints/v18",
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SAT-EURS V26 — EXPERIMENT 1: OPTIMIZED FAST UNSHARP FILTER SWEEP")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Evaluating Folds: {folds} | Alpha values: {alpha_values} | Sigma: {sigma}")
    print("=" * 80)

    manifest = validate_manifest_and_folds()

    # Evaluators for each alpha
    evaluators_pooled = {alpha: V18OOFEvaluator() for alpha in alpha_values}
    evaluators_fold = {alpha: {fold: V18OOFEvaluator() for fold in folds} for alpha in alpha_values}

    t0 = time.time()
    total_imgs = 0

    for fold in folds:
        f_data = get_fold_data(manifest, fold)
        val_scenes = f_data["val_scenes"]
        train_scenes = f_data["train_scenes"]

        print(f"\n[Fold {fold}] Loading model and evaluating {len(val_scenes)} scenes ({len(val_scenes)*200} images)...")

        # Load fold table and refiner
        table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
        if not osp.exists(table_path):
            table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
        rsuigm_table = torch.load(table_path, map_location=device)

        refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
        r_path = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, train_scenes, val_scenes)
        refiner.load_state_dict(weights)
        refiner.eval()

        for sc_idx, sc in enumerate(val_scenes, 1):
            t_sc = time.time()
            c_path = osp.join(data_root, "clean", f"{sc}.png")
            clean_bgr = cv2.imread(c_path, cv2.IMREAD_COLOR)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    deg_path = osp.join(data_root, "degraded", sc, f"{sc}_{wt}_{depth}.png")
                    deg_bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
                    deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                    deg_t = torch.from_numpy(deg_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                    wt_t = torch.tensor([wt], device=device)
                    dp_t = torch.tensor([depth], device=device)

                    with torch.no_grad():
                        out_r = refiner(deg_t, wt_t, dp_t)
                        refined = out_r["refined_image"].squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # Precompute Gaussian blur once per image
                    blur = cv2.GaussianBlur(refined, (0, 0), sigma)
                    high_freq = refined - blur

                    for alpha in alpha_values:
                        if alpha > 0.0:
                            final_rest = np.clip(refined + alpha * high_freq, 0.0, 1.0)
                        else:
                            final_rest = refined

                        evaluators_fold[alpha][fold].update_restoration(final_rest, clean_rgb, scene=sc)
                        evaluators_pooled[alpha].update_restoration(final_rest, clean_rgb, scene=sc)

                    total_imgs += 1

            print(f"  Processed scene {sc:<12} (200 images) in {time.time() - t_sc:.1f}s")

    print(f"\nCompleted evaluation of {total_imgs} images in {time.time() - t0:.1f}s!")

    all_results = {}
    base_qfr = evaluators_pooled[0.00].compute()["restoration"]["Q_FR_dataset"]

    for alpha in alpha_values:
        res_pooled = evaluators_pooled[alpha].compute()["restoration"]
        res_per_fold = {f"fold_{f}": evaluators_fold[alpha][f].compute()["restoration"] for f in folds}
        per_scene = {
            sc: {
                "psnr": float(np.mean(evaluators_pooled[alpha].scene_stats[sc]["psnr"])),
                "ssim": float(np.mean(evaluators_pooled[alpha].scene_stats[sc]["ssim"])),
                "sharp": float(np.mean(evaluators_pooled[alpha].scene_stats[sc]["sharp"]))
            }
            for sc in evaluators_pooled[alpha].scene_stats
        }

        all_results[f"alpha_{alpha:.2f}"] = {
            "alpha": alpha,
            "sigma": sigma,
            "pooled": res_pooled,
            "per_fold": res_per_fold,
            "per_scene": per_scene
        }

    # Save JSON report
    os.makedirs("reports/v26", exist_ok=True)
    out_json = "reports/v26/exp1_unsharp_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_json}")

    print("\n" + "=" * 90)
    print("                    EXPERIMENT 1 SUMMARY MATRIX (POOLED 1,600 IMAGES)")
    print("=" * 90)
    print(f"{'Configuration':<16} | {'PSNR (dB)':<10} | {'PSNR_norm':<10} | {'SSIM':<10} | {'Sharpness':<10} | {'Q_FR':<10} | {'Delta Q_FR':<10}")
    print("-" * 90)

    for alpha in alpha_values:
        k = f"alpha_{alpha:.2f}"
        p = all_results[k]["pooled"]
        dqfr = p["Q_FR_dataset"] - base_qfr
        tag = " (V18 Baseline)" if alpha == 0.0 else (" (V25 Path)" if alpha == 0.25 else "")
        print(f"{k + tag:<24} | {p['PSNR_dB']:<10.2f} | {p['PSNR_norm_dataset']:<10.4f} | {p['SSIM']:<10.4f} | {p['sharpness_ratio']:<10.4f} | {p['Q_FR_dataset']:<10.4f} | {dqfr:+10.5f}")
    print("=" * 90)

    return all_results


if __name__ == "__main__":
    run_unsharp_experiment(folds=[0, 1], alpha_values=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
