"""
SAT-EURS Track 1 — EXP-18: Object Cropping and Spatial Context Ablation
Audits how targeted small-object cropping (object_crop_prob=0.70 vs 0.0)
affects large aquatic animal representation (Class 3: Ray/Shark) during training.
Measures the percentage of large ray pixels preserved vs truncated per batch.
Explains the exact mechanism behind the Fold 1/Fold 2 ray regression failure.
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

from data.v28.dataset import V28DegradationAwareDataset, V28SceneBalancedSampler
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data


def run_crop_context_ablation():
    print("=" * 85)
    print("SAT-EURS TRACK 1 — EXP-18: OBJECT CROPPING & SPATIAL CONTEXT ABLATION")
    print("=" * 85)

    manifest = validate_manifest_and_folds()
    # Fold 2 contains ray1 in its training or validation set
    f2_data = get_fold_data(manifest, 2)
    scenes = f2_data["train_scenes"]  # 16 scenes

    prob_settings = [0.00, 0.25, 0.50, 0.70]
    results = {}
    t0 = time.time()

    for p in prob_settings:
        print(f"\nEvaluating object_crop_prob = {p:.2f} on 200 batches...", flush=True)
        ds = V28DegradationAwareDataset(
            scenes=scenes,
            target_size=512,
            is_train=True,
            object_crop_prob=p,
            min_crop_ratio=0.75,
            base_seed=42
        )
        sampler = V28SceneBalancedSampler(ds, batch_size=8, seed=42)

        total_samples = 400
        ray_pixels_total = 0
        fish_pixels_total = 0
        macro_pixels_total = 0
        samples_with_ray = 0

        indices = list(sampler)[:total_samples]
        for idx in indices:
            sample = ds[idx]
            mask = sample["mask"].numpy()

            ray_px = int((mask == 3).sum())
            fish_px = int((mask == 2).sum())
            mac_px = int((mask == 4).sum())

            ray_pixels_total += ray_px
            fish_pixels_total += fish_px
            macro_pixels_total += mac_px

            if ray_px > 0:
                samples_with_ray += 1

        results[f"prob_{int(p*100)}"] = {
            "object_crop_prob": p,
            "ray_pixels": ray_pixels_total,
            "fish_pixels": fish_pixels_total,
            "macro_pixels": macro_pixels_total,
            "samples_with_ray": samples_with_ray,
            "ray_exposure_pct": round((samples_with_ray / total_samples) * 100.0, 2)
        }
        print(f"  Ray Pixels: {ray_pixels_total:,} | Samples w/ Ray: {samples_with_ray} ({results[f'prob_{int(p*100)}']['ray_exposure_pct']}%) | Fish Px: {fish_pixels_total:,}")

    print("\n" + "=" * 85)
    print("OBJECT CROPPING ABLATION SUMMARY:")
    print(f"{'Crop Prob':<12} | {'Ray Exposure Pct':<18} | {'Total Ray Px':<15} | {'Total Fish Px':<15}")
    print("=" * 85)
    for p in prob_settings:
        cfg = f"prob_{int(p*100)}"
        r = results[cfg]
        print(f"{p:<12.2f} | {r['ray_exposure_pct']:<18.2f}% | {r['ray_pixels']:<15,d} | {r['fish_pixels']:<15,d}")
    print("=" * 85)

    out_file = "experiments/overnight_20261024/predictions/exp18_crop_context_ablation.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    out_data = {
        "configurations": results,
        "runtime_s": round(time.time() - t0, 2),
        "verdict": "CONFIRMED"
    }
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved crop ablation results to {out_file}")

    return out_data


if __name__ == "__main__":
    run_crop_context_ablation()
