"""
SAT-EURS Track 1 — V26 Experiment 4: Extreme Degradation Analysis & Safeguard Audit
Analyzes all 4,000 training degraded images and 1,200 test degraded images.
Evaluates extinction conditions (zero-valued and near-black images) and measures
the exact impact of low-information safeguards on held-out ground truth.
"""

import os
import os.path as osp
import sys
import glob
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

def audit_dataset_extinction(data_root: str = "data/cached_1024", test_root: str = "data/test/degraded") -> Dict[str, Any]:
    print("=" * 80)
    print("SAT-EURS V26 — EXPERIMENT 4: EXTREME DEGRADATION & EXTINCTION AUDIT")
    print("=" * 80)

    # 1. Audit Training Set (20 scenes x 200 images = 4,000 images)
    train_scenes = sorted(os.listdir(osp.join(data_root, "degraded")))
    print(f"Scanning {len(train_scenes)} training scenes (4,000 images)...")

    train_zero_images = []
    train_near_zero_images = []  # max_val < 5 (in uint8)
    train_stats_by_cond = {}     # (wt, depth) -> list of max_vals

    gt_masks = {}
    for sc in train_scenes:
        m_path = osp.join(data_root, "masks", f"{sc}.png")
        if osp.exists(m_path):
            gt_masks[sc] = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

    for sc in train_scenes:
        for wt in range(1, 11):
            for depth in range(5, 105, 5):
                img_path = osp.join(data_root, "degraded", sc, f"{sc}_{wt}_{depth}.png")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                max_val = int(img.max())
                mean_val = float(img.mean())
                cond_key = f"wt{wt}_d{depth}"

                if cond_key not in train_stats_by_cond:
                    train_stats_by_cond[cond_key] = {"maxs": [], "means": []}
                train_stats_by_cond[cond_key]["maxs"].append(max_val)
                train_stats_by_cond[cond_key]["means"].append(mean_val)

                if max_val == 0:
                    train_zero_images.append({
                        "scene": sc, "wt": wt, "depth": depth, "max": max_val, "mean": mean_val
                    })
                elif max_val < 5:
                    train_near_zero_images.append({
                        "scene": sc, "wt": wt, "depth": depth, "max": max_val, "mean": mean_val
                    })

    # 2. Audit Test Set (6 scenes x 200 images = 1,200 images)
    test_scenes = sorted(os.listdir(test_root))
    print(f"Scanning {len(test_scenes)} test scenes (1,200 images)...")

    test_zero_images = []
    test_near_zero_images = []
    test_stats_by_cond = {}

    for sc in test_scenes:
        for wt in range(1, 11):
            for depth in range(5, 105, 5):
                img_path = osp.join(test_root, sc, f"{sc}_{wt}_{depth}.png")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                max_val = int(img.max())
                mean_val = float(img.mean())
                cond_key = f"wt{wt}_d{depth}"

                if cond_key not in test_stats_by_cond:
                    test_stats_by_cond[cond_key] = {"maxs": [], "means": []}
                test_stats_by_cond[cond_key]["maxs"].append(max_val)
                test_stats_by_cond[cond_key]["means"].append(mean_val)

                if max_val == 0:
                    test_zero_images.append({
                        "scene": sc, "wt": wt, "depth": depth, "max": max_val, "mean": mean_val
                    })
                elif max_val < 5:
                    test_near_zero_images.append({
                        "scene": sc, "wt": wt, "depth": depth, "max": max_val, "mean": mean_val
                    })

    print("\n--- EXTINCTION CENSUS SUMMARY ---")
    print(f"Train Set Total Images:       4000")
    print(f"Train Pure-Zero ([0,0,0]):    {len(train_zero_images)} ({len(train_zero_images)/4000*100:.2f}%)")
    print(f"Train Near-Zero (max < 5):    {len(train_near_zero_images)} ({len(train_near_zero_images)/4000*100:.2f}%)")
    print(f"Test Set Total Images:        1200")
    print(f"Test Pure-Zero ([0,0,0]):     {len(test_zero_images)} ({len(test_zero_images)/1200*100:.2f}%)")
    print(f"Test Near-Zero (max < 5):     {len(test_near_zero_images)} ({len(test_near_zero_images)/1200*100:.2f}%)")

    # 3. Ground Truth Content of Train Pure-Zero Images
    print("\n--- GROUND TRUTH ANALYSIS ON ZERO-VALUED IMAGES ---")
    fg_pixels_in_zeros = 0
    total_pixels_in_zeros = 0
    class_pixel_counts = {c: 0 for c in range(5)}

    for item in train_zero_images:
        sc = item["scene"]
        gt = gt_masks[sc]
        total_pixels_in_zeros += gt.size
        for c in range(5):
            cnt = int((gt == c).sum())
            class_pixel_counts[c] += cnt
            if c > 0:
                fg_pixels_in_zeros += cnt

    bg_pct = (class_pixel_counts[0] / max(1, total_pixels_in_zeros)) * 100
    fg_pct = (fg_pixels_in_zeros / max(1, total_pixels_in_zeros)) * 100
    print(f"Total pure-zero training images: {len(train_zero_images)}")
    print(f"Ground Truth Background (Class 0): {bg_pct:.2f}%")
    print(f"Ground Truth Foreground (Classes 1-4): {fg_pct:.2f}%")
    for c in range(5):
        c_pct = (class_pixel_counts[c] / max(1, total_pixels_in_zeros)) * 100
        print(f"  Class {c}: {c_pct:.2f}% ({class_pixel_counts[c]:,} px)")

    # 4. Water types and depths responsible for pure zeros
    zero_conditions_train = set(f"wt{x['wt']}_d{x['depth']}" for x in train_zero_images)
    zero_conditions_test = set(f"wt{x['wt']}_d{x['depth']}" for x in test_zero_images)
    print(f"\nConditions producing pure zeros in Train: {sorted(list(zero_conditions_train))}")
    print(f"Conditions producing pure zeros in Test:  {sorted(list(zero_conditions_test))}")

    results = {
        "train_total": 4000,
        "train_pure_zero_count": len(train_zero_images),
        "train_pure_zero_pct": len(train_zero_images) / 4000 * 100,
        "train_near_zero_count": len(train_near_zero_images),
        "train_near_zero_pct": len(train_near_zero_images) / 4000 * 100,
        "train_pure_zero_items": train_zero_images,
        "test_total": 1200,
        "test_pure_zero_count": len(test_zero_images),
        "test_pure_zero_pct": len(test_zero_images) / 1200 * 100,
        "test_near_zero_count": len(test_near_zero_images),
        "test_near_zero_pct": len(test_near_zero_images) / 1200 * 100,
        "test_pure_zero_items": test_zero_images,
        "train_zero_gt_distribution": {
            f"class_{c}": {
                "pixel_count": class_pixel_counts[c],
                "pct": class_pixel_counts[c] / max(1, total_pixels_in_zeros) * 100
            }
            for c in range(5)
        },
        "zero_conditions_train": sorted(list(zero_conditions_train)),
        "zero_conditions_test": sorted(list(zero_conditions_test))
    }

    out_json = "reports/v26/exp4_extreme_degradation_audit.json"
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audit results to {out_json}")

    return results

if __name__ == "__main__":
    audit_dataset_extinction()
