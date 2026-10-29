"""
SAT-EURS Track 1 — EXP-17: Comprehensive Degradation Robustness Breakdown
Audits segmentation quality across:
  1. 10 Jerlov Water Types:
     - Oceanic: Type 1 (I), Type 2 (IA), Type 3 (IB), Type 4 (II), Type 5 (III)
     - Coastal: Type 6 (1C), Type 7 (3C), Type 8 (5C), Type 9 (7C), Type 10 (9C)
  2. 4 Depth Brackets:
     - Shallow (5m - 25m)
     - Mid-Depth (30m - 50m)
     - Deep (55m - 75m)
     - Extreme (80m - 100m)
Evaluated across all 20 scenes (800 image variants) using the V28-SWA upgraded ensemble.
"""

import os
import os.path as osp
import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data, V18CanonicalPreprocessor
from experiments.overnight_20261024.scripts.benchmark_engine import compute_metrics_from_conf_mat


def run_degradation_breakdown(sample_stride: int = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("SAT-EURS TRACK 1 — EXP-17: COMPREHENSIVE DEGRADATION ROBUSTNESS BREAKDOWN")
    print("=" * 95)

    manifest = validate_manifest_and_folds()
    base_table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    segmenter_ckpts = {
        0: 'checkpoints/v28/segmenter_v28_fold0_best.pt',
        1: 'checkpoints/v18/fold_1/segmenter_best.pt',
        2: 'checkpoints/v18/fold_2/segmenter_best.pt',
        3: 'checkpoints/v18/fold_3/segmenter_best.pt',
        4: 'checkpoints/v18/fold_4/segmenter_best.pt',
    }

    refiner_ckpts = {
        0: 'checkpoints/v18/fold_0/refiner_best.pt',
        1: 'checkpoints/v18/fold_1/refiner_best.pt',
        2: 'checkpoints/v18/fold_2/refiner_best.pt',
        3: 'checkpoints/v18/fold_3/refiner_best.pt',
        4: 'checkpoints/v18/fold_4/refiner_best.pt',
    }

    cal_t = torch.tensor([0.0, 0.0, 0.0, 0.10, 0.35], device=device).view(1, 5, 1, 1)

    # Accumulators
    wt_cmats = {wt: np.zeros((5, 5), dtype=np.int64) for wt in range(1, 11)}
    depth_cmats = {"shallow": np.zeros((5, 5), dtype=np.int64),
                   "mid": np.zeros((5, 5), dtype=np.int64),
                   "deep": np.zeros((5, 5), dtype=np.int64),
                   "extreme": np.zeros((5, 5), dtype=np.int64)}

    t0 = time.time()
    total_imgs = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpts[fold], map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        segmenter = build_v18_segmenter(num_classes=5).to(device)
        s_obj = torch.load(segmenter_ckpts[fold], map_location=device)
        s_sd = s_obj["state_dict"] if "state_dict" in s_obj else s_obj
        segmenter.load_state_dict(s_sd)
        segmenter.eval()

        print(f"Fold {fold} evaluating degradation breakdown on {val_scenes}...", flush=True)

        for sc in val_scenes:
            mask_path = f"data/cached_1024/masks/{sc}.png"
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gh, gw = gt_mask.shape[:2]
            y_t = gt_mask.reshape(-1)

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
                    out_r = refiner(raw_pad.float(), wt_t, dp_t)
                    restored_fused = out_r["refined_image"]

                    raw_pad_flip = torch.flip(raw_pad, dims=[3])
                    restored_flip = torch.flip(restored_fused, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                        out_flip = segmenter(restored_flip, raw_pad_flip, wt_t, dp_t)
                        p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                        prob_fused = 0.5 * (p_orig + p_flip)
                        log_prob = torch.log(prob_fused + 1e-10) + cal_t
                        mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                pred_mask = preprocessor.unpad_output(mask_pad, meta)
                if pred_mask.shape != (gh, gw):
                    pred_mask = cv2.resize(pred_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)

                y_p = pred_mask.reshape(-1)
                indices = 5 * y_t + y_p
                counts = np.bincount(indices, minlength=25).reshape(5, 5)

                wt_cmats[wt] += counts
                if depth <= 25:
                    depth_cmats["shallow"] += counts
                elif depth <= 50:
                    depth_cmats["mid"] += counts
                elif depth <= 75:
                    depth_cmats["deep"] += counts
                else:
                    depth_cmats["extreme"] += counts

                total_imgs += 1

    print(f"\nEvaluated {total_imgs} images in {time.time() - t0:.1f}s.")

    # Water Type Summary
    WT_NAMES = {
        1: "Type 1 (Oceanic I)", 2: "Type 2 (Oceanic IA)", 3: "Type 3 (Oceanic IB)",
        4: "Type 4 (Oceanic II)", 5: "Type 5 (Oceanic III)",
        6: "Type 6 (Coastal 1C)", 7: "Type 7 (Coastal 3C)", 8: "Type 8 (Coastal 5C)",
        9: "Type 9 (Coastal 7C)", 10: "Type 10 (Coastal 9C)"
    }

    wt_summary = {}
    print("\n" + "=" * 95)
    print("WATER TYPE BREAKDOWN:")
    print(f"{'Water Type':<25} | {'Q_seg':<8} | {'mIoU':<8} | {'F1':<8} | {'Fish IoU':<10} | {'Macro IoU':<10}")
    print("=" * 95)
    for wt in range(1, 11):
        m = compute_metrics_from_conf_mat(wt_cmats[wt])
        wt_summary[WT_NAMES[wt]] = m
        print(f"{WT_NAMES[wt]:<25} | {m['Q_seg']:<8.4f} | {m['mIoU']:<8.4f} | {m['Macro_F1']:<8.4f} | {m['class_ious'][2]:<10.4f} | {m['class_ious'][4]:<10.4f}")
    print("=" * 95)

    # Depth Summary
    depth_summary = {}
    print("\n" + "=" * 95)
    print("DEPTH BRACKET BREAKDOWN:")
    print(f"{'Depth Bracket':<25} | {'Q_seg':<8} | {'mIoU':<8} | {'F1':<8} | {'Fish IoU':<10} | {'Macro IoU':<10}")
    print("=" * 95)
    for d_name, label in [("shallow", "Shallow (5-25m)"), ("mid", "Mid-Depth (30-50m)"), ("deep", "Deep (55-75m)"), ("extreme", "Extreme (80-100m)")]:
        m = compute_metrics_from_conf_mat(depth_cmats[d_name])
        depth_summary[label] = m
        print(f"{label:<25} | {m['Q_seg']:<8.4f} | {m['mIoU']:<8.4f} | {m['Macro_F1']:<8.4f} | {m['class_ious'][2]:<10.4f} | {m['class_ious'][4]:<10.4f}")
    print("=" * 95)

    out_data = {
        "water_type_breakdown": wt_summary,
        "depth_breakdown": depth_summary,
        "runtime_s": round(time.time() - t0, 2),
        "total_images": total_imgs
    }

    out_file = "experiments/overnight_20261024/predictions/exp17_degradation_breakdown.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved degradation robustness results to {out_file}")

    return out_data


if __name__ == "__main__":
    run_degradation_breakdown(sample_stride=5)
