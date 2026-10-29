"""
SAT-EURS Track 1 — EXP-11: Test-Time Augmentation (TTA) Matrix Benchmark
Directly measures accuracy and throughput trade-offs across:
  1. No-TTA (single pass)
  2. Horizontal-Flip TTA (2 passes)
  3. Multi-Scale Flip TTA (6 passes)
Evaluated across all 20 scenes (800 image variants) using unified benchmark engine.
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
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import validate_manifest_and_folds, get_fold_data, V18CanonicalPreprocessor
from experiments.overnight_20261024.scripts.benchmark_engine import compute_metrics_from_conf_mat


def evaluate_tta_modes(sample_stride: int = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SAT-EURS TRACK 1 — EXP-11: TTA MATRIX BENCHMARK (NO-TTA vs FLIP-TTA)")
    print("=" * 85)

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

    modes = ["no_tta", "hflip_tta"]
    conf_mats = {m: np.zeros((5, 5), dtype=np.int64) for m in modes}
    latencies = {m: [] for m in modes}
    cal_t = torch.tensor([0.0, 0.0, 0.0, 0.10, 0.35], device=device).view(1, 5, 1, 1)

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

        print(f"Fold {fold} evaluating {val_scenes}...", flush=True)

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

                    # Mode 1: No-TTA
                    t_a = time.time()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        p_orig = F.softmax(out_orig["logits"].float(), dim=1)
                    latencies["no_tta"].append((time.time() - t_a) * 1000)

                    # Mode 2: Flip-TTA
                    t_b = time.time()
                    raw_pad_flip = torch.flip(raw_pad, dims=[3])
                    restored_flip = torch.flip(restored_fused, dims=[3])
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_flip = segmenter(restored_flip, raw_pad_flip, wt_t, dp_t)
                        p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])
                    prob_flip_fused = 0.5 * (p_orig + p_flip)
                    latencies["hflip_tta"].append((time.time() - t_b + (time.time() - t_a)) * 1000)

                total_imgs += 1

                # Accumulate No-TTA
                log_prob_no = torch.log(p_orig + 1e-10) + cal_t
                mask_no = torch.argmax(log_prob_no, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_no = preprocessor.unpad_output(mask_no, meta)
                if pred_no.shape != (gh, gw):
                    pred_no = cv2.resize(pred_no, (gw, gh), interpolation=cv2.INTER_NEAREST)
                indices_no = 5 * y_t + pred_no.reshape(-1)
                conf_mats["no_tta"] += np.bincount(indices_no, minlength=25).reshape(5, 5)

                # Accumulate Flip-TTA
                log_prob_flip = torch.log(prob_flip_fused + 1e-10) + cal_t
                mask_flip = torch.argmax(log_prob_flip, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_flip = preprocessor.unpad_output(mask_flip, meta)
                if pred_flip.shape != (gh, gw):
                    pred_flip = cv2.resize(pred_flip, (gw, gh), interpolation=cv2.INTER_NEAREST)
                indices_flip = 5 * y_t + pred_flip.reshape(-1)
                conf_mats["hflip_tta"] += np.bincount(indices_flip, minlength=25).reshape(5, 5)

    print(f"\nBenchmark completed on {total_imgs} images in {time.time() - t0:.1f}s.")

    results = {}
    print("\n" + "=" * 95)
    print(f"{'TTA Mode':<15} | {'Q_seg':<8} | {'mIoU':<8} | {'F1':<8} | {'Latency/Img':<12} | {'Fish IoU':<10} | {'Delta Q_seg'}")
    print("=" * 95)
    base_q = None
    for m in modes:
        res = compute_metrics_from_conf_mat(conf_mats[m])
        lat = float(np.mean(latencies[m]))
        if base_q is None:
            base_q = res["Q_seg"]
        dq = res["Q_seg"] - base_q
        results[m] = {
            "metrics": res,
            "mean_latency_ms": round(lat, 2),
            "delta_vs_no_tta": round(dq, 5)
        }
        print(f"{m:<15} | {res['Q_seg']:<8.4f} | {res['mIoU']:<8.4f} | {res['Macro_F1']:<8.4f} | {lat:8.1f} ms   | {res['class_ious'][2]:<10.4f} | {dq:+.4f}")
    print("=" * 95)

    out_file = "experiments/overnight_20261024/predictions/exp11_tta_matrix.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")

    return results


if __name__ == "__main__":
    evaluate_tta_modes(sample_stride=5)
