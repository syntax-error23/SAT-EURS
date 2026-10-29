"""
SAT-EURS Track 1 — EXP-10: Nested Cross-Fitted Class-Logit Calibration
Implements strict 5-fold cross-fitting:
For each outer fold k in {0..4}:
  1. Optimize logit offset vector on remaining 4 folds (16 scenes)
  2. Evaluate chosen vector strictly on held-out outer fold k (4 scenes)
  3. Accumulate outer held-out confusion matrices across all 5 folds
Guarantees zero outer-fold leakage in calibration parameter selection.
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

CANDIDATE_VECTORS = {
    "zero_offset": [0.0, 0.0, 0.0, 0.0, 0.0],
    "baseline_25": [0.0, 0.0, 0.0, 0.10, 0.25],
    "macro_30":    [0.0, 0.0, 0.0, 0.10, 0.30],
    "macro_35":    [0.0, 0.0, 0.0, 0.10, 0.35],
    "macro_40":    [0.0, 0.0, 0.0, 0.10, 0.40],
    "fish05_mac35": [0.0, 0.0, 0.05, 0.10, 0.35],
}


def run_nested_calibration():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SAT-EURS TRACK 1 — EXP-10: NESTED CROSS-FITTED CLASS-LOGIT CALIBRATION")
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

    # Pre-build offset tensors on GPU
    cal_tensors = {
        name: torch.tensor(vec, device=device).view(1, 5, 1, 1)
        for name, vec in CANDIDATE_VECTORS.items()
    }

    # Per-fold confusion matrices: conf_mat[fold][vec_name]
    fold_conf_mats = {f: {v: np.zeros((5, 5), dtype=np.int64) for v in CANDIDATE_VECTORS} for f in range(5)}

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

        print(f"Fold {fold} evaluating {val_scenes} across {len(CANDIDATE_VECTORS)} vectors...", flush=True)

        for sc in val_scenes:
            mask_path = f"data/cached_1024/masks/{sc}.png"
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gh, gw = gt_mask.shape[:2]
            y_t = gt_mask.reshape(-1)

            deg_scene_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])[::5]

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
                        log_prob_base = torch.log(prob_fused + 1e-10)

                total_imgs += 1

                for v_name in CANDIDATE_VECTORS:
                    cal_t = cal_tensors[v_name]
                    log_prob = log_prob_base + cal_t
                    mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    pred_mask = preprocessor.unpad_output(mask_pad, meta)
                    if pred_mask.shape != (gh, gw):
                        pred_mask = cv2.resize(pred_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)

                    y_p = pred_mask.reshape(-1)
                    indices = 5 * y_t + y_p
                    counts = np.bincount(indices, minlength=25).reshape(5, 5)
                    fold_conf_mats[fold][v_name] += counts

    print(f"\nCompleted per-fold evaluation ({total_imgs} images) in {time.time() - t0:.1f}s.")

    # 2. Strict Nested Cross-Fitting
    print("\n" + "=" * 85)
    print("STRICT 5-FOLD NESTED CROSS-FITTING RESULTS:")
    print("=" * 85)

    nested_outer_cmat = np.zeros((5, 5), dtype=np.int64)
    baseline_outer_cmat = np.zeros((5, 5), dtype=np.int64)
    fold_selections = {}

    for outer_fold in range(5):
        # Sum inner confusion matrices across the other 4 folds
        inner_cmats = {v: np.zeros((5, 5), dtype=np.int64) for v in CANDIDATE_VECTORS}
        for j in range(5):
            if j != outer_fold:
                for v in CANDIDATE_VECTORS:
                    inner_cmats[v] += fold_conf_mats[j][v]

        # Evaluate inner Q_seg to choose best vector
        best_v = None
        best_inner_q = -1.0
        for v in CANDIDATE_VECTORS:
            m = compute_metrics_from_conf_mat(inner_cmats[v])
            if m["Q_seg"] > best_inner_q:
                best_inner_q = m["Q_seg"]
                best_v = v

        # Apply best_v to outer_fold
        outer_m_chosen = compute_metrics_from_conf_mat(fold_conf_mats[outer_fold][best_v])
        outer_m_base = compute_metrics_from_conf_mat(fold_conf_mats[outer_fold]["baseline_25"])

        nested_outer_cmat += fold_conf_mats[outer_fold][best_v]
        baseline_outer_cmat += fold_conf_mats[outer_fold]["baseline_25"]

        fold_selections[f"fold_{outer_fold}"] = {
            "selected_vector": best_v,
            "inner_q_seg": round(best_inner_q, 5),
            "outer_q_seg_chosen": round(outer_m_chosen["Q_seg"], 5),
            "outer_q_seg_baseline": round(outer_m_base["Q_seg"], 5),
            "outer_delta": round(outer_m_chosen["Q_seg"] - outer_m_base["Q_seg"], 5)
        }
        print(f"Outer Fold {outer_fold}: Selected '{best_v}' (Inner Q_seg={best_inner_q:.4f}) -> Outer Q_seg: {outer_m_chosen['Q_seg']:.4f} (Base: {outer_m_base['Q_seg']:.4f}, Delta: {outer_m_chosen['Q_seg'] - outer_m_base['Q_seg']:+.4f})")

    # Pooled Nested vs Baseline
    nested_metrics = compute_metrics_from_conf_mat(nested_outer_cmat)
    baseline_metrics = compute_metrics_from_conf_mat(baseline_outer_cmat)

    dq = nested_metrics["Q_seg"] - baseline_metrics["Q_seg"]
    dmiou = nested_metrics["mIoU"] - baseline_metrics["mIoU"]
    df1 = nested_metrics["Macro_F1"] - baseline_metrics["Macro_F1"]

    print("\n" + "=" * 85)
    print("FINAL NESTED CROSS-FITTED METRIC SUMMARY (Zero Leakage):")
    print(f"  Baseline 25 Vector:       Q_seg = {baseline_metrics['Q_seg']:.5f} | mIoU = {baseline_metrics['mIoU']:.5f} | F1 = {baseline_metrics['Macro_F1']:.5f}")
    print(f"  Nested Cross-Fitted:      Q_seg = {nested_metrics['Q_seg']:.5f} | mIoU = {nested_metrics['mIoU']:.5f} | F1 = {nested_metrics['Macro_F1']:.5f}")
    print(f"  Net Generalization Delta: Delta Q_seg = {dq:+.5f} | Delta mIoU = {dmiou:+.5f} | Delta F1 = {df1:+.5f}")
    print("=" * 85)

    verdict = "WIN" if dq > 0.001 else ("LOSS" if dq < -0.001 else "INCONCLUSIVE")

    out_data = {
        "metrics": {
            "nested_q_seg": nested_metrics["Q_seg"],
            "baseline_q_seg": baseline_metrics["Q_seg"],
            "delta_q_seg": round(dq, 5),
            "delta_miou": round(dmiou, 5),
            "delta_macro_f1": round(df1, 5),
            "nested_miou": nested_metrics["mIoU"],
            "nested_f1": nested_metrics["Macro_F1"],
            "class_ious": nested_metrics["class_ious"],
        },
        "fold_selections": fold_selections,
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 2)
    }

    out_file = "experiments/overnight_20261024/predictions/exp10_nested_calibration.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote nested calibration results to {out_file}")

    return out_data


if __name__ == "__main__":
    run_nested_calibration()
