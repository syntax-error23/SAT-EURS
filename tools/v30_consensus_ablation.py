"""
SAT-EURS Track 1 — Workstream A: Cross-Degradation Scene Consensus Ablation Engine
Evaluates 7 distinct consensus aggregation algorithms on strictly held-out Fold 0 scenes (800 images):
1. Control: Standard Individual-Image Inference (V28-A protocol)
2. Unweighted Scene-Level Consensus (Average of all 200 variants)
3. Depth-Exponential Weighted Consensus (w = exp(-depth / tau))
4. Visibility-Weighted Consensus (w = exp(-depth / tau) * (11 - wt) / 10)
5. Selective Variant Pooling (Average of top K clearest variants: shallow + clear WT)
6. Robust Trimmed Mean Consensus (15% trimmed mean per pixel)
7. Foreground-Persistence Consensus (High-confidence detection in top-K boosted)

Strictly zero data leakage: uses checkpoint trained strictly without Fold 0 scenes.
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    V18CanonicalPreprocessor,
    parse_scene_wt_depth,
)


def compute_metrics(conf_mat: np.ndarray) -> Dict[str, Any]:
    tp = np.diag(conf_mat).astype(float)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    per_class_iou = np.where(denom_iou > 0, tp / (denom_iou + 1e-10), 1.0)

    denom_f1 = 2 * tp + fp + fn
    per_class_f1 = np.where(denom_f1 > 0, (2 * tp) / (denom_f1 + 1e-10), 1.0)

    miou = float(np.mean(per_class_iou))
    macro_f1 = float(np.mean(per_class_f1))
    q_seg = float(miou * macro_f1)

    return {
        "Q_seg": round(q_seg, 5),
        "mIoU": round(miou, 5),
        "Macro_F1": round(macro_f1, 5),
        "class_ious": [round(float(v), 4) for v in per_class_iou],
        "class_f1s": [round(float(v), 4) for v in per_class_f1],
    }


def run_consensus_ablation(
    fold: int = 0,
    segmenter_ckpt: Optional[str] = None,
    refiner_ckpt: Optional[str] = None,
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    data_root: str = "data/cached_1024",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    device_id: int = 0,
    stride: int = 1  # 1 = all 800 images, 2 = 400 images
) -> Dict[str, Any]:
    if segmenter_ckpt is None:
        if fold == 0:
            segmenter_ckpt = "checkpoints/v27/segmenter_v27_fold0_best.pt"
        else:
            segmenter_ckpt = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
    if refiner_ckpt is None:
        refiner_ckpt = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("SAT-EURS TRACK 1 — WORKSTREAM A: SCENE CONSENSUS ABLATION")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Segmenter Checkpoint: {segmenter_ckpt}")
    print(f"Stride: {stride} (Stride 1 = full 800 images across 4 scenes)")
    print("=" * 85)

    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    val_scenes = sorted(f_data["val_scenes"])
    print(f"Validation Scenes ({len(val_scenes)}): {val_scenes}")

    # 1. Load refiner
    base_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_obj = torch.load(refiner_ckpt, map_location=device)
    refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
    refiner.eval()

    # 2. Load DINOv3 segmenter
    segmenter = build_v18_segmenter(num_classes=5).to(device)
    s_obj = torch.load(segmenter_ckpt, map_location=device)
    segmenter.load_state_dict(s_obj["state_dict"] if "state_dict" in s_obj else s_obj)
    segmenter.eval()

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # 3. Extract and cache all probability maps per scene
    scene_data: Dict[str, List[Dict[str, Any]]] = {}
    gt_masks: Dict[str, np.ndarray] = {}

    t0 = time.time()
    total_imgs = 0

    print("\n[Step 1/2] Extracting full-resolution probability maps across all validation scenes...")
    for sc in val_scenes:
        mask_p = osp.join(data_root, "masks", f"{sc}.png")
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        gt_masks[sc] = gt_mask

        deg_dir = osp.join(data_root, "degraded", sc)
        deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::stride]

        scene_data[sc] = []
        t_sc = time.time()

        for df in deg_files:
            total_imgs += 1
            parts = df.replace(".png", "").split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10

            img_p = osp.join(deg_dir, df)
            bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
            prep = preprocessor.prepare_input(bgr)
            raw_pad = prep["rgb_tensor"].to(device)
            meta = prep["meta"]

            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([depth], device=device)

            with torch.no_grad():
                # Refiner
                out_r = refiner(raw_pad, wt_t, dp_t)
                rest_pad = out_r["refined_image"]

                # Flip TTA
                raw_flip = torch.flip(raw_pad, dims=[3])
                rest_flip = torch.flip(rest_pad, dims=[3])

                out_orig = segmenter(rest_pad, raw_pad, wt_t, dp_t)
                p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                out_flip = segmenter(rest_flip, raw_flip, wt_t, dp_t)
                p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                p_fused = 0.5 * (p_orig + p_flip)

            # Store probability tensor on CPU to save GPU RAM
            scene_data[sc].append({
                "fname": df,
                "wt": wt,
                "depth": depth,
                "meta": meta,
                "prob": p_fused.squeeze(0).cpu().numpy()  # shape (5, H_pad, W_pad)
            })

        print(f"  Scene `{sc}` ({len(scene_data[sc])} images) cached in {time.time() - t_sc:.1f}s.")

    print(f"\nAll {total_imgs} images extracted in {time.time() - t0:.1f}s.")

    # 4. Benchmark Aggregation Algorithms
    print("\n[Step 2/2] Evaluating Consensus Aggregation Algorithms...")

    methods = [
        "A. Control: Individual-Image Inference (No Consensus)",
        "B. Unweighted Scene Consensus (All variants)",
        "C1. Depth-Exponential Consensus (tau=25m)",
        "C2. Depth-Exponential Consensus (tau=40m)",
        "D. Visibility-Weighted Consensus (WT + Depth)",
        "E1. Top-20 Clearest Variants Consensus",
        "E2. Top-40 Clearest Variants Consensus",
        "E3. Top-60 Clearest Variants Consensus",
        "F. Robust Trimmed Mean Consensus (15% trimmed)",
        "G. Hybrid: 70% Individual + 30% Clearest Consensus",
        "H. Hybrid: 50% Individual + 50% Clearest Consensus",
    ]

    cms_global = {m: np.zeros((5, 5), dtype=np.int64) for m in methods}
    cms_per_scene = {m: {sc: np.zeros((5, 5), dtype=np.int64) for sc in val_scenes} for m in methods}

    for sc in val_scenes:
        items = scene_data[sc]
        gt_mask = gt_masks[sc]
        K = len(items)

        # Precompute candidate consensus probability maps for this scene
        probs_all = np.stack([it["prob"] for it in items], axis=0)  # shape (K, 5, H, W)
        wts = np.array([it["wt"] for it in items])
        depths = np.array([it["depth"] for it in items])
        meta0 = items[0]["meta"]

        # Method B: Unweighted Mean
        p_B = np.mean(probs_all, axis=0)

        # Method C1: Depth-Exponential (tau=25)
        w_c1 = np.exp(-depths / 25.0)[:, None, None, None]
        p_C1 = np.sum(probs_all * w_c1, axis=0) / np.sum(w_c1, axis=0)

        # Method C2: Depth-Exponential (tau=40)
        w_c2 = np.exp(-depths / 40.0)[:, None, None, None]
        p_C2 = np.sum(probs_all * w_c2, axis=0) / np.sum(w_c2, axis=0)

        # Method D: Visibility-Weighted (WT + Depth)
        w_d = (np.exp(-depths / 30.0) * (11.0 - wts) / 10.0)[:, None, None, None]
        p_D = np.sum(probs_all * w_d, axis=0) / np.sum(w_d, axis=0)

        # Sort indices by clarity: lowest depth first, then lowest WT
        clarity_score = depths * 2.0 + wts  # lower is clearer
        sorted_indices = np.argsort(clarity_score)

        # Method E1: Top 20 clearest
        top20_idx = sorted_indices[:min(20, K)]
        p_E1 = np.mean(probs_all[top20_idx], axis=0)

        # Method E2: Top 40 clearest
        top40_idx = sorted_indices[:min(40, K)]
        p_E2 = np.mean(probs_all[top40_idx], axis=0)

        # Method E3: Top 60 clearest
        top60_idx = sorted_indices[:min(60, K)]
        p_E3 = np.mean(probs_all[top60_idx], axis=0)

        # Method F: Trimmed mean (15% trimmed)
        trim_lo = int(0.15 * K)
        trim_hi = K - trim_lo
        sorted_probs = np.sort(probs_all, axis=0)
        p_F = np.mean(sorted_probs[trim_lo:trim_hi], axis=0)

        cal_np = np.array(offset_vector, dtype=np.float32).reshape(5, 1, 1)

        # Evaluate every image under each method
        for idx, it in enumerate(items):
            p_indep = it["prob"]
            meta = it["meta"]

            preds = {}

            # A. Control
            l_A = np.log(p_indep + 1e-10) + cal_np
            preds["A. Control: Individual-Image Inference (No Consensus)"] = preprocessor.unpad_output(np.argmax(l_A, axis=0), meta)

            # Shared consensus predictions
            for m_name, p_map in [
                ("B. Unweighted Scene Consensus (All variants)", p_B),
                ("C1. Depth-Exponential Consensus (tau=25m)", p_C1),
                ("C2. Depth-Exponential Consensus (tau=40m)", p_C2),
                ("D. Visibility-Weighted Consensus (WT + Depth)", p_D),
                ("E1. Top-20 Clearest Variants Consensus", p_E1),
                ("E2. Top-40 Clearest Variants Consensus", p_E2),
                ("E3. Top-60 Clearest Variants Consensus", p_E3),
                ("F. Robust Trimmed Mean Consensus (15% trimmed)", p_F),
            ]:
                l_m = np.log(p_map + 1e-10) + cal_np
                preds[m_name] = preprocessor.unpad_output(np.argmax(l_m, axis=0), meta)

            # Hybrids: per-image prob blended with clearest consensus
            p_H1 = 0.70 * p_indep + 0.30 * p_E2
            l_H1 = np.log(p_H1 + 1e-10) + cal_np
            preds["G. Hybrid: 70% Individual + 30% Clearest Consensus"] = preprocessor.unpad_output(np.argmax(l_H1, axis=0), meta)

            p_H2 = 0.50 * p_indep + 0.50 * p_E2
            l_H2 = np.log(p_H2 + 1e-10) + cal_np
            preds["H. Hybrid: 50% Individual + 50% Clearest Consensus"] = preprocessor.unpad_output(np.argmax(l_H2, axis=0), meta)

            # Accumulate confusion matrices
            for m_name, pred_unpad in preds.items():
                inds = 5 * gt_mask.flatten().astype(int) + pred_unpad.flatten().astype(int)
                cnts = np.bincount(inds, minlength=25).reshape(5, 5)
                cms_global[m_name] += cnts
                cms_per_scene[m_name][sc] += cnts

    print("\n" + "=" * 90)
    print(f"{'Consensus Aggregation Algorithm':55s} | {'Q_seg':7s} | {'mIoU':7s} | {'F1':7s} | Delta vs Control")
    print("-" * 90)

    base_q = 0.0
    results_summary = {}

    for m_name in methods:
        m = compute_metrics(cms_global[m_name])
        results_summary[m_name] = m
        if "Control" in m_name:
            base_q = m["Q_seg"]
            delta_str = "BASELINE"
        else:
            delta = m["Q_seg"] - base_q
            delta_str = f"{delta:+7.4f}"

        ious_str = f"[{m['class_ious'][0]:.3f}, {m['class_ious'][1]:.3f}, {m['class_ious'][2]:.3f}, {m['class_ious'][3]:.3f}, {m['class_ious'][4]:.3f}]"
        print(f"{m_name:55s} | {m['Q_seg']:7.4f} | {m['mIoU']:7.4f} | {m['Macro_F1']:7.4f} | {delta_str} | {ious_str}")

    print("\n--- Per-Scene Breakdown (Q_seg) ---")
    sc_hdr = f"{'Algorithm':55s} | " + " | ".join([f"{sc:10s}" for sc in val_scenes])
    print(sc_hdr)
    print("-" * len(sc_hdr))
    for m_name in methods:
        row = f"{m_name:55s} | "
        sc_vals = []
        for sc in val_scenes:
            m_sc = compute_metrics(cms_per_scene[m_name][sc])
            sc_vals.append(f"{m_sc['Q_seg']:10.4f}")
        row += " | ".join(sc_vals)
        print(row)

    out_file = "reports/v24/v30_consensus_benchmark_results.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved full results to {out_file}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    run_consensus_ablation(fold=args.fold, stride=args.stride)
