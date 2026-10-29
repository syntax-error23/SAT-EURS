"""
SAT-EURS Track 1 — Workstream A: Full 5-Fold Scene-Disjoint Consensus Benchmark
Evaluates all 20 scenes (4,000 images) strictly Out-Of-Fold across all 5 folds:
- Fold 0 (fish1, fish11, fish12, seahorse1) using checkpoints/v27/segmenter_v27_fold0_best.pt
- Fold 1 (fish13, fish14, fish15, sharkFish) using checkpoints/v18/fold_1/segmenter_best.pt
- Fold 2 (fish17, fish18, fish2, ray1) using checkpoints/v18/fold_2/segmenter_best.pt
- Fold 3 (fish20, fish21, fish4, starfish2) using checkpoints/v18/fold_3/segmenter_best.pt
- Fold 4 (fish5, fish7, fish9, turtles1) using checkpoints/v18/fold_4/segmenter_best.pt

Zero cross-scene leakage. Compares Control (Individual Inference) vs Visibility-Weighted Consensus.
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

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
)

CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]


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


def benchmark_full_5fold(stride: int = 2, device_id: int = 0):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print("SAT-EURS TRACK 1 — FULL 5-FOLD SCENE-DISJOINT OOF CONSENSUS BENCHMARK")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Stride: {stride} (Stride 2 = 2,000 images across all 20 scenes)")
    print("=" * 90)

    manifest = validate_manifest_and_folds()
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor([0.0, 0.0, 0.0, 0.10, 0.25], device=device).view(1, 5, 1, 1)
    cal_np = np.array([0.0, 0.0, 0.0, 0.10, 0.25], dtype=np.float32).reshape(5, 1, 1)

    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    base_table = torch.load(table_path, map_location=device)

    # Fold checkpoints
    ckpts = {
        0: "checkpoints/v27/segmenter_v27_fold0_best.pt",
        1: "checkpoints/v18/fold_1/segmenter_best.pt",
        2: "checkpoints/v18/fold_2/segmenter_best.pt",
        3: "checkpoints/v18/fold_3/segmenter_best.pt",
        4: "checkpoints/v18/fold_4/segmenter_best.pt",
    }

    ref_ckpts = {f: f"checkpoints/v18/fold_{f}/refiner_best.pt" for f in range(5)}

    cm_control_global = np.zeros((5, 5), dtype=np.int64)
    cm_consensus_global = np.zeros((5, 5), dtype=np.int64)

    per_fold_results = {}
    per_scene_results = {}

    t_all = time.time()
    total_imgs = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])
        s_path = ckpts[fold]
        r_path = ref_ckpts[fold]

        print(f"\n--- FOLD {fold} (Val Scenes: {val_scenes}) ---")
        print(f"  Segmenter Checkpoint: {s_path}")
        print(f"  Refiner Checkpoint:   {r_path}")

        # Load Refiner
        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(r_path, map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        # Load Segmenter
        segmenter = build_v18_segmenter(num_classes=5).to(device)
        s_obj = torch.load(s_path, map_location=device)
        segmenter.load_state_dict(s_obj["state_dict"] if "state_dict" in s_obj else s_obj)
        segmenter.eval()

        cm_ctrl_fold = np.zeros((5, 5), dtype=np.int64)
        cm_cons_fold = np.zeros((5, 5), dtype=np.int64)

        for sc in val_scenes:
            mask_p = f"data/cached_1024/masks/{sc}.png"
            gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            deg_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::stride]

            scene_probs = []
            scene_wts = []
            scene_depths = []
            scene_metas = []

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
                    out_r = refiner(raw_pad, wt_t, dp_t)
                    rest_pad = out_r["refined_image"]

                    raw_flip = torch.flip(raw_pad, dims=[3])
                    rest_flip = torch.flip(rest_pad, dims=[3])

                    out_orig = segmenter(rest_pad, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = segmenter(rest_flip, raw_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    p_fused = 0.5 * (p_orig + p_flip)

                scene_probs.append(p_fused.squeeze(0).cpu().numpy())
                scene_wts.append(wt)
                scene_depths.append(depth)
                scene_metas.append(meta)

            # Build Visibility-Weighted Consensus for this scene
            probs_stack = np.stack(scene_probs, axis=0)  # (K, 5, H, W)
            wts_arr = np.array(scene_wts)
            depths_arr = np.array(scene_depths)

            # Visibility weights: exponential depth + clear water type
            w_vis = (np.exp(-depths_arr / 30.0) * (11.0 - wts_arr) / 10.0)[:, None, None, None]
            p_consensus = np.sum(probs_stack * w_vis, axis=0) / np.sum(w_vis, axis=0)

            # Consensus prediction
            l_cons = np.log(p_consensus + 1e-10) + cal_np
            pred_cons_pad = np.argmax(l_cons, axis=0).astype(np.uint8)
            pred_cons_unpad = preprocessor.unpad_output(pred_cons_pad, scene_metas[0])

            cm_ctrl_sc = np.zeros((5, 5), dtype=np.int64)
            cm_cons_sc = np.zeros((5, 5), dtype=np.int64)

            for idx, p_indep in enumerate(scene_probs):
                meta = scene_metas[idx]

                # Control
                l_ctrl = np.log(p_indep + 1e-10) + cal_np
                pred_ctrl_pad = np.argmax(l_ctrl, axis=0).astype(np.uint8)
                pred_ctrl_unpad = preprocessor.unpad_output(pred_ctrl_pad, meta)

                inds_ctrl = 5 * gt_mask.flatten().astype(int) + pred_ctrl_unpad.flatten().astype(int)
                cm_ctrl_sc += np.bincount(inds_ctrl, minlength=25).reshape(5, 5)

                inds_cons = 5 * gt_mask.flatten().astype(int) + pred_cons_unpad.flatten().astype(int)
                cm_cons_sc += np.bincount(inds_cons, minlength=25).reshape(5, 5)

            m_ctrl_sc = compute_metrics(cm_ctrl_sc)
            m_cons_sc = compute_metrics(cm_cons_sc)
            per_scene_results[sc] = {
                "control": m_ctrl_sc,
                "consensus": m_cons_sc,
                "delta_qseg": round(m_cons_sc["Q_seg"] - m_ctrl_sc["Q_seg"], 5)
            }
            print(f"  Scene `{sc:10s}`: Control Q_seg={m_ctrl_sc['Q_seg']:.4f} -> Consensus Q_seg={m_cons_sc['Q_seg']:.4f} ({m_cons_sc['Q_seg'] - m_ctrl_sc['Q_seg']:+7.4f})")

            cm_ctrl_fold += cm_ctrl_sc
            cm_cons_fold += cm_cons_sc

        m_ctrl_f = compute_metrics(cm_ctrl_fold)
        m_cons_f = compute_metrics(cm_cons_fold)
        per_fold_results[f"fold_{fold}"] = {
            "control": m_ctrl_f,
            "consensus": m_cons_f,
            "delta_qseg": round(m_cons_f["Q_seg"] - m_ctrl_f["Q_seg"], 5)
        }
        print(f"-> FOLD {fold} SUMMARY: Control Q_seg={m_ctrl_f['Q_seg']:.4f} -> Consensus Q_seg={m_cons_f['Q_seg']:.4f} ({m_cons_f['Q_seg'] - m_ctrl_f['Q_seg']:+7.4f})")

        cm_control_global += cm_ctrl_fold
        cm_consensus_global += cm_cons_fold

    print("\n" + "=" * 90)
    print("=== POOLED 5-FOLD GLOBAL OUT-OF-FOLD RESULTS (All 20 Scenes, All 5 Classes Present) ===")
    print("=" * 90)
    m_ctrl_glob = compute_metrics(cm_control_global)
    m_cons_glob = compute_metrics(cm_consensus_global)

    print(f"Control (V28-A Baseline OOF):   Q_seg={m_ctrl_glob['Q_seg']:.4f} | mIoU={m_ctrl_glob['mIoU']:.4f} | F1={m_ctrl_glob['Macro_F1']:.4f}")
    print(f"  IoUs per class: {m_ctrl_glob['class_ious']}")
    print(f"Consensus (Visibility-Weighted): Q_seg={m_cons_glob['Q_seg']:.4f} | mIoU={m_cons_glob['mIoU']:.4f} | F1={m_cons_glob['Macro_F1']:.4f}")
    print(f"  IoUs per class: {m_cons_glob['class_ious']}")
    print(f"  Net Delta Q_seg: {m_cons_glob['Q_seg'] - m_ctrl_glob['Q_seg']:+7.4f}")
    print(f"  Net Delta mIoU:  {m_cons_glob['mIoU'] - m_ctrl_glob['mIoU']:+7.4f}")
    print(f"  Net Delta F1:    {m_cons_glob['Macro_F1'] - m_ctrl_glob['Macro_F1']:+7.4f}")
    print("=" * 90)

    # Count how many of the 20 scenes improved
    improved_scenes = [sc for sc, d in per_scene_results.items() if d["delta_qseg"] > 0]
    regressed_scenes = [sc for sc, d in per_scene_results.items() if d["delta_qseg"] < 0]
    print(f"\nAdvancement Gate Audit across all 20 scenes:")
    print(f"  Improved scenes: {len(improved_scenes)} / 20 ({len(improved_scenes)/20*100:.1f}%)")
    print(f"  Regressed scenes: {len(regressed_scenes)} / 20 ({len(regressed_scenes)/20*100:.1f}%)")

    out_json = "reports/v24/v30_full_5fold_consensus_report.json"
    with open(out_json, "w") as f:
        json.dump({
            "global_control": m_ctrl_glob,
            "global_consensus": m_cons_glob,
            "per_fold": per_fold_results,
            "per_scene": per_scene_results,
            "total_images": total_imgs,
            "runtime_s": round(time.time() - t_all, 1)
        }, f, indent=2)

    print(f"Saved complete 5-fold report to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    benchmark_full_5fold(stride=args.stride)
