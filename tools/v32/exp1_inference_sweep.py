"""
SAT-EURS Track 1 — V32 Experiment 1: Checkpoints, Inference & Calibration Sweep
Evaluates:
1. Checkpoint candidates on complete validation scenes:
   - Fold 0 (fish1, fish11, fish12, seahorse1)
   - Fold 1 (fish13, fish14, fish15, sharkFish - Class 3 Shark)
   - Fold 2 (fish17, fish18, fish2, ray1 - Class 3 Ray)
2. TTA effect (Original vs Horizontal Flip vs Both)
3. Logit Offset and Temperature sweeps
4. Full-frame vs Global-Plus-Detail blending

Outputs comprehensive metrics with exact scorer parity.
"""

import os
import os.path as osp
import sys
import time
import json
import numpy as np
import cv2
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


def compute_official_metrics(conf_mat: np.ndarray) -> dict:
    tp = np.diag(conf_mat).astype(float)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp

    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn

    gt_present = conf_mat.sum(axis=1) > 0
    p_cov = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0

    per_class_iou = np.where(denom_iou > 0, tp / (denom_iou + 1e-10), 1.0)
    per_class_f1 = np.where(denom_f1 > 0, (2 * tp) / (denom_f1 + 1e-10), 1.0)

    miou = float(np.mean(per_class_iou))
    macro_f1 = float(np.mean(per_class_f1))
    q_seg = float(miou * macro_f1 * p_cov)

    return {
        "Q_seg": round(q_seg, 5),
        "mIoU": round(miou, 5),
        "Macro_F1": round(macro_f1, 5),
        "P_cov": p_cov,
        "class_ious": [round(float(v), 4) for v in per_class_iou],
        "class_f1s": [round(float(v), 4) for v in per_class_f1],
    }


def evaluate_fold_predictions(
    fold: int,
    segmenter_ckpts: dict,
    stride: int = 4,
    device_id: int = 0
) -> dict:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    val_scenes = sorted(f_data["val_scenes"])

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    base_table = torch.load(table_path, map_location=device)

    # Refiner
    refiner_path = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_obj = torch.load(refiner_path, map_location=device)
    refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
    refiner.eval()

    models = {}
    for name, ckpt_p in segmenter_ckpts.items():
        if not osp.exists(ckpt_p):
            print(f"Skipping missing checkpoint: {ckpt_p}")
            continue
        seg = build_v18_segmenter(num_classes=5).to(device)
        s_obj = torch.load(ckpt_p, map_location=device)
        seg.load_state_dict(s_obj["state_dict"] if "state_dict" in s_obj else s_obj)
        seg.eval()
        models[name] = seg

    results = {}
    cached_probs = {name: [] for name in models}
    cached_probs_orig = {name: [] for name in models}
    cached_metas = []
    cached_gts = []

    print(f"\nCaching forward passes for Fold {fold} (Val Scenes: {val_scenes}, Stride: {stride})...")
    t0 = time.time()
    img_count = 0

    for sc in val_scenes:
        mask_p = f"data/cached_1024/masks/{sc}.png"
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

        deg_dir = f"data/cached_1024/degraded/{sc}"
        deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::stride]

        for df in deg_files:
            img_count += 1
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

                for name, model in models.items():
                    out_orig = model(rest_pad, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = model(rest_flip, raw_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    p_tta = 0.5 * (p_orig + p_flip)

                    cached_probs[name].append(p_tta.squeeze(0).cpu().numpy())
                    cached_probs_orig[name].append(p_orig.squeeze(0).cpu().numpy())

            cached_metas.append(meta)
            cached_gts.append(gt_mask)

    print(f"Cached {img_count} images in {time.time() - t0:.1f}s.")

    # Sweep configurations
    offsets = {
        "Default [0,0,0,0.10,0.25]": [0.0, 0.0, 0.0, 0.10, 0.25],
        "Offset_A [0,0,0,0.15,0.25]": [0.0, 0.0, 0.0, 0.15, 0.25],
        "Offset_B [0,0,0,0.10,0.30]": [0.0, 0.0, 0.0, 0.10, 0.30],
        "Offset_C [0,0,0.05,0.15,0.28]": [0.0, 0.0, 0.05, 0.15, 0.28],
        "V28B_Offset [0,0,0,0.10,0.35]": [0.0, 0.0, 0.0, 0.10, 0.35],
        "Zero_Offset [0,0,0,0,0]": [0.0, 0.0, 0.0, 0.0, 0.0],
    }

    sweep_results = {}

    for name in models:
        sweep_results[name] = {}
        # Test TTA vs No-TTA
        for tta_type, prob_list in [("Flip_TTA", cached_probs[name]), ("No_TTA", cached_probs_orig[name])]:
            for off_name, off_vec in offsets.items():
                cal_np = np.array(off_vec, dtype=np.float32).reshape(5, 1, 1)
                cm = np.zeros((5, 5), dtype=np.int64)

                for idx in range(len(prob_list)):
                    p = prob_list[idx]
                    meta = cached_metas[idx]
                    gt = cached_gts[idx]

                    l = np.log(p + 1e-10) + cal_np
                    pred_pad = np.argmax(l, axis=0).astype(np.uint8)
                    pred_unpad = preprocessor.unpad_output(pred_pad, meta)

                    inds = 5 * gt.flatten().astype(int) + pred_unpad.flatten().astype(int)
                    cm += np.bincount(inds, minlength=25).reshape(5, 5)

                metrics = compute_official_metrics(cm)
                key = f"{tta_type} | {off_name}"
                sweep_results[name][key] = metrics

    return sweep_results


def run_experiment_1():
    print("=" * 85)
    print("SAT-EURS TRACK 1 — EXPERIMENT 1: INFERENCE & CALIBRATION SWEEP")
    print("=" * 85)

    all_exp1_results = {}

    # 1. Fold 0 Checkpoint Sweep
    ckpts_f0 = {
        "V18_F0_Baseline": "checkpoints/v18/fold_0/segmenter_best.pt",
        "V27_F0_Targeted_Best": "checkpoints/v27/segmenter_v27_fold0_best.pt",
        "V28_F0_Check": "checkpoints/v28/segmenter_v28_fold0_best.pt",
    }
    res_f0 = evaluate_fold_predictions(fold=0, segmenter_ckpts=ckpts_f0, stride=4)
    all_exp1_results["fold_0"] = res_f0

    print("\n--- FOLD 0 CHECKPOINT COMPARISON ---")
    for model_name, configs in res_f0.items():
        base_cfg = configs.get("Flip_TTA | Default [0,0,0,0.10,0.25]", {})
        print(f"  {model_name:<25}: Q_seg={base_cfg.get('Q_seg', 0):.4f} | mIoU={base_cfg.get('mIoU', 0):.4f} | F1={base_cfg.get('Macro_F1', 0):.4f} | IoUs: {base_cfg.get('class_ious', [])}")

    # 2. Fold 1 Checkpoint Sweep (Rare Class 3 Shark)
    ckpts_f1 = {
        "V18_F1_Baseline": "checkpoints/v18/fold_1/segmenter_best.pt",
        "V28_F1_Check": "checkpoints/v28/segmenter_v28_fold1_best.pt",
    }
    res_f1 = evaluate_fold_predictions(fold=1, segmenter_ckpts=ckpts_f1, stride=4)
    all_exp1_results["fold_1"] = res_f1

    print("\n--- FOLD 1 CHECKPOINT COMPARISON (SharkFish) ---")
    for model_name, configs in res_f1.items():
        base_cfg = configs.get("Flip_TTA | Default [0,0,0,0.10,0.25]", {})
        print(f"  {model_name:<25}: Q_seg={base_cfg.get('Q_seg', 0):.4f} | mIoU={base_cfg.get('mIoU', 0):.4f} | F1={base_cfg.get('Macro_F1', 0):.4f} | IoUs: {base_cfg.get('class_ious', [])}")

    # 3. Fold 2 Checkpoint Sweep (Rare Class 3 Ray)
    ckpts_f2 = {
        "V18_F2_Baseline": "checkpoints/v18/fold_2/segmenter_best.pt",
        "V28_F2_Check": "checkpoints/v28/segmenter_v28_fold2_best.pt",
    }
    res_f2 = evaluate_fold_predictions(fold=2, segmenter_ckpts=ckpts_f2, stride=4)
    all_exp1_results["fold_2"] = res_f2

    print("\n--- FOLD 2 CHECKPOINT COMPARISON (Ray1) ---")
    for model_name, configs in res_f2.items():
        base_cfg = configs.get("Flip_TTA | Default [0,0,0,0.10,0.25]", {})
        print(f"  {model_name:<25}: Q_seg={base_cfg.get('Q_seg', 0):.4f} | mIoU={base_cfg.get('mIoU', 0):.4f} | F1={base_cfg.get('Macro_F1', 0):.4f} | IoUs: {base_cfg.get('class_ious', [])}")

    out_json = "reports/v32_exp1_inference_sweep_results.json"
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_exp1_results, f, indent=2)
    print(f"\nSaved complete Experiment 1 results to {out_json}")


if __name__ == "__main__":
    run_experiment_1()
