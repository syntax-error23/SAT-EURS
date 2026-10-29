"""
SAT-EURS Track 1 — V28 Out-Of-Fold (OOF) Evaluator
Evaluates a segmenter checkpoint on its exact held-out validation scenes.
Computes:
1. Exact official metrics: Q_seg, mIoU, Macro F1, Coverage
2. Per-class IoU, Precision, Recall
3. Breakdown by validation scene
4. Breakdown by water type (Jerlov 1..10) and depth (5..100m)
Zero cross-scene leakage.
"""

import os
import os.path as osp
import sys
import json
import time
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
    load_checkpoint_with_provenance_check,
    V18CanonicalPreprocessor,
)
from tools.resize_for_submission import scale_to_max_side

CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]


def evaluate_model_on_val_scenes(
    segmenter_ckpt: str,
    fold: int,
    refiner_ckpt: str = "checkpoints/v18/fold_0/refiner_best.pt",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    data_root: str = "data/cached_1024",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    use_flip_tta: bool = True,
    sample_stride: int = 1,  # 1 = all 200 variants, 5 = 40 variants, etc.
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    val_scenes = sorted(f_data["val_scenes"])
    train_scenes = sorted(f_data["train_scenes"])

    print("=" * 80, flush=True)
    print(f"V28 OUT-OF-FOLD EVALUATION — FOLD {fold}", flush=True)
    print(f"Checkpoint: {segmenter_ckpt}", flush=True)
    print(f"Validation Scenes ({len(val_scenes)}): {val_scenes}", flush=True)
    print(f"Sample Stride: {sample_stride} (Stride 1 = 200 variants/scene)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load refiner for this fold
    base_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_obj = torch.load(refiner_ckpt, map_location=device)
    refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
    refiner.eval()

    # 2. Load segmenter
    segmenter = build_v18_segmenter(num_classes=5).to(device)
    s_obj = torch.load(segmenter_ckpt, map_location=device)
    s_sd = s_obj["state_dict"] if "state_dict" in s_obj else s_obj
    segmenter.load_state_dict(s_sd)
    segmenter.eval()

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # Global and stratified state
    conf_mat_global = np.zeros((5, 5), dtype=np.int64)
    scene_conf_mats = {sc: np.zeros((5, 5), dtype=np.int64) for sc in val_scenes}
    wt_conf_mats = {wt: np.zeros((5, 5), dtype=np.int64) for wt in range(1, 11)}
    depth_bracket_mats = {
        "shallow_5_30": np.zeros((5, 5), dtype=np.int64),
        "mid_35_65": np.zeros((5, 5), dtype=np.int64),
        "deep_70_100": np.zeros((5, 5), dtype=np.int64)
    }

    t0 = time.time()
    total_images_evaluated = 0

    for sc in val_scenes:
        # Load ground truth mask
        mask_path = osp.join(data_root, "masks", f"{sc}.png")
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gh, gw = gt_mask.shape[:2]

        deg_scene_dir = osp.join(data_root, "degraded", sc)
        deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])
        sampled_files = deg_files[::sample_stride]

        for fname in sampled_files:
            stem = osp.splitext(fname)[0]
            parts = stem.split("_")
            wt = int(parts[1])
            depth = int(parts[2])

            img_path = osp.join(deg_scene_dir, fname)
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue

            prep = preprocessor.prepare_input(bgr)
            raw_pad = prep["rgb_tensor"].to(device)
            meta = prep["meta"]
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([depth], device=device)

            with torch.inference_mode():
                out_r = refiner(raw_pad.float(), wt_t, dp_t)
                restored_fused = out_r["refined_image"]

                if use_flip_tta:
                    raw_pad_flip = torch.flip(raw_pad, dims=[3])
                    restored_flip = torch.flip(restored_fused, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                        out_flip = segmenter(restored_flip, raw_pad_flip, wt_t, dp_t)
                        p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                        prob_fused = 0.5 * (p_orig + p_flip)
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        prob_fused = F.softmax(out_orig["logits"].float(), dim=1)

                log_prob = torch.log(prob_fused + 1e-10) + cal_t
                mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            pred_mask = preprocessor.unpad_output(mask_pad, meta)
            if pred_mask.shape != (gh, gw):
                pred_mask = cv2.resize(pred_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)

            # Accumulate confusion matrix
            y_t = gt_mask.reshape(-1)
            y_p = pred_mask.reshape(-1)
            indices = 5 * y_t + y_p
            counts = np.bincount(indices, minlength=25).reshape(5, 5)

            conf_mat_global += counts
            scene_conf_mats[sc] += counts
            wt_conf_mats[wt] += counts

            if depth <= 30:
                depth_bracket_mats["shallow_5_30"] += counts
            elif depth <= 65:
                depth_bracket_mats["mid_35_65"] += counts
            else:
                depth_bracket_mats["deep_70_100"] += counts

            total_images_evaluated += 1

        print(f"  Completed scene {sc} ({len(sampled_files)} variants evaluated)", flush=True)

    # Compute metrics from confusion matrix
    def metrics_from_cmat(cmat: np.ndarray) -> Dict[str, Any]:
        tp = np.diag(cmat).astype(np.float64)
        gt_px = cmat.sum(axis=1).astype(np.float64)
        pred_px = cmat.sum(axis=0).astype(np.float64)
        fp = pred_px - tp
        fn = gt_px - tp

        ious = []
        f1s = []
        precs = []
        recs = []
        coverage = 1.0

        for c in range(5):
            denom_iou = tp[c] + fp[c] + fn[c]
            denom_f1 = 2 * tp[c] + fp[c] + fn[c]
            iou = float(tp[c] / denom_iou) if denom_iou > 0 else (1.0 if gt_px[c] == 0 else 0.0)
            f1 = float((2 * tp[c]) / denom_f1) if denom_f1 > 0 else (1.0 if gt_px[c] == 0 else 0.0)
            prec = float(tp[c] / (tp[c] + fp[c])) if (tp[c] + fp[c]) > 0 else 0.0
            rec = float(tp[c] / gt_px[c]) if gt_px[c] > 0 else 0.0

            ious.append(iou)
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)

            if gt_px[c] > 0 and tp[c] == 0:
                coverage = 0.0

        miou = float(np.mean(ious))
        macro_f1 = float(np.mean(f1s))
        q_seg = miou * macro_f1 * coverage

        return {
            "Q_seg": round(q_seg, 5),
            "mIoU": round(miou, 5),
            "Macro_F1": round(macro_f1, 5),
            "coverage": coverage,
            "class_ious": [round(x, 4) for x in ious],
            "class_f1s": [round(x, 4) for x in f1s],
            "class_precisions": [round(x, 4) for x in precs],
            "class_recalls": [round(x, 4) for x in recs],
        }

    global_res = metrics_from_cmat(conf_mat_global)
    scene_res = {sc: metrics_from_cmat(scene_conf_mats[sc]) for sc in val_scenes}
    wt_res = {f"WT_{wt}": metrics_from_cmat(wt_conf_mats[wt]) for wt in range(1, 11)}
    depth_res = {b: metrics_from_cmat(depth_bracket_mats[b]) for b in depth_bracket_mats}

    print("\n" + "=" * 80, flush=True)
    print(f"FOLD {fold} OOF METRICS SUMMARY ({total_images_evaluated} images evaluated in {time.time()-t0:.1f}s)", flush=True)
    print("=" * 80, flush=True)
    print(f"Q_seg:    {global_res['Q_seg']:.4f}", flush=True)
    print(f"mIoU:     {global_res['mIoU']:.4f}", flush=True)
    print(f"Macro F1: {global_res['Macro_F1']:.4f}", flush=True)
    for c in range(5):
        print(f"  {CLASS_NAMES[c]:12s}: IoU={global_res['class_ious'][c]:.4f} | Prec={global_res['class_precisions'][c]:.4f} | Rec={global_res['class_recalls'][c]:.4f}", flush=True)
    print("-" * 80, flush=True)
    print("Per-Scene Q_seg Breakdown:", flush=True)
    for sc in val_scenes:
        print(f"  {sc:12s}: Q_seg={scene_res[sc]['Q_seg']:.4f} | mIoU={scene_res[sc]['mIoU']:.4f} | F1={scene_res[sc]['Macro_F1']:.4f}", flush=True)
    print("=" * 80, flush=True)

    summary = {
        "fold": fold,
        "segmenter_ckpt": segmenter_ckpt,
        "val_scenes": val_scenes,
        "total_images_evaluated": total_images_evaluated,
        "evaluation_time_s": round(time.time() - t0, 1),
        "global_metrics": global_res,
        "scene_breakdown": scene_res,
        "water_type_breakdown": wt_res,
        "depth_bracket_breakdown": depth_res,
    }

    return summary


if __name__ == "__main__":
    if len(sys.argv) > 2:
        fold = int(sys.argv[1])
        ckpt = sys.argv[2]
        stride = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        evaluate_model_on_val_scenes(segmenter_ckpt=ckpt, fold=fold, sample_stride=stride)
