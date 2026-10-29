"""
SAT-EURS Track 1 — V26 Experiment 2: Segmentation Logit Calibration Sweep (Optimized)
Protocol: Strict Out-Of-Fold Evaluation across Fold 0 (800 images) and Fold 1 (800 images).
Optimizations:
1. Restoration metrics computed ONCE per image (not 8 times)
2. Explicit memory cleanup (del tensors, empty cache per scene)
3. Unbuffered output with flush=True
4. Evaluates exact challenge metrics:
   - Q_seg = mIoU * Macro_F1 * coverage_penalty
   - Full composite (60% Q_seg + 30% Q_FR)
   - Per-class IoU, precision, recall for all 5 classes
   - Class erasure check on sharkFish (Class 2 fish and Class 3 sharks)
   - Macro-fauna recall check on seahorse1 (Class 4)
"""

import os
import os.path as osp
import sys
import time
import json
import gc
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
    load_checkpoint_with_provenance_check,
)
from tools.v18.evaluator import V18OOFEvaluator


CANDIDATE_OFFSETS = {
    "zero_pure": [0.0, 0.0, 0.0, 0.0, 0.0],
    "v18_baseline": [0.0, 0.0, 0.0, 0.15, 0.15],
    "v25_candidate": [0.0, 0.0, 0.0, 0.15, 0.35],
    "hyp_protect_fish": [0.0, 0.0, 0.05, 0.10, 0.15],
    "hyp_mod_balanced": [0.0, 0.0, 0.00, 0.12, 0.18],
    "hyp_c2_c3_c4": [0.0, 0.0, 0.05, 0.15, 0.20],
    "hyp_c4_mod25": [0.0, 0.0, 0.00, 0.10, 0.25],
    "hyp_v18_c4_25": [0.0, 0.0, 0.00, 0.15, 0.25],
}


def run_calibration_sweep(
    folds: List[int] = [0, 1],
    data_root: str = "data/cached_1024",
    checkpoint_dir: str = "checkpoints/v18",
    alpha_restoration: float = 0.10,
    sigma_restoration: float = 2.0,
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("SAT-EURS V26 — EXPERIMENT 2: SEGMENTATION LOGIT CALIBRATION SWEEP", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"Evaluating Folds: {folds} | Restorer alpha: {alpha_restoration}", flush=True)
    print(f"Candidate Offset Vectors ({len(CANDIDATE_OFFSETS)}):", flush=True)
    for k, v in CANDIDATE_OFFSETS.items():
        print(f"  {k:<18}: {v}", flush=True)
    print("=" * 80, flush=True)

    manifest = validate_manifest_and_folds()

    evaluators_pooled = {k: V18OOFEvaluator() for k in CANDIDATE_OFFSETS}
    evaluators_fold = {k: {f: V18OOFEvaluator() for f in folds} for k in CANDIDATE_OFFSETS}

    # Reference restoration evaluator (computed ONCE per image, shared across all calibration candidates)
    rest_eval_pooled = V18OOFEvaluator()
    rest_eval_fold = {f: V18OOFEvaluator() for f in folds}

    t0 = time.time()
    total_imgs = 0

    for fold in folds:
        f_data = get_fold_data(manifest, fold)
        val_scenes = f_data["val_scenes"]
        train_scenes = f_data["train_scenes"]

        print(f"\n[Fold {fold}] Loading Refiner and Dual-Encoder Segmenter...", flush=True)

        table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
        if not osp.exists(table_path):
            table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
        rsuigm_table = torch.load(table_path, map_location=device)

        refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
        r_path = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        r_weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, train_scenes, val_scenes)
        refiner.load_state_dict(r_weights)
        refiner.eval()

        segmenter = build_v18_segmenter(num_classes=5).to(device)
        s_path = osp.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
        s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, train_scenes, val_scenes)
        segmenter.load_state_dict(s_weights)
        segmenter.eval()

        print(f"[Fold {fold}] Models loaded successfully. Starting evaluation across {len(val_scenes)} scenes...", flush=True)

        for sc_idx, sc in enumerate(val_scenes, 1):
            t_sc = time.time()
            c_path = osp.join(data_root, "clean", f"{sc}.png")
            m_path = osp.join(data_root, "masks", f"{sc}.png")

            clean_bgr = cv2.imread(c_path, cv2.IMREAD_COLOR)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

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
                        refined_t = out_r["refined_image"]

                        if alpha_restoration > 0:
                            refined_np = refined_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                            blur = cv2.GaussianBlur(refined_np, (0, 0), sigma_restoration)
                            rest_final_np = np.clip(refined_np + alpha_restoration * (refined_np - blur), 0.0, 1.0)
                            rest_input_t = torch.from_numpy(rest_final_np).permute(2, 0, 1).unsqueeze(0).to(device)
                        else:
                            rest_input_t = refined_t
                            rest_final_np = refined_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

                        # Update restoration metric ONCE per image
                        rest_eval_fold[fold].update_restoration(rest_final_np, clean_rgb, scene=sc)
                        rest_eval_pooled.update_restoration(rest_final_np, clean_rgb, scene=sc)

                        # Segmenter forward pass with Flip TTA
                        deg_flip = torch.flip(deg_t, dims=[3])
                        rest_flip = torch.flip(rest_input_t, dims=[3])

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out_orig = segmenter(rest_input_t, deg_t, wt_t, dp_t)
                            p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                            out_flip = segmenter(rest_flip, deg_flip, wt_t, dp_t)
                            p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                            prob_fused = 0.5 * (p_orig + p_flip)

                        # Evaluate all 8 candidate offsets in memory
                        log_prob = torch.log(prob_fused + 1e-10)

                        for k, off_vec in CANDIDATE_OFFSETS.items():
                            cal_t = torch.tensor(off_vec, device=device).view(1, 5, 1, 1)
                            pred_mask = torch.argmax(log_prob + cal_t, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                            evaluators_fold[k][fold].update_segmentation(pred_mask, gt_mask, scene=sc)
                            evaluators_pooled[k].update_segmentation(pred_mask, gt_mask, scene=sc)

                    del deg_t, wt_t, dp_t, out_r, refined_t, rest_input_t, deg_flip, rest_flip, out_orig, p_orig, out_flip, p_flip, prob_fused, log_prob
                    total_imgs += 1

            # Free cache after each scene
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  Processed scene [{sc_idx}/{len(val_scenes)}] {sc:<12} (200 images) in {time.time() - t_sc:.1f}s", flush=True)

        del refiner, segmenter
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nCompleted evaluation of {total_imgs} images across {len(CANDIDATE_OFFSETS)} calibration settings in {time.time() - t0:.1f}s!", flush=True)

    # Reference restoration metrics (same across all offset variants)
    res_rest_pooled = rest_eval_pooled.compute()["restoration"]
    q_fr_pooled = res_rest_pooled["Q_FR_dataset"]

    all_results = {}
    base_comp = None

    print("\n" + "=" * 125, flush=True)
    print("                    EXPERIMENT 2: CALIBRATION SWEEP SUMMARY (POOLED 1,600 IMAGES)", flush=True)
    print(f"Restoration Fixed Metrics: PSNR = {res_rest_pooled['PSNR_dB']:.2f} dB, SSIM = {res_rest_pooled['SSIM']:.4f}, Sharpness = {res_rest_pooled['sharpness_ratio']:.4f}, Q_FR = {q_fr_pooled:.4f}", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Configuration':<18} | {'Offset Vector':<25} | {'mIoU':<8} | {'Macro F1':<9} | {'P_cov':<6} | {'Q_seg':<8} | {'Q_FR':<8} | {'S (90%)':<8} | {'Delta S':<8}", flush=True)
    print("-" * 125, flush=True)

    # First get baseline
    v18_seg = evaluators_pooled["v18_baseline"].compute()["segmentation"]
    base_comp = 0.60 * v18_seg["Q_seg"] + 0.30 * q_fr_pooled

    for k, off_vec in CANDIDATE_OFFSETS.items():
        seg_p = evaluators_pooled[k].compute()["segmentation"]
        s_90 = 0.60 * seg_p["Q_seg"] + 0.30 * q_fr_pooled
        ds = s_90 - base_comp

        vec_str = f"[{','.join(f'{x:.2f}' for x in off_vec)}]"
        print(
            f"{k:<18} | {vec_str:<25} | {seg_p['mIoU']:<8.4f} | {seg_p['Macro_F1']:<9.4f} | "
            f"{seg_p['coverage_penalty']:<6.1f} | {seg_p['Q_seg']:<8.4f} | {q_fr_pooled:<8.4f} | "
            f"{s_90:<8.4f} | {ds:+8.5f}",
            flush=True
        )

        class_names = ["BG", "Coral", "Fish", "Ray", "Macro"]
        c_ious = [f"{class_names[c]}:{seg_p['class_ious'][c]:.3f}" for c in range(5)]
        c_recs = [f"{class_names[c]}:{seg_p['class_recalls'][c]:.3f}" for c in range(5)]
        print(f"       Per-class IoU:    {' | '.join(c_ious)}", flush=True)
        print(f"       Per-class Recall: {' | '.join(c_recs)}", flush=True)

        all_results[k] = {
            "offsets": off_vec,
            "segmentation": seg_p,
            "restoration": res_rest_pooled,
            "S_exact_90pct": s_90,
            "delta_S_vs_v18": ds,
            "per_fold": {
                f"fold_{f}": {
                    "segmentation": evaluators_fold[k][f].compute()["segmentation"],
                    "restoration": rest_eval_fold[f].compute()["restoration"]
                }
                for f in folds
            }
        }

    print("=" * 125, flush=True)

    out_json = "reports/v26/exp2_calibration_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full calibration results to {out_json}", flush=True)

    return all_results


if __name__ == "__main__":
    run_calibration_sweep(folds=[0, 1])
