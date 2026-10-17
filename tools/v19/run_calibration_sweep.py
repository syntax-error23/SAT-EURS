"""
SAT-EURS Track 1 — V19 Experiment P0: Full 20-Scene Streaming Calibration Sweep
Implementation of EXP-V19-P0:
- Isolated evaluation script for class-4 logit offset sweep.
- Strict scene-disjoint cross-validation: 5 folds, 20 scenes, 4,000 images.
- Preserves all checkpoints, baseline files, and restoration tables.
- Evaluates 11 candidate offsets simultaneously: [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35]
- Uses 2-view Horizontal-Flip TTA with exact production probability aggregation.
- Accumulates streaming confusion matrices with zero disk bloat.
- Performs 5-fold cross-fitted calibration selection to prevent overfitting.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import os.path as osp
import sys
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from data.v18.dataset import V18EURSInvarianceDataset
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    load_checkpoint_with_provenance_check,
    get_fold_data,
    validate_manifest_and_folds
)


def compute_metrics_from_cm(cm: np.ndarray, num_classes: int = 5) -> Dict[str, Any]:
    """Computes all segmentation metrics from a 5x5 confusion matrix."""
    class_ious = []
    class_f1s = []
    class_precisions = []
    class_recalls = []
    per_class_stats = {}

    present_classes = []
    absent_classes = []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        gt = tp + fn
        pred = tp + fp

        status = "PRESENT" if gt > 0 else "ABSENT_GT"
        if gt > 0:
            present_classes.append(c)
        else:
            absent_classes.append(c)

        denom_iou = tp + fp + fn
        iou = float(tp / denom_iou) if denom_iou > 0 else 0.0
        prec = float(tp / pred) if pred > 0 else 0.0
        rec = float(tp / gt) if gt > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        class_ious.append(iou)
        class_f1s.append(f1)
        class_precisions.append(prec)
        class_recalls.append(rec)

        per_class_stats[c] = {
            "class_id": c,
            "status": status,
            "gt_pixels": gt,
            "pred_pixels": pred,
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "iou": round(iou, 5),
            "f1": round(f1, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5)
        }

    # Present class averages
    if len(present_classes) > 0:
        miou_pres = float(np.mean([class_ious[c] for c in present_classes]))
        macro_f1_pres = float(np.mean([class_f1s[c] for c in present_classes]))
    else:
        miou_pres = 0.0
        macro_f1_pres = 0.0

    # 5-class conservative averages
    miou_5cls = float(np.mean(class_ious))
    macro_f1_5cls = float(np.mean(class_f1s))

    # Coverage penalty: 1.0 if all present classes have TP > 0
    cov_penalty = 1.0
    for c in present_classes:
        if per_class_stats[c]["tp_pixels"] == 0:
            cov_penalty = 0.0
            break

    q_seg_pres = miou_pres * macro_f1_pres * cov_penalty
    q_seg_5cls = miou_5cls * macro_f1_5cls * cov_penalty

    return {
        "Q_seg": round(q_seg_pres, 5),
        "mIoU": round(miou_pres, 5),
        "Macro_F1": round(macro_f1_pres, 5),
        "Q_seg_5class": round(q_seg_5cls, 5),
        "mIoU_5class": round(miou_5cls, 5),
        "Macro_F1_5class": round(macro_f1_5cls, 5),
        "coverage_penalty": cov_penalty,
        "present_classes": present_classes,
        "class_ious": [round(x, 5) for x in class_ious],
        "class_f1s": [round(x, 5) for x in class_f1s],
        "class_precisions": [round(x, 5) for x in class_precisions],
        "class_recalls": [round(x, 5) for x in class_recalls],
        "per_class_pixel_stats": per_class_stats,
        "confusion_matrix": cm.tolist()
    }


def main():
    print("=" * 80)
    print("SAT-EURS Track 1 — EXP-V19-P0: Full 20-Scene Streaming Calibration Sweep")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU execution required for calibration sweep.")
    print(f"Hardware: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # 1. Verify V18 baseline archive hash
    zip_path = "deploy/v18/submission_v18_dual_encoder_sota.zip"
    if not osp.exists(zip_path):
        raise FileNotFoundError(f"V18 submission zip missing: {zip_path}")
    with open(zip_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()
    print(f"V18 Baseline Archive SHA256 (Before): {sha_before}")

    manifest = validate_manifest_and_folds()
    checkpoint_dir = "checkpoints/v18"
    out_dir = "reports/v19/exp_p0_calibration"
    os.makedirs(out_dir, exist_ok=True)

    offsets = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35]
    print(f"Candidate Class-4 Offsets to Evaluate ({len(offsets)}): {offsets}")
    print(f"Class-3 Offset Fixed: +0.15")

    # Data structures for accumulation
    # cm_by_fold_and_offset[fold][off] -> np.ndarray (5, 5)
    cm_by_fold = {f: {off: np.zeros((5, 5), dtype=np.int64) for off in offsets} for f in range(5)}
    cm_by_scene = {}

    total_images_processed = 0
    t_global_start = time.time()
    peak_vram_global = 0.0

    for fold in range(5):
        print(f"\n" + "-" * 70)
        print(f"STARTING EVALUATION OF FOLD {fold} / 4")
        fold_data = get_fold_data(manifest, fold)
        train_scenes = fold_data["train_scenes"]
        val_scenes = fold_data["val_scenes"]
        print(f"Held-out Validation Scenes ({len(val_scenes)}): {val_scenes}")

        # Checkpoints & table
        table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
        refiner_path = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        segmenter_path = osp.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")

        rsuigm_table = torch.load(table_path, map_location=device)
        refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
        ref_weights = load_checkpoint_with_provenance_check(refiner_path, "ConditionedRestorationRefinerV18", fold, train_scenes, val_scenes)
        refiner.load_state_dict(ref_weights)
        refiner.eval()

        segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
        seg_weights = load_checkpoint_with_provenance_check(segmenter_path, "EURSAPEXV18Segmenter", fold, train_scenes, val_scenes)
        segmenter.load_state_dict(seg_weights)
        segmenter.eval()

        val_ds = V18EURSInvarianceDataset(scenes=val_scenes, crop_size=1024, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Loaded validation loader: {len(val_ds)} images.")

        t_fold_start = time.time()
        for idx, batch in enumerate(val_loader):
            deg = batch["deg_rgb"].to(device)
            mask_t = batch["mask"].to(device)
            wt = batch["wt"].to(device)
            depth = batch["depth"].to(device)
            scene = batch["scene"][0]

            if scene not in cm_by_scene:
                cm_by_scene[scene] = {off: np.zeros((5, 5), dtype=np.int64) for off in offsets}

            with torch.no_grad():
                out_ref = refiner(deg, wt, depth)
                restored = out_ref["refined_image"]

                # Flip TTA
                deg_f = torch.flip(deg, dims=[3])
                restored_f = torch.flip(restored, dims=[3])

                s_orig = segmenter(restored, deg, wt, depth)
                p_orig = F.softmax(s_orig["logits"], dim=1)

                s_flip = segmenter(restored_f, deg_f, wt, depth)
                p_flip = torch.flip(F.softmax(s_flip["logits"], dim=1), dims=[3])

                p_fused = 0.5 * (p_orig + p_flip)
                log_p = torch.log(p_fused + 1e-10)

                # Class 0..3 baseline with b3 = +0.15
                base_03 = log_p[:, 0:4].clone()
                base_03[:, 3] += 0.15
                max_03, argmax_03 = torch.max(base_03, dim=1) # [1, H, W]
                margin_4 = log_p[:, 4] - max_03               # [1, H, W]

                gt_flat = mask_t.view(-1)
                mask_valid = (gt_flat >= 0) & (gt_flat < 5)
                gt_valid = gt_flat[mask_valid]
                gt_5 = 5 * gt_valid

                for off in offsets:
                    # Class 4 wins iff margin_4 > -off
                    pred = torch.where(margin_4 > -off, 4, argmax_03).view(-1)
                    pred_valid = pred[mask_valid]
                    bin_counts = torch.bincount(gt_5 + pred_valid, minlength=25).view(5, 5).cpu().numpy()

                    cm_by_fold[fold][off] += bin_counts
                    cm_by_scene[scene][off] += bin_counts

            total_images_processed += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == len(val_ds):
                el_f = time.time() - t_fold_start
                ips = (idx + 1) / el_f
                cur_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
                peak_vram_global = max(peak_vram_global, cur_vram)
                print(f"  [Fold {fold}] [{idx+1:03d}/{len(val_ds)}] {ips:.2f} imgs/s | VRAM: {cur_vram:.2f} GB | Scene: {scene}")

        # Clean up models to avoid VRAM creep
        del refiner, segmenter, rsuigm_table, val_ds, val_loader
        torch.cuda.empty_cache()

        # Fold summary table
        print(f"\nFold {fold} Held-out Performance vs. Class-4 Offset:")
        print(f"{'Offset':<8} | {'Q_seg':<8} | {'mIoU':<8} | {'Macro F1':<9} | {'Cls 4 IoU':<10} | {'Cls 4 F1':<9} | {'Cls 4 Rec':<9} | {'Cls 4 Prec':<10}")
        print("-" * 80)
        for off in offsets:
            m_off = compute_metrics_from_cm(cm_by_fold[fold][off])
            s4 = m_off["per_class_pixel_stats"][4]
            print(f"{off:<8.2f} | {m_off['Q_seg']:<8.4f} | {m_off['mIoU']:<8.4f} | {m_off['Macro_F1']:<9.4f} | {s4['iou']:<10.4f} | {s4['f1']:<9.4f} | {s4['recall']:<9.4f} | {s4['precision']:<10.4f}")

    total_runtime_sec = time.time() - t_global_start
    print("\n" + "=" * 80)
    print("SWEEP INFERENCE COMPLETE!")
    print(f"Total Images Processed: {total_images_processed} / 4,000")
    print(f"Total Runtime: {total_runtime_sec / 60:.2f} minutes ({total_runtime_sec:.1f} s)")
    print(f"Overall Throughput: {total_images_processed / total_runtime_sec:.2f} imgs/s")
    print(f"Peak VRAM: {peak_vram_global:.2f} GB")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 2. EXPLORATORY FULL-OOF AGGREGATION ACROSS ALL 20 SCENES
    # -------------------------------------------------------------------------
    print("\n--- 1. EXPLORATORY FULL-OOF RESULTS ACROSS ALL 20 SCENES ---")
    full_oof_cms = {off: np.zeros((5, 5), dtype=np.int64) for off in offsets}
    for f in range(5):
        for off in offsets:
            full_oof_cms[off] += cm_by_fold[f][off]

    full_oof_results = {}
    print(f"{'Offset':<8} | {'Q_seg':<8} | {'mIoU':<8} | {'Macro F1':<9} | {'Cls 4 IoU':<10} | {'Cls 4 F1':<9} | {'Cls 4 Rec':<9} | {'Cls 4 Prec':<10}")
    print("-" * 80)
    for off in offsets:
        res = compute_metrics_from_cm(full_oof_cms[off])
        full_oof_results[str(off)] = res
        s4 = res["per_class_pixel_stats"][4]
        print(f"{off:<8.2f} | {res['Q_seg']:<8.4f} | {res['mIoU']:<8.4f} | {res['Macro_F1']:<9.4f} | {s4['iou']:<10.4f} | {s4['f1']:<9.4f} | {s4['recall']:<9.4f} | {s4['precision']:<10.4f}")

    best_full_oof_off = max(offsets, key=lambda off: full_oof_results[str(off)]["Q_seg"])
    print(f"\nExploratory Best Full-OOF Offset: b4 = {best_full_oof_off:.2f} (Q_seg = {full_oof_results[str(best_full_oof_off)]['Q_seg']:.4f})")
    print("(NOTE: This is calibration-selected on the full dataset and potentially optimistic.)")

    # -------------------------------------------------------------------------
    # 3. FIVE-FOLD CROSS-FITTED CALIBRATION SELECTION (OUTER CV)
    # -------------------------------------------------------------------------
    print("\n--- 2. FIVE-FOLD CROSS-FITTED CALIBRATION SELECTION ---")
    print("For each held-out fold k, selecting optimal b_4 using the other 4 folds, then evaluating on fold k...")

    cross_fitted_selected_offsets = {}
    cross_fitted_cm = np.zeros((5, 5), dtype=np.int64)
    baseline_cm = np.zeros((5, 5), dtype=np.int64)

    for val_fold in range(5):
        # 1. Aggregate training folds
        train_cm = {off: np.zeros((5, 5), dtype=np.int64) for off in offsets}
        for tr_fold in range(5):
            if tr_fold == val_fold:
                continue
            for off in offsets:
                train_cm[off] += cm_by_fold[tr_fold][off]

        # 2. Pick best offset on training folds
        best_tr_off = max(offsets, key=lambda off: compute_metrics_from_cm(train_cm[off])["Q_seg"])
        cross_fitted_selected_offsets[val_fold] = best_tr_off

        # 3. Evaluate selected offset on held-out fold
        cm_held_out = cm_by_fold[val_fold][best_tr_off]
        cross_fitted_cm += cm_held_out

        # Baseline evaluation (b4 = 0.15)
        baseline_cm += cm_by_fold[val_fold][0.15]

        print(f"  Fold {val_fold}: Selected b4 = {best_tr_off:.2f} from folds {[f for f in range(5) if f != val_fold]} -> Evaluated on held-out scenes")

    cross_fitted_metrics = compute_metrics_from_cm(cross_fitted_cm)
    baseline_metrics = compute_metrics_from_cm(baseline_cm)

    print("\n" + "=" * 80)
    print("PRIMARY DECISION: CROSS-FITTED EVALUATION RESULTS")
    print("=" * 80)
    print(f"Selected Offsets per Fold: {cross_fitted_selected_offsets}")
    print(f"Baseline (b4=0.15 across all folds):")
    print(f"  Q_seg:    {baseline_metrics['Q_seg']:.5f}")
    print(f"  mIoU:     {baseline_metrics['mIoU']:.5f}")
    print(f"  Macro F1: {baseline_metrics['Macro_F1']:.5f}")
    print(f"  Class 4 IoU:  {baseline_metrics['per_class_pixel_stats'][4]['iou']:.5f}")
    print(f"  Class 4 F1:   {baseline_metrics['per_class_pixel_stats'][4]['f1']:.5f}")
    print(f"  Class 4 Rec:  {baseline_metrics['per_class_pixel_stats'][4]['recall']:.5f}")
    print(f"  Class 4 Prec: {baseline_metrics['per_class_pixel_stats'][4]['precision']:.5f}")

    print(f"\nCross-Fitted (Tuned strictly out-of-fold):")
    print(f"  Q_seg:    {cross_fitted_metrics['Q_seg']:.5f} (Delta: {cross_fitted_metrics['Q_seg'] - baseline_metrics['Q_seg']:+.5f})")
    print(f"  mIoU:     {cross_fitted_metrics['mIoU']:.5f} (Delta: {cross_fitted_metrics['mIoU'] - baseline_metrics['mIoU']:+.5f})")
    print(f"  Macro F1: {cross_fitted_metrics['Macro_F1']:.5f} (Delta: {cross_fitted_metrics['Macro_F1'] - baseline_metrics['Macro_F1']:+.5f})")
    print(f"  Class 4 IoU:  {cross_fitted_metrics['per_class_pixel_stats'][4]['iou']:.5f} (Delta: {cross_fitted_metrics['per_class_pixel_stats'][4]['iou'] - baseline_metrics['per_class_pixel_stats'][4]['iou']:+.5f})")
    print(f"  Class 4 F1:   {cross_fitted_metrics['per_class_pixel_stats'][4]['f1']:.5f} (Delta: {cross_fitted_metrics['per_class_pixel_stats'][4]['f1'] - baseline_metrics['per_class_pixel_stats'][4]['f1']:+.5f})")
    print(f"  Class 4 Rec:  {cross_fitted_metrics['per_class_pixel_stats'][4]['recall']:.5f} (Delta: {cross_fitted_metrics['per_class_pixel_stats'][4]['recall'] - baseline_metrics['per_class_pixel_stats'][4]['recall']:+.5f})")
    print(f"  Class 4 Prec: {cross_fitted_metrics['per_class_pixel_stats'][4]['precision']:.5f} (Delta: {cross_fitted_metrics['per_class_pixel_stats'][4]['precision'] - baseline_metrics['per_class_pixel_stats'][4]['precision']:+.5f})")

    # 4. Final verification of baseline ZIP SHA256
    with open(zip_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    print(f"\nV18 Baseline Archive SHA256 (After):  {sha_after}")
    assert sha_before == sha_after, "CRITICAL ERROR: V18 baseline archive was modified!"

    # 5. Determine Verdict
    delta_qseg = cross_fitted_metrics["Q_seg"] - baseline_metrics["Q_seg"]
    if delta_qseg > 0.0050:
        verdict = "PASS"
        verdict_msg = f"Cross-fitted calibration demonstrated robust improvement: Q_seg increased by {delta_qseg:+.4f} points."
    elif delta_qseg > 0.0:
        verdict = "PASS (MARGINAL)"
        verdict_msg = f"Cross-fitted calibration yielded marginal improvement: Q_seg increased by {delta_qseg:+.4f} points."
    else:
        verdict = "NO IMPROVEMENT"
        verdict_msg = f"Cross-fitted calibration showed no gain ({delta_qseg:+.4f} Q_seg). Recommend retaining existing V18 calibration."

    print(f"\nFINAL VERDICT: {verdict}")
    print(verdict_msg)

    # 6. Save comprehensive JSON deliverable
    json_deliverable = {
        "experiment_id": "EXP-V19-P0-CALIBRATION-SWEEP",
        "title": "Full 20-Scene Streaming Calibration Sweep (Class-4 Offset)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "verdict": verdict,
        "verdict_summary": verdict_msg,
        "runtime_metrics": {
            "total_images": total_images_processed,
            "total_runtime_sec": round(total_runtime_sec, 2),
            "total_runtime_min": round(total_runtime_sec / 60, 2),
            "throughput_imgs_per_sec": round(total_images_processed / total_runtime_sec, 2),
            "peak_vram_gb": round(peak_vram_global, 2)
        },
        "baseline_archive": {
            "path": zip_path,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "intact": (sha_before == sha_after)
        },
        "baseline_metrics_b4_015": baseline_metrics,
        "cross_fitted_results": {
            "selected_offsets_per_fold": cross_fitted_selected_offsets,
            "metrics": cross_fitted_metrics,
            "deltas_vs_baseline": {
                "Q_seg": round(delta_qseg, 5),
                "mIoU": round(cross_fitted_metrics["mIoU"] - baseline_metrics["mIoU"], 5),
                "Macro_F1": round(cross_fitted_metrics["Macro_F1"] - baseline_metrics["Macro_F1"], 5),
                "class_4_iou": round(cross_fitted_metrics["per_class_pixel_stats"][4]["iou"] - baseline_metrics["per_class_pixel_stats"][4]["iou"], 5),
                "class_4_f1": round(cross_fitted_metrics["per_class_pixel_stats"][4]["f1"] - baseline_metrics["per_class_pixel_stats"][4]["f1"], 5),
                "class_4_recall": round(cross_fitted_metrics["per_class_pixel_stats"][4]["recall"] - baseline_metrics["per_class_pixel_stats"][4]["recall"], 5),
                "class_4_precision": round(cross_fitted_metrics["per_class_pixel_stats"][4]["precision"] - baseline_metrics["per_class_pixel_stats"][4]["precision"], 5)
            }
        },
        "exploratory_full_oof_sweep": full_oof_results,
        "per_fold_sweep": {
            f"fold_{f}": {str(off): compute_metrics_from_cm(cm_by_fold[f][off]) for off in offsets}
            for f in range(5)
        },
        "per_scene_sweep": {
            sc: {str(off): compute_metrics_from_cm(cm_by_scene[sc][off]) for off in offsets}
            for sc in sorted(list(cm_by_scene.keys()))
        }
    }

    json_path = osp.join(out_dir, "CALIBRATION_SWEEP.json")
    with open(json_path, "w") as fp:
        json.dump(json_deliverable, fp, indent=2)
    print(f"\nSerialized JSON Deliverable: {json_path}")


if __name__ == "__main__":
    main()
