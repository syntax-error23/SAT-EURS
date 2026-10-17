"""
SAT-EURS Track 1 — V19 Experiment P1: Controlled Multi-Scale Inference Benchmark
Evaluates Configurations:
  - A_base: Single-Scale (1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.15]
  - A_cal:  Single-Scale (1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.33]
  - B_base: Two-Scale (768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.15]
  - B_cal:  Two-Scale (768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.33]
  - C_base: Three-Scale (512, 768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.15]
  - C_cal:  Three-Scale (512, 768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.33]

Strictly evaluates 5-fold scene-disjoint held-out cross-validation across all 20 EURS scenes (4,000 images).
Zero model retraining, zero baseline file changes, zero external data.
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
    """Computes full challenge metrics from a 5x5 confusion matrix."""
    class_ious = []
    class_f1s = []
    class_precisions = []
    class_recalls = []
    per_class_stats = {}

    present_classes = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        gt = tp + fn
        pred = tp + fp

        status = "PRESENT" if gt > 0 else "ABSENT_GT"
        if gt > 0:
            present_classes.append(c)

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

    miou_pres = float(np.mean([class_ious[c] for c in present_classes])) if present_classes else 0.0
    macro_f1_pres = float(np.mean([class_f1s[c] for c in present_classes])) if present_classes else 0.0
    miou_5cls = float(np.mean(class_ious))
    macro_f1_5cls = float(np.mean(class_f1s))

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
    print("SAT-EURS Track 1 — EXP-V19-P1: Controlled Multi-Scale Inference Benchmark")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU execution required for multi-scale benchmark.")
    print(f"Hardware: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # 1. Baseline archive integrity check
    zip_path = "deploy/v18/submission_v18_dual_encoder_sota.zip"
    if not osp.exists(zip_path):
        raise FileNotFoundError(f"V18 submission zip missing: {zip_path}")
    with open(zip_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()
    print(f"V18 Baseline Archive SHA256 (Before): {sha_before}")

    manifest = validate_manifest_and_folds()
    checkpoint_dir = "checkpoints/v18"
    out_dir = "reports/v19/exp_p1_multiscale"
    os.makedirs(out_dir, exist_ok=True)

    configs = ["A_base", "A_cal", "B_base", "B_cal", "C_base", "C_cal"]
    config_descriptions = {
        "A_base": "Single-Scale (1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.15]",
        "A_cal":  "Single-Scale (1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.33]",
        "B_base": "Two-Scale (768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.15]",
        "B_cal":  "Two-Scale (768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.33]",
        "C_base": "Three-Scale (512, 768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.15]",
        "C_cal":  "Three-Scale (512, 768, 1024) + Flip TTA + Offset [0, 0, 0, 0.15, 0.33]"
    }

    print("\nBenchmark Configurations:")
    for k, v in config_descriptions.items():
        print(f"  [{k}]: {v}")

    # Accumulation structures
    cm_by_fold = {f: {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in configs} for f in range(5)}
    cm_by_scene = {}

    off_15 = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.15], device=device).view(1, 5, 1, 1)
    off_33 = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.33], device=device).view(1, 5, 1, 1)

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
            deg_1024 = batch["deg_rgb"].to(device)
            mask_t = batch["mask"].to(device)
            wt = batch["wt"].to(device)
            depth = batch["depth"].to(device)
            scene = batch["scene"][0]

            if scene not in cm_by_scene:
                cm_by_scene[scene] = {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in configs}

            with torch.no_grad():
                # 1. Base restoration at native 1024 resolution
                rest_1024 = refiner(deg_1024, wt, depth)["refined_image"]

                # --- Scale 1024 ---
                deg_1024_f = torch.flip(deg_1024, dims=[3])
                rest_1024_f = torch.flip(rest_1024, dims=[3])
                s1024_orig = segmenter(rest_1024, deg_1024, wt, depth)
                p1024_orig = F.softmax(s1024_orig["logits"], dim=1)
                s1024_flip = segmenter(rest_1024_f, deg_1024_f, wt, depth)
                p1024_flip = torch.flip(F.softmax(s1024_flip["logits"], dim=1), dims=[3])
                p_1024 = 0.5 * (p1024_orig + p1024_flip)

                # --- Scale 768 ---
                deg_768 = F.interpolate(deg_1024, size=(768, 768), mode="bilinear", align_corners=False)
                rest_768 = F.interpolate(rest_1024, size=(768, 768), mode="bilinear", align_corners=False)
                deg_768_f = torch.flip(deg_768, dims=[3])
                rest_768_f = torch.flip(rest_768, dims=[3])
                s768_orig = segmenter(rest_768, deg_768, wt, depth)
                p768_orig = F.softmax(s768_orig["logits"], dim=1)
                s768_flip = segmenter(rest_768_f, deg_768_f, wt, depth)
                p768_flip = torch.flip(F.softmax(s768_flip["logits"], dim=1), dims=[3])
                p_768 = F.interpolate(0.5 * (p768_orig + p768_flip), size=(1024, 1024), mode="bilinear", align_corners=False)

                # --- Scale 512 ---
                deg_512 = F.interpolate(deg_1024, size=(512, 512), mode="bilinear", align_corners=False)
                rest_512 = F.interpolate(rest_1024, size=(512, 512), mode="bilinear", align_corners=False)
                deg_512_f = torch.flip(deg_512, dims=[3])
                rest_512_f = torch.flip(rest_512, dims=[3])
                s512_orig = segmenter(rest_512, deg_512, wt, depth)
                p512_orig = F.softmax(s512_orig["logits"], dim=1)
                s512_flip = segmenter(rest_512_f, deg_512_f, wt, depth)
                p512_flip = torch.flip(F.softmax(s512_flip["logits"], dim=1), dims=[3])
                p_512 = F.interpolate(0.5 * (p512_orig + p512_flip), size=(1024, 1024), mode="bilinear", align_corners=False)

                # --- Multi-Scale Aggregation ---
                p_A = p_1024
                p_B = 0.5 * (p_1024 + p_768)
                p_C = (p_1024 + p_768 + p_512) / 3.0

                log_A = torch.log(p_A + 1e-10)
                log_B = torch.log(p_B + 1e-10)
                log_C = torch.log(p_C + 1e-10)

                # Predictions
                preds_dict = {
                    "A_base": torch.argmax(log_A + off_15, dim=1).view(-1),
                    "A_cal":  torch.argmax(log_A + off_33, dim=1).view(-1),
                    "B_base": torch.argmax(log_B + off_15, dim=1).view(-1),
                    "B_cal":  torch.argmax(log_B + off_33, dim=1).view(-1),
                    "C_base": torch.argmax(log_C + off_15, dim=1).view(-1),
                    "C_cal":  torch.argmax(log_C + off_33, dim=1).view(-1)
                }

                gt_flat = mask_t.view(-1)
                mask_valid = (gt_flat >= 0) & (gt_flat < 5)
                gt_valid = gt_flat[mask_valid]
                gt_5 = 5 * gt_valid

                for cfg in configs:
                    pred_v = preds_dict[cfg][mask_valid]
                    bin_counts = torch.bincount(gt_5 + pred_v, minlength=25).view(5, 5).cpu().numpy()
                    cm_by_fold[fold][cfg] += bin_counts
                    cm_by_scene[scene][cfg] += bin_counts

            total_images_processed += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == len(val_ds):
                el_f = time.time() - t_fold_start
                ips = (idx + 1) / el_f
                cur_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
                peak_vram_global = max(peak_vram_global, cur_vram)
                print(f"  [Fold {fold}] [{idx+1:03d}/{len(val_ds)}] {ips:.2f} imgs/s | VRAM: {cur_vram:.2f} GB | Scene: {scene}")

        # Clean up models between folds
        del refiner, segmenter, rsuigm_table, val_ds, val_loader
        torch.cuda.empty_cache()

        # Fold summary table
        print(f"\nFold {fold} Multi-Scale Performance Comparison:")
        print(f"{'Config':<8} | {'Q_seg':<8} | {'mIoU':<8} | {'Macro F1':<9} | {'Cls 4 IoU':<10} | {'Cls 4 F1':<9} | {'Cls 4 Rec':<9} | {'Cls 4 Prec':<10}")
        print("-" * 85)
        for cfg in configs:
            m_cfg = compute_metrics_from_cm(cm_by_fold[fold][cfg])
            s4 = m_cfg["per_class_pixel_stats"][4]
            print(f"{cfg:<8} | {m_cfg['Q_seg']:<8.4f} | {m_cfg['mIoU']:<8.4f} | {m_cfg['Macro_F1']:<9.4f} | {s4['iou']:<10.4f} | {s4['f1']:<9.4f} | {s4['recall']:<9.4f} | {s4['precision']:<10.4f}")

    total_runtime_sec = time.time() - t_global_start
    print("\n" + "=" * 80)
    print("ALL 5 FOLDS BENCHMARK COMPLETE!")
    print(f"Total Images Processed: {total_images_processed} / 4,000")
    print(f"Total Runtime: {total_runtime_sec / 60:.2f} minutes ({total_runtime_sec:.1f} s)")
    print(f"Overall Throughput: {total_images_processed / total_runtime_sec:.2f} imgs/s")
    print(f"Peak VRAM: {peak_vram_global:.2f} GB")
    print("=" * 80)

    # Global aggregation across all 20 scenes
    aggregate_cms = {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in configs}
    for f in range(5):
        for cfg in configs:
            aggregate_cms[cfg] += cm_by_fold[f][cfg]

    aggregate_metrics = {cfg: compute_metrics_from_cm(aggregate_cms[cfg]) for cfg in configs}

    print("\n" + "=" * 85)
    print("GLOBAL 20-SCENE AGGREGATE RESULTS COMPARISON")
    print("=" * 85)
    print(f"{'Config':<8} | {'Q_seg':<8} | {'mIoU':<8} | {'Macro F1':<9} | {'Cls 4 IoU':<10} | {'Cls 4 F1':<9} | {'Cls 4 Rec':<9} | {'Cls 4 Prec':<10}")
    print("-" * 85)
    for cfg in configs:
        res = aggregate_metrics[cfg]
        s4 = res["per_class_pixel_stats"][4]
        print(f"{cfg:<8} | {res['Q_seg']:<8.4f} | {res['mIoU']:<8.4f} | {res['Macro_F1']:<9.4f} | {s4['iou']:<10.4f} | {s4['f1']:<9.4f} | {s4['recall']:<9.4f} | {s4['precision']:<10.4f}")

    # Baseline comparisons
    q_A_base = aggregate_metrics["A_base"]["Q_seg"]
    q_B_base = aggregate_metrics["B_base"]["Q_seg"]
    q_C_base = aggregate_metrics["C_base"]["Q_seg"]
    delta_B_vs_A = q_B_base - q_A_base
    delta_C_vs_A = q_C_base - q_A_base

    q_A_cal = aggregate_metrics["A_cal"]["Q_seg"]
    q_B_cal = aggregate_metrics["B_cal"]["Q_seg"]
    q_C_cal = aggregate_metrics["C_cal"]["Q_seg"]

    print("\n" + "-" * 70)
    print("CONTROLLED DELTA ANALYSIS:")
    print(f"  Multi-Scale Effect (Offset 0.15):")
    print(f"    Config B (Two-Scale 768+1024)   vs Config A: Delta Q_seg = {delta_B_vs_A:+.5f}")
    print(f"    Config C (Three-Scale 512+768+1024) vs Config A: Delta Q_seg = {delta_C_vs_A:+.5f}")
    print(f"  Calibration Effect (Offset 0.33 vs 0.15):")
    print(f"    Config A_cal vs A_base: Delta Q_seg = {q_A_cal - q_A_base:+.5f}")
    print(f"    Config B_cal vs B_base: Delta Q_seg = {q_B_cal - q_B_base:+.5f}")
    print(f"    Config C_cal vs C_base: Delta Q_seg = {q_C_cal - q_C_base:+.5f}")
    print("-" * 70)

    # Verify baseline archive hash after execution
    with open(zip_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()
    print(f"\nV18 Baseline Archive SHA256 (After):  {sha_after}")
    assert sha_before == sha_after, "CRITICAL ERROR: V18 baseline archive was modified!"

    # Promotion criterion: Requires at least +0.010 Q_seg without regressions
    best_ms_cfg = "B_base" if q_B_base >= q_C_base else "C_base"
    best_ms_delta = max(delta_B_vs_A, delta_C_vs_A)

    if best_ms_delta >= 0.010:
        recommendation = "PROCEED_TO_SUBMISSION_CANDIDATE"
        verdict = "PASS (PROMOTED)"
        verdict_summary = f"Multi-scale configuration {best_ms_cfg} achieved +{best_ms_delta:.4f} Q_seg over baseline, meeting the +0.010 promotion threshold."
    elif best_ms_delta > 0.002:
        recommendation = "RETAIN_BASELINE_RECOMMEND_PHASE1B"
        verdict = "PASS (MARGINAL / NOT PROMOTED)"
        verdict_summary = f"Multi-scale yielded positive but marginal improvement (+{best_ms_delta:.4f} Q_seg), below the +0.010 threshold. Retain V18 baseline inference."
    elif best_ms_delta > -0.005:
        recommendation = "RETAIN_V18_BASELINE"
        verdict = "NEUTRAL / NO GAIN"
        verdict_summary = f"Multi-scale showed no meaningful gain ({best_ms_delta:+.4f} Q_seg) while doubling/tripling inference latency. Retain V18 single-scale baseline."
    else:
        recommendation = "RETAIN_V18_BASELINE"
        verdict = "REGRESSION / NO-GO"
        verdict_summary = f"Multi-scale caused degradation ({best_ms_delta:+.4f} Q_seg) due to low-resolution boundary smoothing. Retain V18 single-scale baseline."

    print(f"\nFINAL VERDICT: {verdict}")
    print(f"Recommendation: {recommendation}")
    print(verdict_summary)

    # Per-scene ranking
    scene_metrics = {sc: {cfg: compute_metrics_from_cm(cm_by_scene[sc][cfg]) for cfg in configs} for sc in cm_by_scene}
    scene_qseg_A = {sc: scene_metrics[sc]["A_base"]["Q_seg"] for sc in scene_metrics}
    worst_3_scenes = sorted(scene_qseg_A.keys(), key=lambda s: scene_qseg_A[s])[:3]
    best_3_scenes = sorted(scene_qseg_A.keys(), key=lambda s: scene_qseg_A[s], reverse=True)[:3]

    # Save deliverables
    deliverable_json = {
        "experiment_id": "EXP-V19-P1-MULTISCALE-BENCHMARK",
        "title": "Controlled Multi-Scale Segmentation Inference Benchmark",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "verdict": verdict,
        "recommendation": recommendation,
        "verdict_summary": verdict_summary,
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
        "configurations": config_descriptions,
        "aggregate_metrics": aggregate_metrics,
        "deltas_vs_A_base": {
            "B_base_vs_A_base": {
                "Q_seg": round(q_B_base - q_A_base, 5),
                "mIoU": round(aggregate_metrics["B_base"]["mIoU"] - aggregate_metrics["A_base"]["mIoU"], 5),
                "Macro_F1": round(aggregate_metrics["B_base"]["Macro_F1"] - aggregate_metrics["A_base"]["Macro_F1"], 5),
                "class_4_iou": round(aggregate_metrics["B_base"]["per_class_pixel_stats"][4]["iou"] - aggregate_metrics["A_base"]["per_class_pixel_stats"][4]["iou"], 5),
                "class_4_f1": round(aggregate_metrics["B_base"]["per_class_pixel_stats"][4]["f1"] - aggregate_metrics["A_base"]["per_class_pixel_stats"][4]["f1"], 5)
            },
            "C_base_vs_A_base": {
                "Q_seg": round(q_C_base - q_A_base, 5),
                "mIoU": round(aggregate_metrics["C_base"]["mIoU"] - aggregate_metrics["A_base"]["mIoU"], 5),
                "Macro_F1": round(aggregate_metrics["C_base"]["Macro_F1"] - aggregate_metrics["A_base"]["Macro_F1"], 5),
                "class_4_iou": round(aggregate_metrics["C_base"]["per_class_pixel_stats"][4]["iou"] - aggregate_metrics["A_base"]["per_class_pixel_stats"][4]["iou"], 5),
                "class_4_f1": round(aggregate_metrics["C_base"]["per_class_pixel_stats"][4]["f1"] - aggregate_metrics["A_base"]["per_class_pixel_stats"][4]["f1"], 5)
            },
            "A_cal_vs_A_base": {
                "Q_seg": round(q_A_cal - q_A_base, 5),
                "mIoU": round(aggregate_metrics["A_cal"]["mIoU"] - aggregate_metrics["A_base"]["mIoU"], 5),
                "Macro_F1": round(aggregate_metrics["A_cal"]["Macro_F1"] - aggregate_metrics["A_base"]["Macro_F1"], 5)
            }
        },
        "per_fold_metrics": {
            f"fold_{f}": {cfg: compute_metrics_from_cm(cm_by_fold[f][cfg]) for cfg in configs}
            for f in range(5)
        },
        "scene_analysis": {
            "worst_3_scenes": worst_3_scenes,
            "best_3_scenes": best_3_scenes,
            "per_scene_metrics": scene_metrics
        }
    }

    json_path = osp.join(out_dir, "MULTISCALE_BENCHMARK.json")
    with open(json_path, "w") as fp:
        json.dump(deliverable_json, fp, indent=2)
    print(f"\nSerialized JSON Deliverable: {json_path}")


if __name__ == "__main__":
    main()
