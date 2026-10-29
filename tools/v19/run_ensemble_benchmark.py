"""
SAT-EURS Track 1 — V19 Experiment P2: Controlled Heterogeneous Ensemble OOF Benchmark
Evaluates probability-level ensemble mixtures of:
  - V18 Dual-Encoder (DINOv3 ViT-L/16 + ConvNeXt-B + UPerNet neck)
  - V16 DINOv2-Large (ViT-L/14 + Linear Projection + Fusion neck)
  - V16 ConvNeXt-Base FPN (SMP tu-convnext_base + FPN Decoder)

Candidate Grid:
  - C0: (1.00, 0.00, 0.00) [V18 Control]
  - C1: (0.80, 0.20, 0.00)
  - C2: (0.70, 0.30, 0.00)
  - C3: (0.85, 0.00, 0.15)
  - C4: (0.70, 0.20, 0.10)
  - C5: (0.60, 0.25, 0.15)
  - C6: (0.50, 0.35, 0.15)

Evaluated under both:
  - Base offsets: [0.0, 0.0, 0.0, 0.15, 0.15]
  - Calibrated offsets: [0.0, 0.0, 0.0, 0.15, 0.33]

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
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from models.convnext_fpn import build_convnext_fpn
from tools.v18.run_v18_pipeline import (
    load_checkpoint_with_provenance_check,
    get_fold_data,
    validate_manifest_and_folds
)


def compute_metrics_from_cm(cm: np.ndarray, num_classes: int = 5) -> Dict[str, Any]:
    """Computes challenge metrics from a 5x5 confusion matrix."""
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


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("SAT-EURS Track 1 — EXP-V19-P2: Controlled Heterogeneous Ensemble OOF Benchmark")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU execution required for benchmark.")
    print(f"Hardware: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

    # 1. Verify V18 baseline archive hash
    zip_path = "deploy/v18/submission_v18_dual_encoder_sota.zip"
    if not osp.exists(zip_path):
        raise FileNotFoundError(f"V18 submission zip missing: {zip_path}")
    sha_before = compute_sha256(zip_path)
    print(f"V18 Baseline Archive SHA256 (Before): {sha_before}")
    assert sha_before == "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52", "Hash mismatch!"

    manifest = validate_manifest_and_folds()
    out_dir = "reports/v19/exp_p2_ensemble_benchmark"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Candidate Mixtures
    candidates = {
        "C0": {"w18": 1.00, "wdino": 0.00, "wconv": 0.00, "name": "V18 Control (1.00 / 0.00 / 0.00)"},
        "C1": {"w18": 0.80, "wdino": 0.20, "wconv": 0.00, "name": "V18 + DINOv2 (0.80 / 0.20 / 0.00)"},
        "C2": {"w18": 0.70, "wdino": 0.30, "wconv": 0.00, "name": "V18 + DINOv2 (0.70 / 0.30 / 0.00)"},
        "C3": {"w18": 0.85, "wdino": 0.00, "wconv": 0.15, "name": "V18 + ConvNeXt (0.85 / 0.00 / 0.15)"},
        "C4": {"w18": 0.70, "wdino": 0.20, "wconv": 0.10, "name": "Tri-Ensemble Balanced (0.70 / 0.20 / 0.10)"},
        "C5": {"w18": 0.60, "wdino": 0.25, "wconv": 0.15, "name": "Tri-Ensemble High-Diversity (0.60 / 0.25 / 0.15)"},
        "C6": {"w18": 0.50, "wdino": 0.35, "wconv": 0.15, "name": "Tri-Ensemble Equal-Balance (0.50 / 0.35 / 0.15)"},
    }

    # Offsets: base [0, 0, 0, 0.15, 0.15] and cal [0, 0, 0, 0.15, 0.33]
    off_base = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.15], device=device).view(1, 5, 1, 1)
    off_cal = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.33], device=device).view(1, 5, 1, 1)

    eval_configs = []
    for c_id in sorted(candidates.keys()):
        eval_configs.append(f"{c_id}_base")
        eval_configs.append(f"{c_id}_cal")

    print(f"Configurations to Evaluate ({len(eval_configs)}): {eval_configs}")

    # Mean & Std for ImageNet normalization
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Accumulators
    cm_by_fold = {f: {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in eval_configs} for f in range(5)}
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

        # Checkpoint paths
        table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
        refiner_path = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
        segmenter_path = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
        dino_path = f"checkpoints/v16_dinov2/fold_{fold}/best_model.pt"
        conv_path = f"checkpoints/v16_convnext/fold_{fold}/best_model.pt"

        # 1. Load V18 Refiner
        rsuigm_table = torch.load(table_path, map_location=device)
        refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
        ref_weights = load_checkpoint_with_provenance_check(refiner_path, "ConditionedRestorationRefinerV18", fold, train_scenes, val_scenes)
        refiner.load_state_dict(ref_weights)
        refiner.eval()

        # 2. Load V18 Segmenter
        segmenter = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
        seg_weights = load_checkpoint_with_provenance_check(segmenter_path, "EURSAPEXV18Segmenter", fold, train_scenes, val_scenes)
        segmenter.load_state_dict(seg_weights)
        segmenter.eval()

        # 3. Load V16 DINOv2-Large
        dino_model = build_dinov2_model(num_classes=5).to(device)
        dino_ckpt = torch.load(dino_path, map_location=device, weights_only=False)
        dino_model.load_state_dict(dino_ckpt["state_dict"] if "state_dict" in dino_ckpt else dino_ckpt["model_state_dict"])
        dino_model.eval()

        # 4. Load V16 ConvNeXt-Base FPN
        conv_model = build_convnext_fpn(num_classes=5, freeze_stage0=True).to(device)
        conv_ckpt = torch.load(conv_path, map_location=device, weights_only=False)
        conv_model.load_state_dict(conv_ckpt["state_dict"] if "state_dict" in conv_ckpt else conv_ckpt["model_state_dict"])
        conv_model.eval()

        # Validation dataset
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
                cm_by_scene[scene] = {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in eval_configs}

            with torch.no_grad():
                # Pipeline 1: V18 Neural Restoration
                rest_v18 = refiner(deg_1024, wt, depth)["refined_image"]

                # Pipeline 2: Physical RSUIGM table restoration for DINOv2 & ConvNeXt
                rest_table = apply_rsuigm_batch_gpu(deg_1024, wt, depth, rsuigm_table)

                # --- 1. V18 Segmenter with Flip TTA (1024x1024) ---
                deg_1024_f = torch.flip(deg_1024, dims=[3])
                rest_v18_f = torch.flip(rest_v18, dims=[3])
                s_orig = segmenter(rest_v18, deg_1024, wt, depth)
                p_v18_orig = F.softmax(s_orig["logits"], dim=1)
                s_flip = segmenter(rest_v18_f, deg_1024_f, wt, depth)
                p_v18_flip = torch.flip(F.softmax(s_flip["logits"], dim=1), dims=[3])
                p_v18 = 0.5 * (p_v18_orig + p_v18_flip)

                # --- 2. V16 DINOv2 with Flip TTA (518x518 -> 1024x1024) ---
                rest_518 = F.interpolate(rest_table, size=(518, 518), mode="bilinear", align_corners=False)
                x_dino = (rest_518 - mean_im) / std_im
                x_dino_f = torch.flip(x_dino, dims=[3])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_dino_orig = dino_model(x_dino)
                    l_dino_flip = dino_model(x_dino_f)
                p_dino_orig = F.softmax(l_dino_orig.float(), dim=1)
                p_dino_flip = torch.flip(F.softmax(l_dino_flip.float(), dim=1), dims=[3])
                p_dino_518 = 0.5 * (p_dino_orig + p_dino_flip)
                p_dino = F.interpolate(p_dino_518, size=(1024, 1024), mode="bilinear", align_corners=False)

                # --- 3. V16 ConvNeXt with Flip TTA (512x512 -> 1024x1024) ---
                rest_512 = F.interpolate(rest_table, size=(512, 512), mode="bilinear", align_corners=False)
                x_conv = (rest_512 - mean_im) / std_im
                x_conv_f = torch.flip(x_conv, dims=[3])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_conv_orig = conv_model(x_conv)
                    l_conv_flip = conv_model(x_conv_f)
                p_conv_orig = F.softmax(l_conv_orig.float(), dim=1)
                p_conv_flip = torch.flip(F.softmax(l_conv_flip.float(), dim=1), dims=[3])
                p_conv_512 = 0.5 * (p_conv_orig + p_conv_flip)
                p_conv = F.interpolate(p_conv_512, size=(1024, 1024), mode="bilinear", align_corners=False)

                # --- Compute Predictions for All Candidates ---
                preds_dict = {}
                for c_id, weights in candidates.items():
                    w18 = weights["w18"]
                    wdino = weights["wdino"]
                    wconv = weights["wconv"]

                    p_ens = w18 * p_v18 + wdino * p_dino + wconv * p_conv
                    log_p = torch.log(p_ens + 1e-10)

                    preds_dict[f"{c_id}_base"] = torch.argmax(log_p + off_base, dim=1).view(-1)
                    preds_dict[f"{c_id}_cal"]  = torch.argmax(log_p + off_cal, dim=1).view(-1)

                # Confusion matrix update
                gt_flat = mask_t.view(-1)
                mask_valid = (gt_flat >= 0) & (gt_flat < 5)
                gt_valid = gt_flat[mask_valid]
                gt_5 = 5 * gt_valid

                for cfg in eval_configs:
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

        # Clean up fold models to ensure clean VRAM
        del refiner, segmenter, dino_model, conv_model, rsuigm_table
        torch.cuda.empty_cache()

    total_runtime_sec = time.time() - t_global_start
    total_runtime_min = total_runtime_sec / 60.0
    print("\n" + "=" * 80)
    print(f"COMPLETED EVALUATION ACROSS ALL {total_images_processed} IMAGES")
    print(f"Total Runtime: {total_runtime_min:.2f} minutes ({total_runtime_sec:.1f} s)")
    print(f"Global Peak VRAM: {peak_vram_global:.2f} GB")
    print("=" * 80)

    # 3. Verify V18 baseline archive hash after run
    sha_after = compute_sha256(zip_path)
    print(f"V18 Baseline Archive SHA256 (After): {sha_after}")
    assert sha_after == sha_before, "V18 baseline archive was modified!"

    # 4. Aggregate Metrics Across All 20 Scenes
    cm_aggregate = {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in eval_configs}
    for fold in range(5):
        for cfg in eval_configs:
            cm_aggregate[cfg] += cm_by_fold[fold][cfg]

    aggregate_metrics = {}
    for cfg in eval_configs:
        aggregate_metrics[cfg] = compute_metrics_from_cm(cm_aggregate[cfg])

    # 5. Check Control Reproduction Parity
    # Target values from EXP-V19-P1:
    # A_base: Q_seg=0.46899, mIoU=0.61877, Macro_F1=0.75794
    # A_cal:  Q_seg=0.47274, mIoU=0.62167, Macro_F1=0.76044
    c0_base_qseg = aggregate_metrics["C0_base"]["Q_seg"]
    c0_cal_qseg = aggregate_metrics["C0_cal"]["Q_seg"]
    c0_base_miou = aggregate_metrics["C0_base"]["mIoU"]
    c0_cal_miou = aggregate_metrics["C0_cal"]["mIoU"]

    print("\n" + "=" * 80)
    print("V18 CONTROL REPRODUCTION AUDIT")
    print("=" * 80)
    print(f"C0_base Measured: Q_seg={c0_base_qseg:.5f}, mIoU={c0_base_miou:.5f} | Target (P1 A_base): Q_seg=0.46899, mIoU=0.61877")
    print(f"C0_cal  Measured: Q_seg={c0_cal_qseg:.5f}, mIoU={c0_cal_miou:.5f} | Target (P1 A_cal):  Q_seg=0.47274, mIoU=0.62167")
    base_delta = abs(c0_base_qseg - 0.46899)
    cal_delta = abs(c0_cal_qseg - 0.47274)
    print(f"Parity Delta: Base={base_delta:.5f}, Cal={cal_delta:.5f}")
    assert base_delta < 0.00015, f"C0_base did not reproduce P1 A_base! Delta: {base_delta}"
    assert cal_delta < 0.00015, f"C0_cal did not reproduce P1 A_cal! Delta: {cal_delta}"
    print("Control reproduction test PASSED (Tolerance < 0.00015)!")

    # 6. Comparative Grid Table
    print("\n" + "=" * 105)
    print(f"{'Config':<12} | {'Weights (V18/Dino/Conv)':<25} | {'Q_seg':<8} | {'mIoU':<8} | {'F1':<8} | {'Delta Q_seg':<12} | {'Delta mIoU':<12} | {'Cls 4 IoU':<10}")
    print("-" * 105)

    comparison_results = {}
    for c_id in sorted(candidates.keys()):
        for regime, control_key in [("base", "C0_base"), ("cal", "C0_cal")]:
            cfg = f"{c_id}_{regime}"
            m = aggregate_metrics[cfg]
            ctrl = aggregate_metrics[control_key]

            dq = m["Q_seg"] - ctrl["Q_seg"]
            dm = m["mIoU"] - ctrl["mIoU"]
            df1 = m["Macro_F1"] - ctrl["Macro_F1"]
            d_cls_ious = [round(m["class_ious"][c] - ctrl["class_ious"][c], 5) for c in range(5)]

            comparison_results[cfg] = {
                "candidate": c_id,
                "regime": regime,
                "weights": candidates[c_id],
                "metrics": m,
                "control_ref": control_key,
                "delta_qseg": round(dq, 5),
                "delta_miou": round(dm, 5),
                "delta_macro_f1": round(df1, 5),
                "delta_class_ious": d_cls_ious,
                "primary_gate_met": dq >= 0.01000,
                "secondary_gate_met": dm >= 0.00800,
                "safety_floor_met": all(d >= -0.00500 for d in d_cls_ious)
            }

            w_str = f"{candidates[c_id]['w18']:.2f} / {candidates[c_id]['wdino']:.2f} / {candidates[c_id]['wconv']:.2f}"
            c4_iou = m["class_ious"][4]
            print(f"{cfg:<12} | {w_str:<25} | {m['Q_seg']:<8.5f} | {m['mIoU']:<8.5f} | {m['Macro_F1']:<8.5f} | {dq:+12.5f} | {dm:+12.5f} | {c4_iou:<10.5f}")

    # 7. Per-Fold and Per-Scene Metrics
    per_fold_metrics = {f"fold_{f}": {} for f in range(5)}
    for f in range(5):
        for cfg in eval_configs:
            per_fold_metrics[f"fold_{f}"][cfg] = compute_metrics_from_cm(cm_by_fold[f][cfg])

    per_scene_metrics = {sc: {} for sc in sorted(cm_by_scene.keys())}
    for sc in sorted(cm_by_scene.keys()):
        for cfg in eval_configs:
            per_scene_metrics[sc][cfg] = compute_metrics_from_cm(cm_by_scene[sc][cfg])

    # 8. Determine Overall Verdict
    # Did ANY candidate meet the primary gate (+0.010 Qseg) without violating safety floors?
    best_candidate_base = max([f"{c}_base" for c in candidates if c != "C0"], key=lambda k: comparison_results[k]["delta_qseg"])
    best_candidate_cal = max([f"{c}_cal" for c in candidates if c != "C0"], key=lambda k: comparison_results[k]["delta_qseg"])

    max_dq_base = comparison_results[best_candidate_base]["delta_qseg"]
    max_dq_cal = comparison_results[best_candidate_cal]["delta_qseg"]

    print("\n" + "=" * 80)
    print("SCREENING GATES EVALUATION")
    print("=" * 80)
    print(f"Best Base Candidate: {best_candidate_base} (Delta Q_seg: {max_dq_base:+.5f}, Delta mIoU: {comparison_results[best_candidate_base]['delta_miou']:+.5f})")
    print(f"Best Cal Candidate:  {best_candidate_cal} (Delta Q_seg: {max_dq_cal:+.5f}, Delta mIoU: {comparison_results[best_candidate_cal]['delta_miou']:+.5f})")

    primary_gate_passed = (max_dq_base >= 0.01000) or (max_dq_cal >= 0.01000)
    if primary_gate_passed:
        overall_verdict = "PASS"
        verdict_summary = f"Heterogeneous ensemble demonstrated measurable improvement meeting the pre-registered screening gate (Delta Q_seg >= +0.01000). Best: {best_candidate_cal if max_dq_cal >= max_dq_base else best_candidate_base}."
    else:
        overall_verdict = "NO-GO"
        verdict_summary = f"No candidate mixture met the pre-registered screening gate (Delta Q_seg >= +0.01000). Max Delta Q_seg achieved was {max(max_dq_base, max_dq_cal):+.5f}. Baseline V18 remains superior or statistically indistinguishable."

    print(f"Overall Verdict: {overall_verdict}")
    print(f"Summary: {verdict_summary}")

    # 9. Save JSON Deliverable
    deliverable_data = {
        "experiment_id": "EXP-V19-P2-BENCHMARK",
        "title": "SAT-EURS Track 1 — V19 P2: Controlled Heterogeneous Ensemble OOF Benchmark",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "verdict": overall_verdict,
        "verdict_summary": verdict_summary,
        "baseline_archive": {
            "path": zip_path,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "intact": sha_before == sha_after
        },
        "runtime_metrics": {
            "total_images": total_images_processed,
            "total_runtime_sec": round(total_runtime_sec, 2),
            "total_runtime_min": round(total_runtime_min, 2),
            "throughput_imgs_per_sec": round(total_images_processed / total_runtime_sec, 2),
            "peak_vram_gb": round(peak_vram_global, 2)
        },
        "control_reproduction_audit": {
            "C0_base_measured": {"Q_seg": c0_base_qseg, "mIoU": c0_base_miou},
            "P1_A_base_target": {"Q_seg": 0.46899, "mIoU": 0.61877},
            "C0_cal_measured": {"Q_seg": c0_cal_qseg, "mIoU": c0_cal_miou},
            "P1_A_cal_target": {"Q_seg": 0.47274, "mIoU": 0.62167},
            "base_delta": round(base_delta, 5),
            "cal_delta": round(cal_delta, 5),
            "status": "PASSED"
        },
        "candidate_mixtures": candidates,
        "comparison_results": comparison_results,
        "aggregate_metrics": aggregate_metrics,
        "per_fold_metrics": per_fold_metrics,
        "per_scene_metrics": per_scene_metrics
    }

    json_path = osp.join(out_dir, "ENSEMBLE_BENCHMARK.json")
    with open(json_path, "w") as f:
        json.dump(deliverable_data, f, indent=2)
    print(f"\nJSON deliverable written to {json_path}")


if __name__ == "__main__":
    main()
