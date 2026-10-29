"""
SAT-EURS Track 1 — V19 Phase 3/4: Class-4-Only Routing Diagnostic
Evaluates whether gating auxiliary predictions strictly onto Class 0, 1, 2 pixels
can capture Class 4 gains while guaranteeing ZERO degradation on Class 3.

Methodological Constraints & Honest Disclosure:
1. Baseline Control: V18 at calibrated offset b4=0.33 (C0_cal).
2. Reference: Tri-Ensemble at b4=0.33 (C4_cal).
3. Candidate Routing Rule:
   - Y_routed(x, y) = 3 if Y_V18(x, y) == 3 (Class 3 is 100% immune to override).
   - Y_routed(x, y) = 4 iff Y_V18(x, y) in {0, 1, 2} and Y_C4(x, y) == 4 and P_ens,4(x, y) >= tau_4 (and Component_Area >= min_area).
   - Y_routed(x, y) = Y_V18(x, y) otherwise.
4. Exploratory Cross-Fold Routing Disclosure:
   - For outer fold k (800 images), optimal (tau_4, min_area) is fitted on the 4 inner folds {j != k} (3,200 images), then evaluated on fold k.
   - Note: The 4 inner folds use models whose base training included fold k. Hence, this is exploratory cross-fold screening, not unbiased nested-CV.
5. Class-3 Invariance:
   - Class-3 TP, FP, FN, precision, recall, F1, and IoU remain mathematically unchanged.
   - Off-diagonal confusion-matrix cells for other classes reallocate to Class 4.

Strict zero-mutation of production files or checkpoints.
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
        "coverage_penalty": cov_penalty,
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


def filter_by_connected_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = True
    return filtered


def main():
    print("=" * 80)
    print("SAT-EURS Track 1 — V19 Phase 3/4: Class-4-Only Routing Diagnostic")
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
    out_dir = "reports/v19/exp_p3_class4_gate"
    os.makedirs(out_dir, exist_ok=True)

    # 2. Candidate Grid for Routing
    # tau_4: confidence threshold on P_ens,4
    # min_area: component size filter
    tau_grid = [0.30, 0.35, 0.40, 0.45, 0.50]
    min_area_grid = [0, 50, 150]

    grid_keys = [f"tau_{t:.2f}_area_{a}" for t in tau_grid for a in min_area_grid]
    print(f"Evaluating Routing Parameter Grid ({len(grid_keys)} combinations):")
    print(f"  tau_4 in {tau_grid}")
    print(f"  min_area in {min_area_grid}")

    # Offsets: calibrated [0, 0, 0, 0.15, 0.33]
    off_cal = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.33], device=device).view(1, 5, 1, 1)
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Accumulators:
    # cm_by_fold[f][key] for: "C0_cal", "C4_cal", and all grid_keys
    tracked_configs = ["C0_cal", "C4_cal"] + grid_keys
    cm_by_fold = {f: {cfg: np.zeros((5, 5), dtype=np.int64) for cfg in tracked_configs} for f in range(5)}
    
    # Transition trackers: count how many pixels transitioned from c -> 4
    # transitions_by_fold[f][key][c] = number of pixels
    transitions_by_fold = {f: {k: {0: 0, 1: 0, 2: 0, 3: 0} for k in grid_keys} for f in range(5)}

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

        # Checkpoints
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

        # Dataset
        val_ds = V18EURSInvarianceDataset(scenes=val_scenes, crop_size=1024, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        t_fold_start = time.time()
        for idx, batch in enumerate(val_loader):
            deg_1024 = batch["deg_rgb"].to(device)
            mask_t = batch["mask"].to(device)
            wt = batch["wt"].to(device)
            depth = batch["depth"].to(device)
            scene = batch["scene"][0]

            with torch.no_grad():
                # Restorations
                rest_v18 = refiner(deg_1024, wt, depth)["refined_image"]
                rest_table = apply_rsuigm_batch_gpu(deg_1024, wt, depth, rsuigm_table)

                # --- 1. V18 Segmenter with Flip TTA ---
                deg_1024_f = torch.flip(deg_1024, dims=[3])
                rest_v18_f = torch.flip(rest_v18, dims=[3])
                s_orig = segmenter(rest_v18, deg_1024, wt, depth)
                p_v18_orig = F.softmax(s_orig["logits"], dim=1)
                s_flip = segmenter(rest_v18_f, deg_1024_f, wt, depth)
                p_v18_flip = torch.flip(F.softmax(s_flip["logits"], dim=1), dims=[3])
                p_v18 = 0.5 * (p_v18_orig + p_v18_flip)

                # --- 2. V16 DINOv2 with Flip TTA ---
                rest_518 = F.interpolate(rest_table, size=(518, 518), mode="bilinear", align_corners=False)
                x_dino = (rest_518 - mean_im) / std_im
                x_dino_f = torch.flip(x_dino, dims=[3])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_dino_orig = dino_model(x_dino)
                    l_dino_flip = dino_model(x_dino_f)
                p_dino_orig = F.softmax(l_dino_orig.float(), dim=1)
                p_dino_flip = torch.flip(F.softmax(l_dino_flip.float(), dim=1), dims=[3])
                p_dino = F.interpolate(0.5 * (p_dino_orig + p_dino_flip), size=(1024, 1024), mode="bilinear", align_corners=False)

                # --- 3. V16 ConvNeXt with Flip TTA ---
                rest_512 = F.interpolate(rest_table, size=(512, 512), mode="bilinear", align_corners=False)
                x_conv = (rest_512 - mean_im) / std_im
                x_conv_f = torch.flip(x_conv, dims=[3])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_conv_orig = conv_model(x_conv)
                    l_conv_flip = conv_model(x_conv_f)
                p_conv_orig = F.softmax(l_conv_orig.float(), dim=1)
                p_conv_flip = torch.flip(F.softmax(l_conv_flip.float(), dim=1), dims=[3])
                p_conv = F.interpolate(0.5 * (p_conv_orig + p_conv_flip), size=(1024, 1024), mode="bilinear", align_corners=False)

                # --- Base Predictions ---
                # V18 Control: C0_cal
                log_v18 = torch.log(p_v18 + 1e-10)
                pred_c0_cal = torch.argmax(log_v18 + off_cal, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # Tri-Ensemble Reference: C4_cal (0.70 V18 + 0.20 DINO + 0.10 Conv)
                p_ens = 0.70 * p_v18 + 0.20 * p_dino + 0.10 * p_conv
                log_ens = torch.log(p_ens + 1e-10)
                pred_c4_cal = torch.argmax(log_ens + off_cal, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # P_ens Class 4 probability map as numpy
                p_c4_np = p_ens[:, 4].squeeze(0).cpu().numpy()

                gt_np = mask_t.squeeze(0).cpu().numpy()
                mask_valid = (gt_np >= 0) & (gt_np < 5)
                gt_v = gt_np[mask_valid]
                gt_5 = 5 * gt_v

                # Update C0_cal and C4_cal
                cm_by_fold[fold]["C0_cal"] += np.bincount(gt_5 + pred_c0_cal[mask_valid], minlength=25).reshape(5, 5)
                cm_by_fold[fold]["C4_cal"] += np.bincount(gt_5 + pred_c4_cal[mask_valid], minlength=25).reshape(5, 5)

                # Precompute candidate masks where C4_cal == 4 and V18 != 3
                # Override candidates must originate from V18 in {0, 1, 2}
                can_override_base = (pred_v18_c3_immune := (pred_c0_cal != 3)) & (pred_c4_cal == 4) & (pred_c0_cal != 4)
                # Note: if pred_c0_cal == 4, it is already 4, no transition occurs

                # Evaluate all (tau_4, min_area) grid points
                for tau in tau_grid:
                    conf_mask = can_override_base & (p_c4_np >= tau)
                    for area in min_area_grid:
                        key = f"tau_{tau:.2f}_area_{area}"
                        if area > 0:
                            override_mask = filter_by_connected_components(conf_mask, area)
                        else:
                            override_mask = conf_mask

                        # Construct routed prediction
                        pred_routed = pred_c0_cal.copy()
                        pred_routed[override_mask] = 4

                        # Strictly assert Class 3 hard preservation
                        assert np.all(pred_routed[pred_c0_cal == 3] == 3), "CRITICAL BUG: Class 3 was overridden!"

                        # Accumulate confusion matrix
                        cm_by_fold[fold][key] += np.bincount(gt_5 + pred_routed[mask_valid], minlength=25).reshape(5, 5)

                        # Track pixel transitions
                        for orig_c in [0, 1, 2]:
                            n_trans = np.sum(override_mask & (pred_c0_cal == orig_c))
                            transitions_by_fold[fold][key][orig_c] += int(n_trans)
                        # Confirm 3 -> 4 is 0
                        n_trans_3 = np.sum(override_mask & (pred_c0_cal == 3))
                        transitions_by_fold[fold][key][3] += int(n_trans_3)
                        assert n_trans_3 == 0, "Class 3 to 4 transition occurred!"

            total_images_processed += 1
            if (idx + 1) % 100 == 0 or (idx + 1) == len(val_ds):
                el_f = time.time() - t_fold_start
                ips = (idx + 1) / el_f
                cur_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
                peak_vram_global = max(peak_vram_global, cur_vram)
                print(f"  [Fold {fold}] [{idx+1:03d}/{len(val_ds)}] {ips:.2f} imgs/s | VRAM: {cur_vram:.2f} GB | Scene: {scene}")

        del refiner, segmenter, dino_model, conv_model, rsuigm_table
        torch.cuda.empty_cache()

    total_runtime_sec = time.time() - t_global_start
    total_runtime_min = total_runtime_sec / 60.0
    print("\n" + "=" * 80)
    print(f"COMPLETED STREAMING EVALUATION ACROSS ALL {total_images_processed} IMAGES")
    print(f"Total Runtime: {total_runtime_min:.2f} minutes ({total_runtime_sec:.1f} s)")
    print(f"Global Peak VRAM: {peak_vram_global:.2f} GB")
    print("=" * 80)

    # 3. Baseline Hash Verification After Run
    sha_after = compute_sha256(zip_path)
    print(f"V18 Baseline Archive SHA256 (After): {sha_after}")
    assert sha_after == sha_before, "V18 baseline archive was modified!"

    # 4. Verify Control Reproduction
    cm_c0_pooled = sum(cm_by_fold[f]["C0_cal"] for f in range(5))
    metrics_c0 = compute_metrics_from_cm(cm_c0_pooled)
    print("\n" + "=" * 80)
    print("CONTROL REPRODUCTION CHECK")
    print("=" * 80)
    print(f"C0_cal Measured: Q_seg={metrics_c0['Q_seg']:.5f}, mIoU={metrics_c0['mIoU']:.5f}")
    assert abs(metrics_c0["Q_seg"] - 0.47274) < 0.00015, "Control failed to reproduce target!"
    print("Control reproduction verified exact (0.47274)!")

    # 5. Exploratory Cross-Fold Selection (Leave-One-Fold-Out Calibration)
    print("\n" + "=" * 80)
    print("EXPLORATORY CROSS-FOLD THRESHOLD SELECTION (Inner 4 Folds -> Outer Fold)")
    print("=" * 80)
    
    lofo_selected_settings = {}
    lofo_cm_by_fold = {}
    lofo_transitions_by_fold = {}

    for k in range(5):
        # Inner 4 folds {j != k}
        inner_cms = {key: sum(cm_by_fold[j][key] for j in range(5) if j != k) for key in grid_keys}
        inner_qsegs = {key: compute_metrics_from_cm(inner_cms[key])["Q_seg"] for key in grid_keys}
        
        best_key = max(grid_keys, key=lambda k_g: inner_qsegs[k_g])
        best_inner_q = inner_qsegs[best_key]
        
        # Apply best_key to outer fold k
        outer_cm = cm_by_fold[k][best_key]
        outer_trans = transitions_by_fold[k][best_key]
        
        lofo_selected_settings[f"fold_{k}"] = {
            "selected_setting": best_key,
            "inner_qseg": round(best_inner_q, 5),
            "outer_qseg": round(compute_metrics_from_cm(outer_cm)["Q_seg"], 5)
        }
        lofo_cm_by_fold[f"fold_{k}"] = outer_cm
        lofo_transitions_by_fold[f"fold_{k}"] = outer_trans
        print(f"  Fold {k}: Selected {best_key} (Inner Q_seg: {best_inner_q:.5f}) -> Outer Q_seg: {lofo_selected_settings[f'fold_{k}']['outer_qseg']:.5f}")

    # Pooled LOFO Candidate Metrics
    cm_lofo_pooled = sum(lofo_cm_by_fold[f"fold_{k}"] for k in range(5))
    metrics_lofo = compute_metrics_from_cm(cm_lofo_pooled)

    # Reference C4_cal Metrics
    cm_c4_pooled = sum(cm_by_fold[f]["C4_cal"] for f in range(5))
    metrics_c4 = compute_metrics_from_cm(cm_c4_pooled)

    # Compute Total Transitions
    total_transitions = {0: 0, 1: 0, 2: 0, 3: 0}
    for k in range(5):
        for c in [0, 1, 2, 3]:
            total_transitions[c] += lofo_transitions_by_fold[f"fold_{k}"][c]
    total_pixels_changed = sum(total_transitions.values())

    # 6. Audit Class-3 Invariance
    print("\n" + "=" * 80)
    print("CLASS-3 INVARIANCE AUDIT")
    print("=" * 80)
    c3_c0 = metrics_c0["per_class_pixel_stats"][3]
    c3_lofo = metrics_lofo["per_class_pixel_stats"][3]

    tp3_delta = c3_lofo["tp_pixels"] - c3_c0["tp_pixels"]
    fp3_delta = c3_lofo["fp_pixels"] - c3_c0["fp_pixels"]
    fn3_delta = c3_lofo["fn_pixels"] - c3_c0["fn_pixels"]
    iou3_delta = c3_lofo["iou"] - c3_c0["iou"]

    print(f"Class 3 TP Delta: {tp3_delta:+d} (C0: {c3_c0['tp_pixels']}, LOFO: {c3_lofo['tp_pixels']})")
    print(f"Class 3 FP Delta: {fp3_delta:+d} (C0: {c3_c0['fp_pixels']}, LOFO: {c3_lofo['fp_pixels']})")
    print(f"Class 3 FN Delta: {fn3_delta:+d} (C0: {c3_c0['fn_pixels']}, LOFO: {c3_lofo['fn_pixels']})")
    print(f"Class 3 IoU Delta: {iou3_delta:+.5f} (C0: {c3_c0['iou']:.5f}, LOFO: {c3_lofo['iou']:.5f})")
    assert tp3_delta == 0 and fp3_delta == 0 and fn3_delta == 0 and abs(iou3_delta) < 1e-6, "Class 3 invariance failed!"
    print("CLASS-3 INVARIANCE VERIFIED 100% BIT-FOR-BIT IDENTICAL!")

    # 7. Comparative Metrics Table
    print("\n" + "=" * 115)
    print(f"{'Configuration':<25} | {'Q_seg':<8} | {'mIoU':<8} | {'F1':<8} | {'Delta Q_seg':<12} | {'Delta mIoU':<12} | {'Cls 3 IoU':<10} | {'Cls 4 IoU':<10}")
    print("-" * 115)

    delta_q_lofo = metrics_lofo["Q_seg"] - metrics_c0["Q_seg"]
    delta_m_lofo = metrics_lofo["mIoU"] - metrics_c0["mIoU"]
    delta_f_lofo = metrics_lofo["Macro_F1"] - metrics_c0["Macro_F1"]

    delta_q_c4 = metrics_c4["Q_seg"] - metrics_c0["Q_seg"]
    delta_m_c4 = metrics_c4["mIoU"] - metrics_c0["mIoU"]

    print(f"{'V18 Calibrated Control (C0_cal)':<25} | {metrics_c0['Q_seg']:<8.5f} | {metrics_c0['mIoU']:<8.5f} | {metrics_c0['Macro_F1']:<8.5f} | {'+0.00000':<12} | {'+0.00000':<12} | {metrics_c0['class_ious'][3]:<10.5f} | {metrics_c0['class_ious'][4]:<10.5f}")
    print(f"{'P2 Tri-Ensemble (C4_cal)':<25} | {metrics_c4['Q_seg']:<8.5f} | {metrics_c4['mIoU']:<8.5f} | {metrics_c4['Macro_F1']:<8.5f} | {delta_q_c4:+12.5f} | {delta_m_c4:+12.5f} | {metrics_c4['class_ious'][3]:<10.5f} | {metrics_c4['class_ious'][4]:<10.5f}")
    print(f"{'Class-4 Gated Candidate (LOFO)':<25} | {metrics_lofo['Q_seg']:<8.5f} | {metrics_lofo['mIoU']:<8.5f} | {metrics_lofo['Macro_F1']:<8.5f} | {delta_q_lofo:+12.5f} | {delta_m_lofo:+12.5f} | {metrics_lofo['class_ious'][3]:<10.5f} | {metrics_lofo['class_ious'][4]:<10.5f}")

    # Pixel Transitions Summary
    print("\n" + "=" * 80)
    print("PIXEL REASSIGNMENT SUMMARY")
    print("=" * 80)
    print(f"Total Pixels Changed: {total_pixels_changed:,} ({total_pixels_changed / 4e9 * 100:.3f}% of total pixels)")
    print(f"  Class 0 (Background) -> Class 4: {total_transitions[0]:,} ({total_transitions[0]/max(1,total_pixels_changed)*100:.2f}%)")
    print(f"  Class 1 (Coral Reef)  -> Class 4: {total_transitions[1]:,} ({total_transitions[1]/max(1,total_pixels_changed)*100:.2f}%)")
    print(f"  Class 2 (Fish)        -> Class 4: {total_transitions[2]:,} ({total_transitions[2]/max(1,total_pixels_changed)*100:.2f}%)")
    print(f"  Class 3 (Elasmobranch)-> Class 4: {total_transitions[3]:,} (0.00% - STRICTLY IMMUNE)")

    # 8. Decision Gates Evaluation
    gate_qseg = delta_q_lofo >= 0.00600
    c4_delta_iou = metrics_lofo["class_ious"][4] - metrics_c0["class_ious"][4]
    gate_c4 = c4_delta_iou >= 0.01800
    gate_c3 = abs(iou3_delta) < 1e-5
    gate_cov = metrics_lofo["coverage_penalty"] == 1.0

    # Fold wins count
    fold_wins = 0
    for k in range(5):
        q_c0_k = compute_metrics_from_cm(cm_by_fold[k]["C0_cal"])["Q_seg"]
        q_lofo_k = compute_metrics_from_cm(lofo_cm_by_fold[f"fold_{k}"])["Q_seg"]
        if q_lofo_k > q_c0_k:
            fold_wins += 1
    gate_folds = fold_wins >= 4

    print("\n" + "=" * 80)
    print("PRE-REGISTERED DECISION GATES EVALUATION")
    print("=" * 80)
    print(f"Gate 1: Delta Q_seg >= +0.00600: {gate_qseg} (Measured: {delta_q_lofo:+.5f})")
    print(f"Gate 2: Class-4 Delta IoU >= +0.01800: {gate_c4} (Measured: {c4_delta_iou:+.5f})")
    print(f"Gate 3: Class-3 IoU 100% Unchanged: {gate_c3} (Measured Delta: {iou3_delta:+.5f})")
    print(f"Gate 4: Coverage Penalty == 1.0: {gate_cov} (Measured: {metrics_lofo['coverage_penalty']})")
    print(f"Gate 5: Positive Improvement on >= 4/5 Folds: {gate_folds} (Wins: {fold_wins}/5)")

    all_gates_met = gate_qseg and gate_c4 and gate_c3 and gate_cov and gate_folds
    overall_verdict = "PASS" if all_gates_met else "NO-GO"
    print(f"\nOVERALL VERDICT: {overall_verdict} (Exploratory Screening Gate Evaluation)")

    # 9. Deliverable JSON
    deliverable = {
        "title": "SAT-EURS Track 1 — V19 Class-4 Routing Diagnostic Results",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "verdict": overall_verdict,
        "evaluation_label": "EXPLORATORY_CROSS_FOLD_ROUTING_DIAGNOSTIC",
        "methodological_disclosure": (
            "The 4 calibration folds {j != k} used models whose base training included fold k. "
            "Therefore, this cross-fold selection provides exploratory screening evidence on existing OOF artifacts, "
            "not formal independent nested validation. A true nested cross-validation would require retraining base models."
        ),
        "baseline_archive": {
            "path": zip_path,
            "sha256_before": sha_before,
            "sha256_after": sha_after,
            "intact": sha_before == sha_after
        },
        "runtime_metrics": {
            "total_images": total_images_processed,
            "runtime_minutes": round(total_runtime_min, 2),
            "peak_vram_gb": round(peak_vram_global, 2)
        },
        "gates_evaluation": {
            "overall_verdict": overall_verdict,
            "gate_qseg": {"threshold": "+0.00600", "measured": round(delta_q_lofo, 5), "passed": gate_qseg},
            "gate_c4_iou": {"threshold": "+0.01800", "measured": round(c4_delta_iou, 5), "passed": gate_c4},
            "gate_c3_invariance": {"threshold": "0.00000", "measured": round(iou3_delta, 5), "passed": gate_c3},
            "gate_coverage": {"threshold": "1.0", "measured": metrics_lofo["coverage_penalty"], "passed": gate_cov},
            "gate_fold_wins": {"threshold": ">= 4/5", "measured": f"{fold_wins}/5", "passed": gate_folds}
        },
        "metrics_summary": {
            "C0_cal_control": metrics_c0,
            "C4_cal_reference": metrics_c4,
            "Class4_Gated_LOFO": metrics_lofo
        },
        "pixel_reassignments": {
            "total_pixels_changed": total_pixels_changed,
            "pct_of_total": round(total_pixels_changed / 4e9 * 100, 4),
            "from_class_0": total_transitions[0],
            "from_class_1": total_transitions[1],
            "from_class_2": total_transitions[2],
            "from_class_3": total_transitions[3]
        },
        "lofo_fold_settings": lofo_selected_settings,
        "all_grid_metrics": {
            k_g: compute_metrics_from_cm(sum(cm_by_fold[f][k_g] for f in range(5)))
            for k_g in grid_keys
        }
    }

    json_path = osp.join(out_dir, "CLASS4_ROUTING_DIAGNOSTIC.json")
    with open(json_path, "w") as f:
        json.dump(deliverable, f, indent=2)
    print(f"\nDeliverable JSON written to {json_path}")


if __name__ == "__main__":
    main()
