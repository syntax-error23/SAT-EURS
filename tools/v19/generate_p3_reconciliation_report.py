import os
import json
import hashlib
import numpy as np

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def extract_class_metrics(cm, c):
    tp = int(cm[c, c])
    fn = int(cm[c, :].sum() - tp)
    fp = int(cm[:, c].sum() - tp)
    gt = tp + fn
    pred = tp + fp
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    prec = float(tp / pred) if pred > 0 else 0.0
    rec = float(tp / gt) if gt > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "class_id": c,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": gt,
        "pred": pred,
        "iou": round(iou, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5)
    }

def main():
    zip_path = "deploy/v18/submission_v18_dual_encoder_sota.zip"
    expected_hash = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
    actual_hash = compute_sha256(zip_path)
    assert actual_hash == expected_hash, "Hash mismatch!"

    # 1. Load V18 raw OOF evaluation
    with open("reports/v18/OOF_EVALUATION_REPORT.json", "r") as f:
        v18_raw = json.load(f)
    cm_v18_raw = np.array(v18_raw["segmentation"]["confusion_matrix"])

    # 2. Load P2 Benchmark
    with open("reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json", "r") as f:
        p2 = json.load(f)

    cm_c0_base = np.array(p2["aggregate_metrics"]["C0_base"]["confusion_matrix"])
    cm_c0_cal = np.array(p2["aggregate_metrics"]["C0_cal"]["confusion_matrix"])
    cm_c4_cal = np.array(p2["aggregate_metrics"]["C4_cal"]["confusion_matrix"])

    # Class 4 metrics across all 4 configurations
    c4_raw = extract_class_metrics(cm_v18_raw, 4)
    c4_c0_base = extract_class_metrics(cm_c0_base, 4)
    c4_c0_cal = extract_class_metrics(cm_c0_cal, 4)
    c4_c4_cal = extract_class_metrics(cm_c4_cal, 4)

    # Class 3 metrics
    c3_c0_cal = extract_class_metrics(cm_c0_cal, 3)
    c3_c4_cal = extract_class_metrics(cm_c4_cal, 3)

    reconciliation = {
        "title": "SAT-EURS Track 1 — V19 Phase 1 & 2 Evaluation Reconciliation & Routing Preflight",
        "baseline_archive": {
            "path": zip_path,
            "sha256": actual_hash,
            "intact": True
        },
        "phase1_evaluation_protocol_audit": {
            "confusion_matrix_accumulation": "Global micro-pixel accumulation across all 4,000 images in the 20 scenes: cm[gt, pred] += bincount(5*gt + pred).",
            "per_class_metric_formulas": {
                "IoU": "TP / (TP + FP + FN)",
                "Precision": "TP / (TP + FP)",
                "Recall": "TP / (TP + FN)",
                "F1": "2 * TP / (2 * TP + FP + FN)"
            },
            "macro_averaging_policy": {
                "present_classes_mean": "Averaged across classes where GT > 0. For full 20 scenes, all 5 classes are present.",
                "absent_classes_handling": "In local partial splits, absent classes receive 0.0 or are excluded from present mean. In 20-scene pooled evaluation, all 5 classes are present (GT > 32M pixels per class).",
                "coverage_penalty": "1.0 if all present classes have TP > 0, else 0.0.",
                "Q_seg": "mIoU * Macro_F1 * coverage_penalty"
            },
            "local_oof_vs_official_dev_discrepancy_explanation": {
                "local_oof_score": "Q_seg = 0.47274 (Calibrated V18 single model per fold across 20 training scenes, 4,000 images).",
                "official_dev_score": "Q_seg = 0.768, Composite = 0.749 (5-fold ensemble averaged across 6 unseen Dev scenes, 1,200 images).",
                "key_structural_differences": [
                    "Evaluation dataset: 20 training scenes with diverse and severe water degradations (e.g. fish20, fish21 in murky water) vs 6 Dev test scenes.",
                    "Model capacity: Single fold model out-of-fold inference (each scene evaluated by 1 model) vs 5-model probability ensemble on Dev.",
                    "Geometry: Local OOF evaluates on 1024x1024 square geometry; Dev inference uses aspect-preserved max-side 1024 with multiple-of-32 padding followed by final unpadding and resampling to camera dimensions."
                ],
                "verdict_on_scorer": "The local evaluator is mathematically correct and faithful to the official formula. The score difference arises from the dataset difficulty and single-model OOF vs 5-model Dev ensemble inference."
            }
        },
        "phase2_class4_recall_reconciliation": {
            "reported_narrative_value": 0.413,
            "reported_calibrated_table_value": 0.55355,
            "best_ensemble_value": 0.55774,
            "exact_pixel_provenance": {
                "V18_Raw_Baseline_NoTTA_NoOffset": c4_raw,
                "V18_Base_FlipTTA_b4_015": c4_c0_base,
                "V18_Calibrated_FlipTTA_b4_033": c4_c0_cal,
                "P2_Best_Ensemble_C4_cal": c4_c4_cal
            },
            "explanation": "The value 0.413 (0.41297) was V18's raw, unaugmented recall before test-time augmentation and calibration. The narrative erroneously contrasted C4_cal's recall (0.558) with the raw baseline (0.413), creating the illusion of a +0.145 recall jump. However, against the matched calibrated control C0_cal (0.55355), the true recall gain from ensembling was only +0.00419 (+511,513 TP pixels). The primary impact of the ensemble on Class 4 was actually precision (+5.57%, cutting FP pixels from 33.77M to 26.16M)."
        },
        "class3_tradeoff_summary": {
            "C0_cal": c3_c0_cal,
            "C4_cal": c3_c4_cal,
            "delta_tp": c3_c4_cal["tp"] - c3_c0_cal["tp"],
            "delta_fn": c3_c4_cal["fn"] - c3_c0_cal["fn"],
            "delta_fp": c3_c4_cal["fp"] - c3_c0_cal["fp"],
            "delta_iou": round(c3_c4_cal["iou"] - c3_c0_cal["iou"], 5),
            "delta_recall": round(c3_c4_cal["recall"] - c3_c0_cal["recall"], 5),
            "impact": "Blending V16 models caused a severe drop of -0.04615 IoU and lost 1,055,299 Class 3 TP pixels, negating the composite benefit gained on other classes."
        },
        "phase3_artifact_availability_audit": {
            "scene_to_fold_manifest": {"path": "splits/manifest_5fold.json", "status": "PRESENT"},
            "model_checkpoints": {"count": 15, "status": "PRESENT"},
            "aggregated_confusion_matrices": {"path": "reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json", "status": "PRESENT"},
            "raw_probability_maps_on_disk": {"status": "MISSING", "reason": "Streamed in GPU VRAM during P2 to avoid ~250 GB disk storage footprint."},
            "discrete_prediction_masks_on_disk": {"status": "MISSING", "reason": "Streamed in GPU VRAM and accumulated directly into confusion matrices."},
            "phase3_stop_condition_met": True,
            "stop_rationale": "Pixel-level routing rules (e.g., overriding non-Class-3 pixels when auxiliary Class 4 probability exceeds a threshold) cannot be computed from 5x5 confusion matrices alone. Evaluating Phase 3 requires executing a streaming GPU evaluation script across the 5 folds. The user instruction explicitly specifies: 'If the necessary probability maps, masks, or scene-to-fold mappings are missing, stop and report exactly what is missing. Do not regenerate predictions or train models without approval.'"
        },
        "phase3_proposed_routing_specification": {
            "hard_prediction_preservation_rule": "Y_routed(x, y) = Y_V18(x, y) if Y_V18(x, y) == 3. Class 3 pixels are 100% immune to override.",
            "candidate_override_rule": "Y_routed(x, y) = 4 iff Y_V18(x, y) in {0, 1, 2} and P_aux,4(x, y) >= tau_4 and Component_Size >= min_area.",
            "auxiliary_probability_definition": "P_aux,4 = (2/3) * P_dino,4 + (1/3) * P_conv,4 (matching the 0.20 / 0.10 ratio of C4).",
            "tuning_parameters": ["tau_4 in [0.35, 0.40, 0.45, 0.50, 0.55]", "min_area in [0, 50, 150]"],
            "nested_leave_one_fold_out_protocol": "For outer fold k (800 images), optimal (tau_4, min_area) is fitted strictly on the 4 inner folds {j != k} (3,200 images), then applied blindly to fold k."
        }
    }

    out_file = "reports/v19/exp_p3_class4_gate/reconciliation_data.json"
    with open(out_file, "w") as f:
        json.dump(reconciliation, f, indent=2)
    print(f"Reconciliation data saved to {out_file}")

if __name__ == "__main__":
    main()
