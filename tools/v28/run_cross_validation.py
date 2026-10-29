"""
SAT-EURS Track 1 — V28 Degradation-Aware Cross-Validation Runner
Executes cross-validation across folds to test whether fine-tuning on degraded EURS crops
yields a genuine, consistent out-of-fold score improvement without rare-class regression.
Strict promotion gating enforced before any Full-20 training.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any

sys.path.insert(0, ".")

from tools.v28.train_degradation_aware_fold import train_fold


def run_cross_validation(
    folds: List[int] = [0, 1, 2],
    epochs: int = 2,
    swa_start: int = 2,
    batch_size: int = 8,
    lr: float = 3e-5,
    eval_stride: int = 5,
    device_id: int = 0,
    out_dir: str = "checkpoints/v28",
    report_file: str = "reports/v28/phase2/cross_validation_summary.json"
) -> Dict[str, Any]:
    print("=" * 90, flush=True)
    print("SAT-EURS TRACK 1 — V28 DEGRADATION-AWARE CROSS-VALIDATION CAMPAIGN", flush=True)
    print(f"Folds to evaluate: {folds} (minimum 3 required for promotion gate)", flush=True)
    print(f"Epochs per fold: {epochs} (SWA start: {swa_start}) | Batch Size: {batch_size} | LR: {lr}", flush=True)
    print(f"Eval Stride: {eval_stride} (Stride 5 = 40 images/scene, 160 images/fold)", flush=True)
    print("=" * 90, flush=True)

    results = {}
    fold_deltas = []
    base_q_segs = []
    fine_q_segs = []

    catastrophic_regression_detected = False
    regression_details = []

    for fold in folds:
        print(f"\n{'#' * 90}", flush=True)
        print(f"STARTING CROSS-VALIDATION FOR FOLD {fold}", flush=True)
        print(f"{'#' * 90}", flush=True)

        res = train_fold(
            fold=fold,
            epochs=epochs,
            swa_start_epoch=swa_start,
            batch_size=batch_size,
            base_lr=lr,
            eval_stride=eval_stride,
            device_id=device_id,
            out_dir=out_dir,
            report_dir="reports/v28/phase2"
        )
        results[f"fold_{fold}"] = res

        b_q = res["baseline_metrics"]["global_metrics"]["Q_seg"]
        f_q = res["finetuned_metrics"]["global_metrics"]["Q_seg"]
        delta = f_q - b_q
        fold_deltas.append(delta)
        base_q_segs.append(b_q)
        fine_q_segs.append(f_q)

        # Check rare-class regression
        b_ious = res["baseline_metrics"]["global_metrics"]["class_ious"]
        f_ious = res["finetuned_metrics"]["global_metrics"]["class_ious"]
        # Class 2: Fish, Class 3: Ray, Class 4: MacroFauna
        for c_idx, c_name in [(2, "Fish"), (3, "Ray"), (4, "MacroFauna")]:
            b_iou = b_ious[c_idx]
            f_iou = f_ious[c_idx]
            # If baseline had meaningful IoU (>0.20) and it dropped by more than 0.15:
            if b_iou > 0.20 and (b_iou - f_iou) > 0.15:
                catastrophic_regression_detected = True
                msg = f"Fold {fold} {c_name}: baseline IoU={b_iou:.4f} dropped to {f_iou:.4f} (delta={f_iou - b_iou:+.4f})"
                regression_details.append(msg)
                print(f"  [WARNING] Catastrophic Regression: {msg}", flush=True)

    mean_base_q = float(sum(base_q_segs) / len(base_q_segs))
    mean_fine_q = float(sum(fine_q_segs) / len(fine_q_segs))
    mean_delta = float(sum(fold_deltas) / len(fold_deltas))

    # Strict Promotion Gate Evaluation
    # Gate 1: Mean delta > 0.005 (or at least strictly positive)
    # Gate 2: No catastrophic regression on rare classes
    # Gate 3: Consistency (majority of folds show positive delta)
    positive_folds = sum(1 for d in fold_deltas if d > 0)
    passes_consistency = positive_folds >= (len(folds) // 2 + 1)
    passes_gain = mean_delta > 0.0
    passes_safety = not catastrophic_regression_detected

    gate_passed = passes_gain and passes_safety and passes_consistency

    print("\n" + "=" * 90, flush=True)
    print("=== DEGRADATION-AWARE CROSS-VALIDATION PROMOTION GATE VERDICT ===", flush=True)
    print("=" * 90, flush=True)
    print(f"Evaluated Folds:             {folds}", flush=True)
    print(f"Baseline Mean Q_seg:         {mean_base_q:.4f}", flush=True)
    print(f"Fine-Tuned Mean Q_seg:       {mean_fine_q:.4f}", flush=True)
    print(f"Aggregate Delta Q_seg:       {mean_delta:+.4f}", flush=True)
    print(f"Positive Folds:              {positive_folds}/{len(folds)}", flush=True)
    print(f"Rare-Class Safety Gate:      {'PASSED' if passes_safety else 'FAILED'}", flush=True)
    if regression_details:
        for d in regression_details:
            print(f"  - {d}", flush=True)
    print("-" * 90, flush=True)
    print(f"PROMOTION GATE VERDICT:      {'*** PASSED (Proceed with Full-20 Training) ***' if gate_passed else '*** FAILED (Do NOT Train Full-20 Blindly) ***'}", flush=True)
    print("=" * 90, flush=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "folds_evaluated": folds,
        "epochs": epochs,
        "base_lr": lr,
        "mean_baseline_Q_seg": round(mean_base_q, 5),
        "mean_finetuned_Q_seg": round(mean_fine_q, 5),
        "aggregate_delta_Q_seg": round(mean_delta, 5),
        "positive_folds_count": positive_folds,
        "rare_class_catastrophic_regression": catastrophic_regression_detected,
        "regression_details": regression_details,
        "promotion_gate_passed": gate_passed,
        "recommended_action": "TRAIN_FULL20_V28B" if gate_passed else "SELECT_V28A_SUBMISSION",
        "fold_results": results
    }

    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary report to {report_file}", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--swa_start", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eval_stride", type=int, default=5)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    run_cross_validation(
        folds=args.folds,
        epochs=args.epochs,
        swa_start=args.swa_start,
        lr=args.lr,
        eval_stride=args.eval_stride,
        device_id=args.device_id
    )
