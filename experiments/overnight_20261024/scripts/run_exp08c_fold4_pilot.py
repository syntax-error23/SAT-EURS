"""
SAT-EURS Track 1 — EXP-08C: Controlled Fine-Tuning Pilot on Fold 4
Tests whether the validated SWA + low-LR (3e-5) recipe can improve Fold 4
(Baseline Q_seg: 0.3296, mIoU: 0.5223, F1: 0.6310) on its 4 held-out scenes (fish5, fish7, fish9, turtles1).
"""

import os
import os.path as osp
import sys
import time
import json
import torch

sys.path.insert(0, ".")

from tools.v28.train_degradation_aware_fold import train_fold
from tools.v28.evaluate_fold_oof import evaluate_model_on_val_scenes

def main():
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — EXP-08C: CONTROLLED FOLD 4 FINE-TUNING PILOT", flush=True)
    print("=" * 85, flush=True)

    fold = 4
    t0 = time.time()
    out_dir = "checkpoints/overnight_pilots/fold_4"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Run 2-epoch pilot training with conservative SWA and low LR
    print("\n[Training] Running 2-epoch fine-tuning pilot with base_lr=3e-5 and SWA on Fold 4...", flush=True)
    train_res = train_fold(
        fold=fold,
        epochs=2,
        swa_start_epoch=2,
        batch_size=8,
        base_lr=3e-5,
        crop_size=512,
        eval_stride=5,
        device_id=0,
        out_dir=out_dir,
        report_dir="reports/overnight_pilots"
    )

    baseline_metrics = train_res["baseline_metrics"]
    post_metrics = train_res["finetuned_metrics"]
    pilot_ckpt = train_res["checkpoint_path"]

    b_q = baseline_metrics["global_metrics"]["Q_seg"]
    b_miou = baseline_metrics["global_metrics"]["mIoU"]
    b_f1 = baseline_metrics["global_metrics"]["Macro_F1"]

    p_q = post_metrics["global_metrics"]["Q_seg"]
    p_miou = post_metrics["global_metrics"]["mIoU"]
    p_f1 = post_metrics["global_metrics"]["Macro_F1"]

    dq = train_res["delta_Q_seg"]
    dmiou = train_res["delta_mIoU"]
    df1 = p_f1 - b_f1

    print("\n" + "=" * 85, flush=True)
    print("FOLD 4 PILOT EVALUATION SUMMARY:", flush=True)
    print(f"  Pre-training Baseline:  Q_seg = {b_q:.4f} | mIoU = {b_miou:.4f} | F1 = {b_f1:.4f}", flush=True)
    print(f"  Post-training Pilot:    Q_seg = {p_q:.4f} | mIoU = {p_miou:.4f} | F1 = {p_f1:.4f}", flush=True)
    print(f"  Delta on Held-Out:      Delta Q_seg = {dq:+.4f} | Delta mIoU = {dmiou:+.4f} | Delta F1 = {df1:+.4f}", flush=True)
    print("=" * 85, flush=True)

    verdict = "PROMOTED" if dq >= 0.005 else ("POSITIVE" if dq > 0.0 else "REJECTED")

    out_data = {
        "metrics": {
            "baseline_q_seg": round(b_q, 5),
            "pilot_q_seg": round(p_q, 5),
            "delta_q_seg": round(dq, 5),
            "delta_miou": round(dmiou, 5),
            "delta_f1": round(df1, 5),
            "class_ious_baseline": baseline_metrics["global_metrics"]["class_ious"],
            "class_ious_pilot": post_metrics["global_metrics"]["class_ious"],
        },
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 2),
        "checkpoint": pilot_ckpt
    }

    out_file = "experiments/overnight_20261024/predictions/exp08c_fold4_pilot.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote Fold 4 pilot results to {out_file}", flush=True)

    return out_data

if __name__ == "__main__":
    main()
