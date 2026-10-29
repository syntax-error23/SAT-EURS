"""
SAT-EURS Track 1 — EXP-08B: Controlled Fine-Tuning Pilot on Weakest Fold (Fold 3)
Tests whether the validated SWA + low-LR (3e-5) recipe can improve Fold 3
(Baseline Q_seg: 0.2301, mIoU: 0.4327) on its 4 held-out scenes (fish20, fish21, fish4, starfish2)
without suffering the catastrophic regression seen in Fold 1 and Fold 2.
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
    print("=" * 85)
    print("SAT-EURS TRACK 1 — EXP-08B: CONTROLLED FOLD 3 FINE-TUNING PILOT")
    print("=" * 85)

    fold = 3
    t0 = time.time()
    out_dir = "checkpoints/overnight_pilots/fold_3"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Run 2-epoch pilot training with conservative SWA and low LR
    print("\n[Training] Running 2-epoch fine-tuning pilot with base_lr=3e-5 and SWA on Fold 3...")
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

    print("\n" + "=" * 85)
    print("FOLD 3 PILOT EVALUATION SUMMARY:")
    print(f"  Pre-training Baseline:  Q_seg = {b_q:.4f} | mIoU = {b_miou:.4f} | F1 = {b_f1:.4f}")
    print(f"  Post-training Pilot:    Q_seg = {p_q:.4f} | mIoU = {p_miou:.4f} | F1 = {p_f1:.4f}")
    print(f"  Delta on Held-Out:      Delta Q_seg = {dq:+.4f} | Delta mIoU = {dmiou:+.4f} | Delta F1 = {df1:+.4f}")
    print("=" * 85)

    verdict = "PROMOTED" if dq >= 0.005 else "REJECTED"

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

    out_file = "experiments/overnight_20261024/predictions/exp08b_fold3_pilot.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote Fold 3 pilot results to {out_file}")

    return out_data

if __name__ == "__main__":
    main()
