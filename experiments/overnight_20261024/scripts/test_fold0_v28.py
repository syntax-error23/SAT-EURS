import os
import sys
import time
import json
import torch
import cv2
import numpy as np

sys.path.insert(0, ".")

from models.v18.segmenter import build_v18_segmenter
from models.v18.restoration_refiner import build_v18_restoration_refiner
from tools.v18.run_v18_pipeline import V18CanonicalPreprocessor
from experiments.overnight_20261024.scripts.benchmark_engine import evaluate_scene_predictions, compute_metrics_from_conf_mat

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("SAT-EURS TRACK 1 — FOLD 0 COMPARISON: V27 vs V28-SWA CHECKPOINT")
    print("=" * 80)

    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    base_table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)

    # Refiner
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_obj = torch.load("checkpoints/v18/fold_0/refiner_best.pt", map_location=device)
    refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
    refiner.eval()

    # 1. Evaluate V27 Fold 0 (used in V28-A)
    print("\n--- Evaluating V27 Fold 0 Baseline (checkpoints/v27/segmenter_v27_fold0_best.pt) ---")
    s27 = build_v18_segmenter(num_classes=5).to(device)
    s27_obj = torch.load("checkpoints/v27/segmenter_v27_fold0_best.pt", map_location=device)
    s27.load_state_dict(s27_obj["state_dict"] if "state_dict" in s27_obj else s27_obj)
    s27.eval()

    cmat_v27 = evaluate_scene_predictions(
        val_scenes=val_scenes,
        segmenter=s27,
        refiner=refiner,
        preprocessor=preprocessor,
        device=device,
        offset_vector=[0.0, 0.0, 0.0, 0.10, 0.25],
        use_flip_tta=True,
        sample_stride=5
    )
    m_v27 = compute_metrics_from_conf_mat(cmat_v27)
    print(f"V27 Fold 0: Q_seg={m_v27['Q_seg']:.5f} | mIoU={m_v27['mIoU']:.5f} | F1={m_v27['Macro_F1']:.5f}")
    CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]
    for c in range(5):
        print(f"  {CLASS_NAMES[c]:12s}: IoU={m_v27['class_ious'][c]:.4f} | Prec={m_v27['class_precisions'][c]:.4f} | Rec={m_v27['class_recalls'][c]:.4f}")

    # 2. Evaluate V28 Fold 0 (checkpoints/v28/segmenter_v28_fold0_best.pt)
    print("\n--- Evaluating V28 Fold 0 Upgrade (checkpoints/v28/segmenter_v28_fold0_best.pt) ---")
    s28 = build_v18_segmenter(num_classes=5).to(device)
    s28_obj = torch.load("checkpoints/v28/segmenter_v28_fold0_best.pt", map_location=device)
    s28.load_state_dict(s28_obj["state_dict"] if "state_dict" in s28_obj else s28_obj)
    s28.eval()

    cmat_v28 = evaluate_scene_predictions(
        val_scenes=val_scenes,
        segmenter=s28,
        refiner=refiner,
        preprocessor=preprocessor,
        device=device,
        offset_vector=[0.0, 0.0, 0.0, 0.10, 0.25],
        use_flip_tta=True,
        sample_stride=5
    )
    m_v28 = compute_metrics_from_conf_mat(cmat_v28)
    print(f"V28 Fold 0: Q_seg={m_v28['Q_seg']:.5f} | mIoU={m_v28['mIoU']:.5f} | F1={m_v28['Macro_F1']:.5f}")
    for c in range(5):
        print(f"  {CLASS_NAMES[c]:12s}: IoU={m_v28['class_ious'][c]:.4f} | Prec={m_v28['class_precisions'][c]:.4f} | Rec={m_v28['class_recalls'][c]:.4f}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Held-Out Fold 0 Scenes):")
    dq = m_v28['Q_seg'] - m_v27['Q_seg']
    dmiou = m_v28['mIoU'] - m_v27['mIoU']
    df1 = m_v28['Macro_F1'] - m_v27['Macro_F1']
    print(f"  Delta Q_seg:    {dq:+.5f}")
    print(f"  Delta mIoU:     {dmiou:+.5f}")
    print(f"  Delta Macro F1: {df1:+.5f}")
    for c in range(5):
        diou = m_v28['class_ious'][c] - m_v27['class_ious'][c]
        print(f"  Delta {CLASS_NAMES[c]:12s}: {diou:+.5f}")
    print("=" * 80)

    out_data = {
        "metrics": {
            "v27_fold0_q_seg": round(m_v27["Q_seg"], 5),
            "v28_swa_fold0_q_seg": round(m_v28["Q_seg"], 5),
            "delta_q_seg": round(dq, 5),
            "delta_miou": round(dmiou, 5),
            "delta_macro_f1": round(df1, 5),
            "delta_fish_iou": round(m_v28["class_ious"][2] - m_v27["class_ious"][2], 5),
            "delta_macro_iou": round(m_v28["class_ious"][4] - m_v27["class_ious"][4], 5),
            "delta_ray_iou": round(m_v28["class_ious"][3] - m_v27["class_ious"][3], 5)
        },
        "verdict": "PROMOTED" if dq > 0.005 else "REJECTED"
    }
    out_file = "experiments/overnight_20261024/predictions/exp08_reproduce.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote reproduction results to {out_file}")

if __name__ == "__main__":
    main()
