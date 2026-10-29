import os
import json
import time
import sys

sys.path.insert(0, ".")

from experiments.overnight_20261024.scripts.benchmark_engine import run_full_20scene_oof_benchmark

def main():
    print("=" * 80)
    print("SAT-EURS TRACK 1 — EXP-01: BASELINE OUT-OF-FOLD 20-SCENE REPRODUCTION")
    print("=" * 80)

    segmenter_ckpts = {
        0: 'checkpoints/v27/segmenter_v27_fold0_best.pt',
        1: 'checkpoints/v18/fold_1/segmenter_best.pt',
        2: 'checkpoints/v18/fold_2/segmenter_best.pt',
        3: 'checkpoints/v18/fold_3/segmenter_best.pt',
        4: 'checkpoints/v18/fold_4/segmenter_best.pt',
    }

    refiner_ckpts = {
        0: 'checkpoints/v18/fold_0/refiner_best.pt',
        1: 'checkpoints/v18/fold_1/refiner_best.pt',
        2: 'checkpoints/v18/fold_2/refiner_best.pt',
        3: 'checkpoints/v18/fold_3/refiner_best.pt',
        4: 'checkpoints/v18/fold_4/refiner_best.pt',
    }

    res = run_full_20scene_oof_benchmark(
        segmenter_ckpts=segmenter_ckpts,
        refiner_ckpts=refiner_ckpts,
        offset_vector=[0.0, 0.0, 0.0, 0.10, 0.25],
        use_flip_tta=True,
        sample_stride=5,
        min_component_size=0,
        device_id=0
    )

    out_json = "experiments/overnight_20261024/predictions/exp01_baseline_oof.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)

    p_m = res["pooled_metrics"]
    print("\n" + "=" * 80)
    print(f"EXP-01 BASELINE POOLED 20-SCENE BENCHMARK RESULTS ({res['runtime_s']}s):")
    print(f"  Pooled Q_seg:    {p_m['Q_seg']:.4f}")
    print(f"  Pooled mIoU:     {p_m['mIoU']:.4f}")
    print(f"  Pooled Macro F1: {p_m['Macro_F1']:.4f}")
    print(f"  Coverage:        {p_m['coverage']:.1f}")
    print("  Per-Class IoU:")
    CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]
    for c in range(5):
        print(f"    {CLASS_NAMES[c]:12s}: IoU={p_m['class_ious'][c]:.4f} | F1={p_m['class_f1s'][c]:.4f} | Prec={p_m['class_precisions'][c]:.4f} | Rec={p_m['class_recalls'][c]:.4f}")
    print("=" * 80)

    # Fold summary
    print("\nPer-Fold Breakdown (Single-Fold View):")
    q_segs = []
    for f in range(5):
        fm = res["fold_metrics"][f"fold_{f}"]["metrics"]
        q_segs.append(fm["Q_seg"])
        print(f"  Fold {f}: Q_seg={fm['Q_seg']:.4f} | mIoU={fm['mIoU']:.4f} | F1={fm['Macro_F1']:.4f}")
    print(f"  Arithmetic Mean of 5 Folds Q_seg: {sum(q_segs)/5:.4f}")

if __name__ == "__main__":
    main()
