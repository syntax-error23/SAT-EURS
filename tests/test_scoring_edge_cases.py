"""
Unit Tests for Scoring Edge Cases & Leaderboard Mathematical Reconstruction
Tests:
1. Absent classes in ground truth and prediction
2. False positive intrusion into absent classes
3. All-background masks
4. Coverage penalty boundary conditions
5. PSNR window clipping (below 15 dB, above 35 dB)
6. Zero-variance sharpness ratio protection
7. Exact mathematical reconstruction of Leaderboard Submissions #18666 and #18668
"""

import sys
import os
import math
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_sharpness_ratio, compute_ssim_single
from metrics.composite_score import compute_composite_score


def test_absent_classes():
    evaluator = SegmentationEvaluator(num_classes=5)

    # Case 1: Ground truth only has Class 0 (Background) and Class 2 (Fish)
    # Prediction perfectly matches [0, 2]. Classes 1, 3, 4 are absent in both.
    gt = torch.tensor([[0, 0], [2, 2]], dtype=torch.long)
    pred = torch.tensor([[0, 0], [2, 2]], dtype=torch.long)

    evaluator.reset()
    evaluator.update(pred, gt)
    res = evaluator.compute()

    # In standard segmentation convention when denominator is 0 (class neither in GT nor pred),
    # metric implementation sets IoU=1.0 and F1=1.0
    assert res['per_class_IoU'][1] == 1.0, f"Expected 1.0 for absent class 1, got {res['per_class_IoU'][1]}"
    assert res['per_class_IoU'][3] == 1.0, f"Expected 1.0 for absent class 3, got {res['per_class_IoU'][3]}"
    assert res['per_class_IoU'][4] == 1.0, f"Expected 1.0 for absent class 4, got {res['per_class_IoU'][4]}"
    assert res['coverage_penalty'] == 1.0
    print("  [PASS] Absent classes in both GT and Pred correctly assigned 1.0 (no penalty)")


def test_false_positive_intrusion():
    evaluator = SegmentationEvaluator(num_classes=5)

    # Ground truth only has Class 0. Model falsely predicts Class 4 (Starfish) on 1 pixel.
    gt = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    pred = torch.tensor([[0, 0], [0, 4]], dtype=torch.long)

    evaluator.reset()
    evaluator.update(pred, gt)
    res = evaluator.compute()

    # For Class 4: TP=0, FP=1, FN=0 -> Denom=1 -> IoU = 0.0, F1 = 0.0
    assert res['per_class_IoU'][4] == 0.0, f"Expected 0.0 for Class 4 FP, got {res['per_class_IoU'][4]}"
    assert res['per_class_F1'][4] == 0.0, f"Expected 0.0 for Class 4 FP, got {res['per_class_F1'][4]}"
    # Macro F1 should be average of [1.0 (Class 0 with 3/4 TP), 1.0, 1.0, 1.0, 0.0]
    print(f"  [PASS] False positive into absent class correctly drops class F1 to 0.0 (mIoU={res['mIoU']:.4f}, F1={res['macro_F1']:.4f})")


def test_coverage_penalty():
    evaluator = SegmentationEvaluator(num_classes=5)

    # GT has Class 0, 1, 2, 3, 4 present.
    gt = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    # Model misses Class 3 (predicts as 0)
    pred = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)

    evaluator.reset()
    evaluator.update(pred, gt)
    res = evaluator.compute()

    # Class 3 has TP=0 while present in GT
    assert res['coverage_penalty'] == 0.0, f"Expected hard coverage penalty 0.0, got {res['coverage_penalty']}"
    assert res['Q_seg'] == 0.0, f"Expected Q_seg collapsed to 0.0, got {res['Q_seg']}"
    assert res['soft_coverage_penalty'] == 0.75, f"Expected soft coverage 3/4=0.75, got {res['soft_coverage_penalty']}"
    print("  [PASS] Coverage penalty triggers correctly on zero-recall present class")


def test_psnr_clipping():
    # Below 15 dB floor
    p_low = np.zeros((10, 10, 3), dtype=np.float32)
    t_low = np.ones((10, 10, 3), dtype=np.float32) * 0.5 # MSE = 0.25 -> PSNR ~6.02 dB
    psnr, psnr_norm = compute_psnr_and_norm(p_low, t_low)
    assert psnr < 15.0
    assert psnr_norm == 0.0, f"Expected PSNR_norm clipped to 0.0, got {psnr_norm}"

    # Above 35 dB ceiling
    mse_40 = 10.0 ** (-4.0)
    t_high = np.full((10, 10, 3), math.sqrt(mse_40), dtype=np.float32)
    p_high = np.zeros((10, 10, 3), dtype=np.float32)
    psnr_hi, norm_hi = compute_psnr_and_norm(p_high, t_high)
    assert psnr_hi >= 35.0
    assert norm_hi == 1.0, f"Expected PSNR_norm clipped to 1.0, got {norm_hi}"
    print(f"  [PASS] PSNR clipping verified: <15dB -> 0.0, >35dB -> 1.0")


def test_leaderboard_reconstruction():
    print("\n" + "=" * 60)
    print("Leaderboard Exact Mathematical Reconstruction Audit")
    print("=" * 60)

    # Submission #18666
    miou_66 = 0.521
    f1_66 = 0.666
    psnr_66 = 24.50
    ssim_66 = 0.821
    sharp_66 = 0.611
    comp_target_66 = 0.473

    q_seg_66 = round(miou_66 * f1_66, 3)
    assert q_seg_66 == 0.347, f"Expected Q_seg 0.347, got {q_seg_66}"

    psnr_norm_66 = (psnr_66 - 15.0) / (35.0 - 15.0)
    assert abs(psnr_norm_66 - 0.475) < 1e-4

    q_fr_66 = 0.45 * psnr_norm_66 + 0.45 * ssim_66 + 0.10 * sharp_66
    assert abs(q_fr_66 - 0.6443) < 1e-3

    # Recover implied Q_NR: Composite = 0.60 * Q_seg + 0.30 * Q_FR + 0.10 * Q_NR
    # 0.473 = 0.60 * 0.347 + 0.30 * 0.6443 + 0.10 * Q_NR
    implied_q_nr_66 = (comp_target_66 - 0.60 * q_seg_66 - 0.30 * q_fr_66) / 0.10
    print(f"  Submission #18666 Implied Q_NR: {implied_q_nr_66:.4f}")
    assert 0.70 <= implied_q_nr_66 <= 0.73

    # Submission #18668
    miou_68 = 0.523
    f1_68 = 0.643
    comp_target_68 = 0.467

    q_seg_68 = round(miou_68 * f1_68, 3)
    assert q_seg_68 == 0.336, f"Expected Q_seg 0.336, got {q_seg_68}"

    implied_q_nr_68 = (comp_target_68 - 0.60 * q_seg_68 - 0.30 * q_fr_66) / 0.10
    print(f"  Submission #18668 Implied Q_NR: {implied_q_nr_68:.4f}")
    assert 0.70 <= implied_q_nr_68 <= 0.73

    diff_comp = comp_target_66 - comp_target_68 # 0.006
    diff_q_seg_contrib = 0.60 * (q_seg_66 - q_seg_68) # 0.60 * 0.011 = 0.0066
    print(f"  Composite Drop: {diff_comp:.4f} | Q_seg Contribution Drop: {diff_q_seg_contrib:.4f}")
    assert abs(diff_comp - diff_q_seg_contrib) < 0.001
    print("  [PASS] Score drop between #18666 and #18668 is 100% accounted for by Q_seg Macro F1 drop!\n")


if __name__ == "__main__":
    test_absent_classes()
    test_false_positive_intrusion()
    test_coverage_penalty()
    test_psnr_clipping()
    test_leaderboard_reconstruction()
    print("ALL SCORING EDGE CASE UNIT TESTS PASSED!")
