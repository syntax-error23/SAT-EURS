"""
Synthetic Fixture and Scorer Contract Verification Test
Tests exact mathematical parity against known analytic confusion matrices and restoration formulas.
"""

import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import (
    compute_psnr_and_norm,
    compute_sharpness_ratio,
    compute_ssim_single,
    RestorationEvaluator
)
from metrics.composite_score import compute_composite_score


def test_synthetic_confusion_matrix_fixture():
    """
    Known 5x5 Confusion Matrix Synthetic Test:
    Class 0: TP=80, FP=10, FN=20 -> Denom IoU = 80+10+20 = 110 -> IoU = 80/110 = 0.72727...
             Denom F1 = 2*80 + 10 + 20 = 190 -> F1 = 160/190 = 0.84210...
    Class 1: TP=50, FP=15, FN=15 -> Denom IoU = 50+15+15 = 80  -> IoU = 50/80 = 0.625
             Denom F1 = 100 + 30 = 130 -> F1 = 100/130 = 0.76923...
    Class 2: TP=40, FP=10, FN=10 -> Denom IoU = 40+10+10 = 60  -> IoU = 40/60 = 0.66667...
             Denom F1 = 80 + 20 = 100 -> F1 = 80/100 = 0.80000...
    Class 3: TP=20, FP=5,  FN=5  -> Denom IoU = 20+5+5 = 30   -> IoU = 20/30 = 0.66667...
             Denom F1 = 40 + 10 = 50 -> F1 = 40/50 = 0.80000...
    Class 4: TP=10, FP=5,  FN=5  -> Denom IoU = 10+5+5 = 20   -> IoU = 10/20 = 0.500
             Denom F1 = 20 + 10 = 30 -> F1 = 20/30 = 0.66667...
    """
    evaluator = SegmentationEvaluator(num_classes=5)

    # Construct synthetic predictions and targets yielding exact known confusion matrix
    cm_known = np.array([
        [80,  5,  5,  5,  5], # GT Class 0 (sum=100, TP=80, FN=20)
        [ 5, 50,  5,  5,  0], # GT Class 1 (sum=65,  TP=50, FN=15)
        [ 2,  3, 40,  0,  5], # GT Class 2 (sum=50,  TP=40, FN=10)
        [ 1,  2,  2, 20,  0], # GT Class 3 (sum=25,  TP=20, FN=5)
        [ 2,  5,  0,  0, 10], # GT Class 4 (sum=17,  TP=10, FN=7)
    ], dtype=np.int64)

    evaluator.confusion_matrix = cm_known.copy()
    res = evaluator.compute()

    # Verify per-class IoU
    tp = np.diag(cm_known).astype(float)
    fp = cm_known.sum(axis=0) - tp
    fn = cm_known.sum(axis=1) - tp

    expected_iou = tp / (tp + fp + fn)
    expected_f1 = (2 * tp) / (2 * tp + fp + fn)
    expected_prec = tp / (tp + fp)
    expected_rec = tp / (tp + fn)

    for c in range(5):
        assert abs(res['per_class_IoU'][c] - round(expected_iou[c], 4)) <= 1e-4, f"Class {c} IoU mismatch"
        assert abs(res['per_class_F1'][c] - round(expected_f1[c], 4)) <= 1e-4, f"Class {c} F1 mismatch"
        assert abs(res['per_class_precision'][c] - round(expected_prec[c], 4)) <= 1e-4, f"Class {c} Prec mismatch"
        assert abs(res['per_class_recall'][c] - round(expected_rec[c], 4)) <= 1e-4, f"Class {c} Rec mismatch"

    expected_miou = float(np.mean(expected_iou))
    expected_macro_f1 = float(np.mean(expected_f1))
    assert abs(res['mIoU'] - round(expected_miou, 4)) <= 1e-4
    assert abs(res['macro_F1'] - round(expected_macro_f1, 4)) <= 1e-4
    assert res['coverage_penalty'] == 1.0
    assert abs(res['Q_seg'] - round(expected_miou * expected_macro_f1, 4)) <= 1e-4

    print("  [PASS] Synthetic 5x5 confusion matrix test matches mathematical ground truth exactly.")


def test_restoration_evaluator_parity():
    evaluator = RestorationEvaluator()

    # Image 1: perfect match -> PSNR=100.0 (clipped to 1.0), SSIM=1.0, Sharpness=1.0
    p1 = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)
    t1 = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)

    evaluator.update(p1, t1)
    res = evaluator.compute()

    assert res['PSNR_norm'] == 1.0
    assert res['SSIM'] == 1.0
    assert res['sharpness_ratio'] == 1.0
    assert res['Q_rest'] == 1.0

    print("  [PASS] Restoration evaluator perfect reconstruction parity confirmed.")


def test_composite_formula_contract():
    q_seg = 0.50
    q_rest = 0.60
    q_nr = 0.70

    comp = compute_composite_score(q_seg=q_seg, q_rest=q_rest, q_nr=q_nr)
    expected = 0.60 * 0.50 + 0.30 * 0.60 + 0.10 * 0.70
    assert abs(comp['Composite_Score'] - round(expected, 4)) <= 1e-4
    assert comp['Q_seg_weighted'] == 0.30
    assert comp['Q_rest_weighted'] == 0.18
    assert comp['Q_NR_weighted'] == 0.07

    print("  [PASS] Composite score formula contract verified: 0.60/0.30/0.10 weighting confirmed.")


if __name__ == '__main__':
    print("Running Scorer Contract & Synthetic Fixture Suite...")
    test_synthetic_confusion_matrix_fixture()
    test_restoration_evaluator_parity()
    test_composite_formula_contract()
    print("ALL SCORER CONTRACT TESTS PASSED!")
