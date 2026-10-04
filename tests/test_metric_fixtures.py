"""
Metric Fixtures & Parity Unit Tests (Stage 5 Verification)
Tests:
1. Segmentation golden fixtures: perfect match (IoU=1), inverted (IoU=0), coverage penalty.
2. Restoration golden fixtures: MSE=0 -> PSNR=100dB, SSIM=1.0.
3. Sharpness ratio numerator/denominator fixtures and zero-variance protection.
"""

import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from metrics import (
    SegmentationEvaluator,
    RestorationEvaluator,
    compute_psnr_and_norm,
    compute_sharpness_ratio,
    compute_ssim_single,
    compute_composite_score
)


def test_segmentation_fixtures():
    print("=" * 60)
    print("Test 5.1: Segmentation Golden Fixtures & Coverage Penalty")
    print("=" * 60)

    evaluator = SegmentationEvaluator(num_classes=5)

    # Fixture 1: Perfect Match
    target_mask = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    evaluator.reset()
    evaluator.update(target_mask, target_mask)
    res_perfect = evaluator.compute()

    assert res_perfect['mIoU'] == 1.0, f"Expected mIoU 1.0, got {res_perfect['mIoU']}"
    assert res_perfect['macro_F1'] == 1.0, f"Expected macro_F1 1.0, got {res_perfect['macro_F1']}"
    assert res_perfect['coverage_penalty'] == 1.0, "Coverage penalty should be 1.0 for perfect match!"
    assert res_perfect['Q_seg'] == 1.0, f"Expected Q_seg 1.0, got {res_perfect['Q_seg']}"
    print("  [PASS] Segmentation Fixture 1 (Perfect Match): mIoU = 1.0, F1 = 1.0, Q_seg = 1.0")

    # Fixture 2: Inverted / Complete Miss
    evaluator.reset()
    pred_inverted = torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long) # zero correct predictions
    evaluator.update(pred_inverted, target_mask)
    res_miss = evaluator.compute()

    assert res_miss['mIoU'] == 0.0, f"Expected mIoU 0.0, got {res_miss['mIoU']}"
    assert res_miss['macro_F1'] == 0.0, f"Expected macro_F1 0.0, got {res_miss['macro_F1']}"
    assert res_miss['Q_seg'] == 0.0, f"Expected Q_seg 0.0, got {res_miss['Q_seg']}"
    print("  [PASS] Segmentation Fixture 2 (Complete Miss): mIoU = 0.0, F1 = 0.0, Q_seg = 0.0")

    # Fixture 3: Missing Class Coverage Penalty
    # Predict everything as background (class 0)
    evaluator.reset()
    pred_all_bg = torch.zeros_like(target_mask)
    evaluator.update(pred_all_bg, target_mask)
    res_bg = evaluator.compute()
    assert res_bg['coverage_penalty'] == 0.0, "Coverage penalty should trigger (0.0) when classes 1..4 missed!"
    assert res_bg['Q_seg'] == 0.0
    print("  [PASS] Segmentation Fixture 3 (Omitted Classes): coverage_penalty correctly collapsed Q_seg to 0.0\n")


def test_restoration_fixtures():
    print("=" * 60)
    print("Test 5.2: Restoration Golden Fixtures & Sharpness Ratio Guardrails")
    print("=" * 60)

    # Fixture 1: Perfect Match (MSE == 0)
    img_clean = np.random.rand(256, 256, 3).astype(np.float32)
    psnr, psnr_norm = compute_psnr_and_norm(img_clean, img_clean)
    assert psnr == 100.0, f"Expected perfect PSNR 100.0, got {psnr}"
    assert psnr_norm == 1.0, f"Expected perfect PSNR_norm 1.0, got {psnr_norm}"

    ssim_val = compute_ssim_single(img_clean, img_clean)
    assert abs(ssim_val - 1.0) < 1e-4, f"Expected perfect SSIM ~1.0, got {ssim_val}"
    print(f"  [PASS] Restoration Fixture 1 (Perfect Match): PSNR = {psnr} dB, PSNR_norm = {psnr_norm}, SSIM = {ssim_val:.4f}")

    # Fixture 1B: Official 15-35 dB Normalization Window Checks
    # Synthetic MSE producing 25.0 dB: 10 * log10(1 / mse) = 25 -> mse = 10^(-2.5)
    mse_25 = 10.0 ** (-2.5)
    p25 = np.zeros((10, 10, 3), dtype=np.float32)
    t25 = np.full((10, 10, 3), math.sqrt(mse_25), dtype=np.float32)
    psnr_25, norm_25 = compute_psnr_and_norm(p25, t25)
    assert abs(psnr_25 - 25.0) < 0.1, f"Expected PSNR ~25.0, got {psnr_25}"
    assert abs(norm_25 - 0.50) < 1e-2, f"Expected PSNR_norm ~0.50, got {norm_25}"
    print(f"  [PASS] Restoration Fixture 1B (15-35 dB Window): PSNR = 25.0 dB -> PSNR_norm = {norm_25:.4f} (midpoint 0.50)")

    # Fixture 2: Sharpness Ratio Numerator/Denominator
    # Test clean edge variance > restored variance
    img_sharp = np.zeros((100, 100, 3), dtype=np.float32)
    img_sharp[40:60, 40:60] = 1.0 # High contrast edges
    img_blurry = np.full((100, 100, 3), 0.5, dtype=np.float32) # Completely flat (0 edge variance)

    ratio_sharp_to_sharp = compute_sharpness_ratio(img_sharp, img_sharp)
    assert abs(ratio_sharp_to_sharp - 1.0) < 1e-3, f"Expected ratio ~1.0, got {ratio_sharp_to_sharp}"
    print(f"  [PASS] Sharpness Fixture A (Identical Sharp): ratio = {ratio_sharp_to_sharp:.4f}")

    ratio_blurry_to_sharp = compute_sharpness_ratio(img_blurry, img_sharp)
    assert ratio_blurry_to_sharp < 0.1, f"Expected low ratio for blurry image, got {ratio_blurry_to_sharp}"
    print(f"  [PASS] Sharpness Fixture B (Blurry restored): ratio = {ratio_blurry_to_sharp:.4f} (< 0.1)")

    # Fixture 3: Zero-Denominator Protection (Clean image is flat)
    ratio_zero_den = compute_sharpness_ratio(img_sharp, img_blurry)
    assert ratio_zero_den == 0.0, f"Expected fallback 0.0 when clean has 0 variance, got {ratio_zero_den}"

    ratio_both_zero = compute_sharpness_ratio(img_blurry, img_blurry)
    assert ratio_both_zero == 1.0, f"Expected fallback 1.0 when both have 0 variance, got {ratio_both_zero}"
    print("  [PASS] Sharpness Fixture C (Zero-Denominator Guard): correctly handled without NaN/div-0.\n")


def test_composite_score():
    print("=" * 60)
    print("Test 5.3: Composite Score Weighting & Status Audit")
    print("=" * 60)
    comp = compute_composite_score(q_seg=0.80, q_rest=0.70, q_nr=0.50)
    expected = 0.60 * 0.80 + 0.30 * 0.70 + 0.10 * 0.50
    assert abs(comp['Composite_Score'] - expected) < 1e-4
    assert comp['status'] == 'PROVISIONAL_PARTIAL'
    assert comp['official_parity'] is False
    print(f"  [PASS] Composite Score computed: {comp['Composite_Score']} (60% Seg: {comp['Q_seg_weighted']}, 30% Rest: {comp['Q_rest_weighted']}, 10% NR: {comp['Q_NR_weighted']})")
    print("  [PASS] Status explicitly marked as PROVISIONAL_PARTIAL.\n")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("  RUNNING STAGE 5 METRIC FIXTURES & PARITY AUDITS")
    print("#" * 60 + "\n")

    test_segmentation_fixtures()
    test_restoration_fixtures()
    test_composite_score()

    print("=" * 60)
    print("STAGE 5 METRIC FIXTURES VERIFICATION: 100% PASSED")
    print("=" * 60 + "\n")
