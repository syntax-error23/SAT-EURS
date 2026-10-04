"""
Synthetic Coordinate-Alignment & Determinism Test (Stage 3 Verification)
Tests:
1. 4-quadrant coordinate-coded artificial image and mask synchronization.
2. Bitwise transform determinism with fixed seed.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.transforms import SynchronizedTransform


def create_quadrant_synthetic(size=1024):
    half = size // 2
    # Image: 3-channel uint8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Mask: 1-channel uint8
    mask = np.zeros((size, size), dtype=np.uint8)

    # Top-Left: Red [255, 0, 0] -> Class 1
    img[:half, :half] = [255, 0, 0]
    mask[:half, :half] = 1

    # Top-Right: Green [0, 255, 0] -> Class 2
    img[:half, half:] = [0, 255, 0]
    mask[:half, half:] = 2

    # Bottom-Left: Blue [0, 0, 255] -> Class 3
    img[half:, :half] = [0, 0, 255]
    mask[half:, :half] = 3

    # Bottom-Right: White [255, 255, 255] -> Class 4
    img[half:, half:] = [255, 255, 255]
    mask[half:, half:] = 4

    return img, mask


def test_horizontal_flip_alignment():
    print("=" * 60)
    print("Test 3.1: Horizontal Flip Geometric Alignment Audit")
    print("=" * 60)
    img, mask = create_quadrant_synthetic(1024)
    half = 512

    tx = SynchronizedTransform(target_size=(1024, 1024), flip_prob=1.0, crop_prob=0.0)
    deg_aug, clean_aug, mask_aug = tx(img, img, mask, seed=1)

    # After horizontal flip:
    # Top-Left should be Green (Class 2)
    # Top-Right should be Red (Class 1)
    # Bottom-Left should be White (Class 4)
    # Bottom-Right should be Blue (Class 3)

    assert (mask_aug[:half, :half] == 2).all(), "Mask TL is not Class 2 after flip!"
    assert (deg_aug[:half, :half] == [0, 255, 0]).all(), "Degraded TL is not Green after flip!"

    assert (mask_aug[:half, half:] == 1).all(), "Mask TR is not Class 1 after flip!"
    assert (deg_aug[:half, half:] == [255, 0, 0]).all(), "Degraded TR is not Red after flip!"

    assert (mask_aug[half:, :half] == 4).all(), "Mask BL is not Class 4 after flip!"
    assert (deg_aug[half:, :half] == [255, 255, 255]).all(), "Degraded BL is not White after flip!"

    assert (mask_aug[half:, half:] == 3).all(), "Mask BR is not Class 3 after flip!"
    assert (deg_aug[half:, half:] == [0, 0, 255]).all(), "Degraded BR is not Blue after flip!"

    print("  [PASS] Horizontal flip perfectly preserved 100% geometric correspondence across all 4 quadrants.\n")


def test_crop_and_resize_alignment():
    print("=" * 60)
    print("Test 3.2: Crop and Resize Alignment Audit")
    print("=" * 60)
    img, mask = create_quadrant_synthetic(1024)

    # Force a crop purely inside the Top-Left quadrant (pure Red / Class 1)
    tx = SynchronizedTransform(target_size=(1024, 1024), flip_prob=0.0, crop_size=(256, 256), crop_prob=1.0)
    params = {
        'do_flip': False,
        'do_crop': True,
        'crop_box': (50, 50, 256, 256), # inside TL
        'target_size': (1024, 1024)
    }

    deg_aug = tx.apply_spatial_params(img, params, is_mask=False)
    mask_aug = tx.apply_spatial_params(mask, params, is_mask=True)

    assert (mask_aug == 1).all(), f"Cropped TL mask contains non-1 classes: {np.unique(mask_aug)}"
    assert np.allclose(deg_aug, [255, 0, 0]), f"Cropped TL degraded contains non-red pixels: {deg_aug[0,0]}"

    print("  [PASS] Pure quadrant crop produced 100% homogeneous synchronized image and mask.\n")


def test_deterministic_reproducibility():
    print("=" * 60)
    print("Test 3.3: Transform Determinism & Reproducibility Audit")
    print("=" * 60)
    img, mask = create_quadrant_synthetic(1024)
    tx = SynchronizedTransform(target_size=(1024, 1024), flip_prob=0.5, crop_size=(512, 512), crop_prob=0.5)

    # Call with identical seed twice
    d1, c1, m1 = tx(img, img, mask, seed=42)
    d2, c2, m2 = tx(img, img, mask, seed=42)

    assert np.array_equal(d1, d2), "Degraded arrays differ for same seed!"
    assert np.array_equal(c1, c2), "Clean arrays differ for same seed!"
    assert np.array_equal(m1, m2), "Mask arrays differ for same seed!"

    print("  [PASS] Fixed seed produced bitwise-identical output tensors across all 3 channels.\n")


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("  RUNNING STAGE 3 SYNTHETIC COORDINATE ALIGNMENT TESTS")
    print("#" * 60 + "\n")

    test_horizontal_flip_alignment()
    test_crop_and_resize_alignment()
    test_deterministic_reproducibility()

    print("=" * 60)
    print("STAGE 3 SYNTHETIC COORDINATE ALIGNMENT: 100% VERIFIED")
    print("=" * 60 + "\n")
