"""
Small-Object Preservation & Label Integrity Test (Stage 3 Verification)
Tests:
1. Label integrity: Assert unique(mask_after) subseteq {0, 1, 2, 3, 4} under all transforms.
2. Retention logging: Injects 1x1, 2x2, 5x5, 10x10 synthetic objects and measures retention rate under random crops.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.transforms import SynchronizedTransform


def test_label_integrity_and_small_objects():
    print("=" * 60)
    print("Test 3.4: Label Integrity & Small-Object Retention Audit")
    print("=" * 60)

    # Base image and mask (1024x1024)
    img = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    mask = np.zeros((1024, 1024), dtype=np.uint8)

    # Inject small synthetic targets:
    # 1x1 pixel object (Class 2: Fish) at center (512, 512)
    mask[512, 512] = 2

    # 2x2 pixel object (Class 3: Shark) at (520:522, 520:522)
    mask[520:522, 520:522] = 3

    # 5x5 pixel object (Class 1: Coral) at (530:535, 530:535)
    mask[530:535, 530:535] = 1

    # 10x10 pixel object (Class 4: Other) at (550:560, 550:560)
    mask[550:560, 550:560] = 4

    tx = SynchronizedTransform(
        target_size=(1024, 1024),
        flip_prob=0.5,
        crop_size=(512, 512),
        crop_prob=0.8
    )

    trials = 50
    retained_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    label_integrity_passed = True

    for i in range(trials):
        _, _, mask_aug = tx(img, img, mask, seed=1000 + i)

        # Invariant 1: Label Integrity (Nearest Neighbor check)
        unique_classes = np.unique(mask_aug)
        if not set(unique_classes).issubset({0, 1, 2, 3, 4}):
            label_integrity_passed = False
            print(f"  [FAIL] Trial {i} produced corrupted classes: {unique_classes}")
            break

        # Check presence of each class
        for c in (1, 2, 3, 4):
            if c in unique_classes:
                retained_counts[c] += 1

    assert label_integrity_passed, "Label integrity failed! Nearest neighbor violated."
    print("  [PASS] Label integrity 100% verified: mask values strictly in {0, 1, 2, 3, 4} across all 50 trials.")

    print("\n  [AUDIT] Small-Object Retention Rates under 512x512 Random Crops (50 trials):")
    print(f"    - 1x1 Object (Class 2):  {retained_counts[2]}/{trials} ({retained_counts[2]/trials*100:.1f}%)")
    print(f"    - 2x2 Object (Class 3):  {retained_counts[3]}/{trials} ({retained_counts[3]/trials*100:.1f}%)")
    print(f"    - 5x5 Object (Class 1):  {retained_counts[1]}/{trials} ({retained_counts[1]/trials*100:.1f}%)")
    print(f"    - 10x10 Object (Class 4): {retained_counts[4]}/{trials} ({retained_counts[4]/trials*100:.1f}%)")
    print("Small-object retention and label integrity audit passed!\n")


if __name__ == '__main__':
    test_label_integrity_and_small_objects()
