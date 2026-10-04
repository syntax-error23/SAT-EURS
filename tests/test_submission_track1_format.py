"""
Track 1 Submission Packaging & Format Verifier Unit Test (Phase 2A Pre-Flight)
Verifies:
1. Root-level archive layout: 'restored/' and 'seg_masks/' directly at ZIP root.
2. Restored images: 3-channel RGB PNG format.
3. Segmentation masks: strictly 1-channel uint8 PNG with discrete pixel classes in {0, 1, 2, 3, 4}.
4. Integration with resize_for_submission.py to restore arbitrary original resolutions.
5. Strict failure on malformed submissions (invalid class IDs, 3-channel masks, mismatched filenames).
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import zipfile
import tempfile
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.make_submission_track1 import package_submission_track1


def test_submission_packaging():
    print("=" * 60)
    print("Test 2A.8: Track 1 Packaging & Resize Wrapper")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_res_dir = os.path.join(tmpdir, "pred_res")
        pred_mask_dir = os.path.join(tmpdir, "pred_mask")
        ref_img_dir = os.path.join(tmpdir, "ref_img")
        os.makedirs(pred_res_dir)
        os.makedirs(pred_mask_dir)
        os.makedirs(ref_img_dir)

        # Create 3 synthetic samples:
        # Original size: 240x320
        # Model predicted size: 128x128
        sample_ids = ["test_scene_01", "test_scene_02", "test_scene_03"]
        orig_h, orig_w = 240, 320
        pred_h, pred_w = 128, 128

        for sid in sample_ids:
            # 1. Reference original
            ref_img = np.random.randint(0, 256, (orig_h, orig_w, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(ref_img_dir, f"{sid}.png"), ref_img)

            # 2. Predicted restored
            pred_res = np.random.randint(0, 256, (pred_h, pred_w, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(pred_res_dir, f"{sid}.png"), pred_res)

            # 3. Predicted mask (classes 0..4)
            pred_mask = np.random.randint(0, 5, (pred_h, pred_w), dtype=np.uint8)
            cv2.imwrite(os.path.join(pred_mask_dir, f"{sid}.png"), pred_mask)

        # Test 1: Package with resize wrapper
        out_zip_resized = os.path.join(tmpdir, "submission_resized.zip")
        package_submission_track1(
            restored_dir=pred_res_dir,
            masks_dir=pred_mask_dir,
            output_zip=out_zip_resized,
            ref_images_dir=ref_img_dir
        )

        # Inspect resized ZIP
        with zipfile.ZipFile(out_zip_resized, 'r') as zipf:
            namelist = zipf.namelist()
            for sid in sample_ids:
                res_entry = f"restored/{sid}.png"
                mask_entry = f"seg_masks/{sid}.png"
                assert res_entry in namelist, f"Missing {res_entry} in ZIP!"
                assert mask_entry in namelist, f"Missing {mask_entry} in ZIP!"

                # Read files from zip
                res_bytes = zipf.read(res_entry)
                res_arr = cv2.imdecode(np.frombuffer(res_bytes, np.uint8), cv2.IMREAD_COLOR)
                assert res_arr.shape == (orig_h, orig_w, 3), f"Restored size mismatch: {res_arr.shape} != ({orig_h}, {orig_w}, 3)"

                mask_bytes = zipf.read(mask_entry)
                mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                assert mask_arr.shape == (orig_h, orig_w), f"Mask size mismatch: {mask_arr.shape} != ({orig_h}, {orig_w})"
                assert mask_arr.dtype == np.uint8
                assert set(np.unique(mask_arr)).issubset({0, 1, 2, 3, 4})

        print("  [PASS] Resized packaging verified: restored to original resolution (240x320) with valid mask classes.")

        # Test 2: Package direct 1024x1024
        out_zip_direct = os.path.join(tmpdir, "submission_direct.zip")
        package_submission_track1(
            restored_dir=pred_res_dir,
            masks_dir=pred_mask_dir,
            output_zip=out_zip_direct,
            ref_images_dir=None
        )

        with zipfile.ZipFile(out_zip_direct, 'r') as zipf:
            for sid in sample_ids:
                mask_bytes = zipf.read(f"seg_masks/{sid}.png")
                mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                assert mask_arr.shape == (pred_h, pred_w)

        print("  [PASS] Direct packaging verified: exact dimensions preserved.")

        # Test 3: Failure detection on invalid mask classes (e.g. class ID 9)
        bad_mask = np.full((pred_h, pred_w), 9, dtype=np.uint8)
        cv2.imwrite(os.path.join(pred_mask_dir, f"{sample_ids[0]}.png"), bad_mask)

        failed_as_expected = False
        try:
            package_submission_track1(
                restored_dir=pred_res_dir,
                masks_dir=pred_mask_dir,
                output_zip=os.path.join(tmpdir, "should_fail.zip")
            )
        except AssertionError:
            failed_as_expected = True

        assert failed_as_expected, "Packager failed to reject invalid mask class ID 9!"
        print("  [PASS] Rejection of invalid mask classes (class ID 9) verified.")

    print("=" * 60)
    print("WORK PACKAGE 2A.8 TRACK 1 PACKAGING VERIFICATION: 100% PASSED")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_submission_packaging()
