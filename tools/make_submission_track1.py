"""
Track 1 Submission Packager
Formats and packages model predictions for the official Track 1 Leaderboard:
1. Validates restored images (RGB PNG) and segmentation masks (1-channel uint8 PNG, values in {0, 1, 2, 3, 4}).
2. Interfaces with resize_for_submission.py to restore original test resolutions if requested.
3. Produces submission_track1.zip with exact root structure:
     restored/
       <sample_id>.png
     seg_masks/
       <sample_id>.png
4. Audits archive integrity to prevent leaderboard rejection.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import zipfile
import tempfile
import argparse
from typing import Optional
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.resize_for_submission import resize_predictions_to_original


def package_submission_track1(
    restored_dir: str,
    masks_dir: str,
    output_zip: str = "submission_track1.zip",
    ref_images_dir: Optional[str] = None
) -> str:
    print("=" * 60)
    print("[SUBMISSION] Packaging Track 1 submission...")
    print(f"  Restored source: {restored_dir}")
    print(f"  Masks source:    {masks_dir}")
    print(f"  Output ZIP:      {output_zip}")
    print("=" * 60)

    # If reference images provided, run resize wrapper in temp directory
    with tempfile.TemporaryDirectory() as tmp_resize_dir:
        if ref_images_dir and os.path.exists(ref_images_dir):
            active_restored_dir = os.path.join(tmp_resize_dir, "restored")
            active_masks_dir = os.path.join(tmp_resize_dir, "seg_masks")
            print(f"[SUBMISSION] Applying official resize wrapper against: {ref_images_dir}")
            resize_predictions_to_original(
                pred_restored_dir=restored_dir,
                pred_masks_dir=masks_dir,
                ref_images_dir=ref_images_dir,
                out_restored_dir=active_restored_dir,
                out_masks_dir=active_masks_dir
            )
        else:
            active_restored_dir = restored_dir
            active_masks_dir = masks_dir

        # Validate restored files
        restored_files = sorted([f for f in os.listdir(active_restored_dir) if f.lower().endswith('.png')])
        masks_files = sorted([f for f in os.listdir(active_masks_dir) if f.lower().endswith('.png')])

        assert len(restored_files) > 0, "No restored PNG files found!"
        assert len(masks_files) > 0, "No mask PNG files found!"

        restored_bases = {os.path.splitext(f)[0] for f in restored_files}
        masks_bases = {os.path.splitext(f)[0] for f in masks_files}

        assert restored_bases == masks_bases, (
            f"Basename mismatch between restored and masks! "
            f"Restored only: {restored_bases - masks_bases}, "
            f"Masks only: {masks_bases - restored_bases}"
        )

        # Audit content integrity
        for fname in masks_files:
            mask_path = os.path.join(active_masks_dir, fname)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            assert mask is not None, f"Unreadable mask: {mask_path}"
            assert mask.ndim == 2, f"Mask must be 1-channel (2D), got shape {mask.shape} for {fname}"
            assert mask.dtype == np.uint8, f"Mask must be uint8, got {mask.dtype} for {fname}"
            classes = np.unique(mask)
            assert set(classes).issubset({0, 1, 2, 3, 4}), f"Invalid mask classes in {fname}: {classes}"

        # Create output ZIP archive
        os.makedirs(os.path.dirname(os.path.abspath(output_zip)), exist_ok=True)
        with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for fname in restored_files:
                src_path = os.path.join(active_restored_dir, fname)
                arcname = f"restored/{fname}"
                zipf.write(src_path, arcname=arcname)

            for fname in masks_files:
                src_path = os.path.join(active_masks_dir, fname)
                arcname = f"seg_masks/{fname}"
                zipf.write(src_path, arcname=arcname)

    # Audit created ZIP archive layout
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        namelist = zipf.namelist()
        has_restored = any(name.startswith('restored/') for name in namelist)
        has_masks = any(name.startswith('seg_masks/') for name in namelist)
        assert has_restored, "Missing root 'restored/' directory in ZIP!"
        assert has_masks, "Missing root 'seg_masks/' directory in ZIP!"

        # Assert no extra root wrappers (e.g. submission/restored/...)
        for name in namelist:
            assert name.startswith('restored/') or name.startswith('seg_masks/'), f"Invalid path in ZIP root: {name}"

    file_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"[SUBMISSION] Successfully generated: {output_zip} ({file_size_mb:.2f} MB)")
    print(f"  - Total samples: {len(restored_files)}")
    print(f"  - Root layout: restored/ ({len(restored_files)} files), seg_masks/ ({len(masks_files)} files)")
    print("=" * 60)
    return output_zip


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Package Track 1 Submission ZIP")
    parser.add_argument("--restored", type=str, required=True, help="Directory of restored PNGs")
    parser.add_argument("--masks", type=str, required=True, help="Directory of mask PNGs")
    parser.add_argument("--output", type=str, default="submission_track1.zip", help="Output ZIP path")
    parser.add_argument("--ref_images", type=str, default=None, help="Reference images directory for official resizing")
    args = parser.parse_args()

    package_submission_track1(
        restored_dir=args.restored,
        masks_dir=args.masks,
        output_zip=args.output,
        ref_images_dir=args.ref_images
    )
