"""
Official Resize Wrapper for EURS Track 1 Submissions
Resizes 1024x1024 model predictions back to the original test image resolutions:
  - Restored Images: cv2.INTER_LINEAR (Bilinear interpolation)
  - Segmentation Masks: cv2.INTER_NEAREST (Strictly nearest-neighbor to preserve discrete class IDs)
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
from typing import Dict, Tuple
import cv2
import numpy as np


def resize_predictions_to_original(
    pred_restored_dir: str,
    pred_masks_dir: str,
    ref_images_dir: str,
    out_restored_dir: str,
    out_masks_dir: str
):
    os.makedirs(out_restored_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    ref_files = sorted([f for f in os.listdir(ref_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    assert len(ref_files) > 0, f"No reference images found in {ref_images_dir}"

    print("=" * 60)
    print(f"[RESIZE] Resizing predictions to match {len(ref_files)} reference images in: {ref_images_dir}")
    print("=" * 60)

    for fname in ref_files:
        base_name = os.path.splitext(fname)[0]
        ref_path = os.path.join(ref_images_dir, fname)
        ref_img = cv2.imread(ref_path)
        orig_h, orig_w = ref_img.shape[:2]

        # 1. Resize Restored Image
        # Check candidate extensions
        pred_res_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(pred_restored_dir, f"{base_name}{ext}")
            if os.path.exists(cand):
                pred_res_path = cand
                break

        if pred_res_path is not None:
            res_img = cv2.imread(pred_res_path)
            if (res_img.shape[1], res_img.shape[0]) != (orig_w, orig_h):
                res_resized = cv2.resize(res_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                res_resized = res_img
            out_res_path = os.path.join(out_restored_dir, f"{base_name}.png")
            cv2.imwrite(out_res_path, res_resized)

        # 2. Resize Segmentation Mask (strictly Nearest Neighbor)
        pred_mask_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(pred_masks_dir, f"{base_name}{ext}")
            if os.path.exists(cand):
                pred_mask_path = cand
                break

        if pred_mask_path is not None:
            mask_img = cv2.imread(pred_mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img.ndim > 2 and mask_img.shape[2] > 1:
                mask_img = mask_img[:, :, 0]
            if (mask_img.shape[1], mask_img.shape[0]) != (orig_w, orig_h):
                mask_resized = cv2.resize(mask_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask_img

            # Validate discrete classes
            classes = np.unique(mask_resized)
            assert set(classes).issubset({0, 1, 2, 3, 4}), f"Invalid mask classes after resize: {classes}"

            out_mask_path = os.path.join(out_masks_dir, f"{base_name}.png")
            cv2.imwrite(out_mask_path, mask_resized.astype(np.uint8))

    print(f"[RESIZE] Completed resizing. Restored to: {out_restored_dir}, Masks to: {out_masks_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resize model predictions to original test dimensions")
    parser.add_argument("--pred_restored", type=str, required=True, help="Directory of 1024x1024 restored images")
    parser.add_argument("--pred_masks", type=str, required=True, help="Directory of 1024x1024 segmentation masks")
    parser.add_argument("--ref_images", type=str, required=True, help="Directory of original reference test images")
    parser.add_argument("--out_restored", type=str, required=True, help="Output directory for resized restored images")
    parser.add_argument("--out_masks", type=str, required=True, help="Output directory for resized masks")
    args = parser.parse_args()

    resize_predictions_to_original(
        args.pred_restored,
        args.pred_masks,
        args.ref_images,
        args.out_restored,
        args.out_masks
    )
