"""
SAT-EURS Track 1 — V32 Mask Dimension Fix and Repackaging Tool
Resizes all 1,200 segmentation masks to match the official 1024px max-side (aspect-preserved)
contract required by the evaluation server, exactly matching restored/ image dimensions.
"""

import os
import os.path as osp
import sys
import time
import zipfile
import hashlib
import json
import cv2
import numpy as np

sys.path.insert(0, ".")

from tools.resize_for_submission import scale_to_max_side

def fix_and_repackage_v32(
    test_dir: str = "data/test/degraded",
    restored_dir: str = "deploy/v32/v32_outputs/final/restored",
    masks_dir: str = "deploy/v32/v32_outputs/final/seg_masks",
    out_zip: str = "deploy/v32/submission_v32_apex_candidate.zip"
):
    print("=" * 80)
    print("FIXING V32 SUBMISSION MASK DIMENSIONS TO MATCH 1024PX MAX-SIDE CONTRACT")
    print("=" * 80)

    # 1. Gather reference dimensions
    ref_images = {}
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                p = osp.join(root, f)
                ref_images[f] = p

    assert len(ref_images) == 1200, f"Expected 1,200 test images, found {len(ref_images)}"
    print(f"Loaded {len(ref_images)} reference test images.")

    # 2. Process and resize each mask to exact target_w, target_h
    t0 = time.time()
    pixel_counts = {c: 0 for c in range(5)}
    scene_masks = {}
    dimension_mismatches = 0

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])
    assert len(mask_files) == 1200, f"Expected 1,200 masks, found {len(mask_files)}"

    for idx, fname in enumerate(mask_files, 1):
        stem = osp.splitext(fname)[0]
        scene = stem.split("_")[0]
        if scene not in scene_masks:
            scene_masks[scene] = []

        # Find matching ref image
        ref_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = osp.join(test_dir, scene, f"{stem}{ext}")
            if osp.exists(cand):
                ref_path = cand
                break
        if ref_path is None:
            # try flat search
            for k in ref_images:
                if osp.splitext(k)[0] == stem:
                    ref_path = ref_images[k]
                    break
        assert ref_path is not None, f"Could not find ref image for {stem}"

        ref_img = cv2.imread(ref_path)
        orig_h, orig_w = ref_img.shape[:2]
        target_w, target_h = scale_to_max_side(orig_w, orig_h, 1024)

        # Check restored shape
        rest_path = osp.join(restored_dir, f"{stem}.jpg")
        assert osp.exists(rest_path), f"Missing restored image: {rest_path}"
        rest_img = cv2.imread(rest_path)
        assert rest_img.shape[:2] == (target_h, target_w), f"Restored shape mismatch for {stem}: {rest_img.shape} vs {(target_h, target_w)}"

        # Resize mask
        mask_path = osp.join(masks_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim > 2 and mask.shape[2] > 1:
            mask = mask[:, :, 0]

        if (mask.shape[1], mask.shape[0]) != (target_w, target_h):
            dimension_mismatches += 1
            mask_scaled = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(mask_path, mask_scaled)
            mask = mask_scaled

        # Verify classes
        vals = set(np.unique(mask))
        assert vals.issubset({0, 1, 2, 3, 4}), f"Invalid mask classes in {fname}: {vals}"

        # Track uniqueness
        m_hash = hashlib.md5(mask.tobytes()).hexdigest()
        scene_masks[scene].append(m_hash)

        # Track pixel counts
        for c in range(5):
            pixel_counts[c] += int((mask == c).sum())

        if idx % 300 == 0 or idx == len(mask_files):
            print(f"  Verified/Resized [{idx:04d}/1200] masks (Mismatches corrected: {dimension_mismatches})...")

    print(f"\nCompleted dimension normalization in {time.time() - t0:.1f}s.")
    print(f"Total dimension mismatches corrected: {dimension_mismatches} (All 1,200 now exactly match 1024 max-side).")

    # 3. Uniqueness audit
    print("\nScene Mask Uniqueness Audit:")
    for scene, hashes in sorted(scene_masks.items()):
        unique_cnt = len(set(hashes))
        print(f"  Scene '{scene}': {unique_cnt}/200 unique masks")
        assert unique_cnt == 200, f"Scene '{scene}' has {unique_cnt}/200 unique masks!"

    total_pixels = sum(pixel_counts.values())
    print(f"\nTotal Normalized Mask Pixels: {total_pixels:,}")
    print("Class Distribution:")
    for c, cnt in pixel_counts.items():
        pct = (cnt / total_pixels) * 100
        print(f"  Class {c}: {cnt:>12,} ({pct:6.2f}%)")

    # 4. Packaging official submission ZIP
    print(f"\nRepackaging official submission archive: {out_zip}...")
    if osp.exists(out_zip):
        os.remove(out_zip)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(restored_dir)):
            zf.write(osp.join(restored_dir, f), arcname=osp.join("restored", f))
        for f in sorted(os.listdir(masks_dir)):
            zf.write(osp.join(masks_dir, f), arcname=osp.join("seg_masks", f))

    zip_size_mb = osp.getsize(out_zip) / (1024 * 1024)
    with open(out_zip, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"\nArchive created and verified:")
    print(f"  Path:    {out_zip}")
    print(f"  Size:    {zip_size_mb:.2f} MB")
    print(f"  SHA-256: {sha256}")

    manifest_summary = {
        "candidate": "V32-Apex",
        "archive_path": out_zip,
        "sha256": sha256,
        "size_mb": round(zip_size_mb, 2),
        "total_images": 1200,
        "total_pixels": int(total_pixels),
        "pixel_counts": pixel_counts,
        "scene_uniqueness": {sc: len(set(h)) for sc, h in scene_masks.items()},
    }

    manifest_path = out_zip + ".manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_summary, f, indent=2)

    return manifest_summary

if __name__ == "__main__":
    fix_and_repackage_v32()
