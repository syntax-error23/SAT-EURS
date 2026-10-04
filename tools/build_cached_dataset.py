"""
High-Speed Preprocessing Cache Tool
Pre-resizes raw 3000x4000 images and masks to canonical 1024x1024 offline:
  - Clean & Degraded: Bilinear interpolation -> RGB PNG
  - Masks: Strictly Nearest Neighbor (cv2.INTER_NEAREST) -> 1-channel uint8 PNG
Supports:
  --aspect_ratio_mode [stretch|pad_letterbox] (default: stretch)
Eliminates online CPU resizing bottlenecks during RTX 5090 training.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
from typing import Tuple
import cv2
import numpy as np


def resize_image(img: np.ndarray, target_size: Tuple[int, int], mode: str, is_mask: bool) -> np.ndarray:
    target_w, target_h = target_size
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    if mode == "stretch":
        return cv2.resize(img, (target_w, target_h), interpolation=interp)
    elif mode == "pad_letterbox":
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        if is_mask:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        else:
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded
    else:
        raise ValueError(f"Unknown aspect ratio mode: {mode}")


def build_cache(
    data_dir: str = "data",
    output_dir: str = "data/cached_1024",
    aspect_ratio_mode: str = "stretch",
    target_size: Tuple[int, int] = (1024, 1024)
):
    if not os.path.exists(os.path.join(data_dir, 'clean')) and os.path.exists(os.path.join(data_dir, 'mini_dev', 'clean')):
        data_dir = os.path.join(data_dir, 'mini_dev')

    clean_dir = os.path.join(data_dir, 'clean')
    masks_dir = os.path.join(data_dir, 'masks')
    degraded_dir = os.path.join(data_dir, 'degraded')

    out_clean = os.path.join(output_dir, 'clean')
    out_masks = os.path.join(output_dir, 'masks')
    out_degraded = os.path.join(output_dir, 'degraded')

    os.makedirs(out_clean, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_degraded, exist_ok=True)

    print("=" * 60)
    print(f"[CACHE] Generating {target_size[0]}x{target_size[1]} cache (mode: {aspect_ratio_mode})")
    print(f"  Source: {os.path.abspath(data_dir)}")
    print(f"  Target: {os.path.abspath(output_dir)}")
    print("=" * 60)

    # 1. Clean references
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in clean_files:
        src = os.path.join(clean_dir, fname)
        dst = os.path.join(out_clean, os.path.splitext(fname)[0] + '.png')
        img = cv2.imread(src)
        resized = resize_image(img, target_size, aspect_ratio_mode, is_mask=False)
        cv2.imwrite(dst, resized)
    print(f"  [CACHE] Clean references cached: {len(clean_files)}")

    # 2. Masks (strictly Nearest Neighbor)
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in mask_files:
        src = os.path.join(masks_dir, fname)
        dst = os.path.join(out_masks, os.path.splitext(fname)[0] + '.png')
        mask = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if mask.ndim > 2 and mask.shape[2] > 1:
            mask = mask[:, :, 0]
        resized = resize_image(mask, target_size, aspect_ratio_mode, is_mask=True)
        cv2.imwrite(dst, resized)
    print(f"  [CACHE] Masks cached (strictly nearest-neighbor): {len(mask_files)}")

    # 3. Degraded variations
    scene_folders = sorted([d for d in os.listdir(degraded_dir) if os.path.isdir(os.path.join(degraded_dir, d))])
    total_deg = 0
    for scene in scene_folders:
        src_scene_dir = os.path.join(degraded_dir, scene)
        dst_scene_dir = os.path.join(out_degraded, scene)
        os.makedirs(dst_scene_dir, exist_ok=True)

        variants = sorted([f for f in os.listdir(src_scene_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for v in variants:
            src = os.path.join(src_scene_dir, v)
            dst = os.path.join(dst_scene_dir, os.path.splitext(v)[0] + '.png')
            img = cv2.imread(src)
            resized = resize_image(img, target_size, aspect_ratio_mode, is_mask=False)
            cv2.imwrite(dst, resized)
            total_deg += 1

    print(f"  [CACHE] Degraded variations cached: {total_deg}")
    print("=" * 60)
    print("[CACHE] Offline cache generation complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Preprocessing Cache")
    parser.add_argument("--data_dir", type=str, default="data", help="Source data dir")
    parser.add_argument("--output_dir", type=str, default="data/cached_1024", help="Destination cache dir")
    parser.add_argument("--aspect_ratio_mode", type=str, default="stretch", choices=["stretch", "pad_letterbox"])
    parser.add_argument("--target_size", type=int, nargs=2, default=[1024, 1024], help="Width Height")
    args = parser.parse_args()

    build_cache(args.data_dir, args.output_dir, args.aspect_ratio_mode, tuple(args.target_size))
