"""
Track 1 Public Dev-Set Leaderboard Submission Generator for EURS-Net V4
Target: MaCVi @ WACV 2027 — [MaCVi] EURS Public Leaderboard (Track 1)

Implements the exact official MaCVi `resize_for_submission.py` contract on the
1,200 Dev-split images located in `data/dev/degraded`:
  - fish10 (200 images)
  - fish19 (200 images)
  - fish3 (200 images)
  - fish6 (200 images)
  - sharRayFish (200 images)
  - starfish1 (200 images)

1. Computes exact target resolution `(target_w, target_h) = scale_to_max_side(orig_w, orig_h, 1024)`
   for each Dev-split image from its native dimensions.
2. Runs batched GPU inference with EURS-Net V4 (`checkpoints/v4_full20/best_v4_checkpoint.pt`)
   with horizontal-flip Test-Time Augmentation (TTA) across all 1,200 Dev-split images.
3. Resizes each restored image with `PIL.Image.BILINEAR` and each segmentation mask
   with `PIL.Image.NEAREST` (single-channel 'L' PNG, values 0..4) to `(target_w, target_h)`.
4. Packages `restored/<stem>.png` and `seg_masks/<stem>.png` at the root of
   `deploy/eurs_track1_dev_submission.zip` (lossless PNG) and
   `deploy/eurs_track1_dev_compact.zip` (compact Q98 JPEG for restored/ + lossless PNG for seg_masks/).
"""

import os
import os.path as osp
import io
import sys
import time
import shutil
import zipfile
import argparse
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from models.eurs_net_v5 import EURSNetV5

MAX_SIDE = 1024  # Exact constant from official EURS_resize_for_submission.py


def scale_to_max_side(w: int, h: int, max_side: int = MAX_SIDE) -> Tuple[int, int]:
    """Identical to official eurs_evaluator.py / resize_for_submission.py _scale_to_max_side."""
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))


def _save_single_pair(args: Tuple[np.ndarray, np.ndarray, int, int, str, str]) -> Tuple[str, bytes]:
    rest_u8, mask_u8, target_w, target_h, out_rest_path, out_mask_path = args

    rest_img = Image.fromarray(rest_u8, mode="RGB")
    if rest_img.size != (target_w, target_h):
        rest_img = rest_img.resize((target_w, target_h), Image.BILINEAR)
    rest_img.save(out_rest_path, compress_level=6)

    jpg_buf = io.BytesIO()
    rest_img.save(jpg_buf, format="JPEG", quality=97, subsampling=0)

    mask_img = Image.fromarray(mask_u8, mode="L")
    if mask_img.size != (target_w, target_h):
        mask_img = mask_img.resize((target_w, target_h), Image.NEAREST)
    mask_img.save(out_mask_path, compress_level=6)
    return osp.splitext(osp.basename(out_rest_path))[0], jpg_buf.getvalue()


def generate_track1_dev_submission(
    checkpoint_path: str = "checkpoints/v5_gan_attn/best_v5_checkpoint.pt",
    dev_degraded_dir: str = "data/dev/degraded",
    output_dir: str = "deploy/track1_dev_submission",
    zip_png_path: str = "deploy/eurs_track1_dev_submission.zip",
    zip_compact_path: str = "deploy/eurs_track1_dev_compact.zip",
    batch_size: int = 16,
    use_tta: bool = True,
) -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert osp.exists(dev_degraded_dir), f"Dev degraded directory not found: {dev_degraded_dir}"
    scene_folders = sorted([
        d for d in os.listdir(dev_degraded_dir)
        if osp.isdir(osp.join(dev_degraded_dir, d))
    ])
    assert len(scene_folders) == 6, f"Expected 6 Dev scenes in {dev_degraded_dir}, found {len(scene_folders)}: {scene_folders}"

    scene_target_sizes: Dict[str, Tuple[int, int]] = {}
    all_items: List[Tuple[str, str, str]] = []  # (scene_id, stem, raw_img_path)

    for scene_id in scene_folders:
        scene_raw_dir = osp.join(dev_degraded_dir, scene_id)
        raw_files = sorted([
            f for f in os.listdir(scene_raw_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        assert len(raw_files) == 200, f"Expected 200 files in {scene_raw_dir}, found {len(raw_files)}"
        with Image.open(osp.join(scene_raw_dir, raw_files[0])) as deg_img:
            orig_w, orig_h = deg_img.width, deg_img.height
        target_w, target_h = scale_to_max_side(orig_w, orig_h, MAX_SIDE)
        scene_target_sizes[scene_id] = (target_w, target_h)
        print(f"  [Dev Scene] {scene_id:12s}: {len(raw_files)} images | native {orig_w}x{orig_h} -> target {target_w}x{target_h}")

        for rf in raw_files:
            stem = osp.splitext(rf)[0]
            all_items.append((scene_id, stem, osp.join(scene_raw_dir, rf)))

    print("=" * 72)
    print("       MACVI EURS TRACK 1 DEV-SPLIT (1,200 IMAGES) SUBMISSION BUILDER (V5)")
    print("=" * 72)
    print(f"  Checkpoint:       {checkpoint_path}")
    print(f"  Total Dev Images: {len(all_items)} across {len(scene_folders)} scenes")
    print(f"  Device / TTA:     {device} | Horizontal-Flip TTA={use_tta}")
    print(f"  PNG Zip Output:   {zip_png_path}")
    print(f"  Compact Zip Out:  {zip_compact_path}")
    print("=" * 72)

    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    out_restored_dir = osp.join(output_dir, "restored")
    out_masks_dir = osp.join(output_dir, "seg_masks")
    os.makedirs(out_restored_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    # Load EURSNetV5 model
    model = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    save_pool = ThreadPoolExecutor(max_workers=8)
    load_pool = ThreadPoolExecutor(max_workers=8)
    futures = []

    def _load_input_1024(item: Tuple[str, str, str]) -> Tuple[str, str, np.ndarray]:
        sid, st, path = item
        with Image.open(path) as im:
            im_rgb = im.convert("RGB")
            if im_rgb.size != (1024, 1024):
                im_rgb = im_rgb.resize((1024, 1024), Image.BILINEAR)
            arr = np.asarray(im_rgb, dtype=np.float32) / 255.0
        return sid, st, arr

    total = len(all_items)
    for start_idx in range(0, total, batch_size):
        batch_items = all_items[start_idx:start_idx + batch_size]
        loaded = list(load_pool.map(_load_input_1024, batch_items))

        batch_np = np.stack([x[2].transpose(2, 0, 1) for x in loaded], axis=0)
        x_tensor = torch.from_numpy(batch_np).to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            rest_pred, seg_logits = model(x_tensor)
            if use_tta:
                x_flip = torch.flip(x_tensor, dims=[3])
                rest_flip, seg_flip = model(x_flip)
                rest_pred = 0.5 * (rest_pred + torch.flip(rest_flip, dims=[3]))
                seg_logits = 0.5 * (seg_logits + torch.flip(seg_flip, dims=[3]))

        rest_np = (torch.clamp(rest_pred.float(), 0.0, 1.0).cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
        mask_np = seg_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for i, (sid, stem, _) in enumerate(loaded):
            tw, th = scene_target_sizes[sid]
            out_r = osp.join(out_restored_dir, f"{stem}.png")
            out_m = osp.join(out_masks_dir, f"{stem}.png")
            futures.append(save_pool.submit(_save_single_pair, (rest_np[i], mask_np[i], tw, th, out_r, out_m)))

        done_count = min(total, start_idx + batch_size)
        if done_count % 200 == 0 or done_count == total:
            print(f"  [Track 1 Dev Inference] Processed {done_count}/{total} images ({time.time() - t0:.1f}s)")

    jpg_dict: Dict[str, bytes] = {}
    for fut in futures:
        st, jbytes = fut.result()
        jpg_dict[st] = jbytes
    load_pool.shutdown(wait=True)
    save_pool.shutdown(wait=True)

    rest_files = sorted(os.listdir(out_restored_dir))
    mask_files = sorted(os.listdir(out_masks_dir))
    assert len(rest_files) == 1200, f"Expected 1200 restored PNGs, got {len(rest_files)}"
    assert len(mask_files) == 1200, f"Expected 1200 mask PNGs, got {len(mask_files)}"

    print(f"  [Track 1 Packaging] Writing {zip_png_path} (PNG) and {zip_compact_path} (Compact)...")
    with zipfile.ZipFile(zip_png_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("restored", "seg_masks"):
            folder_path = osp.join(output_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                zf.write(osp.join(folder_path, fname), f"{folder}/{fname}")

    with zipfile.ZipFile(zip_compact_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for st in sorted(jpg_dict.keys()):
            zf.writestr(f"restored/{st}.jpg", jpg_dict[st])
        for fname in mask_files:
            zf.write(osp.join(out_masks_dir, fname), f"seg_masks/{fname}")

    png_mb = osp.getsize(zip_png_path) / (1024 * 1024)
    jpg_mb = osp.getsize(zip_compact_path) / (1024 * 1024)
    print("=" * 72)
    print(f"[Track 1 DEV READY] Completed in {time.time() - t0:.1f}s!")
    print(f"  1. PNG Archive:     {zip_png_path} ({png_mb:.1f} MB)")
    print(f"  2. Compact Archive: {zip_compact_path} ({jpg_mb:.1f} MB)")
    print(f"  Verified all 1,200 Dev-split images (fish10, fish19, fish3, fish6, sharRayFish, starfish1).")
    print("=" * 72)


if __name__ == "__main__":
    generate_track1_dev_submission()
