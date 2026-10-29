"""
Generate Multi-Scale SOTA Track 1 Submission Package (Target: S_composite >= 0.70)
MaCVi @ WACV 2027 Challenge (Track 1)

Features:
1. Restoration: 20-Scene Clean-Fitted RSUIGM Inversion Table + Unsharp Filter (JPEG Q98)
2. Segmentation: 5-Fold DINOv2-Large Ensemble with Multi-Scale TTA ([448, 518, 588]) + HFlip
3. Full Per-Image Independence: zero cross-image grouping or consensus
"""

import os
import os.path as osp
import sys
import time
import json
import glob
import zipfile
import hashlib
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from tools.pilot_restoration_evaluation import fit_leak_free_rsuigm_table

MAX_SIDE = 1024

def scale_to_max_side(w: int, h: int, max_side: int = MAX_SIDE) -> Tuple[int, int]:
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))

def generate_multiscale_sota_submission(
    dev_dir: str = "data/dev/degraded",
    out_zip: str = "deploy/eurs_track1_multiscale_sota_submission.zip",
    checkpoint_dir: str = "checkpoints/track1_screening",
    manifest_path: str = "splits/manifest_5fold.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("GENERATING MULTI-SCALE SOTA TRACK 1 SUBMISSION PACKAGE (TARGET >= 0.70)")
    print(f"  Target ZIP: {out_zip}")
    print(f"  Dev Source: {dev_dir}")
    print("=" * 76)

    t0 = time.time()

    # 1. Fit high-diversity RSUIGM table on all 20 training scenes
    print("\n[1/4] Fitting RSUIGM table on all 20 training scenes...")
    clean_scenes = [osp.splitext(osp.basename(f))[0] for f in glob.glob("data/cached_1024/clean/*.png")]
    primary_table = fit_leak_free_rsuigm_table(clean_scenes, device=device)
    print(f"  Fitted 20-scene inversion table! Conditions: {len(primary_table)}")

    # 2. Load all 5 trained fold models
    print("\n[2/4] Loading 5-Fold DINOv2-L Models...")
    models = []
    for fold_idx in range(5):
        ckpt_path = osp.join(checkpoint_dir, f"DINOV2_CP_F{fold_idx}", "best_model.pt")
        assert osp.exists(ckpt_path), f"Missing fold checkpoint: {ckpt_path}"
        m = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  Loaded Fold {fold_idx} from {ckpt_path}")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    scene_folders = sorted(os.listdir(dev_dir))
    out_rest_dir = "deploy/track1_multiscale_dev/restored"
    out_mask_dir = "deploy/track1_multiscale_dev/seg_masks"
    os.makedirs(out_rest_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    scales = [448, 518, 588]
    scale_weights = [0.20, 0.60, 0.20]

    total_images_processed = 0
    print("\n[3/4] Processing Dev images with Multi-Scale 5-Fold TTA...")

    for sc in scene_folders:
        sc_dir = osp.join(dev_dir, sc)
        if not osp.isdir(sc_dir):
            continue
        dev_files = sorted([f for f in os.listdir(sc_dir) if f.endswith(".png")])

        first_im = Image.open(osp.join(sc_dir, dev_files[0]))
        orig_w, orig_h = first_im.size
        target_w, target_h = scale_to_max_side(orig_w, orig_h, MAX_SIDE)

        sc_t0 = time.time()
        print(f"  Processing Dev Scene {sc} ({len(dev_files)} images, {orig_w}x{orig_h} -> {target_w}x{target_h})...", flush=True)

        for df in dev_files:
            stem = osp.splitext(df)[0]
            parts = stem.split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10

            bgr = cv2.imread(osp.join(sc_dir, df))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # 1. Restoration: 20-scene RSUIGM table + calibrated unsharp mask
            rgb_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            x_raw_1024 = torch.from_numpy(rgb_1024.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([depth], device=device)

            with torch.no_grad():
                res_1024 = apply_rsuigm_batch_gpu(x_raw_1024, wt_t, dp_t, primary_table)

            res_np = (res_1024.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
            sharp_np = np.clip(cv2.addWeighted(res_np, 1.30, blur, -0.30, 0), 0, 255).astype(np.uint8)

            if sharp_np.shape[:2] != (target_h, target_w):
                sharp_resized = cv2.resize(sharp_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                sharp_resized = sharp_np

            out_rest_path = osp.join(out_rest_dir, f"{stem}.jpg")
            cv2.imwrite(out_rest_path, cv2.cvtColor(sharp_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])

            # 2. Segmentation: 5-Fold Ensemble with Multi-Scale ([448, 518, 588]) + HFlip TTA
            x_norm_1024 = (res_1024 - mean_im) / std_im

            ensemble_probs = torch.zeros((1, 5, 518, 518), device=device)

            with torch.no_grad():
                for m in models:
                    model_scale_probs = torch.zeros((1, 5, 518, 518), device=device)
                    for sc_sz, sc_w in zip(scales, scale_weights):
                        x_sc = F.interpolate(x_norm_1024, size=(sc_sz, sc_sz), mode="bilinear", align_corners=False)
                        x_sc_flip = torch.flip(x_sc, dims=[3])

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            l_orig = m(x_sc)
                            l_flip = m(x_sc_flip)

                        p_orig = F.softmax(l_orig.float(), dim=1)
                        p_flip = torch.flip(F.softmax(l_flip.float(), dim=1), dims=[3])
                        p_fused = 0.5 * p_orig + 0.5 * p_flip

                        # Interpolate to common 518x518 representation
                        p_518 = F.interpolate(p_fused, size=(518, 518), mode="bilinear", align_corners=False)
                        model_scale_probs += sc_w * p_518

                    ensemble_probs += model_scale_probs

            ensemble_probs /= len(models)
            pred_mask_518 = ensemble_probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Resize mask to target_w, target_h strictly with Nearest Neighbor
            if pred_mask_518.shape != (target_h, target_w):
                mask_final = cv2.resize(pred_mask_518, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_final = pred_mask_518

            out_mask_path = osp.join(out_mask_dir, f"{stem}.png")
            cv2.imwrite(out_mask_path, mask_final)

            total_images_processed += 1

        print(f"    Finished {sc} in {time.time() - sc_t0:.1f}s")

    print(f"\n[4/4] Processed all {total_images_processed} Dev images in {time.time() - t0:.1f}s.")

    # 3. Package ZIP
    print("\nPackaging final Multi-Scale SOTA submission ZIP...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out_rest_dir)):
            zf.write(osp.join(out_rest_dir, f), arcname=f"restored/{f}")
        for f in sorted(os.listdir(out_mask_dir)):
            zf.write(osp.join(out_mask_dir, f), arcname=f"seg_masks/{f}")

    zip_size_mb = os.path.getsize(out_zip) / (1024 ** 2)
    h_sha = hashlib.sha256()
    with open(out_zip, "rb") as fp:
        while chunk := fp.read(8192 * 1024):
            h_sha.update(chunk)
    zip_hash = h_sha.hexdigest()

    print("=" * 76)
    print("FINAL MULTI-SCALE SUBMISSION VERIFICATION:")
    print(f"  Archive: {out_zip}")
    print(f"  Size:    {zip_size_mb:.2f} MB (< 1024 MB constraint: {'PASS' if zip_size_mb < 1024 else 'FAIL'})")
    print(f"  SHA256:  {zip_hash}")
    print(f"  Restored Images: {len(os.listdir(out_rest_dir))} (expected 1,200)")
    print(f"  Seg Masks:       {len(os.listdir(out_mask_dir))} (expected 1,200)")
    print("=" * 76)

if __name__ == '__main__':
    generate_multiscale_sota_submission()
