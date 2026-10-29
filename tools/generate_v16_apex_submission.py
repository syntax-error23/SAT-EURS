"""
Official Track 1 Apex SOTA Submission Generator
Generates deploy/submission_v16_apex_sota.zip in ~7 minutes on RTX 5090.
Combines:
- 20-Scene Calibrated RSUIGM 19-term Inversion Table + 1.34 Unsharp Filter (JPEG Q98)
- 5-Fold SOTA DINOv2-Large Ensemble with Multi-Scale TTA ([448, 518, 588] + Horizontal Flip)
- Zero cross-image leakage, strict per-image evaluation.
"""

import os
import os.path as osp
import sys
import time
import zipfile
import hashlib
from typing import Tuple, List, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")

from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from deploy.generate_phase2_verified_submission import scale_to_max_side, MAX_SIDE

def generate_apex_submission(
    dev_dir: str = "data/dev/degraded",
    out_dir: str = "deploy/submission_v16_apex",
    out_zip: str = "deploy/submission_v16_apex_sota.zip",
    device: torch.device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    print("=" * 80)
    print("GENERATING OFFICIAL TRACK 1 APEX SOTA SUBMISSION PACKAGE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Target Archive: {out_zip}")
    print("=" * 80)

    # 1. Load 20-Scene Calibrated RSUIGM Table
    tbl_path = "checkpoints/track1_screening/tbl19_20sc_clean.pt"
    assert osp.exists(tbl_path), f"Missing table: {tbl_path}"
    table = torch.load(tbl_path, map_location=device)
    print(f"[Step 1/4] Loaded 20-scene calibrated RSUIGM table ({len(table)} conditions)")

    # 2. Load 5-Fold DINOv2-Large Ensemble
    print("[Step 2/4] Loading 5-Fold DINOv2-Large Ensemble...")
    models = []
    ckpt_paths = [
        "checkpoints/v16_dinov2/fold_0/best_model.pt",
        "checkpoints/v16_dinov2/fold_1/best_model.pt",
        "checkpoints/v16_dinov2/fold_2/best_model.pt",
        "checkpoints/v16_dinov2/fold_3/best_model.pt",
        "checkpoints/track1_screening/DINOV2_CP_F4/best_model.pt",
    ]

    for f_idx, ckpt_p in enumerate(ckpt_paths):
        assert osp.exists(ckpt_p), f"Checkpoint missing: {ckpt_p}"
        m = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model_state_dict"]
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)
        print(f"  Loaded Fold {f_idx} from {ckpt_p}")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    scales = [448, 518, 588]
    scale_weights = [0.20, 0.60, 0.20]

    out_rest_dir = osp.join(out_dir, "restored")
    out_mask_dir = osp.join(out_dir, "seg_masks")
    os.makedirs(out_rest_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 3. Process Dev Images with Multi-Scale TTA
    print("\n[Step 3/4] Processing Dev images with Multi-Scale TTA...")
    scene_folders = sorted(os.listdir(dev_dir))
    total_imgs = 0

    for sc in scene_folders:
        sc_dir = osp.join(dev_dir, sc)
        if not osp.isdir(sc_dir):
            continue
        dev_files = sorted([f for f in os.listdir(sc_dir) if f.endswith(".png")])
        if not dev_files:
            continue

        first_im = Image.open(osp.join(sc_dir, dev_files[0]))
        orig_w, orig_h = first_im.size
        target_w, target_h = scale_to_max_side(orig_w, orig_h, MAX_SIDE)

        sc_t0 = time.time()
        print(f"  Processing Scene {sc} ({len(dev_files)} images, {orig_w}x{orig_h} -> {target_w}x{target_h})...", flush=True)

        for df in dev_files:
            stem = osp.splitext(df)[0]
            parts = stem.split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10

            bgr = cv2.imread(osp.join(sc_dir, df))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # 1. Calibrated Physical Restoration
            rgb_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            x_raw = torch.from_numpy(rgb_1024.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([depth], device=device)

            with torch.no_grad():
                rsuigm_1024 = apply_rsuigm_batch_gpu(x_raw, wt_t, dp_t, table)

            res_np = (rsuigm_1024.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
            sharp_np = np.clip(cv2.addWeighted(res_np, 1.34, blur, -0.34, 0), 0, 255).astype(np.uint8)

            if sharp_np.shape[:2] != (target_h, target_w):
                sharp_resized = cv2.resize(sharp_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                sharp_resized = sharp_np

            cv2.imwrite(osp.join(out_rest_dir, f"{stem}.jpg"), cv2.cvtColor(sharp_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])

            # 2. Multi-Scale TTA 5-Fold Segmentation
            x_norm_1024 = (rsuigm_1024 - mean_im) / std_im
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

                        p_518 = F.interpolate(p_fused, size=(518, 518), mode="bilinear", align_corners=False)
                        model_scale_probs += sc_w * p_518

                    ensemble_probs += model_scale_probs

            ensemble_probs /= len(models)
            pred_mask_518 = ensemble_probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            if pred_mask_518.shape != (target_h, target_w):
                mask_final = cv2.resize(pred_mask_518, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_final = pred_mask_518

            cv2.imwrite(osp.join(out_mask_dir, f"{stem}.png"), mask_final)
            total_imgs += 1

        print(f"    Finished {sc} in {time.time() - sc_t0:.1f}s", flush=True)

    # 4. Packaging & Validation
    print(f"\n[Step 4/4] Packaging {total_imgs} images into {out_zip}...", flush=True)
    os.makedirs(osp.dirname(out_zip), exist_ok=True)
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

    n_rest = len(os.listdir(out_rest_dir))
    n_mask = len(os.listdir(out_mask_dir))

    print("=" * 80)
    print("SUBMISSION V16 APEX SOTA COMPLETE:")
    print(f"  Archive:         {out_zip}")
    print(f"  Size:            {zip_size_mb:.2f} MB (< 1024 MB constraint: {'PASS' if zip_size_mb < 1024 else 'FAIL'})")
    print(f"  SHA256:          {zip_hash}")
    print(f"  Restored Images: {n_rest} (expected 1,200: {'PASS' if n_rest == 1200 else 'FAIL'})")
    print(f"  Seg Masks:       {n_mask} (expected 1,200: {'PASS' if n_mask == 1200 else 'FAIL'})")
    print(f"  Total Duration:  {time.time() - t0:.1f}s ({(time.time() - t0)/60:.1f} min)")
    print("=" * 80)

    assert zip_size_mb < 1024, f"ZIP size {zip_size_mb:.2f} MB exceeds 1024 MB"
    assert n_rest == 1200, f"Expected 1,200 restored, got {n_rest}"
    assert n_mask == 1200, f"Expected 1,200 masks, got {n_mask}"

    return zip_hash, zip_size_mb

if __name__ == "__main__":
    generate_apex_submission()
