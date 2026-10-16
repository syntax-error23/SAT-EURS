"""
Generate Phase 2 Verified Track 1 Submission Package
MaCVi @ WACV 2027 Challenge (Track 1)

Components:
1. Restoration: 20-Scene Clean-Fitted 19-Term RSUIGM Inversion Table + Verified Unsharp Filter (cfg1_unsharp_1.34)
   - Resolves the [:8] scene truncation penalty identified in forensic audit (+0.0199 Q_FR confirmed gain).
   - Saved as high-fidelity JPEG Q98 at native target dimensions.
2. Segmentation: 5-Fold DINOv2-Large Ensemble with Multi-Scale TTA ([448, 518, 588] + HFlip)
   - Bilinear probability fusion across scales and models.
   - Nearest-neighbor resizing to native target dimensions preserving discrete classes {0, 1, 2, 3, 4}.
3. Strict Per-Image Independence: zero cross-image communication, grouping, or consensus.
"""

import os
import os.path as osp
import sys
import time
import glob
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
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu

MAX_SIDE = 1024

def scale_to_max_side(w: int, h: int, max_side: int = MAX_SIDE) -> Tuple[int, int]:
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))

def fit_20scene_rsuigm_table(
    cached_dir: str = "data/cached_1024",
    device: torch.device = None
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Fits 19-term RSUIGM inversion table on ALL 20 clean training scenes without truncation.
    Equal pixel representation (20 * 128 * 128 = 327,680 pixels per condition).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_dir = osp.join(cached_dir, "clean")
    deg_dir = osp.join(cached_dir, "degraded")

    clean_files = sorted(glob.glob(osp.join(clean_dir, "*.png")))
    all_scenes = [osp.splitext(osp.basename(f))[0] for f in clean_files]
    assert len(all_scenes) == 20, f"Expected exactly 20 scenes, found {len(all_scenes)}"

    print(f"  Fitting RSUIGM table on all {len(all_scenes)} training scenes: {all_scenes}...")
    clean_list = []
    for s in all_scenes:
        c_bgr = cv2.imread(osp.join(clean_dir, f"{s}.png"), cv2.IMREAD_COLOR)
        c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_list.append(torch.from_numpy(c_rgb).permute(2, 0, 1))

    clean_gpu = torch.stack(clean_list, dim=0).to(device)  # [20, 3, 1024, 1024]
    clean_128 = F.avg_pool2d(clean_gpu, kernel_size=8, stride=8)  # [20, 3, 128, 128]
    Y_target = clean_128.permute(0, 2, 3, 1).reshape(-1, 3)  # [327680, 3]

    reg_eye = 1e-2 * torch.eye(19, device=device, dtype=torch.float32)
    table = {}

    for wt in range(1, 11):
        for depth in range(5, 105, 5):
            deg_list = []
            for s in all_scenes:
                p = osp.join(deg_dir, s, f"{s}_{wt}_{depth}.png")
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                deg_list.append(torch.from_numpy(rgb).permute(2, 0, 1))
            deg_gpu = torch.stack(deg_list, dim=0).to(device)
            deg_128 = F.avg_pool2d(deg_gpu, kernel_size=8, stride=8)

            feats = build_rsuigm_poly_features_gpu(deg_128)
            X_feats = feats.reshape(-1, 19)

            XtX = torch.matmul(X_feats.t(), X_feats) + reg_eye
            XtY = torch.matmul(X_feats.t(), Y_target)
            W = torch.linalg.solve(XtX, XtY)
            table[(wt, depth)] = W

    return table

def generate_phase2_verified_submission(
    dev_dir: str = "data/dev/degraded",
    out_dir: str = "deploy/track1_phase2_dev",
    out_zip: str = "deploy/eurs_track1_phase2_verified.zip",
    checkpoint_dir: str = "checkpoints/track1_screening",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("GENERATING PHASE 2 VERIFIED TRACK 1 SUBMISSION PACKAGE")
    print(f"  Target ZIP: {out_zip}")
    print(f"  Dev Source: {dev_dir}")
    print("=" * 76)

    t0 = time.time()

    # Step 1: Fit 20-scene RSUIGM table
    print("\n[Step 1/4] Fitting full 20-scene RSUIGM 19-term table...")
    table = fit_20scene_rsuigm_table(device=device)
    print(f"  Fitted {len(table)} (wt, depth) conditions successfully.")

    # Save table for provenance
    os.makedirs("checkpoints/track1_screening", exist_ok=True)
    torch.save(table, "checkpoints/track1_screening/tbl19_20sc_clean.pt")
    print("  Saved 20-scene table to checkpoints/track1_screening/tbl19_20sc_clean.pt")

    # Step 2: Load 5-Fold DINOv2-L Models
    print("\n[Step 2/4] Loading 5-Fold DINOv2-L checkpoints...")
    models = []
    for f_idx in range(5):
        ckpt_path = osp.join(checkpoint_dir, f"DINOV2_CP_F{f_idx}", "best_model.pt")
        assert osp.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
        m = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
        print(f"  Loaded Fold {f_idx} from {ckpt_path}")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    scales = [448, 518, 588]
    scale_weights = [0.20, 0.60, 0.20]

    out_rest_dir = osp.join(out_dir, "restored")
    out_mask_dir = osp.join(out_dir, "seg_masks")
    os.makedirs(out_rest_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Step 3: Process Dev Images
    print("\n[Step 3/4] Processing Dev images...")
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

            # 1. Restoration: 20-scene RSUIGM table + verified unsharp (cfg1: 1.34)
            rgb_1024 = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            x_raw_1024 = torch.from_numpy(rgb_1024.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([depth], device=device)

            with torch.no_grad():
                res_1024 = apply_rsuigm_batch_gpu(x_raw_1024, wt_t, dp_t, table)

            res_np = (res_1024.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
            # cfg1_unsharp_1.34
            sharp_np = np.clip(cv2.addWeighted(res_np, 1.34, blur, -0.34, 0), 0, 255).astype(np.uint8)

            if sharp_np.shape[:2] != (target_h, target_w):
                sharp_resized = cv2.resize(sharp_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                sharp_resized = sharp_np

            out_rest_path = osp.join(out_rest_dir, f"{stem}.jpg")
            cv2.imwrite(out_rest_path, cv2.cvtColor(sharp_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])

            # 2. Segmentation: 5-Fold Ensemble with Multi-Scale TTA ([448, 518, 588] + HFlip)
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

                        p_518 = F.interpolate(p_fused, size=(518, 518), mode="bilinear", align_corners=False)
                        model_scale_probs += sc_w * p_518

                    ensemble_probs += model_scale_probs

            ensemble_probs /= len(models)
            pred_mask_518 = ensemble_probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            if pred_mask_518.shape != (target_h, target_w):
                mask_final = cv2.resize(pred_mask_518, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_final = pred_mask_518

            out_mask_path = osp.join(out_mask_dir, f"{stem}.png")
            cv2.imwrite(out_mask_path, mask_final)

            total_imgs += 1

        print(f"    Finished {sc} in {time.time() - sc_t0:.1f}s")

    print(f"\n[Step 4/4] Packaging {total_imgs} images into {out_zip}...")
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
    print("PHASE 2 VERIFIED SUBMISSION SUMMARY:")
    print(f"  Archive:         {out_zip}")
    print(f"  Size:            {zip_size_mb:.2f} MB (< 1024 MB constraint: {'PASS' if zip_size_mb < 1024 else 'FAIL'})")
    print(f"  SHA256:          {zip_hash}")
    print(f"  Restored Images: {len(os.listdir(out_rest_dir))} (expected 1,200)")
    print(f"  Seg Masks:       {len(os.listdir(out_mask_dir))} (expected 1,200)")
    print(f"  Total Time:      {time.time() - t0:.1f}s")
    print("=" * 76)

    # Contract Assertions
    assert zip_size_mb < 1024, f"ZIP size {zip_size_mb:.2f} MB exceeds 1024 MB"
    assert len(os.listdir(out_rest_dir)) == 1200, f"Expected 1,200 restored, got {len(os.listdir(out_rest_dir))}"
    assert len(os.listdir(out_mask_dir)) == 1200, f"Expected 1,200 masks, got {len(os.listdir(out_mask_dir))}"

    # Sample class check
    sample_masks = sorted(os.listdir(out_mask_dir))[::12]
    all_classes = set()
    for sm in sample_masks:
        m = cv2.imread(osp.join(out_mask_dir, sm), cv2.IMREAD_UNCHANGED)
        all_classes.update(np.unique(m).tolist())
    assert all_classes.issubset({0, 1, 2, 3, 4}), f"Invalid classes: {all_classes}"
    print(f"  Sampled Classes: {sorted(list(all_classes))} -> PASS")

if __name__ == '__main__':
    generate_phase2_verified_submission()
