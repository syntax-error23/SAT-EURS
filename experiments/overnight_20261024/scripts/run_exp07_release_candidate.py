"""
SAT-EURS Track 1 — Overnight Campaign Release Candidate (EXP-07)
Integrates:
1. Upgraded Fold 0 Model: checkpoints/v28/segmenter_v28_fold0_best.pt (verified +0.0180 Q_seg gain)
2. Folds 1-4 Models: checkpoints/v18/fold_{1..4}/segmenter_best.pt
3. Zero Full-20 contamination (pure 5-fold ensemble)
4. Validated Logit Calibration: [0.0, 0.0, 0.0, 0.10, 0.35] (tuned on 20 EURS scenes)
5. Horizontal-flip TTA on restored and degraded streams
6. Verified Pareto-optimal restoration: 1,200 locked restored images (PSNR 25.48 dB, SSIM 0.829, Sharpness 0.973)
7. Exact native resolution resizing and atomic packaging into deploy/overnight_candidate/
"""

import os
import os.path as osp
import sys
import time
import json
import hashlib
import shutil
import zipfile
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    load_checkpoint_with_provenance_check,
    parse_scene_wt_depth,
    V18CanonicalPreprocessor,
    resize_predictions_to_original,
)
from tools.v18.package_submission import V18SubmissionPackager

EXPECTED_V18_SHA256 = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
EXPECTED_V27_SHA256 = "722b106ae817e4efbe911a83497ffabe4aa42387f2aedee7e5ac85deb6668c7c"
EXPECTED_V28A_SHA256 = "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39"


def verify_file_sha256(path: str, expected_hash: str, name: str):
    if not osp.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual = h.hexdigest().lower()
    assert actual == expected_hash.lower(), f"{name} HASH MISMATCH! Expected {expected_hash}, got {actual}"
    print(f"  [OK] {name} verified intact: {actual}", flush=True)


def setup_fallback_release():
    print("\n[Step 0/5] Setting up verified fallback release...", flush=True)
    src_v28a = "deploy/v28/submission_v28a_fivefold_ablation.zip"
    dst_fallback = "deploy/overnight_candidate/submission_v28a_fallback.zip"
    os.makedirs(osp.dirname(dst_fallback), exist_ok=True)
    verify_file_sha256(src_v28a, EXPECTED_V28A_SHA256, "V28-A Baseline Archive")
    shutil.copy2(src_v28a, dst_fallback)
    verify_file_sha256(dst_fallback, EXPECTED_V28A_SHA256, "Overnight Fallback Release Archive")
    print(f"  Fallback release archive ready: {dst_fallback}\n", flush=True)


def run_overnight_candidate_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "experiments/overnight_20261024/candidate_outputs",
    v27_raw_restored_dir: str = "deploy/v27/track1_test_outputs/raw_unpadded/restored",
    checkpoint_dir: str = "checkpoints/v18",
    v28_fold0_ckpt: str = "checkpoints/v28/segmenter_v28_fold0_best.pt",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.35],
    out_zip: str = "deploy/overnight_candidate/submission_overnight_candidate_sota.zip",
    device_id: int = 0
):
    print("=" * 85)
    print("SAT-EURS TRACK 1 — OVERNIGHT CANDIDATE GENERATION PIPELINE")
    print("=" * 85)

    # 1. Integrity checks
    verify_file_sha256("deploy/v18/submission_v18_dual_encoder_sota.zip", EXPECTED_V18_SHA256, "V18 Production Baseline")
    verify_file_sha256("deploy/v27/submission_v27_test_hybrid_sota.zip", EXPECTED_V27_SHA256, "V27 Production Baseline")
    verify_file_sha256("deploy/v28/submission_v28a_fivefold_ablation.zip", EXPECTED_V28A_SHA256, "V28-A Baseline Archive")
    setup_fallback_release()

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

    out_rest_dir = osp.join(out_dir, "restored")
    out_mask_dir = osp.join(out_dir, "seg_masks")
    raw_rest_dir = osp.join(out_dir, "raw_unpadded", "restored")
    raw_mask_dir = osp.join(out_dir, "raw_unpadded", "seg_masks")
    os.makedirs(raw_rest_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)
    os.makedirs(out_rest_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 2. Setup restored images
    print(f"\n[Step 1/5] Reusing verified Pareto-optimal V27 restored images...", flush=True)
    assert osp.exists(v27_raw_restored_dir), f"Missing restored directory: {v27_raw_restored_dir}"
    v27_rest_files = [f for f in os.listdir(v27_raw_restored_dir) if f.lower().endswith(".jpg")]
    assert len(v27_rest_files) == 1200, f"Expected 1,200 restored images, found {len(v27_rest_files)}"
    for f in v27_rest_files:
        src = osp.join(v27_raw_restored_dir, f)
        dst = osp.join(raw_rest_dir, f)
        if not osp.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Reused all 1,200 verified raw restored images from {v27_raw_restored_dir}", flush=True)

    # 3. Load Segmenter Ensemble (Fold 0 upgraded to V28-SWA)
    print(f"\n[Step 2/5] Loading 5-Fold Segmenter Ensemble with V28 Fold 0 Upgrade...", flush=True)
    assert osp.exists(v28_fold0_ckpt), f"Missing V28 Fold 0 checkpoint: {v28_fold0_ckpt}"
    manifest = validate_manifest_and_folds()

    segmenter_models = []
    for fold in range(5):
        s_model = build_v18_segmenter(num_classes=5).to(device)
        if fold == 0:
            ckpt_obj = torch.load(v28_fold0_ckpt, map_location=device)
            s_model.load_state_dict(ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj)
            print(f"  Fold 0: Loaded UPGRADED V28-SWA checkpoint from {v28_fold0_ckpt}", flush=True)
        else:
            f_data = get_fold_data(manifest, fold)
            s_path = osp.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
            s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, f_data["train_scenes"], f_data["val_scenes"])
            s_model.load_state_dict(s_weights)
            print(f"  Fold {fold}: Loaded baseline V18 segmenter from {s_path}", flush=True)
        s_model.eval()
        segmenter_models.append(s_model)

    # 4. Discover test images
    test_images = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(osp.join(root, f))
    test_images.sort()
    assert len(test_images) == 1200, f"Expected 1,200 test images, found {len(test_images)}"

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # 5. Run inference on all 1,200 images
    print(f"\n[Step 3/5] Running Upgraded 5-Fold Inference (Calibration: {offset_vector}) on 1,200 test images...", flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(test_images):
        fname = osp.basename(img_path)
        stem = osp.splitext(fname)[0]

        rest_target = osp.join(raw_rest_dir, f"{stem}.jpg")
        mask_target = osp.join(raw_mask_dir, f"{stem}.png")

        assert osp.exists(rest_target), f"Missing restored image: {rest_target}"

        scene, wt, depth = parse_scene_wt_depth(fname)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Corrupt or unreadable image: {img_path}")

        prep = preprocessor.prepare_input(bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]

        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth], device=device)

        # Load restored image and prepare
        rest_u8 = cv2.imread(rest_target, cv2.IMREAD_COLOR)
        rest_prep = preprocessor.prepare_input(rest_u8)
        restored_fused = rest_prep["rgb_tensor"].to(device)

        # Flip tensors for horizontal-flip TTA
        raw_pad_flip = torch.flip(raw_pad, dims=[3])
        restored_flip = torch.flip(restored_fused, dims=[3])

        with torch.no_grad():
            prob_fold_accum = torch.zeros((1, 5, raw_pad.shape[2], raw_pad.shape[3]), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for s_m in segmenter_models:
                    out_orig = s_m(restored_fused, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = s_m(restored_flip, raw_pad_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    prob_fold_accum += 0.5 * (p_orig + p_flip)

            # Pure 5-Fold probability ensemble
            prob_fused = prob_fold_accum / len(segmenter_models)

            # Apply validated logit calibration vector
            log_prob = torch.log(prob_fused + 1e-10) + cal_t
            mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        mask_unpad = preprocessor.unpad_output(mask_pad, meta)
        cv2.imwrite(mask_target, mask_unpad)

        if (idx + 1) % 300 == 0 or idx == len(test_images) - 1:
            print(f"  Processed [{idx+1:04d}/1200] test images in {time.time() - t0:.1f}s...", flush=True)

    # 6. Native Aspect & Geometry Resizing
    print(f"\n[Step 4/5] Applying Official scale_to_max_side(1024) to match exact native resolutions...", flush=True)
    resize_predictions_to_original(
        pred_restored_dir=raw_rest_dir,
        pred_masks_dir=raw_mask_dir,
        ref_images_dir=test_dir,
        out_restored_dir=out_rest_dir,
        out_masks_dir=out_mask_dir,
        restored_ext="jpg",
        jpeg_quality=98,
        max_side=1024
    )

    # 7. Packaging & Strict 8-Point Validation
    print(f"\n[Step 5/5] Packaging and validating final release ZIP: {out_zip}...", flush=True)
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    pkg_res = packager.package_directory(
        restored_dir=out_rest_dir,
        masks_dir=out_mask_dir,
        expected_count=1200
    )

    # Final Verification
    assert osp.exists(out_zip), f"ZIP was not created: {out_zip}"
    zip_size_bytes = osp.getsize(out_zip)
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    assert zip_size_mb < 1024.0, f"ZIP size exceeds 1,024 MB: {zip_size_mb:.2f} MB"

    h = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    zip_sha256 = h.hexdigest().lower()

    print("\n" + "=" * 85)
    print("OVERNIGHT CANDIDATE RELEASE VERIFICATION SUMMARY:")
    print(f"  Archive Path:    {out_zip}")
    print(f"  Archive Size:    {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"  SHA-256 Hash:    {zip_sha256}")
    print("=" * 85)

    # Compute full test pixel census
    print("\nComputing full-census pixel distribution across all 1,200 generated masks...")
    pixel_counts = {c: 0 for c in range(5)}
    total_pixels = 0
    CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]

    for f in os.listdir(out_mask_dir):
        if f.lower().endswith(".png"):
            m = cv2.imread(osp.join(out_mask_dir, f), cv2.IMREAD_GRAYSCALE)
            counts = np.bincount(m.reshape(-1), minlength=5)
            for c in range(5):
                pixel_counts[c] += int(counts[c])
            total_pixels += m.size

    census_summary = {}
    print(f"\n{'Class ID':<10} | {'Class Name':<15} | {'Pixel Count':<16} | {'Share (%)'}")
    print("-" * 55)
    for c in range(5):
        share = 100.0 * pixel_counts[c] / total_pixels
        census_summary[CLASS_NAMES[c]] = {"count": pixel_counts[c], "share_pct": round(share, 3)}
        print(f"{c:<10} | {CLASS_NAMES[c]:<15} | {pixel_counts[c]:<16,d} | {share:6.3f}%")
    print("-" * 55)
    print(f"{'Total':<10} | {'All Classes':<15} | {total_pixels:<16,d} | 100.000%")

    manifest_data = {
        "archive_path": out_zip,
        "archive_size_bytes": zip_size_bytes,
        "archive_size_mb": round(zip_size_mb, 2),
        "sha256": zip_sha256,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "fold0_checkpoint": v28_fold0_ckpt,
            "folds1_4_checkpoint_dir": checkpoint_dir,
            "calibration_vector": offset_vector,
            "use_flip_tta": True,
            "full20_weight": 0.00,
            "restoration": "v27_ensemble_locked_a10_s20"
        },
        "test_census": census_summary,
        "fallback_archive": {
            "path": "deploy/overnight_candidate/submission_v28a_fallback.zip",
            "sha256": EXPECTED_V28A_SHA256,
            "official_composite": 0.634
        }
    }

    manifest_file = "deploy/overnight_candidate/MANIFEST.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nWrote candidate manifest to {manifest_file}")

    return manifest_data


if __name__ == "__main__":
    run_overnight_candidate_pipeline()
