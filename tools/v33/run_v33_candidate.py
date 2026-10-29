"""
SAT-EURS Track 1 — V33 Candidate Generation Pipeline
Candidate: V33-Ensemble-F3Upgrade
Configuration:
1. Restoration: Exact V27/V28-A locked restorations (RSUIGM table + 5-Fold Refiners, 25.48 dB parity)
2. Segmenter Ensemble:
   - Fold 0: checkpoints/v27/segmenter_v27_fold0_best.pt
   - Fold 1: checkpoints/v18/fold_1/segmenter_best.pt
   - Fold 2: checkpoints/v18/fold_2/segmenter_best.pt
   - Fold 3: checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt (Validated +0.1636 Q_seg OOF gain)
   - Fold 4: checkpoints/v18/fold_4/segmenter_best.pt
3. TTA: Horizontal-flip TTA on restored and degraded inputs
4. Calibration: Calibrated logit vector [0.0, 0.0, 0.0, 0.10, 0.25]
5. Output Archive: deploy/v33/submission_v33_ensemble_f3upgrade.zip
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
import hashlib
import shutil
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    parse_scene_wt_depth,
    V18CanonicalPreprocessor,
    resize_predictions_to_original,
)
from tools.v18.package_submission import V18SubmissionPackager

EXPECTED_V28A_ZIP_SHA256 = "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39"
V28A_ZIP_PATH = "deploy/protected_reference/submission_v28a_fivefold_ablation.zip"


def verify_protected_v28a():
    print("[PREFLIGHT] Verifying cryptographic integrity of protected V28-A champion baseline...", flush=True)
    if not osp.exists(V28A_ZIP_PATH):
        raise FileNotFoundError(f"Missing protected V28-A baseline: {V28A_ZIP_PATH}")
    h = hashlib.sha256()
    with open(V28A_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    assert h.hexdigest().lower() == EXPECTED_V28A_ZIP_SHA256, f"PROTECTED V28-A CORRUPTED! Expected {EXPECTED_V28A_ZIP_SHA256}, got {h.hexdigest().lower()}"
    print(f"  [OK] Protected V28-A Baseline verified intact (SHA-256: {EXPECTED_V28A_ZIP_SHA256})", flush=True)


def run_v33_candidate_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v33/v33_outputs",
    v27_raw_restored_dir: str = "deploy/v27/track1_test_outputs/raw_unpadded/restored",
    checkpoint_dir: str = "checkpoints/v18",
    v27_checkpoint_dir: str = "checkpoints/v27",
    v28_f3_checkpoint: str = "checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    out_zip: str = "deploy/v33/submission_v33_ensemble_f3upgrade.zip",
    smoke_test_limit: int = 0,  # 0 = all 1,200 images
    device_id: int = 0
) -> Dict[str, Any]:
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V33 CANDIDATE GENERATION PIPELINE", flush=True)
    print("Ensemble: V28-A + Validated Overnight Pilot Fold 3 Upgrade", flush=True)
    print(f"Logit Calibration Vector: {offset_vector}", flush=True)
    print(f"Input Directory:  {test_dir}", flush=True)
    print(f"Output Candidate: {out_zip}", flush=True)
    print("=" * 85, flush=True)

    # 1. Baseline integrity check
    verify_protected_v28a()

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

    # 2. Check restored image caching: V27 restored images (25.48 dB parity)
    print(f"\n[Step 1/5] Setting up verified raw restored images...", flush=True)
    assert osp.exists(v27_raw_restored_dir), f"Missing raw restored directory: {v27_raw_restored_dir}"
    v27_rest_files = [f for f in os.listdir(v27_raw_restored_dir) if f.lower().endswith(".jpg")]
    assert len(v27_rest_files) == 1200, f"Expected 1,200 restored files, found {len(v27_rest_files)}"
    for f in v27_rest_files:
        src = osp.join(v27_raw_restored_dir, f)
        dst = osp.join(raw_rest_dir, f)
        if not osp.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Reused all 1,200 verified raw restored images from {v27_raw_restored_dir}", flush=True)

    # 3. Load 5-Fold Segmenter Ensemble
    print(f"\n[Step 2/5] Loading 5-Fold Segmenter Ensemble...", flush=True)
    ckpt_paths = {
        0: osp.join(v27_checkpoint_dir, "segmenter_v27_fold0_best.pt"),
        1: osp.join(checkpoint_dir, "fold_1", "segmenter_best.pt"),
        2: osp.join(checkpoint_dir, "fold_2", "segmenter_best.pt"),
        3: v28_f3_checkpoint,
        4: osp.join(checkpoint_dir, "fold_4", "segmenter_best.pt")
    }

    segmenter_models = []
    for fold in range(5):
        ckpt_p = ckpt_paths[fold]
        assert osp.exists(ckpt_p), f"Missing checkpoint for fold {fold}: {ckpt_p}"
        s_model = build_v18_segmenter(num_classes=5).to(device)
        ckpt_obj = torch.load(ckpt_p, map_location=device)
        state_d = ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj
        s_model.load_state_dict(state_d)
        s_model.eval()
        segmenter_models.append(s_model)
        label = "UPGRADED Overnight Pilot" if fold == 3 else ("V27 Targeted" if fold == 0 else "Baseline V18")
        print(f"  Fold {fold} ({label}): Loaded {ckpt_p}", flush=True)

    # 4. Discover all Test images
    test_images = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(osp.join(root, f))
    test_images.sort()
    assert len(test_images) == 1200, f"Expected 1,200 test images, found {len(test_images)}"

    if smoke_test_limit > 0:
        print(f"  [SMOKE TEST] Restricting to first {smoke_test_limit} images...", flush=True)
        test_images = test_images[:smoke_test_limit]

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # 5. Run Inference
    print(f"\n[Step 3/5] Running 5-Fold Ensemble Inference on {len(test_images)} test images...", flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(test_images):
        fname = osp.basename(img_path)
        stem = osp.splitext(fname)[0]

        rest_target = osp.join(raw_rest_dir, f"{stem}.jpg")
        mask_target = osp.join(raw_mask_dir, f"{stem}.png")

        if osp.exists(mask_target) and osp.exists(rest_target):
            if (idx + 1) % 200 == 0 or idx == len(test_images) - 1:
                print(f"  Cached [{idx+1:04d}/{len(test_images)}] predictions...", flush=True)
            continue

        scene, wt, depth = parse_scene_wt_depth(fname)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Corrupt or unreadable image: {img_path}")

        prep = preprocessor.prepare_input(bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]

        wt_t = torch.tensor([wt], dtype=torch.long, device=device)
        dp_t = torch.tensor([depth], dtype=torch.float32, device=device)

        with torch.no_grad():
            rest_u8 = cv2.imread(rest_target, cv2.IMREAD_COLOR)
            rest_prep = preprocessor.prepare_input(rest_u8)
            restored_fused = rest_prep["rgb_tensor"].to(device)

            raw_pad_flip = torch.flip(raw_pad, dims=[3])
            restored_flip = torch.flip(restored_fused, dims=[3])

            prob_fold_accum = torch.zeros((1, 5, raw_pad.shape[2], raw_pad.shape[3]), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for s_m in segmenter_models:
                    out_orig = s_m(restored_fused, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = s_m(restored_flip, raw_pad_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    prob_fold_accum += 0.5 * (p_orig + p_flip)

            prob_fused = prob_fold_accum / len(segmenter_models)
            log_prob = torch.log(prob_fused + 1e-10) + cal_t
            mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        mask_unpad = preprocessor.unpad_output(mask_pad, meta)
        cv2.imwrite(mask_target, mask_unpad)

        if (idx + 1) % 100 == 0 or idx == len(test_images) - 1:
            elapsed = time.time() - t0
            img_sec = (idx + 1) / max(elapsed, 0.001)
            print(f"  Processed [{idx+1:04d}/{len(test_images)}] in {elapsed:.1f}s ({img_sec:.2f} img/s)...", flush=True)

    if smoke_test_limit > 0:
        print(f"\n[SMOKE TEST COMPLETE] Verified {smoke_test_limit} images. Exiting early.", flush=True)
        return {"smoke_test": True, "count": smoke_test_limit}

    # 6. Native Aspect & Geometry Resizing
    print(f"\n[Step 4/5] Applying Official tools/resize_for_submission.py to match exact native resolutions at 1024 max-side...", flush=True)
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

    # 7. Packaging & Strict 7-Point Validation
    print(f"\n[Step 5/5] Packaging candidate into {out_zip} with strict 7-point validation...", flush=True)
    os.makedirs(osp.dirname(out_zip), exist_ok=True)
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    pkg_res = packager.package_directory(
        restored_dir=out_rest_dir,
        masks_dir=out_mask_dir,
        expected_count=1200
    )

    # Re-verify protected baseline integrity
    verify_protected_v28a()

    # Calculate archive SHA-256
    h_out = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(65536):
            h_out.update(chunk)
    out_sha256 = h_out.hexdigest().lower()
    zip_size_mb = osp.getsize(out_zip) / (1024 * 1024)

    # Pixel Census across all 1,200 masks
    print("\nCalculating full pixel census across 1,200 V33 masks...", flush=True)
    total_px = 0
    class_px = np.zeros(5, dtype=np.int64)
    for fname in os.listdir(out_mask_dir):
        if fname.lower().endswith(".png"):
            m = cv2.imread(osp.join(out_mask_dir, fname), cv2.IMREAD_UNCHANGED)
            counts = np.bincount(m.ravel(), minlength=5)
            class_px += counts[:5]
            total_px += m.size

    CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]
    census = {
        CLASS_NAMES[c]: {
            "pixels": int(class_px[c]),
            "percentage": round(float(class_px[c] / total_px) * 100, 3)
        } for c in range(5)
    }

    print("\n" + "=" * 70, flush=True)
    print("V33 CANDIDATE SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Archive:    {out_zip}", flush=True)
    print(f"Size:       {zip_size_mb:.2f} MB", flush=True)
    print(f"SHA-256:    {out_sha256}", flush=True)
    print(f"Total Px:   {total_px:,} (Target parity: 869,580,800)", flush=True)
    for c in range(5):
        print(f"  Class {c} ({CLASS_NAMES[c]:10s}): {class_px[c]:12,d} px ({census[CLASS_NAMES[c]]['percentage']:6.3f}%)", flush=True)
    print("=" * 70, flush=True)

    summary = {
        "candidate": "V33-Ensemble-F3Upgrade",
        "archive_path": out_zip,
        "archive_size_mb": round(zip_size_mb, 2),
        "sha256": out_sha256,
        "total_pixels": int(total_px),
        "census": census,
        "runtime_s": round(time.time() - t0, 1),
    }

    summary_file = osp.join(osp.dirname(out_dir), "v33_candidate_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAT-EURS Track 1 — V33 Candidate Pipeline")
    parser.add_argument("--test_dir", type=str, default="data/test/degraded")
    parser.add_argument("--out_dir", type=str, default="deploy/v33/v33_outputs")
    parser.add_argument("--out_zip", type=str, default="deploy/v33/submission_v33_ensemble_f3upgrade.zip")
    parser.add_argument("--smoke_test_limit", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    run_v33_candidate_pipeline(
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        out_zip=args.out_zip,
        smoke_test_limit=args.smoke_test_limit,
        device_id=args.device_id
    )
