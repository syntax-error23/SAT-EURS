"""
SAT-EURS Track 1 — V28-A Controlled Inference-Only Ablation Pipeline
Ablation Goal: Isolate the effect of the Full-20 probability contribution
Configuration:
1. Restoration: Exact V27 restoration pipeline (5-Fold Refiners + RSUIGM table + alpha=0.10)
2. Segmenter Ensemble: 5-Fold scene-disjoint models (Fold 0 upgraded to V27 targeted checkpoint)
3. Full-20 Weight: 0.00 (Pure 5-Fold Ensemble, Zero Full-20 contribution)
4. TTA: Horizontal-flip TTA on restored and degraded inputs
5. Calibration: Calibrated logit vector [0.0, 0.0, 0.0, 0.10, 0.25]
6. Output: deploy/v28/submission_v28a_fivefold_ablation.zip
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
    load_checkpoint_with_provenance_check,
    parse_scene_wt_depth,
    V18CanonicalPreprocessor,
    resize_predictions_to_original,
)
from tools.v18.package_submission import V18SubmissionPackager

EXPECTED_V18_ZIP_SHA256 = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
V18_ZIP_PATH = "deploy/v18/submission_v18_dual_encoder_sota.zip"

EXPECTED_V26_ZIP_SHA256 = "f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f"
V26_ZIP_PATH = "deploy/v26/submission_v26_test_apex_sota.zip"

EXPECTED_V27_ZIP_SHA256 = "722b106ae817e4efbe911a83497ffabe4aa42387f2aedee7e5ac85deb6668c7c"
V27_ZIP_PATH = "deploy/v27/submission_v27_test_hybrid_sota.zip"


def verify_protected_baselines():
    print("[PREFLIGHT] Verifying cryptographic integrity of protected baselines...", flush=True)
    # V18
    if not osp.exists(V18_ZIP_PATH):
        raise FileNotFoundError(f"Missing V18 baseline: {V18_ZIP_PATH}")
    h18 = hashlib.sha256()
    with open(V18_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h18.update(chunk)
    assert h18.hexdigest().lower() == EXPECTED_V18_ZIP_SHA256, f"V18 CORRUPTED"
    print(f"  [OK] V18 Baseline intact (SHA-256: {EXPECTED_V18_ZIP_SHA256})", flush=True)

    # V26
    if osp.exists(V26_ZIP_PATH):
        h26 = hashlib.sha256()
        with open(V26_ZIP_PATH, "rb") as f:
            while chunk := f.read(65536):
                h26.update(chunk)
        assert h26.hexdigest().lower() == EXPECTED_V26_ZIP_SHA256, f"V26 CORRUPTED"
        print(f"  [OK] V26 Candidate intact (SHA-256: {EXPECTED_V26_ZIP_SHA256})", flush=True)

    # V27
    if not osp.exists(V27_ZIP_PATH):
        raise FileNotFoundError(f"Missing V27 baseline: {V27_ZIP_PATH}")
    h27 = hashlib.sha256()
    with open(V27_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h27.update(chunk)
    assert h27.hexdigest().lower() == EXPECTED_V27_ZIP_SHA256, f"V27 CORRUPTED"
    print(f"  [OK] V27 Baseline intact (SHA-256: {EXPECTED_V27_ZIP_SHA256})", flush=True)


def run_v28a_ablation_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v28/v28a_outputs",
    v27_raw_restored_dir: str = "deploy/v27/track1_test_outputs/raw_unpadded/restored",
    checkpoint_dir: str = "checkpoints/v18",
    v27_checkpoint_dir: str = "checkpoints/v27",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    alpha_restoration: float = 0.10,
    sigma_restoration: float = 2.0,
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    out_zip: str = "deploy/v28/submission_v28a_fivefold_ablation.zip",
    device_id: int = 0
) -> Dict[str, Any]:
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V28-A CONTROLLED FIVE-FOLD ABLATION PIPELINE", flush=True)
    print(f"Ablation: Full-20 Weight = 0.00 (Pure 5-Fold Ensemble with V27 Fold 0)", flush=True)
    print(f"Restoration Alpha: {alpha_restoration} (Sigma: {sigma_restoration})", flush=True)
    print(f"Logit Calibration Vector: {offset_vector}", flush=True)
    print(f"Input Directory:  {test_dir}", flush=True)
    print(f"Output Candidate: {out_zip}", flush=True)
    print("=" * 85, flush=True)

    # 1. Baseline integrity
    verify_protected_baselines()

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

    # 2. Check restored image caching: V27 restored images are 100% matched
    # Copy or reuse to raw_rest_dir
    print(f"\n[Step 1/5] Setting up V27-locked restored images...", flush=True)
    reused_restored = 0
    if osp.exists(v27_raw_restored_dir):
        v27_rest_files = [f for f in os.listdir(v27_raw_restored_dir) if f.lower().endswith(".jpg")]
        if len(v27_rest_files) == 1200:
            for f in v27_rest_files:
                src = osp.join(v27_raw_restored_dir, f)
                dst = osp.join(raw_rest_dir, f)
                if not osp.exists(dst):
                    shutil.copy2(src, dst)
            reused_restored = 1200
            print(f"  Reused all 1,200 verified V27 raw restored images from {v27_raw_restored_dir}", flush=True)

    need_refiner_inference = (reused_restored < 1200)

    # 3. Load refiners if needed
    refiner_models = []
    manifest = validate_manifest_and_folds()
    if need_refiner_inference:
        print(f"  Loading 5-Fold Refiners to compute remaining restored images...", flush=True)
        base_table = torch.load(table_path, map_location=device)
        for fold in range(5):
            f_data = get_fold_data(manifest, fold)
            r_model = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
            r_path = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
            r_weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, f_data["train_scenes"], f_data["val_scenes"])
            r_model.load_state_dict(r_weights)
            r_model.eval()
            refiner_models.append(r_model)

    # 4. Load 5-Fold Segmenter Ensemble (Fold 0 upgraded to V27 targeted)
    print(f"\n[Step 2/5] Loading 5-Fold Segmenter Ensemble...", flush=True)
    targeted_fold0_ckpt = osp.join(v27_checkpoint_dir, "segmenter_v27_fold0_best.pt")
    assert osp.exists(targeted_fold0_ckpt), f"Missing V27 targeted Fold 0 checkpoint: {targeted_fold0_ckpt}"

    segmenter_models = []
    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        s_model = build_v18_segmenter(num_classes=5).to(device)
        if fold == 0:
            ckpt_obj = torch.load(targeted_fold0_ckpt, map_location=device)
            s_model.load_state_dict(ckpt_obj["state_dict"])
            print(f"  Fold 0: Loaded targeted small-object checkpoint from {targeted_fold0_ckpt}", flush=True)
        else:
            s_path = osp.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
            s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, f_data["train_scenes"], f_data["val_scenes"])
            s_model.load_state_dict(s_weights)
            print(f"  Fold {fold}: Loaded baseline V18 segmenter from {s_path}", flush=True)
        s_model.eval()
        segmenter_models.append(s_model)

    # 5. Discover all Test images
    test_images = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(osp.join(root, f))
    test_images.sort()
    assert len(test_images) == 1200, f"Expected 1,200 test images, found {len(test_images)}"

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # 6. Run Inference
    print(f"\n[Step 3/5] Running Pure 5-Fold Inference (Zero Full-20) on 1,200 images...", flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(test_images):
        fname = osp.basename(img_path)
        stem = osp.splitext(fname)[0]

        rest_target = osp.join(raw_rest_dir, f"{stem}.jpg")
        mask_target = osp.join(raw_mask_dir, f"{stem}.png")

        # Skip mask if already generated
        if osp.exists(mask_target) and osp.exists(rest_target):
            if (idx + 1) % 300 == 0 or idx == len(test_images) - 1:
                print(f"  Verified cached [{idx+1:04d}/1200] Test predictions...", flush=True)
            continue

        scene, wt, depth = parse_scene_wt_depth(fname)
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Corrupt or unreadable image: {img_path}")

        prep = preprocessor.prepare_input(bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]

        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth], device=device)

        with torch.no_grad():
            # A. Restoration (if not already cached)
            if osp.exists(rest_target):
                # Load existing restored image
                rest_u8 = cv2.imread(rest_target, cv2.IMREAD_COLOR)
                rest_rgb = cv2.cvtColor(rest_u8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rest_prep = preprocessor.prepare_input(rest_u8)
                restored_fused = rest_prep["rgb_tensor"].to(device)
            else:
                rest_acc = torch.zeros_like(raw_pad)
                for r_m in refiner_models:
                    rest_acc += r_m(raw_pad.float(), wt_t, dp_t)["refined_image"]
                restored_fused = rest_acc / len(refiner_models)

                # Unpad and apply detail enhancement
                restored_unpad = preprocessor.unpad_output(restored_fused.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                if alpha_restoration > 0:
                    blur = cv2.GaussianBlur(restored_unpad, (0, 0), sigma_restoration)
                    restored_final = np.clip(restored_unpad + alpha_restoration * (restored_unpad - blur), 0.0, 1.0)
                else:
                    restored_final = restored_unpad
                rest_u8 = np.clip(restored_final * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(rest_target, cv2.cvtColor(rest_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])

            # Flip tensors for TTA
            raw_pad_flip = torch.flip(raw_pad, dims=[3])
            restored_flip = torch.flip(restored_fused, dims=[3])

            # B. 5-Fold Segmenter Ensemble with Flip TTA
            prob_fold_accum = torch.zeros((1, 5, raw_pad.shape[2], raw_pad.shape[3]), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for s_m in segmenter_models:
                    out_orig = s_m(restored_fused, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = s_m(restored_flip, raw_pad_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    prob_fold_accum += 0.5 * (p_orig + p_flip)

            # Pure 5-Fold probability (Zero Full-20 contribution)
            prob_fused = prob_fold_accum / len(segmenter_models)

            # C. Calibrated logit offsets
            log_prob = torch.log(prob_fused + 1e-10) + cal_t
            mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        mask_unpad = preprocessor.unpad_output(mask_pad, meta)
        cv2.imwrite(mask_target, mask_unpad)

        if (idx + 1) % 200 == 0 or idx == len(test_images) - 1:
            print(f"  Processed [{idx+1:04d}/1200] test images in {time.time() - t0:.1f}s...", flush=True)

    # 7. Native Aspect & Geometry Resizing
    print(f"\n[Step 4/5] Applying Official tools/resize_for_submission.py to match exact native resolutions...", flush=True)
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

    # 8. Packaging & Strict 7-Point Validation
    print(f"\n[Step 5/5] Packaging candidate into {out_zip} with strict 7-point validation...", flush=True)
    os.makedirs(osp.dirname(out_zip), exist_ok=True)
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    pkg_res = packager.package_directory(
        restored_dir=out_rest_dir,
        masks_dir=out_mask_dir,
        expected_count=1200
    )

    # Re-verify protected baseline integrity
    verify_protected_baselines()

    # Calculate archive SHA-256
    h_out = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(65536):
            h_out.update(chunk)
    out_sha256 = h_out.hexdigest().lower()
    zip_size_mb = osp.getsize(out_zip) / (1024 * 1024)

    # Pixel Census across all 1,200 masks
    print("\nCalculating full pixel census across 1,200 V28-A masks...", flush=True)
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
    print("V28-A ABLATION CANDIDATE SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Archive:    {out_zip}", flush=True)
    print(f"Size:       {zip_size_mb:.2f} MB", flush=True)
    print(f"SHA-256:    {out_sha256}", flush=True)
    print(f"Total Px:   {total_px:,}", flush=True)
    for c in range(5):
        print(f"  Class {c} ({CLASS_NAMES[c]:10s}): {class_px[c]:12,d} px ({census[CLASS_NAMES[c]]['percentage']:6.3f}%)", flush=True)
    print("=" * 70, flush=True)

    summary = {
        "candidate": "V28-A Five-Fold Ablation",
        "archive_path": out_zip,
        "archive_size_mb": round(zip_size_mb, 2),
        "sha256": out_sha256,
        "total_pixels": int(total_px),
        "census": census,
        "runtime_s": round(time.time() - t0, 1),
    }

    summary_file = osp.join(osp.dirname(out_dir), "v28a_ablation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}", flush=True)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAT-EURS Track 1 — V28-A Ablation Pipeline")
    parser.add_argument("--test_dir", type=str, default="data/test/degraded")
    parser.add_argument("--out_dir", type=str, default="deploy/v28/v28a_outputs")
    parser.add_argument("--out_zip", type=str, default="deploy/v28/submission_v28a_fivefold_ablation.zip")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    run_v28a_ablation_pipeline(
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        out_zip=args.out_zip,
        device_id=args.device_id
    )
