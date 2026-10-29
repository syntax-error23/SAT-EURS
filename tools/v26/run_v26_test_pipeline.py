"""
SAT-EURS Track 1 — V26 Testing-Phase SOTA Inference & Packaging Pipeline
Official Testing Phase: October 20-27, 2026 (closes 23:59 AoE)
Test Scenes: test04, test09, test10, test11, test19, test20 (1,200 images total)

Key Enhancements over V25:
1. Corrected Restoration: Removed destructive double-unsharp filtering.
   Recalibrated to optimal alpha=0.10 (or 0.00), regaining ~0.4-1.0 dB PSNR.
2. Balanced Calibration: Replaces over-aggressive b4=0.35 with OOF-validated
   balanced offsets, preventing Class 2 (fish) false negatives while protecting rare classes.
3. Multi-View Consensus Fully Removed: Strict single-image inference protocol.
4. Baseline Protection: Verifies immutable V18 development baseline bit-for-bit.
5. Official Sizing & Packaging: tools/resize_for_submission.py + V18SubmissionPackager.
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
import hashlib
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


def verify_baseline():
    if not osp.exists(V18_ZIP_PATH):
        raise FileNotFoundError(f"Baseline ZIP missing: {V18_ZIP_PATH}")
    h = hashlib.sha256()
    with open(V18_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha = h.hexdigest().lower()
    assert sha == EXPECTED_V18_ZIP_SHA256, f"BASELINE CORRUPTED: expected {EXPECTED_V18_ZIP_SHA256}, got {sha}"
    print(f"[PREFLIGHT] Baseline verified intact (SHA-256: {sha})", flush=True)


def run_v26_test_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v26/track1_test_outputs",
    checkpoint_dir: str = "checkpoints/v18",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    alpha_restoration: float = 0.10,
    sigma_restoration: float = 2.0,
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    out_zip: str = "deploy/v26/submission_v26_test_apex_sota.zip",
    device_id: int = 0
) -> Dict[str, Any]:
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V26 TESTING-PHASE SOTA INFERENCE PIPELINE", flush=True)
    print(f"Phase: OFFICIAL TEST PHASE (Deadline: October 27, 2026 23:59 AoE)", flush=True)
    print(f"Protocol: Strict Single-Image Inference (Zero Consensus, Zero Leakage)", flush=True)
    print(f"Restoration Alpha: {alpha_restoration} (Sigma: {sigma_restoration})", flush=True)
    print(f"Logit Calibration Vector: {offset_vector}", flush=True)
    print(f"Input Directory:  {test_dir}", flush=True)
    print(f"Output Candidate: {out_zip}", flush=True)
    print("=" * 85, flush=True)

    # 1. Preflight baseline integrity
    verify_baseline()

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

    # 2. Load 20-scene calibrated RSUIGM table
    if not osp.exists(table_path):
        raise FileNotFoundError(f"Missing restoration table: {table_path}")
    base_table = torch.load(table_path, map_location=device)

    # 3. Load 5-Fold Refiners and Segmenters
    refiner_models = []
    segmenter_models = []
    manifest = validate_manifest_and_folds()

    print(f"\n[Step 1/5] Loading 5-Fold Refiner & Segmenter Ensembles...", flush=True)
    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        t_sc = f_data["train_scenes"]
        v_sc = f_data["val_scenes"]

        r_model = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_path = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        r_weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, t_sc, v_sc)
        r_model.load_state_dict(r_weights)
        r_model.eval()
        refiner_models.append(r_model)

        s_model = build_v18_segmenter(num_classes=5).to(device)
        s_path = osp.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
        s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, t_sc, v_sc)
        s_model.load_state_dict(s_weights)
        s_model.eval()
        segmenter_models.append(s_model)

    print(f"Ensembles Loaded: 5 Refiners and 5 Segmenters ({len(refiner_models)} pairs).", flush=True)

    # 4. Discover all Test images
    test_images = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(osp.join(root, f))
    test_images.sort()

    if len(test_images) != 1200:
        raise RuntimeError(f"Expected exactly 1,200 Test images in {test_dir}, found {len(test_images)}")

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    t0 = time.time()
    print(f"\n[Step 2/5] Running Single-Image Inference on {len(test_images)} Test images...", flush=True)

    for idx, img_path in enumerate(test_images):
        fname = osp.basename(img_path)
        stem = osp.splitext(fname)[0]

        rest_target = osp.join(raw_rest_dir, f"{stem}.jpg")
        mask_target = osp.join(raw_mask_dir, f"{stem}.png")

        # Skip if already generated (resumable)
        if osp.exists(rest_target) and osp.exists(mask_target):
            if (idx + 1) % 300 == 0 or idx == len(test_images) - 1:
                print(f"  Verified cached [{idx+1:04d}/1200] Test images...", flush=True)
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
            # A. 5-Fold Refiner Ensemble
            restored_accum = torch.zeros_like(raw_pad)
            for r_m in refiner_models:
                out_r = r_m(raw_pad.float(), wt_t, dp_t)
                restored_accum += out_r["refined_image"]
            restored_fused = restored_accum / len(refiner_models)

            # B. 5-Fold Segmenter Ensemble with Flip TTA
            raw_pad_flip = torch.flip(raw_pad, dims=[3])
            restored_flip = torch.flip(restored_fused, dims=[3])

            prob_accum = torch.zeros((1, 5, raw_pad.shape[2], raw_pad.shape[3]), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for s_m in segmenter_models:
                    out_orig = s_m(restored_fused, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = s_m(restored_flip, raw_pad_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    prob_accum += 0.5 * (p_orig + p_flip)

            prob_mean = prob_accum / len(segmenter_models)
            log_prob = torch.log(prob_mean + 1e-10) + cal_t
            mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Unpad
        restored_unpad = preprocessor.unpad_output(restored_fused.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
        mask_unpad = preprocessor.unpad_output(mask_pad, meta)

        # Apply calibrated restoration detail enhancement
        if alpha_restoration > 0:
            blur = cv2.GaussianBlur(restored_unpad, (0, 0), sigma_restoration)
            restored_final = np.clip(restored_unpad + alpha_restoration * (restored_unpad - blur), 0.0, 1.0)
        else:
            restored_final = restored_unpad

        rest_u8 = np.clip(restored_final * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(
            rest_target,
            cv2.cvtColor(rest_u8, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 98]
        )
        cv2.imwrite(mask_target, mask_unpad)

        if (idx + 1) % 200 == 0 or idx == len(test_images) - 1:
            print(f"  Processed [{idx+1:04d}/1200] Test images in {time.time() - t0:.1f}s...", flush=True)

    # 5. Native Aspect & Geometry Resizing
    print(f"\n[Step 3/5] Applying Official tools/resize_for_submission.py to match exact native resolutions...", flush=True)
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

    # 6. Packaging & 7-Point Validation
    print(f"\n[Step 4/5] Packaging candidate into {out_zip} with strict 7-point validation...", flush=True)
    os.makedirs(osp.dirname(out_zip), exist_ok=True)
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    pkg_res = packager.package_directory(
        restored_dir=out_rest_dir,
        masks_dir=out_mask_dir,
        expected_count=1200
    )

    # 7. Post-flight baseline check
    print(f"\n[Step 5/5] Re-verifying Baseline Integrity post-packaging...", flush=True)
    verify_baseline()

    # Generate metadata inventory
    inv_path = f"{out_zip}.inventory.json"
    with open(inv_path, "w") as f:
        json.dump({
            "candidate_zip": out_zip,
            "sha256": pkg_res["sha256"],
            "size_mb": pkg_res["size_mb"],
            "file_count": 2400,
            "alpha_restoration": alpha_restoration,
            "sigma_restoration": sigma_restoration,
            "offset_vector": offset_vector,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_census": pkg_res.get("summary", {})
        }, f, indent=2)

    print("\n" + "=" * 85, flush=True)
    print("V26 TEST PHASE SUBMISSION CANDIDATE PACKAGED SUCCESSFULLY:")
    print(f"  Archive:  {pkg_res['zip_path']}", flush=True)
    print(f"  Size:     {pkg_res['size_mb']:.2f} MB", flush=True)
    print(f"  SHA-256:  {pkg_res['sha256']}", flush=True)
    print(f"  Manifest: {inv_path}", flush=True)
    print("=" * 85, flush=True)

    return pkg_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAT-EURS V26 Test Phase Inference")
    parser.add_argument("--alpha", type=float, default=0.10, help="Restoration detail enhancement alpha")
    parser.add_argument("--b3", type=float, default=0.10, help="Logit offset for class 3")
    parser.add_argument("--b4", type=float, default=0.25, help="Logit offset for class 4")
    parser.add_argument("--out_zip", type=str, default="deploy/v26/submission_v26_test_apex_sota.zip")
    args = parser.parse_args()

    offsets = [0.0, 0.0, 0.0, args.b3, args.b4]
    run_v26_test_pipeline(alpha_restoration=args.alpha, offset_vector=offsets, out_zip=args.out_zip)
