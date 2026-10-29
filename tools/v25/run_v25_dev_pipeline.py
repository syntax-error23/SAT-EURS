"""
SAT-EURS Track 1 — V25 Protocol-Compliant Dev Inference & Submission Packaging Pipeline
Architecture:
- Strict Single-Image Inference (zero cross-frame leakage, zero consensus averaging)
- Restoration: 5-Fold Refiner Ensemble + Optimal Calibrated Unsharp Detail Filter (alpha=0.25, sigma=2.0)
- Segmentation: 5-Fold Dual-Encoder DINOv3 ViT-L + ConvNeXt Ensemble with Flip TTA
- Calibration: Cross-Fitted Macro-Fauna Logit Offset vector b = [0.0, 0.0, 0.0, 0.15, 0.35]
- Official Sizing: tools/resize_for_submission.py
- Packager: V18SubmissionPackager with 7-point validation
"""

import os
import os.path as osp
import sys
import time
import json
import hashlib
from typing import Dict, List, Tuple, Any

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
    print(f"[PREFLIGHT] Baseline verified intact (SHA-256: {sha})")


def run_v25_dev_inference(
    dev_dir: str = "data/dev/degraded",
    out_dir: str = "deploy/v25/track1_dev_outputs",
    checkpoint_dir: str = "checkpoints/v18",
    alpha_unsharp: float = 0.25,
    sigma_unsharp: float = 2.0,
    b4_offset: float = 0.35,
    device_id: int = 0
) -> Dict[str, Any]:
    print("=" * 80)
    print("SAT-EURS TRACK 1 — V25 PROTOCOL-COMPLIANT DEV INFERENCE")
    print(f"Protocol: Strict Single-Image Inference")
    print(f"Enhancements: alpha_unsharp={alpha_unsharp}, b4_offset={b4_offset}")
    print("=" * 80)

    verify_baseline()

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    out_rest_dir = osp.join(out_dir, "restored")
    out_mask_dir = osp.join(out_dir, "seg_masks")
    os.makedirs(out_rest_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # 1. Load full 20-scene calibrated RSUIGM table
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    if not osp.exists(table_path):
        raise FileNotFoundError(f"Missing 20-scene restoration table: {table_path}")
    base_table = torch.load(table_path, map_location=device)

    # 2. Load 5-Fold Refiners and Segmenters
    refiner_models = []
    segmenter_models = []
    manifest = validate_manifest_and_folds()

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

    print(f"Loaded 5-Fold Refiner Ensemble ({len(refiner_models)}) and 5-Fold Segmenter Ensemble ({len(segmenter_models)}).")

    # 3. Discover all Dev images
    dev_images = []
    for root, _, files in os.walk(dev_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                dev_images.append(osp.join(root, f))
    dev_images.sort()

    if len(dev_images) != 1200:
        raise RuntimeError(f"Expected exactly 1,200 Dev images in {dev_dir}, found {len(dev_images)}")

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    t0 = time.time()
    processed_count = 0

    raw_rest_dir = osp.join(out_dir, "raw_unpadded", "restored")
    raw_mask_dir = osp.join(out_dir, "raw_unpadded", "seg_masks")
    os.makedirs(raw_rest_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)

    cal_offsets = torch.tensor([0.0, 0.0, 0.0, 0.15, b4_offset], device=device).view(1, 5, 1, 1)

    print(f"\n[Step 1/4] Running Single-Image Inference across {len(dev_images)} Dev images...")
    for idx, img_path in enumerate(dev_images):
        fname = osp.basename(img_path)
        stem = osp.splitext(fname)[0]

        scene, wt, depth = parse_scene_wt_depth(fname)

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Corrupt or unreadable image: {img_path}")

        prep = preprocessor.prepare_input(bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]
        target_h, target_w = meta["target_h"], meta["target_w"]

        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth], device=device)

        with torch.no_grad():
            # A. 5-Fold Refiner Ensemble (run in float32 for exact arithmetic)
            restored_accum = torch.zeros_like(raw_pad)
            for r_m in refiner_models:
                out_r = r_m(raw_pad.float(), wt_t, dp_t)
                restored_accum += out_r["refined_image"]
            restored_fused = restored_accum / len(refiner_models)

            # B. 5-Fold Segmenter Ensemble with Flip TTA (in bfloat16 for fast throughput)
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

            # Apply cross-fitted calibrated logit offsets
            logits_adj = torch.log(prob_mean + 1e-10) + cal_offsets
            mask_pad = torch.argmax(logits_adj, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Unpad to aspect-preserved dimensions
        restored_unpad = preprocessor.unpad_output(restored_fused.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
        mask_unpad = preprocessor.unpad_output(mask_pad, meta)

        # Apply calibrated unsharp filter on restored image
        if alpha_unsharp > 0:
            blur = cv2.GaussianBlur(restored_unpad, (0, 0), sigma_unsharp)
            restored_final = np.clip(restored_unpad + alpha_unsharp * (restored_unpad - blur), 0.0, 1.0)
        else:
            restored_final = restored_unpad

        rest_u8 = np.clip(restored_final * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(
            osp.join(raw_rest_dir, f"{stem}.jpg"),
            cv2.cvtColor(rest_u8, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 98]
        )
        cv2.imwrite(osp.join(raw_mask_dir, f"{stem}.png"), mask_unpad)
        processed_count += 1

        if (idx + 1) % 200 == 0 or idx == len(dev_images) - 1:
            print(f"  Inferred [{idx+1:04d}/1200] Dev images in {time.time() - t0:.1f}s...", flush=True)

    print(f"\n[Step 2/4] Executing Official tools/resize_for_submission.py Wrapper...")
    resize_predictions_to_original(
        pred_restored_dir=raw_rest_dir,
        pred_masks_dir=raw_mask_dir,
        ref_images_dir=dev_dir,
        out_restored_dir=out_rest_dir,
        out_masks_dir=out_mask_dir,
        restored_ext="jpg",
        jpeg_quality=98,
        max_side=1024
    )

    print(f"\n[Step 3/4] Packaging Track 1 Candidate Submission ZIP...")
    out_zip = "deploy/v25/submission_v25_single_image_apex_sota.zip"
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    pkg_res = packager.package_directory(
        restored_dir=out_rest_dir,
        masks_dir=out_mask_dir,
        expected_count=1200
    )

    print(f"\n[Step 4/4] Final Verification of Protected Baseline...")
    verify_baseline()

    print("\n" + "=" * 80)
    print(f"V25 SUBMISSION CANDIDATE READY:")
    print(f"  Path:   {pkg_res['zip_path']}")
    print(f"  Size:   {pkg_res['size_mb']} MB")
    print(f"  SHA256: {pkg_res['sha256']}")
    print("=" * 80)
    return pkg_res


if __name__ == "__main__":
    run_v25_dev_inference()
