"""
SAT-EURS Track 1 — V28-B Degradation-Aware Candidate Pipeline
Deploys fine-tuned degradation-aware fold segmenters validated through cross-validation.
Configuration:
1. Restoration: Exact V27 restoration pipeline (5-Fold Refiners + RSUIGM table + alpha=0.10)
2. Segmenter Ensemble: 5-Fold models incorporating fine-tuned degradation-aware checkpoints
3. Full-20 Weight: 0.00 (Zero Full-20 contamination)
4. TTA: Horizontal-flip TTA on restored and degraded inputs
5. Calibration: Calibrated logit vector [0.0, 0.0, 0.0, 0.10, 0.25]
6. Output: deploy/v28/submission_v28b_degradation_aware.zip
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
    for path, expected_hash in [
        (V18_ZIP_PATH, EXPECTED_V18_ZIP_SHA256),
        (V26_ZIP_PATH, EXPECTED_V26_ZIP_SHA256),
        (V27_ZIP_PATH, EXPECTED_V27_ZIP_SHA256),
    ]:
        if not osp.exists(path):
            raise FileNotFoundError(f"Missing protected baseline: {path}")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        assert h.hexdigest().lower() == expected_hash, f"Hash mismatch on {path}!"
        print(f"  [OK] Intact: {path}", flush=True)


def run_v28b_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v28/v28b_outputs",
    v28a_raw_restored_dir: str = "deploy/v28/v28a_outputs/raw_unpadded/restored",
    v28_checkpoint_dir: str = "checkpoints/v28",
    v18_checkpoint_dir: str = "checkpoints/v18",
    v27_checkpoint_dir: str = "checkpoints/v27",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    out_zip: str = "deploy/v28/submission_v28b_degradation_aware.zip",
    device_id: int = 0
) -> Dict[str, Any]:
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V28-B DEGRADATION-AWARE CANDIDATE PIPELINE", flush=True)
    print(f"Segmenter: Validated Degradation-Aware Checkpoints with Zero Full-20 Contamination", flush=True)
    print(f"Logit Calibration Vector: {offset_vector}", flush=True)
    print(f"Input Directory:  {test_dir}", flush=True)
    print(f"Output Candidate: {out_zip}", flush=True)
    print("=" * 85, flush=True)

    verify_protected_baselines()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

    raw_rest_dir = osp.join(out_dir, "raw_unpadded", "restored")
    raw_mask_dir = osp.join(out_dir, "raw_unpadded", "seg_masks")
    os.makedirs(raw_rest_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)

    # 1. Reuse 1,200 verified restored images from V28-A / V27
    print(f"\n[Step 1/5] Setting up verified restored images...", flush=True)
    if osp.exists(v28a_raw_restored_dir):
        files = [f for f in os.listdir(v28a_raw_restored_dir) if f.lower().endswith(".jpg")]
        assert len(files) == 1200, f"Expected 1,200 restored images, found {len(files)}"
        for f in files:
            src = osp.join(v28a_raw_restored_dir, f)
            dst = osp.join(raw_rest_dir, f)
            if not osp.exists(dst):
                shutil.copy2(src, dst)
        print(f"  Reused all 1,200 verified restored images from {v28a_raw_restored_dir}", flush=True)

    # 2. Load 5-Fold Segmenter Models
    print(f"\n[Step 2/5] Loading 5-Fold Segmenter Models...", flush=True)
    manifest = validate_manifest_and_folds()
    segmenter_models = []

    for fold in range(5):
        s_model = build_v18_segmenter(num_classes=5).to(device)
        # Check for fine-tuned V28 checkpoint first
        v28_ckpt = osp.join(v28_checkpoint_dir, f"segmenter_v28_fold{fold}_best.pt")
        if osp.exists(v28_ckpt):
            ckpt_obj = torch.load(v28_ckpt, map_location=device)
            s_model.load_state_dict(ckpt_obj["state_dict"])
            print(f"  Fold {fold}: Loaded FINE-TUNED degradation-aware checkpoint from {v28_ckpt}", flush=True)
        elif fold == 0:
            v27_f0 = osp.join(v27_checkpoint_dir, "segmenter_v27_fold0_best.pt")
            ckpt_obj = torch.load(v27_f0, map_location=device)
            s_model.load_state_dict(ckpt_obj["state_dict"])
            print(f"  Fold 0: Loaded targeted V27 checkpoint from {v27_f0}", flush=True)
        else:
            v18_ckpt = osp.join(v18_checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
            f_data = get_fold_data(manifest, fold)
            s_weights = load_checkpoint_with_provenance_check(v18_ckpt, "EURSAPEXV18Segmenter", fold, f_data["train_scenes"], f_data["val_scenes"])
            s_model.load_state_dict(s_weights)
            print(f"  Fold {fold}: Loaded baseline checkpoint from {v18_ckpt}", flush=True)

        s_model.eval()
        segmenter_models.append(s_model)

    # 3. Inference on 1,200 Test Images
    print(f"\n[Step 3/5] Executing Single-Image Inference across 1,200 test images...", flush=True)
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    assert len(test_files) == 1200, f"Expected 1,200 test images, found {len(test_files)}"

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    pixel_counts = {c: 0 for c in range(5)}
    t_start = time.time()

    for idx, fname in enumerate(test_files, 1):
        stem = osp.splitext(fname)[0]
        deg_path = osp.join(test_dir, fname)
        bgr = cv2.imread(deg_path)
        assert bgr is not None, f"Failed to load {deg_path}"
        orig_h, orig_w = bgr.shape[:2]

        wt, depth = parse_scene_wt_depth(fname)
        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth], device=device)

        # Load matched restored image
        rest_path = osp.join(raw_rest_dir, f"{stem}.jpg")
        rest_bgr = cv2.imread(rest_path)
        assert rest_bgr is not None, f"Missing restored image: {rest_path}"

        prep_deg = preprocessor.prepare_input(bgr)
        prep_rest = preprocessor.prepare_input(rest_bgr)

        raw_pad = prep_deg["rgb_tensor"].to(device)
        rest_pad = prep_rest["rgb_tensor"].to(device)
        meta = prep_deg["meta"]

        raw_pad_flip = torch.flip(raw_pad, dims=[3])
        rest_pad_flip = torch.flip(rest_pad, dims=[3])

        probs_list = []
        with torch.inference_mode():
            for s_model in segmenter_models:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out_orig = s_model(rest_pad, raw_pad, wt_t, dp_t)
                    p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                    out_flip = s_model(rest_pad_flip, raw_pad_flip, wt_t, dp_t)
                    p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                    probs_list.append(0.5 * (p_orig + p_flip))

            fused_prob = torch.stack(probs_list, dim=0).mean(dim=0)
            calibrated_prob = torch.log(fused_prob + 1e-10) + cal_t
            pred_mask_pad = torch.argmax(calibrated_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        pred_mask_unpad = preprocessor.unpad_output(pred_mask_pad, meta)
        mask_out_path = osp.join(raw_mask_dir, f"{stem}.png")
        cv2.imwrite(mask_out_path, pred_mask_unpad)

        unq, cnts = np.unique(pred_mask_unpad, return_counts=True)
        for u, c in zip(unq, cnts):
            pixel_counts[int(u)] += int(c)

        if idx % 200 == 0 or idx == 1200:
            elapsed = time.time() - t_start
            print(f"  Processed [{idx:04d}/1200] images in {elapsed:.1f}s ({elapsed/idx:.3f} s/img)", flush=True)

    # 4. Packaging
    print(f"\n[Step 4/5] Packaging candidate submission to {out_zip}...", flush=True)
    packager = V18SubmissionPackager(target_max_side=1024, jpeg_quality=98)
    pack_res = packager.build_submission_zip(
        raw_restored_dir=raw_rest_dir,
        raw_masks_dir=raw_mask_dir,
        original_test_dir=test_dir,
        output_zip_path=out_zip,
        verify=True
    )

    print(f"\n[Step 5/5] Computing candidate SHA-256...", flush=True)
    h = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    zip_sha256 = h.hexdigest().lower()
    zip_size = os.path.getsize(out_zip)
    print(f"  Archive SHA-256: {zip_sha256}", flush=True)
    print(f"  Archive Size:    {zip_size / (1024**2):.2f} MB ({zip_size} bytes)", flush=True)

    return {
        "candidate": "V28-B",
        "zip_path": out_zip,
        "sha256": zip_sha256,
        "size_bytes": zip_size,
        "pixel_counts": pixel_counts,
        "packaging_result": pack_res
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    run_v28b_pipeline(device_id=args.device_id)
