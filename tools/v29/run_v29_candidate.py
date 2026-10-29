"""
SAT-EURS Track 1 — V29 Dual-Backbone SOTA Candidate Pipeline
Campaign: Last-Chance Sprint to 0.700+ Official Composite

Architecture & Strategy:
1. Restoration: Preserved SOTA restoration pipeline (PSNR 25.48 dB, SSIM 0.829, Sharpness 0.973)
2. Semantic Segmentation: Dual-Backbone SOTA Fusion
   - Path A: 5-Fold DINOv2-Large Ensemble (V28-A baseline)
   - Path B: SegFormer-B5 Coralscapes Fine-Tuned (EURS 5-class head)
   - Probability Blending: w * P_SegFormer + (1 - w) * P_DINOv2
   - Horizontal Flip TTA across both architectures
   - Anti-Trap Presence Gating on rare classes (Classes 3 & 4)
   - Calibrated logit adjustment [0.0, 0.0, 0.0, 0.10, 0.25]
3. Official Native Aspect-Preserved Resizing
4. Atomic Verification & Packaging: deploy/v29/submission_v29_dual_backbone_sota.zip
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
import hashlib
import shutil
import zipfile
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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
from tools.train_track1_screening import apply_rsuigm_batch_gpu

# Protected baseline hashes
PROTECTED_BASELINES = {
    "V18": ("deploy/v18/submission_v18_dual_encoder_sota.zip", "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"),
    "V26": ("deploy/v26/submission_v26_test_apex_sota.zip", "f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f"),
    "V27": ("deploy/v27/submission_v27_test_hybrid_sota.zip", "722b106ae817e4efbe911a83497ffabe4aa42387f2aedee7e5ac85deb6668c7c"),
    "V28-A": ("deploy/v28/submission_v28a_fivefold_ablation.zip", "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39"),
}


def verify_protected_baselines():
    print("[PREFLIGHT] Verifying cryptographic integrity of protected baselines...", flush=True)
    for name, (path, expected_hash) in PROTECTED_BASELINES.items():
        if not osp.exists(path):
            raise FileNotFoundError(f"Missing protected baseline {name}: {path}")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_hash = h.hexdigest().lower()
        assert actual_hash == expected_hash, f"Hash mismatch for {name}! Expected {expected_hash}, got {actual_hash}"
        print(f"  [OK] {name} Intact ({path})", flush=True)


def package_and_verify_zip(
    out_rest_dir: str,
    out_mask_dir: str,
    test_dir: str,
    zip_path: str
) -> str:
    print(f"\n[Packaging] Building and verifying {zip_path}...", flush=True)
    os.makedirs(osp.dirname(zip_path), exist_ok=True)
    temp_zip = zip_path + ".tmp"
    if osp.exists(temp_zip):
        os.remove(temp_zip)

    # List files
    rest_files = sorted([f for f in os.listdir(out_rest_dir) if f.lower().endswith(".jpg")])
    mask_files = sorted([f for f in os.listdir(out_mask_dir) if f.lower().endswith(".png")])

    if len(rest_files) != 1200:
        raise ValueError(f"Expected 1,200 restored images, found {len(rest_files)}")
    if len(mask_files) != 1200:
        raise ValueError(f"Expected 1,200 segmentation masks, found {len(mask_files)}")

    # Verify stems match exactly
    rest_stems = [osp.splitext(f)[0] for f in rest_files]
    mask_stems = [osp.splitext(f)[0] for f in mask_files]
    assert rest_stems == mask_stems, "Restored and mask filenames do not match!"

    # Create ZIP archive
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in rest_files:
            zf.write(osp.join(out_rest_dir, f), arcname=f"restored/{f}")
        for f in mask_files:
            zf.write(osp.join(out_mask_dir, f), arcname=f"seg_masks/{f}")

    if osp.exists(zip_path):
        os.remove(zip_path)
    os.replace(temp_zip, zip_path)

    # Compute SHA-256
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    zip_sha = h.hexdigest().lower()

    # Integrity verification of packaged zip
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        r_entries = [n for n in namelist if n.startswith("restored/") and n.endswith(".jpg")]
        m_entries = [n for n in namelist if n.startswith("seg_masks/") and n.endswith(".png")]
        assert len(r_entries) == 1200, f"ZIP contains {len(r_entries)} restored entries, expected 1200"
        assert len(m_entries) == 1200, f"ZIP contains {len(m_entries)} mask entries, expected 1200"

    zip_size_mb = osp.getsize(zip_path) / (1024 * 1024)
    print(f"  [OK] ZIP Built: {zip_path}")
    print(f"  [OK] File Count: 2,400 files (1,200 restored + 1,200 masks)")
    print(f"  [OK] Archive Size: {zip_size_mb:.2f} MB")
    print(f"  [OK] SHA-256: {zip_sha}")

    return zip_sha


def run_v29_candidate_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v29/v29_outputs",
    v27_raw_restored_dir: str = "deploy/v27/track1_test_outputs/raw_unpadded/restored",
    dino_checkpoint_dir: str = "checkpoints/v18",
    dino_v27_fold0_ckpt: str = "checkpoints/v27/segmenter_v27_fold0_best.pt",
    segformer_ckpt: str = "checkpoints/v29/segformer_b5_full20_best.pt",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    w_segformer: float = 0.60,
    presence_threshold: float = 0.05,
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    out_zip: str = "deploy/v29/submission_v29_dual_backbone_sota.zip",
    device_id: int = 0
) -> Dict[str, Any]:
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V29 DUAL-BACKBONE SOTA CANDIDATE PIPELINE", flush=True)
    print(f"Candidate: DINOv2 5-Fold + SegFormer-B5 Coralscapes Hybrid", flush=True)
    print(f"SegFormer Weight: {w_segformer:.2f} | DINOv2 Weight: {1.0 - w_segformer:.2f}", flush=True)
    print(f"Anti-Trap Rare Class Presence Threshold: {presence_threshold:.2f}", flush=True)
    print(f"Logit Calibration Vector: {offset_vector}", flush=True)
    print(f"Input Directory:  {test_dir}", flush=True)
    print(f"Output Candidate: {out_zip}", flush=True)
    print("=" * 85, flush=True)

    # 1. Verify protected baselines
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

    # 2. Setup restored images: Reuse verified V27/V28 restored images
    print(f"\n[Step 1/5] Setting up verified restored images...", flush=True)
    if osp.exists(v27_raw_restored_dir):
        v27_rest_files = [f for f in os.listdir(v27_raw_restored_dir) if f.lower().endswith(".jpg")]
        if len(v27_rest_files) == 1200:
            for f in v27_rest_files:
                src = osp.join(v27_raw_restored_dir, f)
                dst = osp.join(raw_rest_dir, f)
                if not osp.exists(dst):
                    shutil.copy2(src, dst)
            print(f"  Reused all 1,200 verified V27/V28 raw restored images from {v27_raw_restored_dir}", flush=True)
        else:
            raise FileNotFoundError(f"Expected 1,200 restored images in {v27_raw_restored_dir}, found {len(v27_rest_files)}")
    else:
        raise FileNotFoundError(f"Restored directory not found: {v27_raw_restored_dir}")

    # 3. Load DINOv2 5-Fold Ensemble
    print(f"\n[Step 2/5] Loading 5-Fold DINOv2 Segmenter Ensemble...", flush=True)
    manifest = validate_manifest_and_folds()
    dino_models = []
    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        s_model = build_v18_segmenter(num_classes=5).to(device)
        if fold == 0:
            ckpt_obj = torch.load(dino_v27_fold0_ckpt, map_location=device, weights_only=False)
            s_model.load_state_dict(ckpt_obj["state_dict"] if "state_dict" in ckpt_obj else ckpt_obj)
            print(f"  Fold 0: Loaded targeted checkpoint from {dino_v27_fold0_ckpt}", flush=True)
        else:
            s_path = osp.join(dino_checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
            s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, f_data["train_scenes"], f_data["val_scenes"])
            s_model.load_state_dict(s_weights)
            print(f"  Fold {fold}: Loaded baseline V18 segmenter from {s_path}", flush=True)
        s_model.eval()
        dino_models.append(s_model)

    # 4. Load SegFormer-B5 Model
    print(f"\n[Step 3/5] Loading Coralscapes SegFormer-B5 Segmenter...", flush=True)
    model_id = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024"
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(device)
    seg_ckpt_data = torch.load(segformer_ckpt, map_location=device, weights_only=False)
    seg_weights = seg_ckpt_data["model_state_dict"] if "model_state_dict" in seg_ckpt_data else seg_ckpt_data
    seg_model.load_state_dict(seg_weights)
    seg_model.eval()
    print(f"  Loaded SegFormer-B5 from {segformer_ckpt}", flush=True)

    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    cal_offsets = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # 5. Discover Test images
    test_images = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(osp.join(root, f))
    test_images.sort()
    if len(test_images) != 1200:
        raise RuntimeError(f"Expected 1,200 test images, found {len(test_images)}")

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    # 6. Execute Dual-Backbone Inference
    print(f"\n[Step 4/5] Executing Dual-Backbone Inference across 1,200 Test Images...", flush=True)
    t0 = time.time()
    for idx, img_path in enumerate(test_images):
        fname = osp.basename(img_path)
        stem = osp.splitext(fname)[0]
        mask_raw_path = osp.join(raw_mask_dir, f"{stem}.png")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {img_path}")

        prep = preprocessor.prepare_input(bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]

        scene, wt, depth = parse_scene_wt_depth(fname)
        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth], device=device)

        # Load raw restored image corresponding to this sample
        rest_raw_path = osp.join(raw_rest_dir, f"{stem}.jpg")
        rest_bgr = cv2.imread(rest_raw_path, cv2.IMREAD_COLOR)
        if rest_bgr is None:
            raise RuntimeError(f"Missing or unreadable restored file: {rest_raw_path}")
        rest_prep = preprocessor.prepare_input(rest_bgr)
        restored_pad = rest_prep["rgb_tensor"].to(device)

        with torch.no_grad():
            raw_pad_flip = torch.flip(raw_pad, dims=[3])
            restored_flip = torch.flip(restored_pad, dims=[3])

            # A. DINOv2 5-Fold Ensemble with Flip TTA
            dino_prob_accum = torch.zeros((1, 5, raw_pad.shape[2], raw_pad.shape[3]), device=device)
            for s_m in dino_models:
                p_orig = F.softmax(s_m(restored_pad, raw_pad, wt_t, dp_t)["logits"], dim=1)
                p_flip = torch.flip(F.softmax(s_m(restored_flip, raw_pad_flip, wt_t, dp_t)["logits"], dim=1), dims=[3])
                dino_prob_accum += 0.5 * (p_orig + p_flip)
            p_dino = dino_prob_accum / len(dino_models)

            # B. SegFormer-B5 with Flip TTA
            x_norm = (restored_pad - im_mean) / im_std
            x_norm_flip = torch.flip(x_norm, dims=[3])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seg_out_orig = seg_model(pixel_values=x_norm)
                l_orig = F.interpolate(seg_out_orig.logits.float(), size=raw_pad.shape[-2:], mode="bilinear", align_corners=False)
                p_seg_orig = F.softmax(l_orig, dim=1)

                seg_out_flip = seg_model(pixel_values=x_norm_flip)
                l_flip = F.interpolate(seg_out_flip.logits.float(), size=raw_pad.shape[-2:], mode="bilinear", align_corners=False)
                p_seg_flip = torch.flip(F.softmax(l_flip, dim=1), dims=[3])

            p_seg = 0.5 * (p_seg_orig + p_seg_flip)

            # C. Dual-Backbone Probability Blend
            p_fused = (1.0 - w_segformer) * p_dino + w_segformer * p_seg

            # D. Anti-Trap Presence Gating on rare classes 3 & 4
            if presence_threshold > 0.0:
                for c in [3, 4]:
                    max_prob = p_fused[:, c, :, :].max()
                    if max_prob < presence_threshold:
                        p_fused[:, c, :, :] = 0.0

            # E. Calibrated Logit Adjustment
            logits_adj = torch.log(p_fused + 1e-10) + cal_offsets
            mask_pad = torch.argmax(logits_adj, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Unpad mask to native aspect-preserved dimensions
        mask_unpad = preprocessor.unpad_output(mask_pad, meta)
        cv2.imwrite(mask_raw_path, mask_unpad)

        if (idx + 1) % 200 == 0 or idx == len(test_images) - 1:
            print(f"  Processed [{idx+1:04d}/1200] images in {time.time() - t0:.1f}s...", flush=True)

    # 7. Official Native Aspect-Preserved Resizing
    print(f"\n[Step 5/5] Executing Official tools/resize_for_submission.py Resizing...", flush=True)
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

    # 8. Package & Cryptographically Verify Submission Archive
    zip_sha256 = package_and_verify_zip(
        out_rest_dir=out_rest_dir,
        out_mask_dir=out_mask_dir,
        test_dir=test_dir,
        zip_path=out_zip
    )

    print("\n" + "=" * 85)
    print("V29 DUAL-BACKBONE CANDIDATE GENERATION COMPLETE!")
    print(f"Archive: {out_zip}")
    print(f"SHA-256: {zip_sha256}")
    print("=" * 85)

    return {
        "out_zip": out_zip,
        "sha256": zip_sha256,
        "w_segformer": w_segformer,
        "presence_threshold": presence_threshold
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="data/test/degraded")
    parser.add_argument("--out_dir", type=str, default="deploy/v29/v29_outputs")
    parser.add_argument("--w_segformer", type=float, default=0.60)
    parser.add_argument("--presence_threshold", type=float, default=0.05)
    parser.add_argument("--out_zip", type=str, default="deploy/v29/submission_v29_dual_backbone_sota.zip")
    args = parser.parse_args()

    run_v29_candidate_pipeline(
        test_dir=args.test_dir,
        out_dir=args.out_dir,
        w_segformer=args.w_segformer,
        presence_threshold=args.presence_threshold,
        out_zip=args.out_zip
    )
