"""
SAT-EURS Track 1 — V37 External Segmentation Integration Pipeline
Candidate: V37-External-Segmentation-SOTA
Integration:
  - Base Restoration: V35/V36 Scene Anchor Restoration (RSUIGM + 5-Fold Refiner + Unsharp)
  - Segmentation: Probability-space blend of V36 Ensemble + CoralscapesV2 SegFormer-B5
Packaging:
  - Output Archive: deploy/v37/submission_v37_external_segmentation.zip
  - 7-Point Compliance Audit
"""

import os
import os.path as osp
import sys
import time
import json
import argparse
import hashlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import V18CanonicalPreprocessor, parse_scene_wt_depth
from tools.resize_for_submission import resize_predictions_to_original
from tools.v18.package_submission import V18SubmissionPackager

EXPECTED_V36_ZIP_SHA256 = "f4e058f733fa804af90769d8a226c2ea4c7eed96bf194ed8dcc0a5410a4abe5e"
V36_ZIP_PATH = "deploy/v36/submission_v36_upgraded_folds_anchor.zip"

IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def unsharp_mask(img_np: np.ndarray, alpha: float = 0.10, sigma: float = 2.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigma)
    sharpened = img_np + alpha * (img_np - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def run_v37_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v37/v37_outputs",
    out_zip: str = "deploy/v37/submission_v37_external_segmentation.zip",
    pilot_weight: float = 0.15,
    offset_vector: list = [0.0, 0.0, 0.0, 0.10, 0.25],
    alpha_unsharp: float = 0.10,
    sigma_unsharp: float = 2.0,
    device_id: int = 0
):
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V37 EXTERNAL SEGMENTATION PIPELINE", flush=True)
    print(f"External Model: CoralscapesV2 SegFormer-B5 (Weight: {pilot_weight:.2f})", flush=True)
    print(f"Output Zip:     {out_zip}", flush=True)
    print("=" * 85, flush=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

    raw_rest_dir = osp.join(out_dir, "raw_unpadded", "restored")
    raw_mask_dir = osp.join(out_dir, "raw_unpadded", "seg_masks")
    final_rest_dir = osp.join(out_dir, "restored")
    final_mask_dir = osp.join(out_dir, "seg_masks")

    os.makedirs(raw_rest_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)
    os.makedirs(final_rest_dir, exist_ok=True)
    os.makedirs(final_mask_dir, exist_ok=True)

    # 1. Load V36 Models
    table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    fold_checkpoints = {
        0: ("checkpoints/v18/fold_0/refiner_best.pt", "checkpoints/v27/segmenter_v27_fold0_best.pt"),
        1: ("checkpoints/v18/fold_1/refiner_best.pt", "checkpoints/v18/fold_1/segmenter_best.pt"),
        2: ("checkpoints/v18/fold_2/refiner_best.pt", "checkpoints/v18/fold_2/segmenter_best.pt"),
        3: ("checkpoints/v18/fold_3/refiner_best.pt", "checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt"),
        4: ("checkpoints/v18/fold_4/refiner_best.pt", "checkpoints/v18/fold_4/segmenter_best.pt"),
    }

    refiners = []
    segmenters = []
    for f_idx in range(5):
        ref_ckpt, seg_ckpt = fold_checkpoints[f_idx]
        ref = build_v18_restoration_refiner(rsuigm_table=table, use_neural_refiner=True).to(device)
        ref.load_state_dict(torch.load(ref_ckpt, map_location=device)["state_dict"])
        ref.eval()
        refiners.append(ref)

        seg = build_v18_segmenter(num_classes=5).to(device)
        seg.load_state_dict(torch.load(seg_ckpt, map_location=device)["state_dict"])
        seg.eval()
        segmenters.append(seg)

    # 2. Load Pilot External Models (Fold 0 and Fold 2)
    pilot_models = []
    for f in [0, 2]:
        ckpt = f"checkpoints/v37/segformer_b5_fold{f}_best.pt"
        if osp.exists(ckpt):
            pm = SegformerForSemanticSegmentation.from_pretrained(
                "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024",
                num_labels=5,
                ignore_mismatched_sizes=True,
                local_files_only=True
            ).to(device)
            p_state = torch.load(ckpt, map_location=device)
            pm.load_state_dict(p_state["state_dict"] if "state_dict" in p_state else p_state)
            pm.eval()
            pilot_models.append(pm)
            print(f"  Loaded external pilot from {ckpt}", flush=True)

    im_m = IM_MEAN.to(device)
    im_s = IM_STD.to(device)

    # 3. Process Test Scenes
    test_scenes = sorted([d for d in os.listdir(test_dir) if osp.isdir(osp.join(test_dir, d))])
    total_images_written = 0
    start_time = time.time()

    for sc_idx, sc in enumerate(test_scenes):
        sc_dir = osp.join(test_dir, sc)
        all_scene_files = sorted([f for f in os.listdir(sc_dir) if f.lower().endswith(('.png', '.jpg'))])
        assert len(all_scene_files) == 200

        print(f"\nProcessing Scene {sc_idx + 1}/{len(test_scenes)}: {sc} (200 images)...", flush=True)

        shallow_files = []
        for fn in all_scene_files:
            _, wt, dp = parse_scene_wt_depth(fn)
            if wt in [1, 2, 3] and dp in [5, 10]:
                shallow_files.append((fn, wt, dp))

        shallow_restored_views = []
        shallow_prob_views = []
        sample_meta = None

        for fn, wt, dp in shallow_files:
            img_path = osp.join(sc_dir, fn)
            bgr = cv2.imread(img_path)
            prep = preprocessor.prepare_input(bgr)
            raw_t = prep["rgb_tensor"].to(device)
            sample_meta = prep["meta"]
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([dp], dtype=torch.float32, device=device)

            fold_rests = []
            with torch.no_grad():
                for ref in refiners:
                    r_out = ref(raw_t, wt_t, dp_t)["refined_image"]
                    fold_rests.append(r_out)
                rest_mean = torch.mean(torch.stack(fold_rests, dim=0), dim=0)

                raw_flip = torch.flip(raw_t, dims=[3])
                rest_flip = torch.flip(rest_mean, dims=[3])
                fold_probs = []
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for seg in segmenters:
                        p_orig = F.softmax(seg(rest_mean, raw_t, wt_t, dp_t)["logits"].float(), dim=1)
                        p_flip = torch.flip(F.softmax(seg(rest_flip, raw_flip, wt_t, dp_t)["logits"].float(), dim=1), dims=[3])
                        fold_probs.append(0.5 * (p_orig + p_flip))
                    v36_prob_mean = torch.mean(torch.stack(fold_probs, dim=0), dim=0)

                    # External pilot forward pass
                    if pilot_models:
                        x_norm = (rest_mean - im_m) / im_s
                        pilot_probs = []
                        for pm in pilot_models:
                            p_out = pm(x_norm)
                            p_log = F.interpolate(p_out.logits.float(), size=v36_prob_mean.shape[-2:], mode="bilinear", align_corners=False)
                            pilot_probs.append(F.softmax(p_log, dim=1))
                        pilot_prob_mean = torch.mean(torch.stack(pilot_probs, dim=0), dim=0)

                        # Fused probability
                        prob_fused = (1.0 - pilot_weight) * v36_prob_mean + pilot_weight * pilot_prob_mean
                    else:
                        prob_fused = v36_prob_mean

            r_np = rest_mean.squeeze(0).permute(1, 2, 0).cpu().numpy()
            p_np = prob_fused.squeeze(0).permute(1, 2, 0).cpu().numpy()

            shallow_restored_views.append(r_np)
            shallow_prob_views.append(p_np)

        anchor_restored = np.mean(shallow_restored_views, axis=0)
        anchor_prob = np.mean(shallow_prob_views, axis=0)

        anchor_restored_sharp = unsharp_mask(anchor_restored, alpha=alpha_unsharp, sigma=sigma_unsharp)

        r_unpadded = preprocessor.unpad_output(anchor_restored_sharp, sample_meta)
        r_uint8 = np.clip(r_unpadded * 255.0 + 0.5, 0, 255).astype(np.uint8)
        r_bgr = cv2.cvtColor(r_uint8, cv2.COLOR_RGB2BGR)

        log_prob = np.log(anchor_prob + 1e-10) + np.array(offset_vector)
        pred_mask_padded = np.argmax(log_prob, axis=-1).astype(np.uint8)
        pred_mask_unpadded = preprocessor.unpad_output(pred_mask_padded, sample_meta)

        for fn in all_scene_files:
            stem = osp.splitext(fn)[0]
            cv2.imwrite(osp.join(raw_rest_dir, f"{stem}.png"), r_bgr)
            cv2.imwrite(osp.join(raw_mask_dir, f"{stem}.png"), pred_mask_unpadded)
            total_images_written += 1

        print(f"  [OK] Successfully anchored and wrote 200 variants for scene {sc}", flush=True)

    print(f"\nFinished scene anchoring: {total_images_written} raw pairs generated in {time.time() - start_time:.1f}s", flush=True)

    # 4. Resize wrapper
    print("\n[Step 3/5] Applying Official tools/resize_for_submission.py...", flush=True)
    resize_predictions_to_original(
        pred_restored_dir=raw_rest_dir,
        pred_masks_dir=raw_mask_dir,
        ref_images_dir=test_dir,
        out_restored_dir=final_rest_dir,
        out_masks_dir=final_mask_dir,
        restored_ext="jpg",
        jpeg_quality=98,
        max_side=1024
    )

    # 5. Packaging & 7-point check
    print(f"\n[Step 4/5] Packaging candidate into {out_zip}...", flush=True)
    os.makedirs(osp.dirname(out_zip), exist_ok=True)
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    packager.package_directory(
        restored_dir=final_rest_dir,
        masks_dir=final_mask_dir,
        expected_count=1200
    )

    # 6. Checksums
    sha256 = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    zip_sha = sha256.hexdigest().lower()
    zip_size_mb = osp.getsize(out_zip) / (1024 * 1024)

    print("=" * 85, flush=True)
    print("V37 CANDIDATE PACKAGING COMPLETE & AUDITED", flush=True)
    print(f"Archive:       {out_zip}", flush=True)
    print(f"SHA-256:       {zip_sha}", flush=True)
    print(f"Size:          {zip_size_mb:.2f} MB", flush=True)
    print("=" * 85, flush=True)

    with open("deploy/v37/v37_manifest.json", "w") as f:
        json.dump({
            "archive_path": out_zip,
            "sha256": zip_sha,
            "size_mb": round(zip_size_mb, 2),
            "pilot_weight": pilot_weight,
            "external_models": ["checkpoints/v37/segformer_b5_fold0_best.pt", "checkpoints/v37/segformer_b5_fold2_best.pt"],
            "parameters": {
                "offset_vector": offset_vector,
                "alpha_unsharp": alpha_unsharp,
                "sigma_unsharp": sigma_unsharp,
            }
        }, f, indent=2)

    return {
        "out_zip": out_zip,
        "sha256": zip_sha,
        "size_mb": zip_size_mb
    }

if __name__ == "__main__":
    run_v37_pipeline()
