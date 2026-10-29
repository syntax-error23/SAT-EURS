"""
SAT-EURS Track 1 — V38 4-Model-Slot Per-Scene Batched Quad-Architecture SOTA Pipeline
Executes the promoted 22-checkpoint Hierarchical Bayesian Quad-Architecture Ensemble:
  - Recipe: HB5_QuadArch_HighPromotedSegF_InvEntC4 | Off_20_40 (0, 0, 0, 0.20, 0.40)
  - Global 20-Scene OOF Q_seg: 0.59570 (+0.07072 over V33, +0.05597 over V28-A)
  - Global 20-Scene OOF mIoU:  0.71626 | Macro F1: 0.83169
  - Memory-Safe 4-Slot Architecture:
      Instantiates strictly 1 model instance per architecture family (4 slots total:
      V18Segmenter, SegFormer-B5, DINOv2-Large, ConvNeXt-FPN => ~3.1 GB VRAM total)
      and processes 1 scene (200 images) at a time with B=6 (B_cat=12 with 2-pass Flip TTA),
      keeping CPU RAM < 9.5 GB and GPU VRAM < 6.5 GB while finishing all 1,200 images in ~4.5 minutes.
  - Strictly Single-Image Inference:
      Every image's output depends solely on its own input tensor X_i and filename metadata (wt, depth).
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import os.path as osp
import sys
import gc
import time
import json
import hashlib
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

from models.v18.segmenter import build_v18_segmenter
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from models.convnext_fpn import build_convnext_fpn
from tools.v18.run_v18_pipeline import (
    parse_scene_wt_depth,
    V18CanonicalPreprocessor,
    resize_predictions_to_original,
)
from tools.v18.package_submission import V18SubmissionPackager
from tools.v38.evaluate_bayesian_quad_oof import hierarchical_bayesian_blend

EXPECTED_V33_SHA256 = "de741a84ce4601f82286ef156458b07e9fe2d4c4f75559a4589010f90ae6510d"
V33_ZIP_PATH = "deploy/protected_reference/submission_v33_ensemble_f3upgrade.zip"

EXPECTED_V28A_SHA256 = "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39"
V28A_ZIP_PATH = "deploy/protected_reference/submission_v28a_fivefold_ablation.zip"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_protected_baselines():
    for name, path, expected in [
        ("V33 Champion", V33_ZIP_PATH, EXPECTED_V33_SHA256),
        ("V28-A Fallback", V28A_ZIP_PATH, EXPECTED_V28A_SHA256),
    ]:
        if not osp.exists(path):
            raise FileNotFoundError(f"Protected baseline missing: {path}")
        actual = compute_sha256(path)
        if actual != expected:
            raise RuntimeError(f"CRITICAL: Protected baseline {name} ({path}) hash mismatch! Expected {expected}, got {actual}")
        print(f"[SAFEGUARD] {name} verified intact: {actual}", flush=True)


def load_weights_inplace(model: nn.Module, path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "state_dict" in obj:
            state = obj["state_dict"]
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        else:
            state = obj
    else:
        state = obj
    model.load_state_dict(state)
    model.eval()
    del obj, state


def run_uniqueness_and_release_audit(zip_path: str) -> Dict[str, Any]:
    print("\n" + "=" * 95, flush=True)
    print(f"[RELEASE GATE] RUNNING UNIQUENESS & COMPLIANCE AUDIT ON: {zip_path}", flush=True)
    print("=" * 95, flush=True)

    rest_hashes = defaultdict(set)
    mask_hashes = defaultdict(set)
    class_pixel_totals = {c: 0 for c in range(5)}
    invalid_values = set()
    rest_count = 0
    mask_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            sc = osp.basename(name).split("_")[0]
            data = zf.read(name)
            md5 = hashlib.md5(data).hexdigest()
            if name.startswith("restored/") and name.endswith(".jpg"):
                rest_count += 1
                rest_hashes[sc].add(md5)
            elif name.startswith("seg_masks/") and name.endswith(".png"):
                mask_count += 1
                mask_hashes[sc].add(md5)
                arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if arr.ndim != 2 or arr.dtype != np.uint8:
                    raise RuntimeError(f"Invalid mask format in {name}: shape={arr.shape}, dtype={arr.dtype}")
                u_vals, counts = np.unique(arr, return_counts=True)
                for u, cnt in zip(u_vals, counts):
                    if int(u) not in {0, 1, 2, 3, 4}:
                        invalid_values.add(int(u))
                    else:
                        class_pixel_totals[int(u)] += int(cnt)

    assert rest_count == 1200, f"Expected 1200 restored JPEGs, found {rest_count}"
    assert mask_count == 1200, f"Expected 1200 segmentation PNGs, found {mask_count}"
    assert len(invalid_values) == 0, f"Found invalid mask label values: {invalid_values}"

    scene_uniqueness = {}
    for sc in sorted(rest_hashes.keys()):
        u_r = len(rest_hashes[sc])
        u_m = len(mask_hashes[sc])
        scene_uniqueness[sc] = {"unique_restored_of_200": u_r, "unique_masks_of_200": u_m}
        print(f"  Scene {sc:8s}: Restored unique = {u_r:3d}/200 | Masks unique = {u_m:3d}/200", flush=True)
        if u_r < 200:
            raise RuntimeError(f"RELEASE GATE FAILURE: Scene {sc} has fewer than 200 unique restored images ({u_r}/200)!")

    total_px = sum(class_pixel_totals.values())
    class_pcts = {c: round(100.0 * class_pixel_totals[c] / max(1, total_px), 3) for c in range(5)}
    print(f"  Global Test Class Distribution (%): {class_pcts}", flush=True)

    size_bytes = osp.getsize(zip_path)
    sha256 = compute_sha256(zip_path)
    print(f"  Archive Size:   {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)", flush=True)
    print(f"  Archive SHA256: {sha256}", flush=True)

    return {
        "zip_path": zip_path,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "sha256": sha256,
        "restored_count": rest_count,
        "mask_count": mask_count,
        "scene_uniqueness": scene_uniqueness,
        "class_pixel_totals": class_pixel_totals,
        "class_percentages": class_pcts,
    }


def run_v38_quad_pipeline(
    test_dir: str = "data/test/degraded",
    out_dir: str = "deploy/v38/track1_test_outputs",
    out_zip: str = "deploy/v38/submission_v38_bayesian_quad_sota.zip",
    offset_vector: List[float] = [0.0, 0.0, 0.0, 0.20, 0.40],
    batch_size: int = 6,
):
    print("=" * 100, flush=True)
    print("SAT-EURS TRACK 1 — V38 4-SLOT BATCHED HIERARCHICAL BAYESIAN QUAD-ARCHITECTURE PIPELINE")
    print("Recipe:         HB5_QuadArch_HighPromotedSegF_InvEntC4 | Off_20_40 (0, 0, 0, 0.20, 0.40)")
    print("OOF Validation: Global 20-Scene Q_seg = 0.59570 (+0.07072 vs V33) | mIoU = 0.71626 | F1 = 0.83169")
    print(f"Output Archive: {out_zip}")
    print("=" * 100, flush=True)

    verify_protected_baselines()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

    raw_rest_dir = osp.join(out_dir, "raw_unpadded", "restored")
    raw_mask_dir = osp.join(out_dir, "raw_unpadded", "seg_masks")
    out_rest_dir = osp.join(out_dir, "restored")
    out_mask_dir = osp.join(out_dir, "seg_masks")
    os.makedirs(raw_rest_dir, exist_ok=True)
    os.makedirs(raw_mask_dir, exist_ok=True)
    os.makedirs(out_rest_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Group test images by scene
    scene_to_files = defaultdict(list)
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                sc = f.split("_")[0]
                scene_to_files[sc].append(osp.join(root, f))
    for sc in scene_to_files:
        scene_to_files[sc].sort()

    total_imgs = sum(len(v) for v in scene_to_files.values())
    assert total_imgs == 1200, f"Expected 1200 test images, found {total_imgs}"

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    tbl_20sc = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device, weights_only=False)

    # Instantiate strictly 1 model slot per architecture family (4 slots total -> ~3.1 GB VRAM)
    print("\n[Stage 1/4] Instantiating 4 Architecture Slots on GPU...", flush=True)
    slot_v18 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device).eval()
    slot_segf = SegformerForSemanticSegmentation.from_pretrained(
        "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024",
        num_labels=5,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device).eval()
    slot_dino = build_dinov2_model(num_classes=5).to(device).eval()
    slot_conv = build_convnext_fpn(num_classes=5, freeze_stage0=True).to(device).eval()
    print(f"  4 Architecture Slots ready (VRAM allocated: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB)", flush=True)

    # Exact checkpoint specifications and weights matching HB5_QuadArch_HighPromotedSegF_InvEntC4
    # 1. V18/V27 Family (8 checkpoints): (ckpt_path, w_c3, w_c4, w_maj)
    v18_specs = [
        ("checkpoints/v27/segmenter_v27_fold0_best.pt",                   0.20, 0.14, 0.15),
        ("checkpoints/v20/matched_pilot/fold_0/treatment_v20_best.pt",    0.00, 0.06, 0.05),
        ("checkpoints/v18/fold_1/segmenter_best.pt",                      0.20, 0.20, 0.20),
        ("checkpoints/v18/fold_2/segmenter_best.pt",                      0.20, 0.20, 0.10),
        ("checkpoints/v20/matched_pilot/fold_2/control_v18_best.pt",      0.00, 0.00, 0.10),
        ("checkpoints/v18/fold_3/segmenter_best.pt",                      0.15, 0.20, 0.15),
        ("checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt", 0.05, 0.00, 0.05),
        ("checkpoints/v18/fold_4/segmenter_best.pt",                      0.20, 0.20, 0.20),
    ]
    # 2. Promoted SegFormer-B5 Family (4 checkpoints): (ckpt_path, w_segf, is_f0)
    segf_specs = [
        ("checkpoints/v29/segformer_b5_fold_0_best.pt", 0.25, True),
        ("checkpoints/v38/segformer_b5_fold_1_best.pt", 0.25, False),
        ("checkpoints/v38/segformer_b5_fold_2_best.pt", 0.25, False),
        ("checkpoints/v38/segformer_b5_fold_4_best.pt", 0.25, False),
    ]
    # 3. DINOv2-Large Family (5 checkpoints)
    dino_specs = [(f"checkpoints/v16_dinov2/fold_{f}/best_model.pt", 0.20) for f in range(5)]
    # 4. ConvNeXt-Base FPN Family (5 checkpoints)
    conv_specs = [(f"checkpoints/v16_convnext/fold_{f}/best_model.pt", 0.20) for f in range(5)]

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # Order scenes so uncomputed scenes finish first, then test09 and test04 get upgraded to full HB5 Quad-Arch
    ordered_scenes = ["test10", "test11", "test19", "test20", "test09", "test04"]

    t0_all = time.time()
    print(f"\n[Stage 2/4] Running 22-Checkpoint Quad-Arch + Flip TTA across 6 Scenes ({ ordered_scenes })...", flush=True)

    for sc_idx, sc in enumerate(ordered_scenes):
        t_sc = time.time()
        img_paths = scene_to_files[sc]
        N = len(img_paths)

        # Peek first image for padded dimensions
        bgr0 = cv2.imread(img_paths[0], cv2.IMREAD_COLOR)
        prep0 = preprocessor.prepare_input(bgr0)
        H_pad, W_pad = prep0["rgb_tensor"].shape[-2:]
        nh0, nw0 = prep0["meta"]["target_h"], prep0["meta"]["target_w"]

        # Pre-allocate scene input buffers on CPU (float16)
        raw_cpu = torch.empty((N, 3, H_pad, W_pad), dtype=torch.float16)
        rest_cpu = torch.empty((N, 3, H_pad, W_pad), dtype=torch.float16)
        xd_cpu = torch.empty((N, 3, 518, 518), dtype=torch.float16)
        xc_cpu = torch.empty((N, 3, 512, 512), dtype=torch.float16)
        wt_cpu = torch.empty((N,), dtype=torch.long)
        dp_cpu = torch.empty((N,), dtype=torch.float32)
        metas = []
        stems = []

        with torch.no_grad():
            for i, img_path in enumerate(img_paths):
                fname = osp.basename(img_path)
                stem = osp.splitext(fname)[0]
                stems.append(stem)
                _, wt, depth = parse_scene_wt_depth(fname)
                wt_cpu[i] = wt
                dp_cpu[i] = float(depth)

                bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                rest_u8 = cv2.imread(osp.join(raw_rest_dir, f"{stem}.jpg"), cv2.IMREAD_COLOR)
                prep = preprocessor.prepare_input(bgr)
                rest_prep = preprocessor.prepare_input(rest_u8)
                metas.append(prep["meta"])

                raw_pad = prep["rgb_tensor"].to(device)
                rest_pad = rest_prep["rgb_tensor"].to(device)
                raw_cpu[i].copy_(raw_pad.squeeze(0).half())
                rest_cpu[i].copy_(rest_pad.squeeze(0).half())

                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], dtype=torch.long, device=device)
                rest_tbl_20 = apply_rsuigm_batch_gpu(raw_pad, wt_t, dp_t, tbl_20sc)
                rest_valid = rest_tbl_20[:, :, :nh0, :nw0]

                r_518 = F.interpolate(rest_valid, size=(518, 518), mode="bilinear", align_corners=False)
                xd = (r_518 - mean_im) / std_im
                xd_cpu[i].copy_(xd.squeeze(0).half())

                r_512 = F.interpolate(rest_valid, size=(512, 512), mode="bilinear", align_corners=False)
                xc = (r_512 - mean_im) / std_im
                xc_cpu[i].copy_(xc.squeeze(0).half())

        # Allocate scene probability accumulators on CPU (float16)
        acc_v18_c3 = torch.zeros((N, 5, H_pad, W_pad), dtype=torch.float16)
        acc_v18_c4 = torch.zeros((N, 5, H_pad, W_pad), dtype=torch.float16)
        acc_v18_maj = torch.zeros((N, 5, H_pad, W_pad), dtype=torch.float16)
        acc_segf = torch.zeros((N, 5, H_pad, W_pad), dtype=torch.float16)
        acc_segf_f0 = torch.zeros((N, 5, H_pad, W_pad), dtype=torch.float16)
        acc_dino_518 = torch.zeros((N, 5, 518, 518), dtype=torch.float16)
        acc_conv_512 = torch.zeros((N, 5, 512, 512), dtype=torch.float16)

        with torch.no_grad():
            # 1. Run 8 V18/V27 Checkpoints via slot_v18
            for ckpt_path, w_c3, w_c4, w_maj in v18_specs:
                load_weights_inplace(slot_v18, ckpt_path)
                for s in range(0, N, batch_size):
                    e = min(N, s + batch_size)
                    b = e - s
                    raw_b = raw_cpu[s:e].to(device, dtype=torch.float32)
                    rest_b = rest_cpu[s:e].to(device, dtype=torch.float32)
                    wt_b = wt_cpu[s:e].to(device)
                    dp_b = dp_cpu[s:e].to(device)

                    raw_cat = torch.cat([raw_b, torch.flip(raw_b, dims=[3])], dim=0)
                    rest_cat = torch.cat([rest_b, torch.flip(rest_b, dims=[3])], dim=0)
                    wt_cat = torch.cat([wt_b, wt_b], dim=0)
                    dp_cat = torch.cat([dp_b, dp_b], dim=0)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits_cat = slot_v18(rest_cat, raw_cat, wt_cat, dp_cat)["logits"].float()
                    probs_cat = F.softmax(logits_cat, dim=1)
                    p_b = 0.5 * (probs_cat[:b] + torch.flip(probs_cat[b:], dims=[3]))
                    p_cpu = p_b.half().cpu()

                    if w_c3 > 0:
                        acc_v18_c3[s:e].add_(p_cpu, alpha=w_c3)
                    if w_c4 > 0:
                        acc_v18_c4[s:e].add_(p_cpu, alpha=w_c4)
                    if w_maj > 0:
                        acc_v18_maj[s:e].add_(p_cpu, alpha=w_maj)

            # 2. Run 4 Promoted SegFormer-B5 Checkpoints via slot_segf
            for ckpt_path, w_sf, is_f0 in segf_specs:
                load_weights_inplace(slot_segf, ckpt_path)
                for s in range(0, N, batch_size):
                    e = min(N, s + batch_size)
                    b = e - s
                    rest_b = rest_cpu[s:e].to(device, dtype=torch.float32)
                    xn_b = (rest_b - mean_im) / std_im
                    xn_cat = torch.cat([xn_b, torch.flip(xn_b, dims=[3])], dim=0)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits_cat = slot_segf(pixel_values=xn_cat).logits.float()
                    up_cat = F.interpolate(logits_cat, size=(H_pad, W_pad), mode="bilinear", align_corners=False)
                    probs_cat = F.softmax(up_cat, dim=1)
                    p_b = 0.5 * (probs_cat[:b] + torch.flip(probs_cat[b:], dims=[3]))
                    p_cpu = p_b.half().cpu()

                    acc_segf[s:e].add_(p_cpu, alpha=w_sf)
                    if is_f0:
                        acc_segf_f0[s:e].copy_(p_cpu)

            # 3. Run 5 DINOv2-Large Checkpoints via slot_dino
            for ckpt_path, w_d in dino_specs:
                load_weights_inplace(slot_dino, ckpt_path)
                for s in range(0, N, batch_size):
                    e = min(N, s + batch_size)
                    b = e - s
                    xd_b = xd_cpu[s:e].to(device, dtype=torch.float32)
                    xd_cat = torch.cat([xd_b, torch.flip(xd_b, dims=[3])], dim=0)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits_cat = slot_dino(xd_cat).float()
                    probs_cat = F.softmax(logits_cat, dim=1)
                    p_b = 0.5 * (probs_cat[:b] + torch.flip(probs_cat[b:], dims=[3]))
                    acc_dino_518[s:e].add_(p_b.half().cpu(), alpha=w_d)

            # 4. Run 5 ConvNeXt-FPN Checkpoints via slot_conv
            for ckpt_path, w_c in conv_specs:
                load_weights_inplace(slot_conv, ckpt_path)
                for s in range(0, N, batch_size):
                    e = min(N, s + batch_size)
                    b = e - s
                    xc_b = xc_cpu[s:e].to(device, dtype=torch.float32)
                    xc_cat = torch.cat([xc_b, torch.flip(xc_b, dims=[3])], dim=0)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits_cat = slot_conv(xc_cat).float()
                    probs_cat = F.softmax(logits_cat, dim=1)
                    p_b = 0.5 * (probs_cat[:b] + torch.flip(probs_cat[b:], dims=[3]))
                    acc_conv_512[s:e].add_(p_b.half().cpu(), alpha=w_c)

            # 5. Decode all 200 images of scene using HB5_QuadArch_HighPromotedSegF_InvEntC4 | Off_20_40
            for i in range(N):
                p_v18_c3 = acc_v18_c3[i:i+1].to(device, dtype=torch.float32)
                p_v18_c4 = acc_v18_c4[i:i+1].to(device, dtype=torch.float32)
                p_v18_maj = acc_v18_maj[i:i+1].to(device, dtype=torch.float32)
                p_segf = acc_segf[i:i+1].to(device, dtype=torch.float32)
                p_segf_f0 = acc_segf_f0[i:i+1].to(device, dtype=torch.float32)

                p_dino = torch.zeros_like(p_v18_c3)
                p_dino[:, :, :nh0, :nw0] = F.interpolate(
                    acc_dino_518[i:i+1].to(device, dtype=torch.float32),
                    size=(nh0, nw0), mode="bilinear", align_corners=False
                )

                p_conv = torch.zeros_like(p_v18_c3)
                p_conv[:, :, :nh0, :nw0] = F.interpolate(
                    acc_conv_512[i:i+1].to(device, dtype=torch.float32),
                    size=(nh0, nw0), mode="bilinear", align_corners=False
                )

                c4_list = [(p_v18_c4, 0.75), (p_dino, 0.20), (p_segf_f0, 0.05)]
                maj_list = [
                    (p_v18_maj, [0.58, 0.52, 0.50]),
                    (p_segf,    [0.17, 0.24, 0.28]),
                    (p_dino,    [0.15, 0.14, 0.14]),
                    (p_conv,    [0.10, 0.10, 0.08]),
                ]

                p_final = hierarchical_bayesian_blend(
                    p_c3_auth=p_v18_c3,
                    c4_models=c4_list,
                    maj_models=maj_list,
                    use_inv_entropy_c4=True,
                )

                log_p = torch.log(p_final + 1e-10) + cal_t
                mask_pad = torch.argmax(log_p, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                mask_unpad = preprocessor.unpad_output(mask_pad, metas[i])
                cv2.imwrite(osp.join(raw_mask_dir, f"{stems[i]}.png"), mask_unpad)

        del raw_cpu, rest_cpu, xd_cpu, xc_cpu, acc_v18_c3, acc_v18_c4, acc_v18_maj, acc_segf, acc_segf_f0, acc_dino_518, acc_conv_512
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [Scene {sc_idx+1}/6: {sc}] Completed {N} images in {time.time() - t_sc:.1f}s (Total elapsed: {time.time() - t0_all:.1f}s)", flush=True)

    del slot_v18, slot_segf, slot_dino, slot_conv
    gc.collect()
    torch.cuda.empty_cache()

    # Stage 3: Official Resizing
    print("\n[Stage 3/4] Applying Official tools/resize_for_submission.py (max_side=1024)...", flush=True)
    resize_predictions_to_original(
        pred_restored_dir=raw_rest_dir,
        pred_masks_dir=raw_mask_dir,
        ref_images_dir=test_dir,
        out_restored_dir=out_rest_dir,
        out_masks_dir=out_mask_dir,
        restored_ext="jpg",
        jpeg_quality=98,
        max_side=1024,
    )

    # Stage 4: Packaging & 7-Point + Uniqueness Audit
    print(f"\n[Stage 4/4] Packaging Submission Archive -> {out_zip}...", flush=True)
    os.makedirs(osp.dirname(out_zip), exist_ok=True)
    packager = V18SubmissionPackager(out_zip_path=out_zip, jpeg_quality=98)
    pkg_res = packager.package_directory(
        restored_dir=out_rest_dir,
        masks_dir=out_mask_dir,
        reference_dir=test_dir,
    )

    audit_res = run_uniqueness_and_release_audit(out_zip)
    verify_protected_baselines()

    manifest_out = osp.join(osp.dirname(out_zip), "v38_submission_manifest.json")
    with open(manifest_out, "w") as fp:
        json.dump({
            "candidate": "V38-Bayesian-QuadArch-SOTA",
            "oof_q_seg_20scene": 0.59570,
            "oof_miou_20scene": 0.71626,
            "oof_macro_f1_20scene": 0.83169,
            "oof_delta_q_seg_vs_v33": +0.07072,
            "recipe": "HB5_QuadArch_HighPromotedSegF_InvEntC4 | Off_20_40 (0,0,0,0.20,0.40)",
            "package_validation": pkg_res,
            "release_audit": audit_res,
        }, fp, indent=2)
    print(f"\nSaved complete V38 submission manifest to {manifest_out}", flush=True)
    return audit_res


if __name__ == "__main__":
    run_v38_quad_pipeline()
