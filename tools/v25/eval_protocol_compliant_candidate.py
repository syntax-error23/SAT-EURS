"""
SAT-EURS Track 1 — V25 Protocol-Compliant Candidate Evaluation
Evaluates single-image calibrated restoration and segmentation on Fold 0 and Fold 1.
Protocol: Strict Single-Image Inference
Evaluator: Exact tools.v18.evaluator.V18OOFEvaluator
"""

import os
import os.path as osp
import sys
import json
import time
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from tools.v18.evaluator import V18OOFEvaluator
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter

EXPECTED_V18_ZIP_SHA256 = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
V18_ZIP_PATH = "deploy/v18/submission_v18_dual_encoder_sota.zip"


def verify_baseline():
    import hashlib
    if not osp.exists(V18_ZIP_PATH):
        raise FileNotFoundError(f"Baseline ZIP missing: {V18_ZIP_PATH}")
    h = hashlib.sha256()
    with open(V18_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha = h.hexdigest().lower()
    assert sha == EXPECTED_V18_ZIP_SHA256, f"BASELINE CORRUPTED: expected {EXPECTED_V18_ZIP_SHA256}, got {sha}"
    print(f"[PREFLIGHT] Baseline verified intact (SHA-256: {sha})")


def preload_scene_data(scenes: List[str], data_root: str = "data/cached_1024"):
    items = []
    print(f"Preloading {len(scenes)} scenes into RAM...")
    t0 = time.time()
    for sc in scenes:
        c_path = osp.join(data_root, "clean", f"{sc}.png")
        clean_bgr = cv2.imread(c_path, cv2.IMREAD_COLOR)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        m_path = osp.join(data_root, "masks", f"{sc}.png")
        gt_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

        for wt in range(1, 11):
            for depth in range(5, 105, 5):
                d_path = osp.join(data_root, "degraded", sc, f"{sc}_{wt}_{depth}.png")
                deg_bgr = cv2.imread(d_path, cv2.IMREAD_COLOR)
                deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                items.append({
                    "scene": sc,
                    "wt": wt,
                    "depth": depth,
                    "deg_rgb": deg_rgb,
                    "clean_rgb": clean_rgb,
                    "gt_mask": gt_mask
                })
    print(f"Preloaded {len(items)} items in {time.time() - t0:.1f}s.")
    return items


def run_evaluation(
    fold: int,
    alpha_unsharp: float = 0.25,
    sigma_unsharp: float = 2.0,
    b4_offset: float = 0.33,
    device_id: int = 0
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "=" * 80)
    print(f"EVALUATING FOLD {fold}: unsharp(alpha={alpha_unsharp}, sigma={sigma_unsharp}), b4={b4_offset}")
    print(f"=" * 80)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    val_scenes = manifest["folds"][f"fold_{fold}"]["val_scenes"]
    print(f"Held-out scenes: {val_scenes}")

    # Load Refiner
    table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    rsuigm_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
    refiner_ckpt = torch.load(f"checkpoints/v18/fold_{fold}/refiner_best.pt", map_location=device)
    refiner.load_state_dict(refiner_ckpt.get("state_dict", refiner_ckpt))
    refiner.eval()

    # Load Segmenter
    segmenter = build_v18_segmenter(num_classes=5).to(device)
    seg_ckpt = torch.load(f"checkpoints/v18/fold_{fold}/segmenter_best.pt", map_location=device)
    segmenter.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))
    segmenter.eval()

    val_data = preload_scene_data(val_scenes)

    # 1. Baseline V18 evaluation (b4=0.15, alpha=0.0)
    evaluator_base = V18OOFEvaluator(num_classes=5)
    # 2. Enhanced candidate evaluation (b4=b4_offset, alpha=alpha_unsharp)
    evaluator_cand = V18OOFEvaluator(num_classes=5)

    cal_offsets_base = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.15], device=device).view(1, 5, 1, 1)
    cal_offsets_cand = torch.tensor([0.0, 0.0, 0.0, 0.15, b4_offset], device=device).view(1, 5, 1, 1)

    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch = val_data[i:i+batch_size]
            deg_tensors = [torch.from_numpy(it["deg_rgb"]).permute(2, 0, 1) for it in batch]
            deg_batch = torch.stack(deg_tensors, dim=0).to(device)
            wt_batch = torch.tensor([it["wt"] for it in batch], device=device)
            dp_batch = torch.tensor([it["depth"] for it in batch], device=device)

            # Restoration forward pass in float32 (avoids numpy BFloat16 issue)
            out_ref = refiner(deg_batch.float(), wt_batch, dp_batch)
            rest_base_batch = out_ref["refined_image"].float()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Segmenter forward pass on restored image
                out_seg_orig = segmenter(rest_base_batch, deg_batch, wt_batch, dp_batch, unpadded_size=(1024, 1024))
                p_orig = F.softmax(out_seg_orig["logits"].float(), dim=1)

                # Flip TTA
                rest_flip = torch.flip(rest_base_batch, dims=[3])
                deg_flip = torch.flip(deg_batch, dims=[3])
                out_seg_flip = segmenter(rest_flip, deg_flip, wt_batch, dp_batch, unpadded_size=(1024, 1024))
                p_flip = torch.flip(F.softmax(out_seg_flip["logits"].float(), dim=1), dims=[3])

                p_fused = 0.5 * (p_orig + p_flip)

                # Base mask
                log_p_base = torch.log(p_fused + 1e-10) + cal_offsets_base
                pred_base_masks = torch.argmax(log_p_base, dim=1).cpu().numpy().astype(np.uint8)

                # Cand mask
                log_p_cand = torch.log(p_fused + 1e-10) + cal_offsets_cand
                pred_cand_masks = torch.argmax(log_p_cand, dim=1).cpu().numpy().astype(np.uint8)

            if (i // batch_size + 1) % 50 == 0:
                print(f"  Evaluated {i + len(batch)} / {len(val_data)} images...", flush=True)

            rest_base_np = rest_base_batch.permute(0, 2, 3, 1).cpu().numpy()

            for b_idx, it in enumerate(batch):
                r_base = np.clip(rest_base_np[b_idx], 0.0, 1.0)
                m_base = pred_base_masks[b_idx]
                evaluator_base.update_restoration(r_base, it["clean_rgb"], scene=it["scene"])
                evaluator_base.update_segmentation(m_base, it["gt_mask"], scene=it["scene"])

                # Enhanced restoration: post-hoc unsharp filter
                if alpha_unsharp > 0:
                    blur = cv2.GaussianBlur(r_base, (0, 0), sigma_unsharp)
                    r_cand = np.clip(r_base + alpha_unsharp * (r_base - blur), 0.0, 1.0)
                else:
                    r_cand = r_base

                m_cand = pred_cand_masks[b_idx]
                evaluator_cand.update_restoration(r_cand, it["clean_rgb"], scene=it["scene"])
                evaluator_cand.update_segmentation(m_cand, it["gt_mask"], scene=it["scene"])

    res_base = evaluator_base.compute()
    res_cand = evaluator_cand.compute()

    q_seg_base = res_base["segmentation"]["Q_seg"]
    q_fr_base = res_base["restoration"]["Q_FR_dataset"]
    comp_base = 0.60 * q_seg_base + 0.30 * q_fr_base + 0.10 * 0.737

    q_seg_cand = res_cand["segmentation"]["Q_seg"]
    q_fr_cand = res_cand["restoration"]["Q_FR_dataset"]
    comp_cand = 0.60 * q_seg_cand + 0.30 * q_fr_cand + 0.10 * 0.737

    delta_comp = comp_cand - comp_base

    print(f"\n--- FOLD {fold} RESULTS ---")
    print(f"BASELINE:  PSNR: {res_base['restoration']['PSNR_dB']:.2f} dB | SSIM: {res_base['restoration']['SSIM']:.4f} | Sharp: {res_base['restoration']['sharpness_ratio']:.4f} | Q_FR: {q_fr_base:.4f} | Q_seg: {q_seg_base:.4f} | Comp: {comp_base:.4f}")
    print(f"CANDIDATE: PSNR: {res_cand['restoration']['PSNR_dB']:.2f} dB | SSIM: {res_cand['restoration']['SSIM']:.4f} | Sharp: {res_cand['restoration']['sharpness_ratio']:.4f} | Q_FR: {q_fr_cand:.4f} | Q_seg: {q_seg_cand:.4f} | Comp: {comp_cand:.4f}")
    print(f"DELTA:     Q_FR: {q_fr_cand - q_fr_base:+.4f} | Q_seg: {q_seg_cand - q_seg_base:+.4f} | COMPOSITE DELTA: {delta_comp:+.4f}")

    return {
        "fold": fold,
        "baseline": res_base,
        "candidate": res_cand,
        "delta_comp": delta_comp,
        "delta_q_fr": q_fr_cand - q_fr_base,
        "delta_q_seg": q_seg_cand - q_seg_base
    }


if __name__ == "__main__":
    verify_baseline()
    f0 = run_evaluation(fold=0, alpha_unsharp=0.25, sigma_unsharp=2.0, b4_offset=0.33)
    f1 = run_evaluation(fold=1, alpha_unsharp=0.25, sigma_unsharp=2.0, b4_offset=0.33)

    mean_delta = 0.5 * (f0["delta_comp"] + f1["delta_comp"])
    print("\n" + "=" * 80)
    print(f"SUMMARY ACROSS FOLDS 0 & 1: Mean Composite Delta = {mean_delta:+.4f}")
    print(f"Fold 0 Delta: {f0['delta_comp']:+.4f} (Q_seg: {f0['delta_q_seg']:+.4f}, Q_FR: {f0['delta_q_fr']:+.4f})")
    print(f"Fold 1 Delta: {f1['delta_comp']:+.4f} (Q_seg: {f1['delta_q_seg']:+.4f}, Q_FR: {f1['delta_q_fr']:+.4f})")
    print("=" * 80)
    verify_baseline()
