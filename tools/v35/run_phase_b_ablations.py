"""
SAT-EURS Track 1 — Phase B Controlled Ablation Benchmark
Ablation Matrix:
Restoration:
  R-0: Per-image Baseline (V28-A restoration, single image)
  R-A: Shallow Anchor (mean of shallow views: WT 1-3, depth 5-10m)
  R-B: Shallow Anchor + Unsharp Masking (alpha in [0.05, 0.10, 0.15], sigma=2.0)
  R-C: Depth-Weighted Blend: R_final = (1 - w(z)) * R_anchor + w(z) * R_single
  R-D: Full Scene-Anchor Restoration (optimal anchor + unsharp)

Segmentation:
  S-0: Per-image Baseline (V28-A single image segmentation)
  S-A: Shallow Anchor Probability (no TTA)
  S-B: Shallow Anchor + TTA (Horizontal Flip)
  S-C: Scene Anchor + Logit Calibration ([0.0, 0.0, 0.0, 0.10, 0.25])
  S-D: Full Scene-Anchor Segmentation (optimal anchor + TTA + calibration)

Strict Leakage-Free Protocol:
  Evaluated Out-Of-Fold across all 20 scenes using splits/manifest_5fold.json.
  Each scene is strictly evaluated by the fold model that held it out.
"""

import os
import os.path as osp
import sys
import json
import time
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import V18CanonicalPreprocessor
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio
from metrics.segmentation_metrics import SegmentationEvaluator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Executing Phase B Ablations on {device} ({torch.cuda.get_device_name(device)})")

def unsharp_mask(img_np: np.ndarray, alpha: float = 0.10, sigma: float = 2.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigma)
    sharpened = img_np + alpha * (img_np - blurred)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)

def compute_composite(q_seg: float, q_fr: float, q_nr: float = 0.60) -> float:
    return 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr

def run_ablations():
    with open("splits/manifest_5fold.json", "r") as f:
        splits = json.load(f)["folds"]

    table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    fold_checkpoints = {
        0: ("checkpoints/v18/fold_0/refiner_best.pt", "checkpoints/v27/segmenter_v27_fold0_best.pt"),
        1: ("checkpoints/v18/fold_1/refiner_best.pt", "checkpoints/v18/fold_1/segmenter_best.pt"),
        2: ("checkpoints/v18/fold_2/refiner_best.pt", "checkpoints/v18/fold_2/segmenter_best.pt"),
        3: ("checkpoints/v18/fold_3/refiner_best.pt", "checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt"),
        4: ("checkpoints/v18/fold_4/refiner_best.pt", "checkpoints/v18/fold_4/segmenter_best.pt"),
    }

    # Tracking metrics per ablation
    # Restoration accumulators
    rest_metrics = {
        "R-0_base": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
        "R-A_shallow_anchor": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
        "R-B_unsharp_05": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
        "R-B_unsharp_10": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
        "R-B_unsharp_15": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
        "R-C_depth_blend": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
        "R-D_full_anchor": {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": []},
    }

    # Segmentation evaluators
    seg_evals = {
        "S-0_base": SegmentationEvaluator(num_classes=5),
        "S-A_shallow_anchor": SegmentationEvaluator(num_classes=5),
        "S-B_shallow_tta": SegmentationEvaluator(num_classes=5),
        "S-C_prior_calib": SegmentationEvaluator(num_classes=5),
        "S-D_full_anchor": SegmentationEvaluator(num_classes=5),
    }

    start_time = time.time()
    total_scenes = 0

    for fold_idx in sorted(fold_checkpoints.keys()):
        val_scenes = splits[f"fold_{fold_idx}"]["val_scenes"]
        print(f"\nEvaluating Fold {fold_idx} held-out scenes: {val_scenes} (Zero leakage)")

        ref_ckpt, seg_ckpt = fold_checkpoints[fold_idx]
        refiner = build_v18_restoration_refiner(rsuigm_table=table, use_neural_refiner=True).to(device)
        refiner.load_state_dict(torch.load(ref_ckpt, map_location=device)["state_dict"])
        refiner.eval()

        segmenter = build_v18_segmenter(num_classes=5).to(device)
        segmenter.load_state_dict(torch.load(seg_ckpt, map_location=device)["state_dict"])
        segmenter.eval()

        for sc in val_scenes:
            clean_bgr = cv2.imread(f"data/cached_1024/clean/{sc}.png")
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_mask = cv2.imread(f"data/cached_1024/masks/{sc}.png", cv2.IMREAD_UNCHANGED)
            gt_t = torch.from_numpy(gt_mask).long()

            # 1. Compute shallow anchors across WT 1-3, depth 5-10m (6 views)
            shallow_rests = []
            shallow_probs_notta = []
            shallow_probs_tta = []

            for wt in [1, 2, 3]:
                for dp in [5, 10]:
                    deg_fn = f"data/cached_1024/degraded/{sc}/{sc}_{wt}_{dp}.png"
                    if not osp.exists(deg_fn):
                        continue
                    bgr = cv2.imread(deg_fn)
                    prep = preprocessor.prepare_input(bgr)
                    raw_t = prep["rgb_tensor"].to(device)
                    meta = prep["meta"]
                    wt_t = torch.tensor([wt], device=device)
                    dp_t = torch.tensor([dp], dtype=torch.float32, device=device)

                    with torch.no_grad():
                        rest = refiner(raw_t, wt_t, dp_t)["refined_image"]
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out_orig = segmenter(rest, raw_t, wt_t, dp_t)["logits"].float()
                            p_orig = F.softmax(out_orig, dim=1)

                            # TTA flip
                            raw_flip = torch.flip(raw_t, dims=[3])
                            rest_flip = torch.flip(rest, dims=[3])
                            out_flip = segmenter(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                            p_flip = torch.flip(F.softmax(out_flip, dim=1), dims=[3])
                            p_tta = 0.5 * (p_orig + p_flip)

                    r_np = rest.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    p_orig_np = p_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    p_orig_unpad = preprocessor.unpad_output(p_orig_np, meta)
                    p_orig_res = cv2.resize(p_orig_unpad, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

                    p_tta_np = p_tta.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    p_tta_unpad = preprocessor.unpad_output(p_tta_np, meta)
                    p_tta_res = cv2.resize(p_tta_unpad, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

                    shallow_rests.append(r_np)
                    shallow_probs_notta.append(p_orig_res)
                    shallow_probs_tta.append(p_tta_res)

            # Build Anchor representations
            r_anchor = np.mean(shallow_rests, axis=0)
            p_anchor_notta = np.mean(shallow_probs_notta, axis=0)
            p_anchor_tta = np.mean(shallow_probs_tta, axis=0)

            # R-A: Shallow Anchor restoration metrics
            p, pn = compute_psnr_and_norm(r_anchor, clean_rgb)
            s = compute_ssim_single(r_anchor, clean_rgb)
            sh = compute_sharpness_ratio(r_anchor, clean_rgb)
            rest_metrics["R-A_shallow_anchor"]["psnr"].append(p)
            rest_metrics["R-A_shallow_anchor"]["psnr_norm"].append(pn)
            rest_metrics["R-A_shallow_anchor"]["ssim"].append(s)
            rest_metrics["R-A_shallow_anchor"]["sharp"].append(sh)

            # R-B: Unsharp variants
            for alpha_val, key in [(0.05, "R-B_unsharp_05"), (0.10, "R-B_unsharp_10"), (0.15, "R-B_unsharp_15")]:
                r_unsharp = unsharp_mask(r_anchor, alpha=alpha_val, sigma=2.0)
                p_u, pn_u = compute_psnr_and_norm(r_unsharp, clean_rgb)
                s_u = compute_ssim_single(r_unsharp, clean_rgb)
                sh_u = compute_sharpness_ratio(r_unsharp, clean_rgb)
                rest_metrics[key]["psnr"].append(p_u)
                rest_metrics[key]["psnr_norm"].append(pn_u)
                rest_metrics[key]["ssim"].append(s_u)
                rest_metrics[key]["sharp"].append(sh_u)

            # R-D: Full anchor (alpha=0.10)
            r_full = unsharp_mask(r_anchor, alpha=0.10, sigma=2.0)
            p_f, pn_f = compute_psnr_and_norm(r_full, clean_rgb)
            s_f = compute_ssim_single(r_full, clean_rgb)
            sh_f = compute_sharpness_ratio(r_full, clean_rgb)
            rest_metrics["R-D_full_anchor"]["psnr"].append(p_f)
            rest_metrics["R-D_full_anchor"]["psnr_norm"].append(pn_f)
            rest_metrics["R-D_full_anchor"]["ssim"].append(s_f)
            rest_metrics["R-D_full_anchor"]["sharp"].append(sh_f)

            # Evaluate Segmentation Anchors
            # S-A: Shallow anchor no TTA
            mask_sa = np.argmax(p_anchor_notta, axis=-1).astype(np.uint8)
            seg_evals["S-A_shallow_anchor"].update(torch.from_numpy(mask_sa).unsqueeze(0), gt_t.unsqueeze(0))

            # S-B: Shallow anchor with TTA
            mask_sb = np.argmax(p_anchor_tta, axis=-1).astype(np.uint8)
            seg_evals["S-B_shallow_tta"].update(torch.from_numpy(mask_sb).unsqueeze(0), gt_t.unsqueeze(0))

            # S-C: Shallow anchor with logit calibration ([0,0,0, 0.10, 0.25])
            log_p_c = np.log(p_anchor_tta + 1e-10) + np.array([0.0, 0.0, 0.0, 0.10, 0.25])
            mask_sc = np.argmax(log_p_c, axis=-1).astype(np.uint8)
            seg_evals["S-C_prior_calib"].update(torch.from_numpy(mask_sc).unsqueeze(0), gt_t.unsqueeze(0))

            # S-D: Full anchor segmentation
            seg_evals["S-D_full_anchor"].update(torch.from_numpy(mask_sc).unsqueeze(0), gt_t.unsqueeze(0))

            # Evaluate R-0 and S-0 Baseline across representative depths (20 views per scene)
            deg_files = sorted(os.listdir(f"data/cached_1024/degraded/{sc}"))[::10]
            scene_base_p, scene_base_pn, scene_base_s, scene_base_sh = [], [], [], []
            scene_blend_p, scene_blend_pn, scene_blend_s, scene_blend_sh = [], [], [], []

            for deg_fn in deg_files:
                parts = deg_fn.replace(".png", "").split("_")
                wt, dp = int(parts[1]), int(parts[2])
                bgr = cv2.imread(f"data/cached_1024/degraded/{sc}/{deg_fn}")
                prep = preprocessor.prepare_input(bgr)
                raw_t = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([dp], dtype=torch.float32, device=device)

                with torch.no_grad():
                    rest = refiner(raw_t, wt_t, dp_t)["refined_image"]
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_b = segmenter(rest, raw_t, wt_t, dp_t)["logits"].float()
                        p_b = F.softmax(out_b, dim=1)

                r_single = rest.squeeze(0).permute(1, 2, 0).cpu().numpy()
                p_b_np = p_b.squeeze(0).permute(1, 2, 0).cpu().numpy()
                p_b_unpad = preprocessor.unpad_output(p_b_np, meta)
                p_b_res = cv2.resize(p_b_unpad, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask_b = np.argmax(p_b_res, axis=-1).astype(np.uint8)
                seg_evals["S-0_base"].update(torch.from_numpy(mask_b).unsqueeze(0), gt_t.unsqueeze(0))

                # R-0 baseline metrics
                bp, bpn = compute_psnr_and_norm(r_single, clean_rgb)
                bs = compute_ssim_single(r_single, clean_rgb)
                bsh = compute_sharpness_ratio(r_single, clean_rgb)
                scene_base_p.append(bp)
                scene_base_pn.append(bpn)
                scene_base_s.append(bs)
                scene_base_sh.append(bsh)

                # R-C: Depth-weighted blend: w(z) = (dp - 5) / 100 * 0.15
                w_z = max(0.0, min(1.0, (dp - 5.0) / 95.0)) * 0.15
                r_blend = (1.0 - w_z) * r_anchor + w_z * r_single
                cp, cpn = compute_psnr_and_norm(r_blend, clean_rgb)
                cs = compute_ssim_single(r_blend, clean_rgb)
                csh = compute_sharpness_ratio(r_blend, clean_rgb)
                scene_blend_p.append(cp)
                scene_blend_pn.append(cpn)
                scene_blend_s.append(cs)
                scene_blend_sh.append(csh)

            rest_metrics["R-0_base"]["psnr"].append(float(np.mean(scene_base_p)))
            rest_metrics["R-0_base"]["psnr_norm"].append(float(np.mean(scene_base_pn)))
            rest_metrics["R-0_base"]["ssim"].append(float(np.mean(scene_base_s)))
            rest_metrics["R-0_base"]["sharp"].append(float(np.mean(scene_base_sh)))

            rest_metrics["R-C_depth_blend"]["psnr"].append(float(np.mean(scene_blend_p)))
            rest_metrics["R-C_depth_blend"]["psnr_norm"].append(float(np.mean(scene_blend_pn)))
            rest_metrics["R-C_depth_blend"]["ssim"].append(float(np.mean(scene_blend_s)))
            rest_metrics["R-C_depth_blend"]["sharp"].append(float(np.mean(scene_blend_sh)))

            total_scenes += 1
            print(f"  Scene {sc:10s} Done | R-0 PSNR: {np.mean(scene_base_p):.2f} dB -> Anchor: {p:.2f} dB | Anchor SSIM: {s:.4f}")

    elapsed = time.time() - start_time
    print(f"\nAblations Completed in {elapsed:.1f}s across {total_scenes} scenes.")

    # Aggregate Restoration Metrics
    rest_summary = {}
    for k, v in rest_metrics.items():
        m_psnr = float(np.mean(v["psnr"]))
        m_psnr_norm = min(1.0, max(0.0, (m_psnr - 15.0) / 20.0))
        m_ssim = float(np.mean(v["ssim"]))
        m_sharp = float(np.mean(v["sharp"]))
        q_fr = 0.45 * m_psnr_norm + 0.45 * m_ssim + 0.10 * m_sharp
        rest_summary[k] = {
            "PSNR_dB": round(m_psnr, 2),
            "PSNR_norm": round(m_psnr_norm, 4),
            "SSIM": round(m_ssim, 4),
            "Sharpness": round(m_sharp, 4),
            "Q_FR": round(q_fr, 4),
        }

    # Aggregate Segmentation Metrics
    seg_summary = {}
    for k, ev in seg_evals.items():
        res = ev.compute()
        seg_summary[k] = {
            "mIoU": res["mIoU"],
            "macro_F1": res["macro_F1"],
            "Q_seg": res["Q_seg"],
            "per_class_IoU": res["per_class_IoU"],
        }

    # Build Combined Matrix
    print("\n" + "=" * 90)
    print("PHASE B CONTROLLED ABLATION RESULTS (STRICT 20-SCENE OUT-OF-FOLD LEAKAGE-FREE)")
    print("=" * 90)
    print(f"{'Ablation':<22} | {'PSNR (dB)':<9} | {'SSIM':<7} | {'Sharp':<6} | {'Q_FR':<6} | {'mIoU':<6} | {'MacroF1':<7} | {'Q_seg':<6} | {'Composite':<9}")
    print("-" * 90)

    ablation_pairs = [
        ("R-0 + S-0 (Baseline)", "R-0_base", "S-0_base"),
        ("R-A + S-A (Shallow)", "R-A_shallow_anchor", "S-A_shallow_anchor"),
        ("R-B + S-B (+TTA/Sharp)", "R-B_unsharp_10", "S-B_shallow_tta"),
        ("R-C + S-C (+PriorCalib)", "R-C_depth_blend", "S-C_prior_calib"),
        ("R-D + S-D (Full Anchor)", "R-D_full_anchor", "S-D_full_anchor"),
    ]

    ablation_results = {}
    for label, r_k, s_k in ablation_pairs:
        r_info = rest_summary[r_k]
        s_info = seg_summary[s_k]
        comp = compute_composite(s_info["Q_seg"], r_info["Q_FR"])
        ablation_results[label] = {
            "restoration": r_info,
            "segmentation": s_info,
            "composite": round(comp, 4)
        }
        print(f"{label:<22} | {r_info['PSNR_dB']:<9.2f} | {r_info['SSIM']:<7.4f} | {r_info['Sharpness']:<6.4f} | {r_info['Q_FR']:<6.4f} | {s_info['mIoU']:<6.4f} | {s_info['macro_F1']:<7.4f} | {s_info['Q_seg']:<6.4f} | {comp:<9.4f}")

    print("=" * 90)

    # Save to JSON
    output_path = "runs/master_execution/phase_b_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "restoration_details": rest_summary,
            "segmentation_details": seg_summary,
            "ablation_matrix": ablation_results
        }, f, indent=2)
    print(f"Results successfully saved to {output_path}")

if __name__ == "__main__":
    run_ablations()
