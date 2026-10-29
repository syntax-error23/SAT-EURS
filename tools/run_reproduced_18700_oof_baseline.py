"""
Reproduce and Lock #18700 Baseline across Strict Scene-Disjoint OOF Folds
Evaluates exact #18700 pipeline:
- DINOv2-L with Exact-(wt, depth) Copy-Paste
- Multi-Scale TTA [448, 518, 588] + HFlip
- Leak-Free Fold RSUIGM Table + Unsharp Filter (alpha=1.30, sigma=2.0)
Computes:
- Pooled Screening Folds (0-1) [1,600 images]
- Pooled Confirmation Folds (2-4) [2,400 images]
- Pooled All-5-Folds (0-4) [4,000 images]
- Per-Fold breakdowns (Fold 0 through Fold 4)
Saves:
- reports/track1_research/v7_locked_baseline.json
"""

import os
import os.path as osp
import sys
import time
import json
from typing import Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import RestorationEvaluator
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from tools.pilot_restoration_evaluation import fit_leak_free_rsuigm_table

def run_locked_baseline(
    checkpoint_dir: str = "checkpoints/track1_screening",
    manifest_path: str = "splits/manifest_5fold.json",
    data_root: str = "data/cached_1024",
    save_path: str = "reports/track1_research/v7_locked_baseline.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("LOCKING #18700 REPRODUCED POOLED OOF BASELINE ACROSS ALL 20 SCENES")
    print("=" * 76)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    clean_dir = osp.join(data_root, "clean")
    masks_dir = osp.join(data_root, "masks")
    deg_dir = osp.join(data_root, "degraded")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    scales = [448, 518, 588]
    scale_weights = [0.20, 0.60, 0.20]

    # Evaluators
    eval_seg_all = SegmentationEvaluator(num_classes=5)
    eval_rest_all = RestorationEvaluator()

    eval_seg_screen = SegmentationEvaluator(num_classes=5)
    eval_rest_screen = RestorationEvaluator()

    eval_seg_confirm = SegmentationEvaluator(num_classes=5)
    eval_rest_confirm = RestorationEvaluator()

    per_fold_results = {}
    t0 = time.time()
    total_evaluated = 0

    for fold_idx in range(5):
        fold_id = f"fold_{fold_idx}"
        f_meta = manifest["folds"][fold_id]
        train_scenes = f_meta["train_scenes"]
        val_scenes = f_meta["val_scenes"]

        ckpt_path = osp.join(checkpoint_dir, f"DINOV2_CP_F{fold_idx}", "best_model.pt")
        print(f"\n[Fold {fold_idx}/4] Evaluating held-out scenes: {val_scenes} using {ckpt_path}...")

        model = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Leak-free table fitted on train_scenes
        rsuigm_table = fit_leak_free_rsuigm_table(train_scenes, device=device)

        eval_seg_fold = SegmentationEvaluator(num_classes=5)
        eval_rest_fold = RestorationEvaluator()

        fold_t0 = time.time()
        for sc in val_scenes:
            # Clean target & mask
            c_bgr = cv2.imread(osp.join(clean_dir, f"{sc}.png"), cv2.IMREAD_COLOR)
            c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            c_t = torch.from_numpy(c_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            gt_mask = cv2.imread(osp.join(masks_dir, f"{sc}.png"), cv2.IMREAD_GRAYSCALE)
            h_orig, w_orig = gt_mask.shape

            sc_deg_dir = osp.join(deg_dir, sc)
            deg_files = sorted([f for f in os.listdir(sc_deg_dir) if f.endswith(".png")])

            for df in deg_files:
                parts = osp.splitext(df)[0].split("_")
                wt = int(parts[1])
                dp = int(parts[2])

                bgr = cv2.imread(osp.join(sc_deg_dir, df))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                x_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([dp], device=device)

                with torch.no_grad():
                    # 1. Restoration
                    x_rest = apply_rsuigm_batch_gpu(x_t, wt_t, dp_t, rsuigm_table)
                    res_np = (x_rest.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    blur = cv2.GaussianBlur(res_np, (0, 0), 2.0)
                    sharp_np = np.clip(cv2.addWeighted(res_np, 1.30, blur, -0.30, 0), 0, 255).astype(np.float32) / 255.0
                    sharp_t = torch.from_numpy(sharp_np).permute(2, 0, 1).unsqueeze(0).to(device)

                    # 2. Multi-Scale Segmentation
                    x_norm = (x_rest - mean_im) / std_im
                    p_accum = torch.zeros((1, 5, 518, 518), device=device)

                    for sc_sz, sc_w in zip(scales, scale_weights):
                        x_sc = F.interpolate(x_norm, size=(sc_sz, sc_sz), mode="bilinear", align_corners=False)
                        x_sc_flip = torch.flip(x_sc, dims=[3])

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            l_orig = model(x_sc)
                            l_flip = model(x_sc_flip)

                        p_orig = F.softmax(l_orig.float(), dim=1)
                        p_flip = torch.flip(F.softmax(l_flip.float(), dim=1), dims=[3])
                        p_fused = 0.5 * p_orig + 0.5 * p_flip

                        p_518 = F.interpolate(p_fused, size=(518, 518), mode="bilinear", align_corners=False)
                        p_accum += sc_w * p_518

                # Resize prediction to native mask dimension
                p_orig_size = F.interpolate(p_accum, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
                y_t = torch.from_numpy(gt_mask.astype(np.int64)).long().unsqueeze(0).to(device)

                # Update fold evaluators
                eval_seg_fold.update(p_orig_size, y_t)
                eval_rest_fold.update(sharp_t, c_t)

                # Update global evaluators
                eval_seg_all.update(p_orig_size, y_t)
                eval_rest_all.update(sharp_t, c_t)

                if fold_idx in (0, 1):
                    eval_seg_screen.update(p_orig_size, y_t)
                    eval_rest_screen.update(sharp_t, c_t)
                else:
                    eval_seg_confirm.update(p_orig_size, y_t)
                    eval_rest_confirm.update(sharp_t, c_t)

                total_evaluated += 1

        fold_seg = eval_seg_fold.compute()
        fold_rest = eval_rest_fold.compute()
        per_fold_results[fold_id] = {
            "val_scenes": val_scenes,
            "segmentation": fold_seg,
            "restoration": fold_rest,
            "elapsed_sec": round(time.time() - fold_t0, 1)
        }
        print(f"  {fold_id} finished in {time.time() - fold_t0:.1f}s:")
        print(f"    Seg:  mIoU={fold_seg['mIoU']:.4f} | F1={fold_seg['macro_F1']:.4f} | Q_seg={fold_seg['Q_seg']:.4f}")
        print(f"    Rest: PSNR={fold_rest['PSNR_dB']:.2f} dB | SSIM={fold_rest['SSIM']:.4f} | Sharp={fold_rest['sharpness_ratio']:.4f}")

    print("\n" + "=" * 76)
    print("COMPUTING FINAL POOLED OOF METRICS")
    print("=" * 76)

    res_all_seg = eval_seg_all.compute()
    res_all_rest = eval_rest_all.compute()

    res_screen_seg = eval_seg_screen.compute()
    res_screen_rest = eval_rest_screen.compute()

    res_confirm_seg = eval_seg_confirm.compute()
    res_confirm_rest = eval_rest_confirm.compute()

    # Exact official composite calculation over pooled sets (with provisional Q_NR = 0.50 stub)
    def compute_composite(q_seg, psnr_norm, ssim, sharp, q_nr=0.50):
        q_fr = 0.45 * psnr_norm + 0.45 * ssim + 0.10 * sharp
        s_comp = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr
        return round(s_comp, 4), round(q_fr, 4)

    s_all, q_fr_all = compute_composite(
        res_all_seg["Q_seg"], res_all_rest["PSNR_norm"], res_all_rest["SSIM"], res_all_rest["sharpness_ratio"]
    )
    s_screen, q_fr_screen = compute_composite(
        res_screen_seg["Q_seg"], res_screen_rest["PSNR_norm"], res_screen_rest["SSIM"], res_screen_rest["sharpness_ratio"]
    )
    s_confirm, q_fr_confirm = compute_composite(
        res_confirm_seg["Q_seg"], res_confirm_rest["PSNR_norm"], res_confirm_rest["SSIM"], res_confirm_rest["sharpness_ratio"]
    )

    locked_data = {
        "metadata": {
            "baseline_description": "Exact reproduced #18700 MultiScale OOF baseline",
            "pipeline": "DINOv2-L + Exact-(wt, depth) CP + Multi-Scale TTA [448, 518, 588] + RSUIGM 19-term unsharp",
            "total_images_evaluated": total_evaluated,
            "total_elapsed_sec": round(time.time() - t0, 1),
            "date": "2026-10-15",
            "commit": "9a17437"
        },
        "pooled_screening_folds_0_1": {
            "description": "Pooled Folds 0 & 1 (1,600 images)",
            "composite_provisional_est": s_screen,
            "Q_FR": q_fr_screen,
            "segmentation": res_screen_seg,
            "restoration": res_screen_rest,
            "hard_floors": {
                "Q_seg": res_screen_seg["Q_seg"],
                "mIoU": res_screen_seg["mIoU"],
                "macro_F1": res_screen_seg["macro_F1"],
                "per_class_IoU": res_screen_seg["per_class_IoU"],
                "per_class_F1": res_screen_seg["per_class_F1"],
                "Sharpness_OOF_Base": res_screen_rest["sharpness_ratio"]
            }
        },
        "pooled_confirmation_folds_2_4": {
            "description": "Pooled Folds 2, 3, & 4 (2,400 images)",
            "composite_provisional_est": s_confirm,
            "Q_FR": q_fr_confirm,
            "segmentation": res_confirm_seg,
            "restoration": res_confirm_rest,
            "hard_floors": {
                "Q_seg": res_confirm_seg["Q_seg"],
                "mIoU": res_confirm_seg["mIoU"],
                "macro_F1": res_confirm_seg["macro_F1"],
                "per_class_IoU": res_confirm_seg["per_class_IoU"],
                "per_class_F1": res_confirm_seg["per_class_F1"],
                "Sharpness_OOF_Base": res_confirm_rest["sharpness_ratio"]
            }
        },
        "pooled_all_5_folds_0_4": {
            "description": "Pooled All 5 Folds (4,000 images across all 20 scenes)",
            "composite_provisional_est": s_all,
            "Q_FR": q_fr_all,
            "segmentation": res_all_seg,
            "restoration": res_all_rest,
            "hard_floors": {
                "Q_seg": res_all_seg["Q_seg"],
                "mIoU": res_all_seg["mIoU"],
                "macro_F1": res_all_seg["macro_F1"],
                "per_class_IoU": res_all_seg["per_class_IoU"],
                "per_class_F1": res_all_seg["per_class_F1"],
                "Sharpness_OOF_Base": res_all_rest["sharpness_ratio"]
            }
        },
        "per_fold_breakdown": per_fold_results
    }

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(locked_data, fp, indent=2)

    print("\n" + "=" * 76)
    print(f"LOCKED BASELINE SAVED TO {save_path}:")
    print(f"  Pooled All-5-Folds Q_seg:          {res_all_seg['Q_seg']:.4f}")
    print(f"  Pooled All-5-Folds mIoU:           {res_all_seg['mIoU']:.4f}")
    print(f"  Pooled All-5-Folds Macro F1:       {res_all_seg['macro_F1']:.4f}")
    print(f"  Pooled All-5-Folds PSNR:           {res_all_rest['PSNR_dB']:.2f} dB")
    print(f"  Pooled All-5-Folds SSIM:           {res_all_rest['SSIM']:.4f}")
    print(f"  Pooled All-5-Folds Sharpness Base: {res_all_rest['sharpness_ratio']:.4f}")
    print(f"  Pooled All-5-Folds Q_FR:           {q_fr_all:.4f}")
    print("=" * 76)

    return locked_data

if __name__ == '__main__':
    run_locked_baseline()
