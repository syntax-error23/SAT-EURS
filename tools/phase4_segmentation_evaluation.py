"""
Phase 4 — Downstream Segmentation Evaluation without Confounding
MaCVi @ WACV 2027 Challenge — SAT-EURS Track 1
Hardware: Windows 11 | RTX 5090 32GB | 64GB DDR5

Evaluates downstream segmentation with frozen DINOv2-Large Fold 0 on:
1. Raw Degraded Input (Floor)
2. RSUIGM (No Unsharp)
3. RSUIGM + Unsharp Mask 1.34 (Official V16 Pipeline)

Measures:
- PSNR, SSIM, Sharpness, Q_FR
- Per-class IoU and F1 for all 5 classes: [Background, Coral, Fish, Elasmobranch, Other]
- Macro mIoU, Macro F1, Q_seg, and Composite Score
"""

import os
import os.path as osp
import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")

from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio
from metrics.segmentation_metrics import SegmentationEvaluator


def run_phase4_evaluation():
    print("=" * 80)
    print("PHASE 4: DOWNSTREAM SEGMENTATION EVALUATION (FROZEN DINOV2-L FOLD 0)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Frozen DINOv2-Large Fold 0 Model
    dino_path = "checkpoints/v16_dinov2/fold_0/best_model.pt"
    assert osp.exists(dino_path), f"Missing {dino_path}"
    model = build_dinov2_model(num_classes=5).to(device)
    ckpt = torch.load(dino_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded Frozen DINOv2-Large Fold 0 from: {dino_path}")

    # 2. Load Calibration Table (Leak-free Fold 0 table)
    tbl_path = "checkpoints/track1_screening/tbl19_16sc_fold0.pt"
    tbl16 = torch.load(tbl_path, map_location=device)
    print(f"Loaded Leak-Free Fold 0 RSUIGM table from: {tbl_path}")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 3. Setup Dataset (Fold 0 Val scenes)
    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    data_root = "data/cached_1024"

    clean_cache = {}
    mask_cache = {}
    for sc in val_scenes:
        c_bgr = cv2.imread(osp.join(data_root, "clean", f"{sc}.png"))
        clean_cache[sc] = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m_arr = np.asarray(Image.open(osp.join(data_root, "masks", f"{sc}.png")), dtype=np.uint8)
        mask_cache[sc] = m_arr

    items = []
    for sc in val_scenes:
        sc_p = osp.join(data_root, "degraded", sc)
        deg_files = sorted([f for f in os.listdir(sc_p) if f.endswith(".png")])
        for df in deg_files[::10]: # 80 images
            parts = osp.splitext(df)[0].split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
            items.append((sc, osp.join(sc_p, df), wt, depth))

    print(f"Sampled {len(items)} evaluation samples across {val_scenes}")

    configs = [
        {"name": "raw_degraded", "mode": "raw", "desc": "Raw Degraded Input (No Restoration)"},
        {"name": "rsuigm_raw", "mode": "rsuigm_raw", "desc": "RSUIGM Inversion (No Unsharp)"},
        {"name": "rsuigm_unsharp134", "mode": "rsuigm_unsharp", "desc": "RSUIGM + 1.34 Unsharp (Official V16 Pipeline)"},
    ]

    results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        mode = cfg["mode"]
        print(f"\n--- Evaluating: {cfg['desc']} ---")

        evaluator = SegmentationEvaluator(num_classes=5)
        psnrs, ssims, sharps = [], [], []
        t0 = time.time()

        for sc, deg_p, wt, depth in items:
            clean_rgb = clean_cache[sc]
            mask_arr = mask_cache[sc]

            deg_bgr = cv2.imread(deg_p)
            deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Restoration stage
            if mode == "raw":
                res_rgb = deg_rgb
            else:
                deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)
                with torch.no_grad():
                    res_t = apply_rsuigm_batch_gpu(deg_t, wt_t, dp_t, tbl16)
                res_np = res_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

                if mode == "rsuigm_unsharp":
                    res_u8 = (np.clip(res_np, 0.0, 1.0) * 255.0).astype(np.uint8)
                    blur = cv2.GaussianBlur(res_u8, (0, 0), 2.0)
                    sharp_u8 = np.clip(cv2.addWeighted(res_u8, 1.34, blur, -0.34, 0), 0, 255)
                    res_rgb = sharp_u8.astype(np.float32) / 255.0
                else:
                    res_rgb = np.clip(res_np, 0.0, 1.0)

            # Restoration metrics
            p, _ = compute_psnr_and_norm(res_rgb, clean_rgb)
            s = compute_ssim_single(res_rgb, clean_rgb)
            sh = compute_sharpness_ratio(res_rgb, clean_rgb)
            psnrs.append(p)
            ssims.append(s)
            sharps.append(sh)

            # Segmentation stage (DINOv2-Large 518x518 with horizontal flip TTA)
            res_t = torch.from_numpy(res_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
            res_518 = F.interpolate(res_t, size=(518, 518), mode="bilinear", align_corners=False)
            x_norm = (res_518 - mean_im) / std_im
            x_flip = torch.flip(x_norm, dims=[3])

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_orig = model(x_norm)
                    l_flip = model(x_flip)
                p_orig = F.softmax(l_orig.float(), dim=1)
                p_flip = torch.flip(F.softmax(l_flip.float(), dim=1), dims=[3])
                p_fused = 0.5 * (p_orig + p_flip)
                p_1024 = F.interpolate(p_fused, size=(1024, 1024), mode="bilinear", align_corners=False)

            pred_mask = p_1024.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            evaluator.update(torch.from_numpy(pred_mask).unsqueeze(0), torch.from_numpy(mask_arr).unsqueeze(0))

        elapsed = time.time() - t0
        seg_res = evaluator.compute()

        mean_p = float(np.mean(psnrs))
        p_norm = float(np.clip((mean_p - 15.0) / 20.0, 0.0, 1.0))
        mean_s = float(np.mean(ssims))
        mean_sh = float(np.mean(sharps))
        q_fr = 0.45 * p_norm + 0.45 * mean_s + 0.10 * mean_sh
        q_seg = seg_res["Q_seg"]
        q_nr = 0.7417 # Locked server baseline value
        composite = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr

        results[cfg_name] = {
            "description": cfg["desc"],
            "psnr_db": round(mean_p, 2),
            "ssim": round(mean_s, 4),
            "sharpness": round(mean_sh, 4),
            "q_fr": round(q_fr, 4),
            "miou": round(seg_res["mIoU"], 4),
            "f1": round(seg_res["macro_F1"], 4),
            "q_seg": round(q_seg, 4),
            "composite": round(composite, 4),
            "per_class_iou": [round(float(v), 4) for v in seg_res["per_class_IoU"]]
        }

        print(f"  Restoration: PSNR={mean_p:.2f} dB, SSIM={mean_s:.4f}, Sharp={mean_sh:.4f}, Q_FR={q_fr:.4f}")
        print(f"  Segmentation: mIoU={seg_res['mIoU']:.4f}, F1={seg_res['macro_F1']:.4f}, Q_seg={q_seg:.4f}")
        print(f"  Composite Score: {composite:.4f}")
        print(f"  Per-Class IoU: {results[cfg_name]['per_class_iou']}")

    print("\n" + "=" * 80)
    print("PHASE 4: DOWNSTREAM SEGMENTATION COMPARATIVE SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'PSNR':<6} | {'SSIM':<6} | {'Q_FR':<6} | {'mIoU':<6} | {'F1':<6} | {'Q_seg':<6} | {'Composite':<9}")
    print("-" * 95)
    for k, v in results.items():
        print(f"{v['description'][:35]:<35} | {v['psnr_db']:<6.2f} | {v['ssim']:<6.4f} | {v['q_fr']:<6.4f} | {v['miou']:<6.4f} | {v['f1']:<6.4f} | {v['q_seg']:<6.4f} | {v['composite']:<9.4f}")
    print("=" * 80)

    out_json = "reports/PHASE4_DOWNSTREAM_SEGMENTATION_RESULTS.json"
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to: {out_json}")

    return results

if __name__ == "__main__":
    run_phase4_evaluation()
