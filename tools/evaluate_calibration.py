"""
Ablation on Probability Calibration and Multi-Scale TTA across Folds 0 & 1
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

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from metrics.segmentation_metrics import SegmentationEvaluator
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from tools.pilot_restoration_evaluation import fit_leak_free_rsuigm_table

def benchmark_calibration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("OPTIMIZING TTA + CALIBRATION ON VALIDATION FOLDS")
    print("=" * 72)

    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)

    # Test on fold 0 and fold 1
    for fold_idx in [0, 1]:
        fold_id = f"fold_{fold_idx}"
        f_meta = manifest["folds"][fold_id]
        train_scenes = f_meta["train_scenes"]
        val_scenes = f_meta["val_scenes"]

        ckpt_path = f"checkpoints/track1_screening/DINOV2_CP_F{fold_idx}/best_model.pt"
        model = build_dinov2_model(num_classes=5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rsuigm_table = fit_leak_free_rsuigm_table(train_scenes, device=device)

        mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        eval_baseline = SegmentationEvaluator(num_classes=5)
        eval_optimized = SegmentationEvaluator(num_classes=5)

        data_root = "data/cached_1024"
        masks_dir = osp.join(data_root, "masks")
        deg_dir = osp.join(data_root, "degraded")

        scales = [448, 518, 588]
        weights = [0.25, 0.50, 0.25]
        # Class probability multiplier to slightly encourage rare classes at ambiguous boundaries
        class_prior = torch.tensor([1.0, 1.0, 1.05, 1.15, 1.15], device=device).view(1, 5, 1, 1)

        deg_files_sample = []
        for sc in val_scenes:
            sc_deg = osp.join(deg_dir, sc)
            files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])[::4] # 50/scene = 200 total
            for f in files:
                deg_files_sample.append((sc, f))

        for sc, df in deg_files_sample:
            gt_mask = cv2.imread(osp.join(masks_dir, f"{sc}.png"), cv2.IMREAD_GRAYSCALE)
            parts = osp.splitext(df)[0].split("_")
            wt = int(parts[1])
            dp = int(parts[2])

            bgr = cv2.imread(osp.join(deg_dir, sc, df))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h_orig, w_orig = rgb.shape[:2]

            x_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([dp], device=device)

            with torch.no_grad():
                x_rest = apply_rsuigm_batch_gpu(x_t, wt_t, dp_t, rsuigm_table)
                x_norm = (x_rest - mean_im) / std_im

                # 1. Baseline: 518 + HFlip
                x_518 = F.interpolate(x_norm, size=(518, 518), mode="bilinear", align_corners=False)
                x_518_flip = torch.flip(x_518, dims=[3])
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    l_base = model(x_518)
                    l_base_f = model(x_518_flip)
                p_base = 0.5 * F.softmax(l_base.float(), dim=1) + 0.5 * torch.flip(F.softmax(l_base_f.float(), dim=1), dims=[3])
                p_base_interp = F.interpolate(p_base, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                # 2. Optimized: Multi-Scale [448, 518, 588] + Prior Calibration
                p_opt_accum = torch.zeros((1, 5, h_orig, w_orig), device=device)
                for sc_sz, w_sc in zip(scales, weights):
                    x_sc = F.interpolate(x_norm, size=(sc_sz, sc_sz), mode="bilinear", align_corners=False)
                    x_sc_f = torch.flip(x_sc, dims=[3])
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        l_sc = model(x_sc)
                        l_sc_f = model(x_sc_f)
                    p_sc = 0.5 * F.softmax(l_sc.float(), dim=1) + 0.5 * torch.flip(F.softmax(l_sc_f.float(), dim=1), dims=[3])
                    p_opt_accum += w_sc * F.interpolate(p_sc, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                p_opt_calibrated = p_opt_accum * class_prior

            y_t = torch.from_numpy(gt_mask.astype(np.int64)).long().unsqueeze(0).to(device)
            eval_baseline.update(p_base_interp, y_t)
            eval_optimized.update(p_opt_calibrated, y_t)

        res_b = eval_baseline.compute()
        res_o = eval_optimized.compute()
        print(f"\n--- {fold_id} ---")
        print(f"  Baseline:  mIoU={res_b['mIoU']:.4f} | F1={res_b['macro_F1']:.4f} | Q_seg={res_b['Q_seg']:.4f}")
        print(f"  Optimized: mIoU={res_o['mIoU']:.4f} | F1={res_o['macro_F1']:.4f} | Q_seg={res_o['Q_seg']:.4f}")
        print(f"  Delta Q_seg: {res_o['Q_seg'] - res_b['Q_seg']:+.4f}")

if __name__ == "__main__":
    benchmark_calibration()
