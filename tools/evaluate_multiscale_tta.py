"""
Test Multi-Scale TTA vs Standard Single-Scale TTA on Fold 0 Held-Out Scenes
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

def benchmark_multiscale_fold0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("BENCHMARKING MULTI-SCALE TTA ON FOLD 0 HELD-OUT VALIDATION SCENES")
    print("=" * 72)

    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)

    f_meta = manifest["folds"]["fold_0"]
    train_scenes = f_meta["train_scenes"]
    val_scenes = f_meta["val_scenes"]

    ckpt_path = "checkpoints/track1_screening/DINOV2_CP_F0/best_model.pt"
    model = build_dinov2_model(num_classes=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rsuigm_table = fit_leak_free_rsuigm_table(train_scenes, device=device)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    eval_single = SegmentationEvaluator(num_classes=5)
    eval_multi = SegmentationEvaluator(num_classes=5)

    scales = [448, 518, 588]  # All divisible by 14
    weights = [0.25, 0.50, 0.25]

    data_root = "data/cached_1024"
    masks_dir = osp.join(data_root, "masks")
    deg_dir = osp.join(data_root, "degraded")

    # Evaluate on a stride-sample of 50 images per scene (200 images total) for quick comparison
    total_images = 0
    t0 = time.time()

    for sc in val_scenes:
        gt_mask = cv2.imread(osp.join(masks_dir, f"{sc}.png"), cv2.IMREAD_GRAYSCALE)
        sc_deg = osp.join(deg_dir, sc)
        deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])[::4] # 50 per scene

        for df in deg_files:
            parts = osp.splitext(df)[0].split("_")
            wt = int(parts[1])
            dp = int(parts[2])

            bgr = cv2.imread(osp.join(sc_deg, df))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h_orig, w_orig = rgb.shape[:2]

            x_t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([dp], device=device)

            with torch.no_grad():
                x_rest = apply_rsuigm_batch_gpu(x_t, wt_t, dp_t, rsuigm_table)
                x_norm = (x_rest - mean_im) / std_im

                # 1. Single-Scale (518) + HFlip
                x_518 = F.interpolate(x_norm, size=(518, 518), mode="bilinear", align_corners=False)
                x_518_flip = torch.flip(x_518, dims=[3])

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    log_518 = model(x_518)
                    log_518_flip = model(x_518_flip)

                p_single = 0.5 * F.softmax(log_518.float(), dim=1) + 0.5 * torch.flip(F.softmax(log_518_flip.float(), dim=1), dims=[3])
                p_single_interp = F.interpolate(p_single, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

                # 2. Multi-Scale ([448, 518, 588]) + HFlip
                p_multi_accum = torch.zeros((1, 5, h_orig, w_orig), device=device)
                for sc_sz, w_sc in zip(scales, weights):
                    x_sc = F.interpolate(x_norm, size=(sc_sz, sc_sz), mode="bilinear", align_corners=False)
                    x_sc_flip = torch.flip(x_sc, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        l_sc = model(x_sc)
                        l_sc_flip = model(x_sc_flip)

                    p_sc = 0.5 * F.softmax(l_sc.float(), dim=1) + 0.5 * torch.flip(F.softmax(l_sc_flip.float(), dim=1), dims=[3])
                    p_sc_orig = F.interpolate(p_sc, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
                    p_multi_accum += w_sc * p_sc_orig

            y_t = torch.from_numpy(gt_mask.astype(np.int64)).long().unsqueeze(0).to(device)

            eval_single.update(p_single_interp, y_t)
            eval_multi.update(p_multi_accum, y_t)
            total_images += 1

    res_single = eval_single.compute()
    res_multi = eval_multi.compute()

    print(f"\nEvaluated {total_images} images in {time.time() - t0:.1f}s.")
    print("-" * 72)
    print(f"Single-Scale (518 + HFlip):")
    print(f"  mIoU: {res_single['mIoU']:.4f} | Macro F1: {res_single['macro_F1']:.4f} | Q_seg: {res_single['Q_seg']:.4f}")
    print(f"  Per-Class IoU: {res_single['per_class_IoU']}")
    print("-" * 72)
    print(f"Multi-Scale ([448, 518, 588] + HFlip):")
    print(f"  mIoU: {res_multi['mIoU']:.4f} | Macro F1: {res_multi['macro_F1']:.4f} | Q_seg: {res_multi['Q_seg']:.4f}")
    print(f"  Per-Class IoU: {res_multi['per_class_IoU']}")
    print("-" * 72)
    delta_q = res_multi['Q_seg'] - res_single['Q_seg']
    print(f"Delta Q_seg: {delta_q:+.4f} ({'IMPROVEMENT' if delta_q > 0 else 'NEUTRAL/DROP'})")

if __name__ == "__main__":
    benchmark_multiscale_fold0()
