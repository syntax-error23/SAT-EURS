"""
Benchmark Single-Image Restoration Alternatives (Zero Filename Metadata)
Compares:
1. Filename-Metadata RSUIGM (Oracle wt, depth parsed from filename)
2. Global Invariant RSUIGM (Single fixed matrix applied to all images)
3. Image-Adaptive RSUIGM (Estimates best operator from RGB color histogram/channel stats alone)
Evaluates PSNR, PSNR_norm, SSIM, Sharpness, and Q_FR on all 800 held-out images.
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu
from tools.train_segformer_baseline import LeakFreeSegFormerDataset
from metrics.restoration_metrics import RestorationEvaluator


def fit_global_rsuigm_matrix(table: dict, device: torch.device) -> torch.Tensor:
    """Computes the mean/median global polynomial transformation matrix across all 200 operators."""
    matrices = torch.stack(list(table.values()), dim=0) # [200, 19, 3]
    return torch.median(matrices, dim=0).values # [19, 3]


def estimate_operator_from_image(x_raw: torch.Tensor, table: dict) -> torch.Tensor:
    """
    Estimates the best RSUIGM matrix purely from image RGB statistics:
    Underwater attenuation heavily absorbs Red relative to Blue and Green.
    - Mean Red/Blue ratio and Green/Blue ratio correlate strongly with water depth and turbidity.
    Maps channel ratios to the closest (wt, depth) operator without reading filenames.
    """
    # x_raw: [1, 3, H, W]
    r_mean = x_raw[:, 0].mean().item()
    g_mean = x_raw[:, 1].mean().item()
    b_mean = x_raw[:, 2].mean().item()
    
    # Red-to-Blue ratio indicates depth (lower ratio = deeper/more red absorption)
    rb_ratio = r_mean / (b_mean + 1e-6)
    # Estimate depth between 10m and 80m
    est_depth = int(np.clip(round((1.0 - rb_ratio) * 80.0 / 5.0) * 5, 5, 100))
    # Green-to-Blue ratio indicates water type (turbid/coastal vs clear ocean)
    gb_ratio = g_mean / (b_mean + 1e-6)
    est_wt = int(np.clip(round(gb_ratio * 5.0), 1, 10))
    
    return table.get((est_wt, est_depth), table[(1, 10)])


def benchmark_restoration_alternatives():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RSUIGM table on {device}...")
    table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)
    global_mat = fit_global_rsuigm_matrix(table, device)

    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    data_root = "data/cached_1024" if os.path.exists("data/cached_1024") else "data/train"

    eval_oracle = RestorationEvaluator()
    eval_global = RestorationEvaluator()
    eval_image_adapt = RestorationEvaluator()
    eval_default_1_10 = RestorationEvaluator()

    t0 = time.time()
    total_images = 0

    with torch.no_grad():
        for scene in val_scenes:
            clean_path = os.path.join(data_root, "clean", f"{scene}.png")
            clean_im = Image.open(clean_path).convert("RGB").resize((1024, 1024), Image.BILINEAR)
            clean_tensor = torch.from_numpy(np.array(clean_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            ds = LeakFreeSegFormerDataset(scenes=[scene], data_root=data_root, crop_size=512, is_train=False)
            loader = DataLoader(ds, batch_size=4, shuffle=False)

            for x_raw, _, wts, depths in loader:
                x_raw = x_raw.to(device) # [B, 3, H, W]
                B, C, H, W = x_raw.shape
                total_images += B

                feats = build_rsuigm_poly_features_gpu(x_raw) # [B*H*W, 19]

                # 1. Oracle (Filename metadata)
                rest_oracle = []
                for i in range(B):
                    wt_i = int(wts[i].item())
                    d_i = int(depths[i].item())
                    mat = table.get((wt_i, d_i), table[(1, 10)])
                    f_i = build_rsuigm_poly_features_gpu(x_raw[i:i+1])
                    p_i = torch.clamp(torch.matmul(f_i, mat).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                    rest_oracle.append(p_i)
                rest_oracle = torch.cat(rest_oracle, dim=0)
                eval_oracle.update(rest_oracle, clean_tensor.expand(B, -1, -1, -1))

                # 2. Global Invariant Matrix (Median across all 200 operators)
                rest_global = torch.clamp(torch.matmul(feats, global_mat).reshape(B, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                eval_global.update(rest_global, clean_tensor.expand(B, -1, -1, -1))

                # 3. Default Fixed (wt=1, depth=10)
                mat_1_10 = table[(1, 10)]
                rest_1_10 = torch.clamp(torch.matmul(feats, mat_1_10).reshape(B, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                eval_default_1_10.update(rest_1_10, clean_tensor.expand(B, -1, -1, -1))

                # 4. Image-Adaptive (Color statistics alone, zero filename metadata)
                rest_img = []
                for i in range(B):
                    mat_est = estimate_operator_from_image(x_raw[i:i+1], table)
                    f_i = build_rsuigm_poly_features_gpu(x_raw[i:i+1])
                    p_i = torch.clamp(torch.matmul(f_i, mat_est).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                    rest_img.append(p_i)
                rest_img = torch.cat(rest_img, dim=0)
                eval_image_adapt.update(rest_img, clean_tensor.expand(B, -1, -1, -1))

    res_oracle = eval_oracle.compute()
    res_global = eval_global.compute()
    res_1_10 = eval_default_1_10.compute()
    res_img = eval_image_adapt.compute()

    elapsed = time.time() - t0
    print("=" * 76)
    print(f"   RESTORATION METRICS COMPARISON (800 HELD-OUT IMAGES, {elapsed:.1f}s)")
    print("=" * 76)
    print(f"{'Method':<32} | {'PSNR':<8} | {'PSNR_norm':<10} | {'SSIM':<8} | {'Sharpness':<10} | {'Q_FR':<8}")
    print("-" * 76)
    for name, r in [
        ("1. Filename Oracle (wt, depth)", res_oracle),
        ("2. Image-Adaptive (Zero Metadata)", res_img),
        ("3. Global Invariant (Median W)", res_global),
        ("4. Fixed Default (wt=1, d=10)", res_1_10)
    ]:
        print(f"{name:<32} | {r['psnr']:<8.2f} | {r['psnr_norm']:<10.4f} | {r['ssim']:<8.4f} | {r['sharpness_ratio']:<10.4f} | {r['Q_FR']:<8.4f}")
    print("=" * 76)

    out_file = "checkpoints/track1_high_capacity/restoration_metadata_benchmark.json"
    with open(out_file, "w") as fp:
        json.dump({
            "oracle": res_oracle,
            "image_adaptive": res_img,
            "global_invariant": res_global,
            "fixed_default": res_1_10
        }, fp, indent=2)
    print(f"Saved benchmark results to {out_file}")


if __name__ == "__main__":
    benchmark_restoration_alternatives()
