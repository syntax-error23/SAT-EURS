"""
100% CUDA-Accelerated Benchmark of Single-Image Restoration Alternatives (Zero Filename Metadata)
Evaluates 800 held-out images:
  1. Filename Oracle: W[(wt, depth)] parsed from filename
  2. Single-Image Adaptive: Estimates (wt, depth) from image RGB color ratios alone (Zero metadata)
  3. Global Invariant Matrix: Single median matrix across all 200 operators (Zero metadata)
  4. Fixed Default Matrix: W[(1, 10)] (Zero metadata)
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu
from tools.train_segformer_baseline import LeakFreeSegFormerDataset
from torch.utils.data import DataLoader


def run_benchmark():
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    print("=" * 76)
    print("   100% CUDA BENCHMARK: SINGLE-IMAGE RESTORATION ALTERNATIVES")
    print("=" * 76)

    table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)
    # Median global matrix
    all_mats = torch.stack(list(table.values()), dim=0) # [200, 19, 3]
    global_mat = torch.median(all_mats, dim=0).values # [19, 3]
    mat_1_10 = table[(1, 10)]

    # GPU Kernels
    def get_gaussian_kernel2d(kernel_size=11, sigma=1.5):
        k = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        gauss = torch.exp(-k**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        kernel2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        return kernel2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    g_kernel = get_gaussian_kernel2d()
    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    def ssim_gpu(pred, target):
        c1, c2 = 0.01**2, 0.03**2
        mu_p = F.conv2d(pred, g_kernel, padding=5, groups=3)
        mu_t = F.conv2d(target, g_kernel, padding=5, groups=3)
        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_pt = mu_p * mu_t
        sigma_p_sq = F.conv2d(pred * pred, g_kernel, padding=5, groups=3) - mu_p_sq
        sigma_t_sq = F.conv2d(target * target, g_kernel, padding=5, groups=3) - mu_t_sq
        sigma_pt = F.conv2d(pred * target, g_kernel, padding=5, groups=3) - mu_pt
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2) + 1e-10)
        return ssim_map.mean(dim=(1, 2, 3))

    def laplacian_var_gpu(img):
        gray = (0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]) * 255.0
        lap = F.conv2d(gray, lap_kernel, padding=1)
        return lap.var(dim=(1, 2, 3))

    methods = ["oracle", "image_adaptive", "global_median", "fixed_1_10"]
    stats = {m: {"psnr_list": [], "ssim_list": [], "sharp_list": []} for m in methods}

    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    data_root = "data/cached_1024" if os.path.exists("data/cached_1024") else "data/train"

    t0 = time.time()
    with torch.no_grad():
        for scene in val_scenes:
            clean_path = os.path.join(data_root, "clean", f"{scene}.png")
            clean_im = Image.open(clean_path).convert("RGB").resize((1024, 1024), Image.BILINEAR)
            clean_t = torch.from_numpy(np.array(clean_im, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            var_t = laplacian_var_gpu(clean_t)

            ds = LeakFreeSegFormerDataset(scenes=[scene], data_root=data_root, crop_size=512, is_train=False)
            loader = DataLoader(ds, batch_size=8, shuffle=False)

            for x_raw, _, wts, depths in loader:
                x_raw = x_raw.to(device)
                B, C, H, W = x_raw.shape
                clean_batch = clean_t.expand(B, -1, -1, -1)
                var_t_batch = var_t.expand(B)

                feats = build_rsuigm_poly_features_gpu(x_raw) # [B, H*W, 19]

                # 1. Oracle (wt, depth from filename)
                rest_oracle_list = []
                # 2. Image-adaptive (wt, depth from image color stats)
                rest_adapt_list = []

                for i in range(B):
                    # Oracle
                    wt_i = int(wts[i].item())
                    d_i = int(depths[i].item())
                    mat_o = table.get((wt_i, d_i), mat_1_10)
                    f_i = feats[i] # [H*W, 19]
                    p_o = torch.clamp(torch.matmul(f_i, mat_o).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                    rest_oracle_list.append(p_o)

                    # Adaptive from image
                    r_m = x_raw[i, 0].mean().item()
                    g_m = x_raw[i, 1].mean().item()
                    b_m = x_raw[i, 2].mean().item()
                    rb = r_m / (b_m + 1e-6)
                    gb = g_m / (b_m + 1e-6)
                    est_d = int(np.clip(round((1.0 - rb) * 80.0 / 5.0) * 5, 5, 100))
                    est_w = int(np.clip(round(gb * 5.0), 1, 10))
                    mat_a = table.get((est_w, est_d), mat_1_10)
                    p_a = torch.clamp(torch.matmul(f_i, mat_a).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                    rest_adapt_list.append(p_a)

                rest_oracle = torch.cat(rest_oracle_list, dim=0)
                rest_adapt = torch.cat(rest_adapt_list, dim=0)
                rest_global = torch.clamp(torch.matmul(feats, global_mat).reshape(B, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
                rest_fixed = torch.clamp(torch.matmul(feats, mat_1_10).reshape(B, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)

                for m_name, rest_batch in [
                    ("oracle", rest_oracle),
                    ("image_adaptive", rest_adapt),
                    ("global_median", rest_global),
                    ("fixed_1_10", rest_fixed)
                ]:
                    mse = F.mse_loss(rest_batch, clean_batch, reduction='none').mean(dim=(1, 2, 3))
                    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-10))
                    ssim = ssim_gpu(rest_batch, clean_batch)
                    var_p = laplacian_var_gpu(rest_batch)
                    sharp = torch.clamp(var_p / (var_t_batch + 1e-7), 0.0, 1.0)

                    stats[m_name]["psnr_list"].extend(psnr.cpu().tolist())
                    stats[m_name]["ssim_list"].extend(ssim.cpu().tolist())
                    stats[m_name]["sharp_list"].extend(sharp.cpu().tolist())

    elapsed = time.time() - t0
    results = {}
    print("\n" + "=" * 80)
    print(f"   RESTORATION METRICS BENCHMARK: METADATA ORACLE VS. ZERO-METADATA ({elapsed:.1f}s)")
    print("=" * 80)
    print(f"{'Restoration Method':<36} | {'PSNR':<8} | {'PSNR_norm':<10} | {'SSIM':<8} | {'Sharpness':<10} | {'Q_FR':<8}")
    print("-" * 80)

    for m_name, label in [
        ("oracle", "1. Filename Oracle (wt, depth)"),
        ("image_adaptive", "2. Single-Image Adaptive (RGB Stats)"),
        ("global_median", "3. Global Median Operator"),
        ("fixed_1_10", "4. Default Fixed Operator (1, 10)")
    ]:
        p = float(np.mean(stats[m_name]["psnr_list"]))
        p_norm = float(np.clip((p - 15.0) / 20.0, 0.0, 1.0))
        s = float(np.mean(stats[m_name]["ssim_list"]))
        sh = float(np.mean(stats[m_name]["sharp_list"]))
        q_fr = 0.45 * p_norm + 0.45 * s + 0.10 * sh

        results[m_name] = {
            "name": label,
            "psnr": round(p, 2),
            "psnr_norm": round(p_norm, 4),
            "ssim": round(s, 4),
            "sharpness_ratio": round(sh, 4),
            "Q_FR": round(q_fr, 4)
        }
        print(f"{label:<36} | {p:<8.2f} | {p_norm:<10.4f} | {s:<8.4f} | {sh:<10.4f} | {q_fr:<8.4f}")
    print("=" * 80)

    with open("checkpoints/track1_high_capacity/restoration_metadata_benchmark.json", "w") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    run_benchmark()
