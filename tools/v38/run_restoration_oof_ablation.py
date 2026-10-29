"""
SAT-EURS Track 1 — V38 Strict 5-Fold Scene-Disjoint OOF Restoration Ablation Suite
Evaluates strictly out-of-fold across all 20 scenes (splits/manifest_5fold.json):
  - R0: Baseline 19-term fold table + ConditionedRestorationRefinerV18 + unsharp (0.10, 2.0)
  - R1: Refiner Horizontal-Flip TTA: 0.5 * (refiner(x) + flip(refiner(flip(x))))
  - R2: Expanded 35-term per-image physics/spatial/veil-normalized basis (fit strictly on 16 train scenes/fold)
        blended with neural refiner at weights w in {0.15, 0.25, 0.35, 1.00}
  - R3: Unsharp mask calibration (alpha in {0.0, 0.06, 0.10, 0.14}, sigma in {1.5, 2.0})
Measures exact official Q_FR = 0.45 * PSNR_norm + 0.45 * SSIM + 0.10 * Sharpness.
"""

import os
import os.path as osp
import sys
import time
import json
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    V18CanonicalPreprocessor,
)
from metrics.restoration_metrics import (
    compute_psnr_and_norm,
    compute_ssim_single,
    compute_sharpness_ratio,
)
from tools.exp4_1_rsuigm_scale_consistent import get_stratified_sample_indices


def build_rsuigm_35_features_gpu(x: torch.Tensor) -> torch.Tensor:
    """
    Constructs a 35-term per-image physics, multi-scale spatial, and global-veil-normalized
    feature basis for a batch of degraded RGB images x of shape [B, 3, H, W] in [0, 1].
    Strictly single-image: uses only statistics of x[b] itself.
    Returns tensor of shape [B, H*W, 35].
    """
    B, C, H, W = x.shape
    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]

    ones = torch.ones_like(r)
    # 1. Polynomial terms (13 terms)
    r2, g2, b2 = r * r, g * g, b * b
    rg, rb, gb = r * g, r * b, g * b
    r3, g3, b3 = r2 * r, g2 * g, b2 * b

    # 2. Multi-scale spatial context pools (12 terms: 4, 16, 64, 128)
    def pool_up(t: torch.Tensor, k: int) -> torch.Tensor:
        kh = min(k, H)
        kw = min(k, W)
        p = F.avg_pool2d(t, kernel_size=(kh, kw), stride=(kh, kw))
        return F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)

    p4 = pool_up(x, 4)
    p16 = pool_up(x, 16)
    p64 = pool_up(x, 64)
    p128 = pool_up(x, 128)

    # 3. Per-image global mean, std, and standardized RGB (9 terms)
    mu = x.mean(dim=(2, 3), keepdim=True)
    sigma = x.std(dim=(2, 3), keepdim=True)
    x_norm = (x - mu) / (sigma + 0.05)
    mu_b = mu.expand(B, 3, H, W)
    sig_b = sigma.expand(B, 3, H, W)

    # 4. Vertical water-column coordinate y/H (1 term)
    y_coord = torch.linspace(0.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)

    feat_map = torch.cat([
        ones, r, g, b,
        r2, g2, b2, rg, rb, gb,
        r3, g3, b3,
        p16, p64,
        p4, p128,
        mu_b, sig_b, x_norm,
        y_coord
    ], dim=1)  # [B, 35, H, W]

    return feat_map.permute(0, 2, 3, 1).reshape(B, H * W, 35)


def apply_table35_gpu(x: torch.Tensor, wt: int, depth: int, table35: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
    B, C, H, W = x.shape
    W_mat = table35[(wt, depth)]  # [35, 3]
    feats = build_rsuigm_35_features_gpu(x)  # [B, H*W, 35]
    out = torch.matmul(feats, W_mat)  # [B, H*W, 3]
    out = out.view(B, H, W, 3).permute(0, 3, 1, 2)
    return torch.clamp(out, 0.0, 1.0)


def fit_table35_on_scenes(
    scenes: List[str],
    device: torch.device,
    reg_lambda: float = 5e-2,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Fits the 35-term per-(wt, depth) ridge regression operator strictly on the provided scenes.
    Uses downsampled 128x128 spatial resolution on GPU so fitting all 200 conditions takes ~3 seconds per fold.
    """
    clean_imgs = []
    for sc in scenes:
        bgr = cv2.imread(f"data/cached_1024/clean/{sc}.png", cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if rgb.shape[:2] != (1024, 1024):
            rgb = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        clean_imgs.append(rgb)
    clean_gpu = torch.from_numpy(np.stack(clean_imgs, axis=0)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    clean_128 = F.avg_pool2d(clean_gpu, kernel_size=8, stride=8)
    Y_target = clean_128.permute(0, 2, 3, 1).reshape(-1, 3)

    reg_eye = reg_lambda * torch.eye(35, device=device, dtype=torch.float32)
    reg_eye[0, 0] = 1e-5  # minimal regularization on intercept
    table35: Dict[Tuple[int, int], torch.Tensor] = {}

    for wt in range(1, 11):
        for depth in range(5, 105, 5):
            deg_list = []
            for sc in scenes:
                p = f"data/cached_1024/degraded/{sc}/{sc}_{wt}_{depth}.png"
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (1024, 1024):
                    rgb = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                deg_list.append(rgb)
            deg_gpu = torch.from_numpy(np.stack(deg_list, axis=0)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            deg_128 = F.avg_pool2d(deg_gpu, kernel_size=8, stride=8)
            with torch.no_grad():
                feats = build_rsuigm_35_features_gpu(deg_128).reshape(-1, 35)
                XtX = torch.matmul(feats.T, feats) + reg_eye
                XtY = torch.matmul(feats.T, Y_target)
                W = torch.linalg.solve(XtX, XtY)
                table35[(wt, depth)] = W
    return table35


def unsharp(img: np.ndarray, alpha: float, sigma: float = 2.0) -> np.ndarray:
    if alpha <= 0.0:
        return np.clip(img, 0.0, 1.0)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(img + alpha * (img - blur), 0.0, 1.0)


def eval_single_pair(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float, float, float]:
    psnr, psnr_norm = compute_psnr_and_norm(pred, target)
    ssim_val = compute_ssim_single(pred, target)
    sharp = compute_sharpness_ratio(pred, target)
    q_fr = 0.45 * psnr_norm + 0.45 * ssim_val + 0.10 * sharp
    return psnr, psnr_norm, ssim_val, sharp, q_fr


def run_restoration_ablation(sample_stride: int = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 100, flush=True)
    print(f"SAT-EURS TRACK 1 — V38 5-FOLD OOF RESTORATION ABLATION SUITE (stride={sample_stride})", flush=True)
    print("=" * 100, flush=True)

    manifest = validate_manifest_and_folds()
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    accum: Dict[str, List[Tuple[float, float, float, float, float]]] = {}

    def record(name: str, pred: np.ndarray, clean: np.ndarray):
        if name not in accum:
            accum[name] = []
        accum[name].append(eval_single_pair(pred, clean))

    t0 = time.time()
    total_imgs = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        train_scenes = sorted(f_data["train_scenes"])
        val_scenes = sorted(f_data["val_scenes"])
        tf = time.time()
        print(f"\n[FOLD {fold}] Fitting leak-free 35-term table on 16 train scenes & evaluating on {val_scenes}...", flush=True)

        tbl35_fold = fit_table35_on_scenes(train_scenes, device=device, reg_lambda=5e-2)
        print(f"  Fit tbl35_fold_{fold} in {time.time() - tf:.1f}s", flush=True)

        tbl19_fold = torch.load(f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt", map_location=device, weights_only=False)
        refiner = build_v18_restoration_refiner(rsuigm_table=tbl19_fold, use_neural_refiner=True).to(device)
        r_obj = torch.load(f"checkpoints/v18/fold_{fold}/refiner_best.pt", map_location=device, weights_only=False)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        for sc in val_scenes:
            clean_bgr = cv2.imread(f"data/cached_1024/clean/{sc}.png", cv2.IMREAD_COLOR)
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            deg_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::sample_stride]

            for df in deg_files:
                total_imgs += 1
                parts = osp.splitext(df)[0].split("_")
                wt, depth = int(parts[1]), int(parts[2])

                bgr = cv2.imread(osp.join(deg_dir, df), cv2.IMREAD_COLOR)
                prep = preprocessor.prepare_input(bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], dtype=torch.float32, device=device)

                with torch.no_grad():
                    r_orig = refiner(raw_pad, wt_t, dp_t)["refined_image"]
                    r_flip = torch.flip(refiner(torch.flip(raw_pad, dims=[3]), wt_t, dp_t)["refined_image"], dims=[3])
                    r_tta = 0.5 * (r_orig + r_flip)
                    r_t35 = apply_table35_gpu(raw_pad, wt, depth, tbl35_fold)

                np_orig = preprocessor.unpad_output(r_orig.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                np_tta = preprocessor.unpad_output(r_tta.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                np_t35 = preprocessor.unpad_output(r_t35.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)

                if np_orig.shape[:2] != clean_rgb.shape[:2]:
                    np_orig = cv2.resize(np_orig, (clean_rgb.shape[1], clean_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                    np_tta = cv2.resize(np_tta, (clean_rgb.shape[1], clean_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                    np_t35 = cv2.resize(np_t35, (clean_rgb.shape[1], clean_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

                # Evaluate candidates
                record("R0_Base_NoUnsharp", np_orig, clean_rgb)
                record("R0_Base_Unsharp_010_s20 (V33)", unsharp(np_orig, 0.10, 2.0), clean_rgb)
                record("R1_FlipTTA_NoUnsharp", np_tta, clean_rgb)
                record("R1_FlipTTA_Unsharp_010_s20", unsharp(np_tta, 0.10, 2.0), clean_rgb)
                record("R1_FlipTTA_Unsharp_012_s20", unsharp(np_tta, 0.12, 2.0), clean_rgb)
                record("R1_FlipTTA_Unsharp_008_s15", unsharp(np_tta, 0.08, 1.5), clean_rgb)
                record("R2_Tbl35_Alone", np_t35, clean_rgb)
                record("R2_Blend15_Tbl35_FlipTTA_U010", unsharp(0.85 * np_tta + 0.15 * np_t35, 0.10, 2.0), clean_rgb)
                record("R2_Blend25_Tbl35_FlipTTA_U010", unsharp(0.75 * np_tta + 0.25 * np_t35, 0.10, 2.0), clean_rgb)

    print("\n" + "=" * 105, flush=True)
    print(f"GLOBAL 5-FOLD OOF RESTORATION RESULTS ({total_imgs} images, {time.time() - t0:.1f}s)", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Configuration':35s} | {'Q_FR':8s} | {'dQ_FR':8s} | {'PSNR(dB)':9s} | {'PSNR_n':8s} | {'SSIM':8s} | {'Sharp':8s}", flush=True)
    print("-" * 105, flush=True)

    summary = {}
    for k, vals in accum.items():
        arr = np.array(vals)
        summary[k] = {
            "PSNR": round(float(arr[:, 0].mean()), 4),
            "PSNR_norm": round(float(arr[:, 1].mean()), 5),
            "SSIM": round(float(arr[:, 2].mean()), 5),
            "Sharpness": round(float(arr[:, 3].mean()), 5),
            "Q_FR": round(float(arr[:, 4].mean()), 5),
        }

    base_qfr = summary["R0_Base_Unsharp_010_s20 (V33)"]["Q_FR"]
    for k, m in summary.items():
        dq = m["Q_FR"] - base_qfr
        print(f"{k:35s} | {m['Q_FR']:8.5f} | {dq:+8.5f} | {m['PSNR']:9.4f} | {m['PSNR_norm']:8.5f} | {m['SSIM']:8.5f} | {m['Sharpness']:8.5f}", flush=True)

    os.makedirs("reports/v38", exist_ok=True)
    with open("reports/v38/oof_restoration_ablation_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_restoration_ablation(sample_stride=10)
