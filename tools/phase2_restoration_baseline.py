"""
Phase 2 — Establish the Restoration-Only Baseline
MaCVi @ WACV 2027 Challenge — SAT-EURS Track 1
Hardware: Windows 11 | RTX 5090 32GB | 64GB DDR5

Evaluates the physical restoration pipeline on held-out Fold 0 scenes:
Val scenes: ['fish1', 'fish11', 'fish12', 'seahorse1']

Configurations Evaluated:
1. Raw Degraded (Lower Bound Control)
2. Production RSUIGM Table (tbl19_20sc_clean.pt) RAW (no unsharp)
3. Production RSUIGM Table (tbl19_20sc_clean.pt) + Unsharp Mask 1.34 (Historical V16/V17 comparator)
4. Leak-Free Fold 0 RSUIGM Table (tbl19_16sc_fold0.pt) + Unsharp Mask 1.34 (Scientific OOF Control)

Outputs:
- Full Reference: PSNR (dB), PSNR_norm, SSIM, Sharpness Ratio, Q_FR
- No Reference: UIQM, UCIQE
- Visual Artifacts: Pixel Clipping %, Dynamic Range, Saturation
- System Profile: GPU Memory (allocated/reserved), Throughput (FPS), Latency (ms)
"""

import os
import os.path as osp
import sys
import json
import time
import math
import psutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, ".")

from tools.train_track1_screening import apply_rsuigm_batch_gpu
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio
from tools.diagnose_v6_bottlenecks import compute_uiqm, compute_uciqe


def run_phase2_baseline():
    print("=" * 80)
    print("PHASE 2: ESTABLISHING RESTORATION-ONLY BASELINE ON HELD-OUT VAL SCENES")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA required for Phase 2 benchmark!"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Fold 0 Held-Out Scenes
    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    data_root = "data/cached_1024"
    clean_dir = osp.join(data_root, "clean")
    deg_dir = osp.join(data_root, "degraded")

    print(f"\n[Step 1/4] Loading Clean Ground Truths for Val Scenes: {val_scenes}...")
    clean_cache = {}
    for sc in val_scenes:
        p = osp.join(clean_dir, f"{sc}.png")
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_cache[sc] = rgb

    # Build evaluation item list (stride 5 -> 160 representative images covering all 10 WT x 20 depths)
    items = []
    for sc in val_scenes:
        sc_p = osp.join(deg_dir, sc)
        deg_files = sorted([f for f in os.listdir(sc_p) if f.endswith(".png")])
        for df in deg_files[::5]: # Stride of 5: 40 images per scene x 4 scenes = 160 images
            stem = osp.splitext(df)[0]
            parts = stem.split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
            items.append((sc, osp.join(sc_p, df), wt, depth))

    print(f"  Sampled {len(items)} evaluation images (40 per scene across 10 water types and 20 depths)")

    # 2. Load Physical Restoration Tables
    print("\n[Step 2/4] Loading Calibration Tables...")
    tbl_20_path = "checkpoints/track1_screening/tbl19_20sc_clean.pt"
    tbl_16_path = "checkpoints/track1_screening/tbl19_16sc_fold0.pt"

    assert osp.exists(tbl_20_path), f"Missing {tbl_20_path}"
    assert osp.exists(tbl_16_path), f"Missing {tbl_16_path}"

    table_20 = torch.load(tbl_20_path, map_location=device)
    table_16 = torch.load(tbl_16_path, map_location=device)
    print(f"  Loaded Production 20-scene table: {tbl_20_path}")
    print(f"  Loaded Leak-Free Fold 0 table:    {tbl_16_path}")

    # 3. Define Configurations to Benchmark
    configs = [
        {"name": "raw_degraded", "table": None, "unsharp": False, "desc": "Raw Degraded Input (Baseline Floor)"},
        {"name": "rsuigm_20sc_raw", "table": table_20, "unsharp": False, "desc": "Production RSUIGM Table (No Unsharp)"},
        {"name": "rsuigm_20sc_unsharp134", "table": table_20, "unsharp": True, "desc": "Production RSUIGM + Unsharp 1.34 (Official #18747/#18752 Reference)"},
        {"name": "rsuigm_16sc_unsharp134", "table": table_16, "unsharp": True, "desc": "Leak-Free Fold 0 RSUIGM + Unsharp 1.34 (Strict Scientific Baseline)"},
    ]

    results = {}

    print("\n[Step 3/4] Benchmarking Configurations...")
    for cfg in configs:
        cfg_name = cfg["name"]
        tbl = cfg["table"]
        use_unsharp = cfg["unsharp"]
        print(f"\n  --> Evaluating: {cfg['desc']}...")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        psnrs, ssims, sharps, uiqms, uciqes = [], [], [], [], []
        clippings = []
        latencies = []

        t_start = time.time()

        for idx, (sc, deg_p, wt, depth) in enumerate(items):
            clean_rgb = clean_cache[sc]
            deg_bgr = cv2.imread(deg_p)
            deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if tbl is None:
                # Raw degraded
                res_rgb = deg_rgb
            else:
                deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)

                with torch.no_grad():
                    res_t = apply_rsuigm_batch_gpu(deg_t, wt_t, dp_t, tbl)

                res_np = res_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

                if use_unsharp:
                    res_u8 = (np.clip(res_np, 0.0, 1.0) * 255.0).astype(np.uint8)
                    blur = cv2.GaussianBlur(res_u8, (0, 0), 2.0)
                    sharp_u8 = np.clip(cv2.addWeighted(res_u8, 1.34, blur, -0.34, 0), 0, 255)
                    res_rgb = sharp_u8.astype(np.float32) / 255.0
                else:
                    res_rgb = np.clip(res_np, 0.0, 1.0)

            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            # Full-reference metrics
            p, _ = compute_psnr_and_norm(res_rgb, clean_rgb)
            s = compute_ssim_single(res_rgb, clean_rgb)
            sh = compute_sharpness_ratio(res_rgb, clean_rgb)
            psnrs.append(p)
            ssims.append(s)
            sharps.append(sh)

            # Pixel clipping percentage (artifacts)
            clip_low = np.mean(res_rgb <= 0.004) # Near black
            clip_high = np.mean(res_rgb >= 0.996) # Near saturated white
            clippings.append((clip_low + clip_high) * 100.0)

            # Sample subset for UIQM / UCIQE (CPU-intensive, evaluate on 40 samples)
            if idx % 4 == 0:
                uiqms.append(compute_uiqm(res_rgb))
                uciqes.append(compute_uciqe(res_rgb))

        elapsed = time.time() - t_start
        peak_vram_mb = torch.cuda.max_memory_reserved() / (1024**2)

        mean_psnr = float(np.mean(psnrs))
        psnr_norm = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
        mean_ssim = float(np.mean(ssims))
        mean_sharp = float(np.mean(sharps))
        q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
        mean_uiqm = float(np.mean(uiqms))
        mean_uciqe = float(np.mean(uciqes))
        mean_clip = float(np.mean(clippings))
        avg_lat = float(np.mean(latencies))
        fps = len(items) / elapsed

        results[cfg_name] = {
            "description": cfg["desc"],
            "psnr_db": round(mean_psnr, 2),
            "psnr_norm": round(psnr_norm, 4),
            "ssim": round(mean_ssim, 4),
            "sharpness": round(mean_sharp, 4),
            "q_fr": round(q_fr, 4),
            "uiqm": round(mean_uiqm, 4),
            "uciqe": round(mean_uciqe, 4),
            "clipping_pct": round(mean_clip, 2),
            "latency_ms": round(avg_lat, 2),
            "throughput_fps": round(fps, 1),
            "peak_vram_mb": round(peak_vram_mb, 1)
        }

        print(f"    PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | Sharp: {mean_sharp:.4f} | Q_FR: {q_fr:.4f}")
        print(f"    UIQM: {mean_uiqm:.4f} | UCIQE: {mean_uciqe:.4f} | Clipping: {mean_clip:.2f}% | Latency: {avg_lat:.2f} ms | VRAM: {peak_vram_mb:.1f} MB")

    # 4. Comparative Report Generation
    print("\n" + "=" * 80)
    print("PHASE 2 BASELINE BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<38} | {'PSNR (dB)':<9} | {'SSIM':<6} | {'Sharp':<6} | {'Q_FR':<6} | {'UIQM':<6} | {'UCIQE':<6} | {'Clip %':<6} | {'FPS':<5}")
    print("-" * 105)
    for k, v in results.items():
        print(f"{v['description'][:38]:<38} | {v['psnr_db']:<9.2f} | {v['ssim']:<6.4f} | {v['sharpness']:<6.4f} | {v['q_fr']:<6.4f} | {v['uiqm']:<6.4f} | {v['uciqe']:<6.4f} | {v['clipping_pct']:<6.2f} | {v['throughput_fps']:<5.1f}")
    print("=" * 80)

    out_json = "reports/PHASE2_RESTORATION_BASELINE_RESULTS.json"
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full benchmark data saved to: {out_json}")

    return results

if __name__ == "__main__":
    run_phase2_baseline()
