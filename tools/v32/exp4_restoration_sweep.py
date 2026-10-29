"""
SAT-EURS Track 1 — V32 Experiment 4: Depth-Adaptive Restoration Sweep
Evaluates whether depth/water-type adaptive sharpening outperforms fixed unsharp masking.
Strictly scene-held-out out-of-fold evaluation across all 5 folds.
Uses exact official scorer weights: Q_FR = 0.45 * PSNR_norm + 0.45 * SSIM + 0.10 * Sharpness
"""

import os
import os.path as osp
import sys
import time
import json
import numpy as np
import cv2
import torch

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    V18CanonicalPreprocessor,
)
from tools.resize_for_submission import scale_to_max_side
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_restoration_sweep(
    checkpoint_dir: str = "checkpoints/v18",
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    data_root: str = "data/cached_1024",
    sample_stride: int = 5,  # 40 variants/scene across 20 scenes = 800 evaluations
    device_id: int = 0
):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    manifest = validate_manifest_and_folds()
    base_table = torch.load(table_path, map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    configs = {
        "Baseline (Static alpha=0.10, sigma=2.0)": {
            "type": "static",
            "alpha": 0.10,
            "sigma": 2.0,
        },
        "Static (alpha=0.08, sigma=2.0)": {
            "type": "static",
            "alpha": 0.08,
            "sigma": 2.0,
        },
        "Static (alpha=0.12, sigma=2.0)": {
            "type": "static",
            "alpha": 0.12,
            "sigma": 2.0,
        },
        "Adaptive A (alpha0=0.12, d0=40, sigma=2.0)": {
            "type": "adaptive",
            "alpha0": 0.12,
            "d0": 40.0,
            "sigma": 2.0,
        },
        "Adaptive B (alpha0=0.10, d0=30, sigma=2.0)": {
            "type": "adaptive",
            "alpha0": 0.10,
            "d0": 30.0,
            "sigma": 2.0,
        },
        "Adaptive C (alpha0=0.15, d0=50, sigma=2.0)": {
            "type": "adaptive",
            "alpha0": 0.15,
            "d0": 50.0,
            "sigma": 2.0,
        },
        "Adaptive Depth-Only (alpha0=0.12, d0=35, sigma=2.0)": {
            "type": "adaptive_depth_only",
            "alpha0": 0.12,
            "d0": 35.0,
            "sigma": 2.0,
        },
    }

    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — EXPERIMENT 4: RESTORATION DEPTH-ADAPTIVE SWEEP", flush=True)
    print(f"Configurations to test: {list(configs.keys())}", flush=True)
    print(f"Sample Stride: {sample_stride} (800 total held-out images)", flush=True)
    print("=" * 85, flush=True)

    # Pre-cache unpadded restored outputs from each fold refiner
    print("\nPhase 1: Computing neural refiner outputs on held-out scenes...", flush=True)
    t0 = time.time()
    cached_samples = []

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        refiner_ckpt = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpt, map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        print(f"  Fold {fold}: Processing held-out scenes {val_scenes}...", flush=True)

        for sc in val_scenes:
            clean_path = osp.join(data_root, "clean", f"{sc}.png")
            clean_bgr = cv2.imread(clean_path)
            ch, cw = clean_bgr.shape[:2]
            tw, th = scale_to_max_side(cw, ch, 1024)
            clean_scaled = cv2.resize(clean_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
            clean_rgb_f = cv2.cvtColor(clean_scaled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            deg_scene_dir = osp.join(data_root, "degraded", sc)
            deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])
            sampled_files = deg_files[::sample_stride]

            for fname in sampled_files:
                parts = osp.splitext(fname)[0].split("_")
                wt = int(parts[1])
                depth = int(parts[2])

                img_path = osp.join(deg_scene_dir, fname)
                deg_bgr = cv2.imread(img_path)
                if deg_bgr is None:
                    continue

                prep = preprocessor.prepare_input(deg_bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)

                with torch.inference_mode():
                    out_r = refiner(raw_pad.float(), wt_t, dp_t)
                    rest_t = out_r["refined_image"]

                rest_unpad = preprocessor.unpad_output(rest_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                if rest_unpad.shape[:2] != (th, tw):
                    rest_unpad = cv2.resize(rest_unpad, (tw, th), interpolation=cv2.INTER_LINEAR)

                cached_samples.append({
                    "fold": fold,
                    "scene": sc,
                    "wt": wt,
                    "depth": depth,
                    "rest_unpad": rest_unpad,
                    "clean_rgb_f": clean_rgb_f,
                    "tw": tw,
                    "th": th,
                })

    print(f"Cached {len(cached_samples)} images in {time.time() - t0:.1f}s.", flush=True)

    print("\nPhase 2: Sweeping post-processing configurations...", flush=True)
    sweep_results = {}

    for cfg_name, cfg in configs.items():
        t_cfg = time.time()
        psnrs, psnr_norms, ssims, sharps, q_frs = [], [], [], [], []

        for sample in cached_samples:
            unpad = sample["rest_unpad"]
            clean_rgb_f = sample["clean_rgb_f"]
            wt = sample["wt"]
            depth = sample["depth"]

            if cfg["type"] == "static":
                alpha = cfg["alpha"]
                sigma = cfg["sigma"]
            elif cfg["type"] == "adaptive":
                alpha0 = cfg["alpha0"]
                d0 = cfg["d0"]
                sigma = cfg["sigma"]
                # Decay with depth, moderate by water type
                alpha = alpha0 * np.exp(-depth / d0) * ((11.0 - wt) / 10.0)
            elif cfg["type"] == "adaptive_depth_only":
                alpha0 = cfg["alpha0"]
                d0 = cfg["d0"]
                sigma = cfg["sigma"]
                alpha = alpha0 * np.exp(-depth / d0)
            else:
                alpha = 0.0
                sigma = 2.0

            if alpha > 0.001:
                blur = cv2.GaussianBlur(unpad, (0, 0), sigma)
                rest_enh = np.clip(unpad + alpha * (unpad - blur), 0.0, 1.0)
            else:
                rest_enh = unpad

            # Simulate JPEG 98 official roundtrip
            rest_u8 = np.clip(rest_enh * 255.0, 0, 255).astype(np.uint8)
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(rest_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec_rgb_f = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            p_res, pn_res = compute_psnr_and_norm(dec_rgb_f, clean_rgb_f)
            s_res = compute_ssim_single(dec_rgb_f, clean_rgb_f)
            sh_res = compute_sharpness_ratio(dec_rgb_f, clean_rgb_f)
            q_res = 0.45 * pn_res + 0.45 * s_res + 0.10 * sh_res

            psnrs.append(p_res)
            psnr_norms.append(pn_res)
            ssims.append(s_res)
            sharps.append(sh_res)
            q_frs.append(q_res)

        mean_psnr = float(np.mean(psnrs))
        mean_psnr_norm = float(np.mean(psnr_norms))
        mean_ssim = float(np.mean(ssims))
        mean_sharp = float(np.mean(sharps))
        mean_qfr = float(np.mean(q_frs))

        # Composite delta relative to baseline V28-A:
        # S_restoration = 0.30 * Q_FR + 0.10 * Q_NR
        # Delta composite = 0.30 * (mean_qfr - baseline_qfr)
        sweep_results[cfg_name] = {
            "PSNR": round(mean_psnr, 4),
            "PSNR_norm": round(mean_psnr_norm, 5),
            "SSIM": round(mean_ssim, 5),
            "Sharpness": round(mean_sharp, 5),
            "Q_FR": round(mean_qfr, 5),
            "time_sec": round(time.time() - t_cfg, 2),
        }

        print(f"  {cfg_name}:")
        print(f"    PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | Sharpness: {mean_sharp:.4f} | Q_FR: {mean_qfr:.5f}")

    baseline_qfr = sweep_results["Baseline (Static alpha=0.10, sigma=2.0)"]["Q_FR"]
    for k, v in sweep_results.items():
        v["delta_Q_FR"] = round(v["Q_FR"] - baseline_qfr, 5)
        v["delta_Composite"] = round(0.30 * v["delta_Q_FR"], 5)

    out_file = "reports/v32/exp4_restoration_sweep_results.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\nSaved restoration sweep results to {out_file}", flush=True)
    return sweep_results


if __name__ == "__main__":
    evaluate_restoration_sweep()
