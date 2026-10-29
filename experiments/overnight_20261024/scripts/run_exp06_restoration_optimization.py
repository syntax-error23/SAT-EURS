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


def evaluate_restoration_configs(
    refiner_ckpts: Dict[int, str],
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    data_root: str = "data/cached_1024",
    sample_stride: int = 5,
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    manifest = validate_manifest_and_folds()
    base_table = torch.load(table_path, map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    # Define candidate restoration parameter configurations
    configs = {
        "cfg0_no_unsharp": {"alpha": 0.00, "sigma": 2.0, "raw_blend": 0.0},
        "cfg1_baseline_a10_s20": {"alpha": 0.10, "sigma": 2.0, "raw_blend": 0.0},
        "cfg2_a05_s20": {"alpha": 0.05, "sigma": 2.0, "raw_blend": 0.0},
        "cfg3_a15_s20": {"alpha": 0.15, "sigma": 2.0, "raw_blend": 0.0},
        "cfg4_a20_s20": {"alpha": 0.20, "sigma": 2.0, "raw_blend": 0.0},
        "cfg5_a10_s15": {"alpha": 0.10, "sigma": 1.5, "raw_blend": 0.0},
        "cfg6_a10_s10": {"alpha": 0.10, "sigma": 1.0, "raw_blend": 0.0},
        "cfg7_a15_s15": {"alpha": 0.15, "sigma": 1.5, "raw_blend": 0.0},
        "cfg8_blend_raw05": {"alpha": 0.10, "sigma": 2.0, "raw_blend": 0.05},
        "cfg9_blend_raw10": {"alpha": 0.10, "sigma": 2.0, "raw_blend": 0.10},
    }

    # Tracking metrics per config
    results = {k: {"psnr": [], "psnr_norm": [], "ssim": [], "sharpness": []} for k in configs}

    print("=" * 80)
    print("SAT-EURS TRACK 1 — EXP-06: RESTORATION FIDELITY OPTIMIZATION BENCHMARK")
    print(f"Sweeping {len(configs)} restoration configurations across all 20 held-out scenes...")
    print("=" * 80)

    t0 = time.time()
    total_images = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        # Load fold refiner
        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpts[fold], map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        print(f"Fold {fold} evaluating scenes: {val_scenes}...", flush=True)

        for sc in val_scenes:
            clean_path = osp.join(data_root, "clean", f"{sc}.png")
            clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            if clean_bgr is None:
                continue
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gh, gw = clean_rgb.shape[:2]

            deg_scene_dir = osp.join(data_root, "degraded", sc)
            deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])
            sampled_files = deg_files[::sample_stride]

            for fname in sampled_files:
                stem = osp.splitext(fname)[0]
                parts = stem.split("_")
                wt = int(parts[1])
                depth = int(parts[2])

                img_path = osp.join(deg_scene_dir, fname)
                bgr = cv2.imread(img_path)
                if bgr is None:
                    continue

                prep = preprocessor.prepare_input(bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)

                with torch.inference_mode():
                    out_r = refiner(raw_pad.float(), wt_t, dp_t)
                    ref_pad = out_r["refined_image"]

                ref_unpad = preprocessor.unpad_output(ref_pad.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
                raw_unpad = preprocessor.unpad_output(raw_pad.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)

                if ref_unpad.shape[:2] != (gh, gw):
                    ref_unpad = cv2.resize(ref_unpad, (gw, gh), interpolation=cv2.INTER_LINEAR)
                    raw_unpad = cv2.resize(raw_unpad, (gw, gh), interpolation=cv2.INTER_LINEAR)

                ref_unpad = np.clip(ref_unpad, 0.0, 1.0)
                raw_unpad = np.clip(raw_unpad, 0.0, 1.0)
                total_images += 1

                for c_name, c_params in configs.items():
                    alpha = c_params["alpha"]
                    sigma = c_params["sigma"]
                    raw_blend = c_params["raw_blend"]

                    cur = ref_unpad
                    if raw_blend > 0.0:
                        cur = (1.0 - raw_blend) * cur + raw_blend * raw_unpad

                    if alpha > 0.0:
                        blur = cv2.GaussianBlur(cur, (0, 0), sigma)
                        cur = np.clip(cur + alpha * (cur - blur), 0.0, 1.0)

                    psnr, psnr_norm = compute_psnr_and_norm(cur, clean_rgb)
                    ssim = compute_ssim_single(cur, clean_rgb)
                    sharp = compute_sharpness_ratio(cur, clean_rgb)

                    results[c_name]["psnr"].append(psnr)
                    results[c_name]["psnr_norm"].append(psnr_norm)
                    results[c_name]["ssim"].append(ssim)
                    results[c_name]["sharpness"].append(sharp)

    elapsed = time.time() - t0
    print(f"\nEvaluated {total_images} images across 20 scenes in {elapsed:.1f}s.")
    print("\n" + "=" * 95)
    print(f"{'Configuration':<25} | {'PSNR (dB)':<10} | {'PSNR_norm':<10} | {'SSIM':<8} | {'Sharpness':<10} | {'Q_FR':<8} | {'Delta vs Base'}")
    print("=" * 95)

    base_q_fr = None
    summary = {}
    for c_name in configs:
        m_psnr = float(np.mean(results[c_name]["psnr"]))
        m_pnorm = float(np.mean(results[c_name]["psnr_norm"]))
        m_ssim = float(np.mean(results[c_name]["ssim"]))
        m_sharp = float(np.mean(results[c_name]["sharpness"]))
        q_fr = float(0.45 * m_pnorm + 0.45 * m_ssim + 0.10 * m_sharp)

        if base_q_fr is None:
            base_q_fr = q_fr  # cfg0 or cfg1

        delta_str = f"{q_fr - base_q_fr:+.5f}" if base_q_fr is not None else "+0.00000"

        summary[c_name] = {
            "params": configs[c_name],
            "psnr": round(m_psnr, 4),
            "psnr_norm": round(m_pnorm, 4),
            "ssim": round(m_ssim, 4),
            "sharpness": round(m_sharp, 4),
            "Q_FR": round(q_fr, 5),
            "delta_vs_first": round(q_fr - base_q_fr, 5)
        }
        print(f"{c_name:<25} | {m_psnr:<10.3f} | {m_pnorm:<10.4f} | {m_ssim:<8.4f} | {m_sharp:<10.4f} | {q_fr:<8.4f} | {delta_str}")
    print("=" * 95)

    return {"summary": summary, "elapsed_s": round(elapsed, 2), "num_images": total_images}


def main():
    refiner_ckpts = {
        0: 'checkpoints/v18/fold_0/refiner_best.pt',
        1: 'checkpoints/v18/fold_1/refiner_best.pt',
        2: 'checkpoints/v18/fold_2/refiner_best.pt',
        3: 'checkpoints/v18/fold_3/refiner_best.pt',
        4: 'checkpoints/v18/fold_4/refiner_best.pt',
    }

    out = evaluate_restoration_configs(refiner_ckpts=refiner_ckpts, sample_stride=5)
    out_file = "experiments/overnight_20261024/predictions/exp06_restoration_results.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
