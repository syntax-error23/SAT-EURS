"""
SAT-EURS Track 1 — Phase 1 Restoration Forensics Diagnostic
Measures:
1. Exact image processing graph (dtypes, normalization, clipping)
2. Quantitative impact of unsharp filter alpha in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25} on paired validation data
3. Restoration metrics (PSNR, PSNR_norm, SSIM, Sharpness Ratio, Q_FR) across water types and depths
4. Dynamic range, saturation, luminance shift, clipping rates
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    load_checkpoint_with_provenance_check,
    parse_scene_wt_depth,
    V18CanonicalPreprocessor,
)
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[PREFLIGHT] Device: {device} ({torch.cuda.get_device_name(device)})")

    out_dir = "reports/v28/phase1_diagnostic"
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    checkpoint_dir = "checkpoints/v18"

    # 1. Load 5-Fold Refiners
    print("[1/4] Loading 5-Fold Refiners...")
    base_table = torch.load(table_path, map_location=device)
    manifest = validate_manifest_and_folds()
    refiner_models = []
    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        r_model = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_path = os.path.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        r_weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, f_data["train_scenes"], f_data["val_scenes"])
        r_model.load_state_dict(r_weights)
        r_model.eval()
        refiner_models.append(r_model)

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    # 2. Select validation pairs
    # Use 4 diverse validation scenes: fish1 (coral/shallow), sharkFish (pelagic/deep), seahorse1 (macro/benthic), fish12 (turbid)
    val_scenes = ["fish1", "sharkFish", "seahorse1", "fish12"]
    conditions = [
        ("mild", 1, 10),
        ("mid", 3, 30),
        ("deep", 5, 60),
        ("turbid", 8, 40),
        ("extreme", 10, 90),
    ]

    test_pairs = []
    for sc in val_scenes:
        clean_file = f"data/clean/{sc}.png"
        if not os.path.exists(clean_file):
            continue
        for lbl, wt, dp in conditions:
            deg_file = f"data/degraded/{sc}/{sc}_{wt}_{dp}.png"
            if os.path.exists(deg_file):
                test_pairs.append((sc, clean_file, deg_file, wt, dp, lbl))

    print(f"[2/4] Assembled {len(test_pairs)} paired validation samples.")

    # 3. Sweep alpha values: 0.00, 0.05, 0.10, 0.15, 0.20, 0.25
    alphas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    alpha_metrics = {a: {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": [], "q_fr": [], "clip_high": [], "clip_low": []} for a in alphas}
    raw_degraded_metrics = {"psnr": [], "psnr_norm": [], "ssim": [], "sharp": [], "q_fr": []}

    print("[3/4] Running paired restoration benchmark...")
    t0 = time.time()
    for idx, (sc, clean_path, deg_path, wt, dp, lbl) in enumerate(test_pairs):
        clean_bgr = cv2.imread(clean_path)
        deg_bgr = cv2.imread(deg_path)

        # Scale clean to 1024 max side for matched comparison
        ch, cw = clean_bgr.shape[:2]
        from tools.resize_for_submission import scale_to_max_side
        tw, th = scale_to_max_side(cw, ch, 1024)
        clean_scaled = cv2.resize(clean_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        clean_rgb_f = cv2.cvtColor(clean_scaled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Preprocess degraded
        prep = preprocessor.prepare_input(deg_bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]
        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([dp], device=device)

        with torch.inference_mode():
            rest_acc = torch.zeros_like(raw_pad)
            for r_m in refiner_models:
                rest_acc += r_m(raw_pad.float(), wt_t, dp_t)["refined_image"]
            rest_fused = rest_acc / len(refiner_models)

        # Unpad
        rest_unpad = preprocessor.unpad_output(rest_fused.squeeze(0).permute(1, 2, 0).cpu().numpy(), meta)
        # Ensure exact shape match with clean_scaled
        if rest_unpad.shape[:2] != (th, tw):
            rest_unpad = cv2.resize(rest_unpad, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Evaluate raw degraded
        deg_scaled = cv2.resize(deg_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        deg_rgb_f = cv2.cvtColor(deg_scaled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        p_raw, pn_raw = compute_psnr_and_norm(deg_rgb_f, clean_rgb_f)
        s_raw = compute_ssim_single(deg_rgb_f, clean_rgb_f)
        sh_raw = compute_sharpness_ratio(deg_rgb_f, clean_rgb_f)
        q_raw = 0.45 * pn_raw + 0.45 * s_raw + 0.10 * sh_raw
        raw_degraded_metrics["psnr"].append(p_raw)
        raw_degraded_metrics["psnr_norm"].append(pn_raw)
        raw_degraded_metrics["ssim"].append(s_raw)
        raw_degraded_metrics["sharp"].append(sh_raw)
        raw_degraded_metrics["q_fr"].append(q_raw)

        # Evaluate across alphas
        blur = cv2.GaussianBlur(rest_unpad, (0, 0), 2.0)
        high_freq = rest_unpad - blur

        for a in alphas:
            if a > 0:
                enhanced = rest_unpad + a * high_freq
            else:
                enhanced = rest_unpad

            # Count clipping before clipping
            c_low = float(np.mean(enhanced < 0.0)) * 100
            c_high = float(np.mean(enhanced > 1.0)) * 100
            clipped = np.clip(enhanced, 0.0, 1.0)

            # Simulate JPEG quality 98 compression
            clipped_u8 = np.clip(clipped * 255.0, 0, 255).astype(np.uint8)
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(clipped_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])
            dec_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec_rgb = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            psnr, psnr_norm = compute_psnr_and_norm(dec_rgb, clean_rgb_f)
            ssim_val = compute_ssim_single(dec_rgb, clean_rgb_f)
            sharp_val = compute_sharpness_ratio(dec_rgb, clean_rgb_f)
            q_fr = 0.45 * psnr_norm + 0.45 * ssim_val + 0.10 * sharp_val

            alpha_metrics[a]["psnr"].append(psnr)
            alpha_metrics[a]["psnr_norm"].append(psnr_norm)
            alpha_metrics[a]["ssim"].append(ssim_val)
            alpha_metrics[a]["sharp"].append(sharp_val)
            alpha_metrics[a]["q_fr"].append(q_fr)
            alpha_metrics[a]["clip_low"].append(c_low)
            alpha_metrics[a]["clip_high"].append(c_high)

    # 4. Synthesize results
    print("\n" + "=" * 95)
    print("RESTORATION BENCHMARK RESULTS (Across 20 Paired Validation Conditions)")
    print("=" * 95)
    print(f"{'CONFIGURATION':<22} | {'PSNR (dB)':<9} | {'PSNR_norm':<9} | {'SSIM':<7} | {'SHARPNESS':<9} | {'Q_FR':<7} | {'CLIPPING %':<10}")
    print("-" * 95)
    
    r_psnr = np.mean(raw_degraded_metrics["psnr"])
    r_pnorm = np.mean(raw_degraded_metrics["psnr_norm"])
    r_ssim = np.mean(raw_degraded_metrics["ssim"])
    r_sharp = np.mean(raw_degraded_metrics["sharp"])
    r_qfr = np.mean(raw_degraded_metrics["q_fr"])
    print(f"{'Raw Degraded Input':<22} | {r_psnr:8.3f}  | {r_pnorm:8.4f}  | {r_ssim:6.4f}  | {r_sharp:8.4f}  | {r_qfr:6.4f} | 0.00%")

    summary = {
        "raw_degraded": {
            "psnr": round(float(r_psnr), 4),
            "psnr_norm": round(float(r_pnorm), 4),
            "ssim": round(float(r_ssim), 4),
            "sharpness": round(float(r_sharp), 4),
            "q_fr": round(float(r_qfr), 4),
        },
        "alpha_sweep": {}
    }

    for a in alphas:
        p = np.mean(alpha_metrics[a]["psnr"])
        pn = np.mean(alpha_metrics[a]["psnr_norm"])
        s = np.mean(alpha_metrics[a]["ssim"])
        sh = np.mean(alpha_metrics[a]["sharp"])
        q = np.mean(alpha_metrics[a]["q_fr"])
        cl = np.mean(alpha_metrics[a]["clip_low"])
        ch = np.mean(alpha_metrics[a]["clip_high"])
        clip_tot = cl + ch

        label = f"Refiner alpha={a:.2f}"
        if a == 0.00:
            label += " (Pure)"
        elif a == 0.10:
            label += " (V26/V27)"
        elif a == 0.25:
            label += " (V25)"

        print(f"{label:<22} | {p:8.3f}  | {pn:8.4f}  | {s:6.4f}  | {sh:8.4f}  | {q:6.4f} | {clip_tot:5.2f}%")

        summary["alpha_sweep"][f"alpha_{a:.2f}"] = {
            "alpha": a,
            "psnr_dB": round(float(p), 4),
            "psnr_norm": round(float(pn), 4),
            "ssim": round(float(s), 4),
            "sharpness_ratio": round(float(sh), 4),
            "q_fr": round(float(q), 4),
            "clip_low_pct": round(float(cl), 3),
            "clip_high_pct": round(float(ch), 3),
            "total_clipping_pct": round(float(clip_tot), 3),
        }

    print("=" * 95)

    out_file = os.path.join(out_dir, "restoration_forensics_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved restoration forensics summary to {out_file}")

if __name__ == "__main__":
    main()
