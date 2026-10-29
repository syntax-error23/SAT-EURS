"""
SAT-EURS Track 1 — Phase 1 Scale Mismatch & Tiled Inference Diagnostic
Tests whether running sliding-window / tiled 512x512 inference improves small-object
(Fish / Ray / MacroFauna) mIoU on validation scenes vs canonical 1024 full-scene inference.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    load_checkpoint_with_provenance_check,
    V18CanonicalPreprocessor,
)
from tools.resize_for_submission import scale_to_max_side

def predict_full_image(model, refiner, bgr, wt, depth, preprocessor, device, cal_t):
    prep = preprocessor.prepare_input(bgr)
    raw_pad = prep["rgb_tensor"].to(device)
    meta = prep["meta"]
    wt_t = torch.tensor([wt], device=device)
    dp_t = torch.tensor([depth], device=device)

    with torch.inference_mode():
        rest_t = refiner(raw_pad.float(), wt_t, dp_t)["refined_image"]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(rest_t, raw_pad, wt_t, dp_t)
            logits = out["logits"].float()
            log_prob = torch.log(F.softmax(logits, dim=1) + 1e-10) + cal_t
            mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return preprocessor.unpad_output(mask_pad, meta)

def predict_tiled_512(model, refiner, bgr, wt, depth, preprocessor, device, cal_t, tile_size=512, stride=384):
    """
    Sliding window inference with 512x512 tiles matching the training crop distribution.
    """
    orig_h, orig_w = bgr.shape[:2]
    tw, th = scale_to_max_side(orig_w, orig_h, 1024)
    scaled_bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    # Pad scaled_bgr so it's at least tile_size in each dim
    pad_h = max(0, tile_size - th)
    pad_w = max(0, tile_size - tw)
    if pad_h > 0 or pad_w > 0:
        scaled_bgr = cv2.copyMakeBorder(scaled_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    cur_h, cur_w = scaled_bgr.shape[:2]
    
    # Generate tile coordinates
    y_starts = list(range(0, cur_h - tile_size + 1, stride))
    if y_starts[-1] + tile_size < cur_h:
        y_starts.append(cur_h - tile_size)
        
    x_starts = list(range(0, cur_w - tile_size + 1, stride))
    if x_starts[-1] + tile_size < cur_w:
        x_starts.append(cur_w - tile_size)

    prob_accum = np.zeros((5, cur_h, cur_w), dtype=np.float32)
    count_accum = np.zeros((cur_h, cur_w), dtype=np.float32)

    wt_t = torch.tensor([wt], device=device)
    dp_t = torch.tensor([depth], device=device)

    # Cosine window for seamless tile blending
    win_y = np.hanning(tile_size)
    win_x = np.hanning(tile_size)
    window_2d = np.outer(win_y, win_x) + 1e-3

    rgb_f = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    with torch.inference_mode():
        for ys in y_starts:
            for xs in x_starts:
                tile_rgb = rgb_f[ys:ys+tile_size, xs:xs+tile_size]
                tile_t = torch.from_numpy(tile_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

                rest_t = refiner(tile_t.float(), wt_t, dp_t)["refined_image"]

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(rest_t, tile_t, wt_t, dp_t)
                    p_tile = F.softmax(out["logits"].float(), dim=1).squeeze(0).cpu().numpy()

                prob_accum[:, ys:ys+tile_size, xs:xs+tile_size] += p_tile * window_2d
                count_accum[ys:ys+tile_size, xs:xs+tile_size] += window_2d

    prob_fused = prob_accum / np.maximum(count_accum, 1e-5)
    # Strip padding
    prob_fused = prob_fused[:, :th, :tw]

    # Apply calibration offsets
    cal_np = np.array([0.0, 0.0, 0.0, 0.10, 0.25]).reshape(5, 1, 1)
    log_p = np.log(prob_fused + 1e-10) + cal_np
    mask = np.argmax(log_p, axis=0).astype(np.uint8)
    return mask

def compute_miou_and_f1(pred, gt, num_classes=5):
    ious = []
    f1s = []
    class_stats = {}
    for c in range(num_classes):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c))
        fn = np.sum((pred != c) & (gt == c))
        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else (1.0 if np.sum(gt == c) == 0 else 0.0)
        f1 = (2 * tp) / denom_f1 if denom_f1 > 0 else (1.0 if np.sum(gt == c) == 0 else 0.0)
        ious.append(iou)
        f1s.append(f1)
        class_stats[c] = {"iou": round(float(iou), 4), "f1": round(float(f1), 4), "tp": int(tp), "gt": int(np.sum(gt == c))}
    return float(np.mean(ious)), float(np.mean(f1s)), class_stats

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[PREFLIGHT] Device: {device} ({torch.cuda.get_device_name(device)})")

    out_dir = "reports/v28/phase1_diagnostic"
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    v27_checkpoint_dir = "checkpoints/v27"

    # Load Full-20 Model
    print("Loading Full-20 SWA model...")
    base_table = torch.load(table_path, map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
    r_path = "checkpoints/v18/fold_0/refiner_best.pt"
    refiner.load_state_dict(torch.load(r_path, map_location=device)["state_dict"])
    refiner.eval()

    full20_model = build_v18_segmenter(num_classes=5).to(device)
    f20_ckpt = torch.load(os.path.join(v27_checkpoint_dir, "segmenter_full20_swa.pt"), map_location=device)
    full20_model.load_state_dict(f20_ckpt["state_dict"] if "state_dict" in f20_ckpt else f20_ckpt)
    full20_model.eval()

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    cal_t = torch.tensor([0.0, 0.0, 0.0, 0.10, 0.25], device=device).view(1, 5, 1, 1)

    # Test scenes with ground truth:
    # fish1: dense small fish
    # sharkFish: shark + fish
    # seahorse1: macro fauna + coral
    scenes = ["fish1", "sharkFish", "seahorse1"]
    conditions = [(1, 10), (3, 30), (5, 60)]

    results = []

    print("\nRunning Scale Mismatch Benchmark (Full 1024 vs Tiled 512)...")
    for sc in scenes:
        mask_gt_raw = cv2.imread(f"data/masks/{sc}.png", cv2.IMREAD_GRAYSCALE)
        gh, gw = mask_gt_raw.shape[:2]
        tw, th = scale_to_max_side(gw, gh, 1024)
        gt = cv2.resize(mask_gt_raw, (tw, th), interpolation=cv2.INTER_NEAREST)

        for wt, dp in conditions:
            deg_path = f"data/degraded/{sc}/{sc}_{wt}_{dp}.png"
            if not os.path.exists(deg_path):
                continue
            deg_bgr = cv2.imread(deg_path)

            # 1. Full image inference
            t_start = time.time()
            mask_full = predict_full_image(full20_model, refiner, deg_bgr, wt, dp, preprocessor, device, cal_t)
            time_full = time.time() - t_start

            # 2. Tiled 512 inference
            t_start = time.time()
            mask_tiled = predict_tiled_512(full20_model, refiner, deg_bgr, wt, dp, preprocessor, device, cal_t, tile_size=512, stride=384)
            time_tiled = time.time() - t_start

            miou_full, f1_full, cls_full = compute_miou_and_f1(mask_full, gt)
            miou_tiled, f1_tiled, cls_tiled = compute_miou_and_f1(mask_tiled, gt)

            q_full = miou_full * f1_full
            q_tiled = miou_tiled * f1_tiled

            results.append({
                "scene": sc,
                "wt": wt,
                "depth": dp,
                "mIoU_full": round(miou_full, 4),
                "mIoU_tiled": round(miou_tiled, 4),
                "delta_mIoU": round(miou_tiled - miou_full, 4),
                "Q_seg_full": round(q_full, 4),
                "Q_seg_tiled": round(q_tiled, 4),
                "delta_Q_seg": round(q_tiled - q_full, 4),
                "fish_iou_full": cls_full[2]["iou"],
                "fish_iou_tiled": cls_tiled[2]["iou"],
                "cls_full": cls_full,
                "cls_tiled": cls_tiled,
                "time_full_s": round(time_full, 3),
                "time_tiled_s": round(time_tiled, 3),
            })

            delta_str = f"+{q_tiled - q_full:.4f}" if q_tiled >= q_full else f"{q_tiled - q_full:.4f}"
            print(f"  {sc:10s} (WT={wt}, D={dp}m): Full Q_seg={q_full:.4f} | Tiled Q_seg={q_tiled:.4f} (Delta: {delta_str}) | Fish IoU: {cls_full[2]['iou']:.3f} -> {cls_tiled[2]['iou']:.3f}")

    mean_delta_miou = np.mean([r["delta_mIoU"] for r in results])
    mean_delta_qseg = np.mean([r["delta_Q_seg"] for r in results])
    fish_deltas = [r["fish_iou_tiled"] - r["fish_iou_full"] for r in results if r["cls_full"][2]["gt"] > 0]
    mean_fish_delta = np.mean(fish_deltas) if fish_deltas else 0.0

    print("\n" + "=" * 80)
    print("SCALE MISMATCH BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Mean Delta mIoU (Tiled vs Full):    {mean_delta_miou:+.4f}")
    print(f"Mean Delta Q_seg (Tiled vs Full):   {mean_delta_qseg:+.4f}")
    print(f"Mean Fish IoU Delta:                {mean_fish_delta:+.4f}")
    print("=" * 80)

    out_file = os.path.join(out_dir, "scale_mismatch_summary.json")
    with open(out_file, "w") as f:
        json.dump({
            "mean_delta_mIoU": round(float(mean_delta_miou), 4),
            "mean_delta_Q_seg": round(float(mean_delta_qseg), 4),
            "mean_fish_iou_delta": round(float(mean_fish_delta), 4),
            "runs": results
        }, f, indent=2)
    print(f"Saved scale mismatch summary to {out_file}")

if __name__ == "__main__":
    main()
