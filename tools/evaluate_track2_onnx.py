"""
Comprehensive Track 2 ONNX Hardware & Metric Evaluator
Runs the exported standalone ONNX engine directly via ONNX Runtime:
- Input: [1, 3, 1024, 1024] in [0, 1]
- Output 1: 'restored_image' [1, 3, 1024, 1024]
- Output 2: 'seg_logits' [1, 5, 1024, 1024]
Measures:
- Latency (ms/frame) on execution provider
- Full-reference restoration: PSNR, PSNR_norm, SSIM, Sharpness, Q_FR
- Segmentation: mIoU, Macro F1, Coverage Penalty, Q_seg (with verified official class names)
- Composite Track 2 competition score
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, ".")
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_track2_onnx(onnx_path: str = "deploy/eurs_net_v5_calibrated_track2.onnx", stride: int = 5):
    print("=" * 76)
    print("   EVALUATING TRACK 2 STANDALONE ONNX ENGINE (ZERO METADATA)")
    print(f"   Model: {onnx_path}")
    print(f"   Stride: 1 in every {stride} images across held-out scenes")
    print("=" * 76)

    assert os.path.exists(onnx_path), f"ONNX model {onnx_path} not found!"

    # Session options
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    data_clean_dir = "data/cached_1024/clean"
    data_masks_dir = "data/cached_1024/masks"
    data_deg_dir = "data/cached_1024/degraded"

    evaluator_seg = SegmentationEvaluator(num_classes=5)

    psnrs, ssims, sharps = [], [], []
    latencies = []

    total_images = 0
    t0 = time.time()

    for sc in val_scenes:
        clean_p = os.path.join(data_clean_dir, f"{sc}.png")
        mask_p = os.path.join(data_masks_dir, f"{sc}.png")
        deg_folder = os.path.join(data_deg_dir, sc)

        clean_im = Image.open(clean_p).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        clean_np = np.asarray(clean_im, dtype=np.float32) / 255.0

        mask_im = Image.open(mask_p).resize((1024, 1024), Image.NEAREST)
        mask_np = np.asarray(mask_im, dtype=np.int64)

        deg_files = sorted([os.path.join(deg_folder, f) for f in os.listdir(deg_folder) if f.endswith(".png")])[::stride]

        for df in deg_files:
            total_images += 1
            deg_im = Image.open(df).convert("RGB").resize((1024, 1024), Image.BILINEAR)
            deg_np = np.asarray(deg_im, dtype=np.float32) / 255.0
            inp = deg_np.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            # Measure inference latency
            t_start = time.perf_counter()
            outputs = session.run(None, {"input": inp})
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

            rest_np = outputs[0][0].transpose(1, 2, 0) # [1024, 1024, 3] in [0, 1]
            logits_np = outputs[1][0] # [5, 1024, 1024]

            # Segmentation update
            pred_mask = np.argmax(logits_np, axis=0) # [1024, 1024]
            # Convert to torch tensor for SegmentationEvaluator
            import torch
            evaluator_seg.update(torch.from_numpy(pred_mask).unsqueeze(0), torch.from_numpy(mask_np).unsqueeze(0))

            # Restoration metrics
            p, _ = compute_psnr_and_norm(rest_np, clean_np)
            s = compute_ssim_single(rest_np, clean_np)
            sh = compute_sharpness_ratio(rest_np, clean_np)

            psnrs.append(p)
            ssims.append(s)
            sharps.append(sh)

    elapsed = time.time() - t0
    seg_m = evaluator_seg.compute()

    mean_psnr = float(np.mean(psnrs))
    mean_psnr_norm = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
    mean_ssim = float(np.mean(ssims))
    mean_sharp = float(np.mean(sharps))
    q_fr = 0.45 * mean_psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp

    q_seg = seg_m["Q_seg"]
    q_nr_local = 0.5000
    q_nr_proj = 0.7180

    comp_local = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr_local
    comp_proj = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr_proj

    avg_lat = float(np.mean(latencies))
    fps = 1000.0 / avg_lat

    print("\n" + "=" * 76)
    print("   TRACK 2 ONNX STANDALONE BENCHMARK RESULTS")
    print("=" * 76)
    print(f"Total Evaluated:       {total_images} images ({elapsed:.1f}s)")
    print(f"Mean Latency (CPU):    {avg_lat:.2f} ms/frame ({fps:.1f} FPS)")
    print("-" * 76)
    print("SEGMENTATION METRICS (Official EURS Classes):")
    print(f"  mIoU:                {seg_m['mIoU']:.4f}")
    print(f"  Macro F1:            {seg_m['macro_F1']:.4f}")
    print(f"  Coverage Penalty:    {seg_m['coverage_penalty']:.4f}")
    print(f"  Q_seg:               {q_seg:.4f}")
    classes = ["Background", "Coral", "Fish", "Elasmobranch", "Other"]
    for i, c_name in enumerate(classes):
        print(f"    Class {i} ({c_name:<13}): IoU = {seg_m['per_class_IoU'][i]:.4f} | F1 = {seg_m['per_class_F1'][i]:.4f}")
    print("-" * 76)
    print("RESTORATION METRICS (Single-Pass Neural Restoration):")
    print(f"  PSNR:                {mean_psnr:.2f} dB")
    print(f"  PSNR_norm (15-35):   {mean_psnr_norm:.4f}")
    print(f"  SSIM:                {mean_ssim:.4f}")
    print(f"  Sharpness Ratio:     {mean_sharp:.4f}")
    print(f"  Q_FR:                {q_fr:.4f}")
    print("-" * 76)
    print("COMPOSITE SCORE:")
    print(f"  Measured Local S:    {comp_local:.4f} (with local Q_NR = 0.50)")
    print(f"  Projected Server S:  {comp_proj:.4f} (with server Q_NR = 0.718)")
    print("=" * 76)

    res = {
        "onnx_path": onnx_path,
        "onnx_size_mb": round(os.path.getsize(onnx_path) / 1e6, 2),
        "total_evaluated": total_images,
        "mean_latency_ms": round(avg_lat, 2),
        "fps_cpu": round(fps, 1),
        "segmentation": {
            "mIoU": round(seg_m['mIoU'], 4),
            "macro_F1": round(seg_m['macro_F1'], 4),
            "coverage_penalty": round(seg_m['coverage_penalty'], 4),
            "Q_seg": round(q_seg, 4),
            "per_class_IoU": [round(x, 4) for x in seg_m['per_class_IoU']],
            "per_class_F1": [round(x, 4) for x in seg_m['per_class_F1']]
        },
        "restoration": {
            "psnr_dB": round(mean_psnr, 2),
            "psnr_norm": round(mean_psnr_norm, 4),
            "ssim": round(mean_ssim, 4),
            "sharpness_ratio": round(mean_sharp, 4),
            "Q_FR": round(q_fr, 4)
        },
        "composite": {
            "measured_local": round(comp_local, 4),
            "projected_server": round(comp_proj, 4)
        }
    }

    out_json = "deploy/track2_calibrated_onnx_evaluation.json"
    with open(out_json, "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"Saved evaluation report to {out_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="deploy/eurs_edgesota_v2_final.onnx")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    evaluate_track2_onnx(onnx_path=args.onnx, stride=args.stride)

