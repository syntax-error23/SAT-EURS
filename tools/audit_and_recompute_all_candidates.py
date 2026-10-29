"""
SAT-EURS Track 2 — Multi-Candidate Comprehensive Benchmark
Evaluates all existing candidate ONNX models on identical validation sets:
1. eurs_net_v5_calibrated_track2.onnx (Baseline Fallback, #18699, score 0.3380)
2. eurs_edgesota_v2_final.onnx
3. EXP-COMBINED-BEST.onnx
4. eurs_litev3_track2.onnx (Submission #19119, score 0.251)
5. eurs_litev3_production.onnx (Submission #19130, score 0.253)

Computes exact official metrics:
- Per-class IoU and F1
- mIoU, Macro-F1, Coverage Penalty (P_cov), Q_seg
- PSNR (dB), PSNR_norm, SSIM, Sharpness Ratio (against clean GT), Q_FR
- Composite Score
"""

import os
import sys
import time
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, ".")
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_onnx_candidate(
    onnx_path: str,
    scenes: List[str],
    step: int = 10
) -> Dict:
    print(f"\n=======================================================")
    print(f"Evaluating Candidate: {onnx_path}")
    print(f"=======================================================")
    
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    in_meta = session.get_inputs()[0]
    in_shape = in_meta.shape # e.g. [1, 3, 1024, 1024] or [1, 3, 384, 768]
    req_h, req_w = in_shape[2], in_shape[3]
    print(f"  Input contract: name={in_meta.name}, shape=[1, 3, {req_h}, {req_w}]")
    
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    psnr_list, ssim_list, sharp_list = [], [], []
    latencies = []
    
    t0 = time.time()
    total_imgs = 0
    
    for sc in scenes:
        c_p = f"data/cached_1024/clean/{sc}.png"
        m_p = f"data/cached_1024/masks/{sc}.png"
        deg_dir = f"data/cached_1024/degraded/{sc}"
        
        c_im = cv2.imread(c_p)
        c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
        
        deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::step]
        
        for df in deg_files:
            total_imgs += 1
            d_p = os.path.join(deg_dir, df)
            d_im = cv2.imread(d_p)
            if d_im is None:
                continue
            d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Resize input to model's required resolution
            d_in = cv2.resize(d_rgb, (req_w, req_h), interpolation=cv2.INTER_LINEAR)
            inp = d_in.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            
            t_s = time.perf_counter()
            outputs = session.run(None, {in_meta.name: inp})
            latencies.append((time.perf_counter() - t_s) * 1000.0)
            
            rest_out = outputs[0][0].transpose(1, 2, 0) # [H, W, 3]
            logits_out = outputs[1][0] # [5, H, W]
            
            # Resize outputs back to 1024x1024 for ground truth comparison
            if rest_out.shape[:2] != (1024, 1024):
                rest_1024 = cv2.resize(rest_out, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            else:
                rest_1024 = rest_out
                
            pred_mask_small = np.argmax(logits_out, axis=0).astype(np.uint8)
            if pred_mask_small.shape != (1024, 1024):
                pred_mask = cv2.resize(pred_mask_small, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            else:
                pred_mask = pred_mask_small
                
            valid = (m_im >= 0) & (m_im < num_classes)
            bins = num_classes * m_im[valid].astype(np.int64) + pred_mask[valid].astype(np.int64)
            counts = np.bincount(bins, minlength=num_classes * num_classes)
            conf_matrix += counts.reshape(num_classes, num_classes)
            
            p, _ = compute_psnr_and_norm(rest_1024, c_rgb)
            s = compute_ssim_single(rest_1024, c_rgb)
            sh = compute_sharpness_ratio(rest_1024, c_rgb)
            psnr_list.append(p)
            ssim_list.append(s)
            sharp_list.append(sh)
            
    tp = np.diag(conf_matrix).astype(float)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    
    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    
    per_class_iou = [float(tp[c] / denom_iou[c]) if denom_iou[c] > 0 else 0.0 for c in range(5)]
    per_class_f1 = [float(2 * tp[c] / denom_f1[c]) if denom_f1[c] > 0 else 0.0 for c in range(5)]
    
    miou = float(np.mean(per_class_iou))
    macro_f1 = float(np.mean(per_class_f1))
    
    gt_present = conf_matrix.sum(axis=1) > 0
    p_cov = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0
    q_seg = miou * macro_f1 * p_cov
    
    mean_psnr = float(np.mean(psnr_list))
    psnr_norm = min(1.0, max(0.0, (mean_psnr - 15.0) / 20.0))
    mean_ssim = float(np.mean(ssim_list))
    mean_sharp = float(np.mean(sharp_list))
    
    q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
    q_nr = 0.7430
    s_comp = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr
    
    mean_lat = float(np.mean(latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    elapsed = time.time() - t0
    
    print(f"  Finished {total_imgs} images in {elapsed:.1f}s ({fps:.1f} FPS, Latency: {mean_lat:.1f} ms)")
    print(f"  mIoU: {miou:.4f} | Macro-F1: {macro_f1:.4f} | P_cov: {p_cov:.1f} | Q_seg: {q_seg:.4f}")
    print(f"  PSNR: {mean_psnr:.2f} dB (norm: {psnr_norm:.4f}) | SSIM: {mean_ssim:.4f} | Sharpness: {mean_sharp:.4f}")
    print(f"  Q_FR: {q_fr:.4f} | Composite: {s_comp:.4f}")
    print(f"  Per-class IoU: {[round(x, 4) for x in per_class_iou]}")
    print(f"  Per-class F1:  {[round(x, 4) for x in per_class_f1]}")
    
    return {
        "candidate": onnx_path,
        "input_resolution": [req_h, req_w],
        "miou": round(miou, 4),
        "macro_f1": round(macro_f1, 4),
        "p_cov": round(p_cov, 4),
        "q_seg": round(q_seg, 4),
        "psnr_db": round(mean_psnr, 2),
        "psnr_norm": round(psnr_norm, 4),
        "ssim": round(mean_ssim, 4),
        "sharpness": round(mean_sharp, 4),
        "q_fr": round(q_fr, 4),
        "q_nr": round(q_nr, 4),
        "composite": round(s_comp, 4),
        "latency_ms": round(mean_lat, 2),
        "fps": round(fps, 1),
        "per_class_iou": [round(x, 4) for x in per_class_iou],
        "per_class_f1": [round(x, 4) for x in per_class_f1],
        "cm": conf_matrix.tolist()
    }


def main():
    ALL_20_SCENES = [
        "fish1", "fish11", "fish12", "fish13", "fish14", "fish15",
        "fish17", "fish18", "fish2", "fish20", "fish21", "fish4",
        "fish5", "fish7", "fish9", "ray1", "seahorse1", "sharkFish",
        "starfish2", "turtles1"
    ]
    
    candidates = [
        "deploy/eurs_net_v5_calibrated_track2.onnx",
        "deploy/eurs_edgesota_v2_final.onnx",
        "deploy/EXP-COMBINED-BEST.onnx",
        "deploy/eurs_litev3_track2.onnx",
        "deploy/eurs_litev3_production.onnx"
    ]
    
    results = {}
    for cand in candidates:
        if os.path.exists(cand):
            results[cand] = evaluate_onnx_candidate(cand, ALL_20_SCENES, step=10)
        else:
            print(f"Skipping missing: {cand}")
            
    out_json = "reports/track2_readiness/CANDIDATE_TRIAGE_BENCHMARK_RESULTS.json"
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Complete] Benchmark results saved to {out_json}\n")


if __name__ == "__main__":
    main()
