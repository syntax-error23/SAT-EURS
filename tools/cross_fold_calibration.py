"""
SAT-EURS Track 2 — 5-Fold Cross-Fold Logit Offset Calibration Engine
Performs strict cross-fold calibration:
For each fold k in [0, 1, 2, 3, 4]:
  1. Fit offsets beta^{(-k)} on the remaining 4 held-out folds (k not in {k}) to maximize official Q_seg.
  2. Evaluate beta^{(-k)} on the truly held-out fold k.
Pools the out-of-fold test confusion matrices to measure true generalization.
Compares:
  - Baseline (beta = [0, 0, 0, 0, 0])
  - Out-of-fold Calibrated (cross-validated beta)
  - Argon's Hardcoded Offsets [0.0, 0.0, 0.0, 0.10, 0.25]
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.track2_litev3 import build_track2_litev3_model


def compute_metrics_from_cm(cm: np.ndarray) -> Dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    
    per_class_iou = [float(tp[c] / denom_iou[c]) if denom_iou[c] > 0 else 0.0 for c in range(5)]
    per_class_f1 = [float(2 * tp[c] / denom_f1[c]) if denom_f1[c] > 0 else 0.0 for c in range(5)]
    
    miou = float(np.mean(per_class_iou))
    macro_f1 = float(np.mean(per_class_f1))
    
    gt_present = cm.sum(axis=1) > 0
    p_cov = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0
    q_seg = miou * macro_f1 * p_cov
    
    return {
        "miou": round(miou, 4),
        "macro_f1": round(macro_f1, 4),
        "p_cov": round(p_cov, 4),
        "q_seg": round(q_seg, 4),
        "per_class_iou": [round(x, 4) for x in per_class_iou],
        "per_class_f1": [round(x, 4) for x in per_class_f1],
        "cm": cm.tolist()
    }


def extract_fold_logits(
    ckpt_path: str,
    val_scenes: List[str],
    device: torch.device,
    step: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Runs model inference once per validation image.
    Returns list of (downsampled_logits [5, 192, 384], mask [192, 384]).
    """
    model = build_track2_litev3_model(pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    samples = []
    with torch.no_grad():
        for sc in val_scenes:
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"
            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            m_down = cv2.resize(m_im, (384, 192), interpolation=cv2.INTER_NEAREST)
            
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::step]
            for df in deg_files:
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                d_in = cv2.resize(d_rgb, (768, 384))
                x_th = torch.from_numpy(d_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                _, logits_t = model(x_th)
                # Downsample logits to [192, 384] for fast calibration search
                logits_down = F.interpolate(logits_t, size=(192, 384), mode="bilinear", align_corners=False)
                logits_np = logits_down.squeeze(0).cpu().numpy().astype(np.float32)
                
                samples.append((logits_np, m_down))
                
    return samples


def evaluate_offsets_on_samples(samples: List[Tuple[np.ndarray, np.ndarray]], beta: np.ndarray) -> np.ndarray:
    """
    Evaluates a candidate offset vector beta [5] on pre-extracted samples.
    Returns 5x5 confusion matrix.
    """
    cm = np.zeros((5, 5), dtype=np.int64)
    beta_reshaped = beta[:, np.newaxis, np.newaxis] # [5, 1, 1]
    
    for logits, mask in samples:
        calib_logits = logits + beta_reshaped
        pred = calib_logits.argmax(axis=0)
        
        valid = (mask >= 0) & (mask < 5)
        bins = 5 * mask[valid].astype(np.int64) + pred[valid].astype(np.int64)
        counts = np.bincount(bins, minlength=25)
        cm += counts.reshape(5, 5)
        
    return cm


def optimize_offsets(train_samples: List[Tuple[np.ndarray, np.ndarray]], max_iter: int = 3) -> np.ndarray:
    """
    Coordinate descent to find beta [0, beta1, beta2, beta3, beta4] maximizing Q_seg.
    Fixes beta[0] = 0.
    """
    beta = np.zeros(5, dtype=np.float32)
    
    # Grid search candidates
    candidate_steps = np.arange(-1.5, 2.0, 0.25, dtype=np.float32)
    
    best_cm = evaluate_offsets_on_samples(train_samples, beta)
    best_score = compute_metrics_from_cm(best_cm)["q_seg"]
    
    for it in range(max_iter):
        improved = False
        for c in [1, 2, 3, 4]: # Tune coral, fish, elasmobranch, other
            current_best_val = beta[c]
            for step_val in candidate_steps:
                if abs(step_val - current_best_val) < 1e-4:
                    continue
                trial_beta = beta.copy()
                trial_beta[c] = step_val
                
                trial_cm = evaluate_offsets_on_samples(train_samples, trial_beta)
                metrics = compute_metrics_from_cm(trial_cm)
                
                # Check for strictly better Q_seg with coverage == 1.0
                if metrics["p_cov"] == 1.0 and metrics["q_seg"] > best_score:
                    best_score = metrics["q_seg"]
                    beta[c] = step_val
                    improved = True
                    
        if not improved:
            break
            
    return beta


def run_cross_fold_calibration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print("SAT-EURS Track 2: 5-Fold Cross-Fold Logit Calibration")
    print(f"Device: {device}")
    print("=======================================================\n")
    
    with open("splits/manifest_5fold.json") as f:
        manifest_5fold = json.load(f)
        
    fold_samples = {}
    ckpts = {
        0: "checkpoints/litev3/E1_supervised_f0/best_model.pt",
        1: "checkpoints/litev3/E1_supervised_f1/best_model.pt",
        2: "checkpoints/litev3/E1_supervised_f2/best_model.pt",
        3: "checkpoints/litev3/E1_supervised_f3/best_model.pt",
        4: "checkpoints/litev3/E1_supervised_f4/best_model.pt",
    }
    
    # 1. Extract validation logits for all 5 folds
    for k in range(5):
        val_sc = manifest_5fold["folds"][f"fold_{k}"]["val_scenes"]
        ckpt = ckpts[k]
        if not os.path.exists(ckpt):
            print(f"Error: Missing checkpoint for fold {k}: {ckpt}")
            return
        print(f"Extracting held-out logits for Fold {k} (scenes: {val_sc})...", flush=True)
        fold_samples[k] = extract_fold_logits(ckpt, val_sc, device, step=5)
        print(f"  Extracted {len(fold_samples[k])} validation samples for Fold {k}.")
        
    # 2. Baseline evaluation (all beta = 0)
    baseline_cm = np.zeros((5, 5), dtype=np.int64)
    zero_beta = np.zeros(5, dtype=np.float32)
    for k in range(5):
        baseline_cm += evaluate_offsets_on_samples(fold_samples[k], zero_beta)
        
    baseline_res = compute_metrics_from_cm(baseline_cm)
    print("\n-------------------------------------------------------")
    print("BASELINE OUT-OF-FOLD PERFORMANCE (beta = [0, 0, 0, 0, 0]):")
    print(f"  mIoU:     {baseline_res['miou']:.4f}")
    print(f"  Macro-F1: {baseline_res['macro_f1']:.4f}")
    print(f"  P_cov:    {baseline_res['p_cov']:.1f}")
    print(f"  Q_seg:    {baseline_res['q_seg']:.4f}")
    print(f"  Per-class IoU: {baseline_res['per_class_iou']}")
    print(f"  Per-class F1:  {baseline_res['per_class_f1']}")
    print("-------------------------------------------------------\n")
    
    # 3. Argon's Proposed Offsets Diagnostic
    argon_beta = np.array([0.0, 0.0, 0.0, 0.10, 0.25], dtype=np.float32)
    argon_cm = np.zeros((5, 5), dtype=np.int64)
    for k in range(5):
        argon_cm += evaluate_offsets_on_samples(fold_samples[k], argon_beta)
    argon_res = compute_metrics_from_cm(argon_cm)
    print("ARGON PROPOSED OFFSETS [0.0, 0.0, 0.0, 0.10, 0.25]:")
    print(f"  mIoU:     {argon_res['miou']:.4f}")
    print(f"  Macro-F1: {argon_res['macro_f1']:.4f}")
    print(f"  P_cov:    {argon_res['p_cov']:.1f}")
    print(f"  Q_seg:    {argon_res['q_seg']:.4f}")
    print(f"  Per-class IoU: {argon_res['per_class_iou']}")
    print(f"  Per-class F1:  {argon_res['per_class_f1']}")
    print("-------------------------------------------------------\n")
    
    # 4. Strict Cross-Fold Calibration (Train on 4 folds, Test on 5th)
    cv_oof_cm = np.zeros((5, 5), dtype=np.int64)
    cv_betas = {}
    
    for k in range(5):
        print(f"[Fold {k} Rotation] Fitting beta on 4 remaining folds...")
        train_samples = []
        for j in range(5):
            if j != k:
                train_samples.extend(fold_samples[j])
                
        best_beta_k = optimize_offsets(train_samples)
        cv_betas[k] = [round(float(x), 4) for x in best_beta_k]
        print(f"  Fold {k} Optimal Training Offsets beta: {cv_betas[k]}")
        
        # Test on strictly held-out Fold k
        test_cm_k = evaluate_offsets_on_samples(fold_samples[k], best_beta_k)
        cv_oof_cm += test_cm_k
        
    calib_oof_res = compute_metrics_from_cm(cv_oof_cm)
    print("\n=======================================================")
    print("STRICT CROSS-FOLD OUT-OF-SAMPLE EVALUATION RESULTS:")
    print(f"  mIoU:     {calib_oof_res['miou']:.4f} (Baseline: {baseline_res['miou']:.4f})")
    print(f"  Macro-F1: {calib_oof_res['macro_f1']:.4f} (Baseline: {baseline_res['macro_f1']:.4f})")
    print(f"  P_cov:    {calib_oof_res['p_cov']:.1f} (Baseline: {baseline_res['p_cov']:.1f})")
    print(f"  Q_seg:    {calib_oof_res['q_seg']:.4f} (Baseline: {baseline_res['q_seg']:.4f})")
    print(f"  Delta Q_seg: {calib_oof_res['q_seg'] - baseline_res['q_seg']:+.4f}")
    print(f"  Per-class IoU: {calib_oof_res['per_class_iou']}")
    print(f"  Per-class F1:  {calib_oof_res['per_class_f1']}")
    print("=======================================================\n")
    
    # 5. Fit pooled global beta for production export (if validated)
    all_samples = []
    for k in range(5):
        all_samples.extend(fold_samples[k])
    production_beta = optimize_offsets(all_samples)
    print(f"Production Calibrated Offsets beta*: {[round(float(x), 4) for x in production_beta]}\n")
    
    # Save Report
    report = {
        "baseline_oof": baseline_res,
        "argon_offsets_oof": argon_res,
        "cv_calibrated_oof": calib_oof_res,
        "cv_fold_betas": cv_betas,
        "production_beta": [round(float(x), 4) for x in production_beta]
    }
    
    out_json = "reports/track2_sprint/PHASE1_CALIBRATION_RESULTS.json"
    with open(out_json, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"[Complete] Calibration report saved to {out_json}")


if __name__ == "__main__":
    run_cross_fold_calibration()
