"""
SAT-EURS Track 2 — Pooled Out-of-Sample Evaluation Engine
Evaluates Student S0 on Fold 0 held-out scenes and Student S1 on Fold 1 held-out scenes.
Pools the out-of-sample confusion matrix across all 8 scenes (320-800 images).
Computes:
- Macro Q_seg over all 5 classes
- Per-class IoU and Recall
- Coverage P_cov
- PSNR, SSIM, Sharpness, and normalized Q_FR
- Local composite score: S_local_90 = 0.60 * Q_seg + 0.30 * Q_FR
- Official verification gates
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.track2_litev3 import build_track2_litev3_model
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_single_fold(
    ckpt_path: str,
    val_scenes: List[str],
    device: torch.device
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    psnr_list, ssim_list, sharp_list = [], [], []
    
    model = build_track2_litev3_model(pretrained_backbone=False, unsharp_factor=1.20).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    
    with torch.no_grad():
        for sc in val_scenes:
            c_p = f"data/cached_1024/clean/{sc}.png"
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"
            
            c_im = cv2.imread(c_p)
            c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::5] # 40 per scene = 160 images
            
            for df in deg_files:
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                d_in = cv2.resize(d_rgb, (768, 384))
                x_th = torch.from_numpy(d_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                restored_t, logits_t = model(x_th)
                
                res_np = restored_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                res_1024 = cv2.resize(res_np, (1024, 1024))
                
                logits_1024 = F.interpolate(logits_t, size=(1024, 1024), mode="bilinear", align_corners=False)
                pred_mask = logits_1024.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)
                
                valid = (m_im >= 0) & (m_im < num_classes)
                bins = num_classes * m_im[valid].astype(np.int64) + pred_mask[valid].astype(np.int64)
                counts = np.bincount(bins, minlength=num_classes * num_classes)
                conf_matrix += counts.reshape(num_classes, num_classes)
                
                p, _ = compute_psnr_and_norm(res_1024, c_rgb)
                s = compute_ssim_single(res_1024, c_rgb)
                sh = compute_sharpness_ratio(res_1024, d_rgb)
                psnr_list.append(p)
                ssim_list.append(s)
                sharp_list.append(sh)
                
    return conf_matrix, psnr_list, ssim_list, sharp_list


def evaluate_pooled(ckpt_f0: str, ckpt_f1: str, exp_name: str) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"SAT-EURS Track 2 Pooled Evaluation: {exp_name}")
    print(f"Device: {device}")
    print(f"Student S0 (eval on Fold 0): {ckpt_f0}")
    print(f"Student S1 (eval on Fold 1): {ckpt_f1}")
    print(f"=======================================================\n")
    
    with open("splits/manifest_5fold.json") as f:
        manifest_5fold = json.load(f)
        
    val_scenes_f0 = manifest_5fold["folds"]["fold_0"]["val_scenes"]
    val_scenes_f1 = manifest_5fold["folds"]["fold_1"]["val_scenes"]
    
    print(f"[Fold 0] Evaluating S0 on {val_scenes_f0}...", flush=True)
    c0, psnr0, ssim0, sharp0 = evaluate_single_fold(ckpt_f0, val_scenes_f0, device)
    
    print(f"[Fold 1] Evaluating S1 on {val_scenes_f1}...", flush=True)
    c1, psnr1, ssim1, sharp1 = evaluate_single_fold(ckpt_f1, val_scenes_f1, device)
    
    # Pool confusion matrices and restoration metrics
    conf_pooled = c0 + c1
    psnr_all = psnr0 + psnr1
    ssim_all = ssim0 + ssim1
    sharp_all = sharp0 + sharp1
    
    num_classes = 5
    ious = []
    recalls = []
    precisions = []
    f1s = []
    
    for c in range(num_classes):
        tp = conf_pooled[c, c]
        fn = conf_pooled[c, :].sum() - tp
        fp = conf_pooled[:, c].sum() - tp
        denom_iou = tp + fn + fp
        gt_c = conf_pooled[c, :].sum()
        pred_c = conf_pooled[:, c].sum()
        
        iou_c = float(tp / denom_iou) if denom_iou > 0 else 0.0
        rec_c = float(tp / gt_c) if gt_c > 0 else 0.0
        prec_c = float(tp / pred_c) if pred_c > 0 else 0.0
        f1_c = float(2.0 * tp / (gt_c + pred_c)) if (gt_c + pred_c) > 0 else 0.0
        
        ious.append(iou_c)
        recalls.append(rec_c)
        precisions.append(prec_c)
        f1s.append(f1_c)
        
    q_seg = float(np.mean(ious))
    macro_f1 = float(np.mean(f1s))
    
    mean_psnr = float(np.mean(psnr_all))
    mean_ssim = float(np.mean(ssim_all))
    mean_sharp = float(np.mean(sharp_all))
    
    q_fr = float(
        0.40 * np.clip((mean_psnr - 10.0) / 25.0, 0.0, 1.0) +
        0.35 * np.clip(mean_ssim, 0.0, 1.0) +
        0.25 * np.clip(mean_sharp / 1.50, 0.0, 1.0)
    )
    
    s_local_90 = 0.60 * q_seg + 0.30 * q_fr
    p_cov = 1.0 if all(conf_pooled[:, c].sum() > 0 for c in range(num_classes)) else 0.0
    
    class_names = ["Background", "Fish", "Starfish", "Shark/Ray", "Seahorse/Turtle"]
    
    print("\n--- POOLED OUT-OF-SAMPLE METRICS (8 Scenes, 320 Images) ---")
    print(f"Composite S_local_90: {s_local_90:.4f}")
    print(f"Segmentation Q_seg:   {q_seg:.4f} (Macro-F1: {macro_f1:.4f})")
    print(f"Restoration Q_FR:     {q_fr:.4f} (PSNR: {mean_psnr:.2f} dB, SSIM: {mean_ssim:.4f}, Sharpness: {mean_sharp:.4f})")
    print(f"Coverage P_cov:       {p_cov:.1f}")
    print("\nPer-Class Breakdown:")
    for c in range(num_classes):
        print(f"  Class {c} ({class_names[c]:<16}): IoU={ious[c]:.4f}, Recall={recalls[c]:.4f}, Precision={precisions[c]:.4f}, GT_Pixels={conf_pooled[c, :].sum():,}")
        
    result = {
        "exp_name": exp_name,
        "s_local_90": round(s_local_90, 4),
        "q_seg": round(q_seg, 4),
        "macro_f1": round(macro_f1, 4),
        "q_fr": round(q_fr, 4),
        "psnr": round(mean_psnr, 2),
        "ssim": round(mean_ssim, 4),
        "sharpness": round(mean_sharp, 4),
        "p_cov": p_cov,
        "ious": [round(x, 4) for x in ious],
        "recalls": [round(x, 4) for x in recalls],
        "precisions": [round(x, 4) for x in precisions],
        "f1s": [round(x, 4) for x in f1s],
        "conf_matrix_pooled": conf_pooled.tolist()
    }
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_f0", type=str, required=True)
    parser.add_argument("--ckpt_f1", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()
    
    res = evaluate_pooled(args.ckpt_f0, args.ckpt_f1, args.exp_name)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as fp:
            json.dump(res, fp, indent=2)
        print(f"\nSaved metrics to {args.out_json}")
