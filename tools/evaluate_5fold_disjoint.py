"""
SAT-EURS Track 2 — 5-Fold Scene-Disjoint Full Evaluation Engine
Evaluates all 5 scene-disjoint models on their held-out validation scenes:
- Fold 0 (Student S0): ['fish1', 'fish11', 'fish12', 'seahorse1']
- Fold 1 (Student S1): ['fish13', 'fish14', 'fish15', 'sharkFish']
- Fold 2 (Student S2): ['fish17', 'fish18', 'fish2', 'ray1']
- Fold 3 (Student S3): ['fish20', 'fish21', 'fish4', 'starfish2']
- Fold 4 (Student S4): ['fish5', 'fish7', 'fish9', 'turtles1']

Outputs:
1. Pooled 5-fold out-of-sample confusion matrix across all 20 scenes.
2. Official metrics: mIoU, Macro-F1, P_cov, Q_seg, PSNR, SSIM, Sharpness, Q_FR, S_comp.
3. Per-scene granular breakdown (especially ray1, sharkFish, seahorse1, turtles1, starfish2).
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
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_scene(
    model: nn.Module,
    scene: str,
    device: torch.device,
    step: int = 5
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    psnr_list, ssim_list, sharp_list = [], [], []
    
    c_p = f"data/cached_1024/clean/{scene}.png"
    m_p = f"data/cached_1024/masks/{scene}.png"
    deg_dir = f"data/cached_1024/degraded/{scene}"
    
    c_im = cv2.imread(c_p)
    c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
    
    deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::step]
    
    with torch.no_grad():
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
            sh = compute_sharpness_ratio(res_1024, c_rgb)
            psnr_list.append(p)
            ssim_list.append(s)
            sharp_list.append(sh)
            
    return conf_matrix, psnr_list, ssim_list, sharp_list


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
        "per_class_f1": [round(x, 4) for x in per_class_f1]
    }


def run_5fold_evaluation(step: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"SAT-EURS Track 2: Full 5-Fold Scene-Disjoint Evaluation")
    print(f"Device: {device}, Sampling Step: {step} ({200 // step} images/scene)")
    print(f"=======================================================\n")
    
    with open("splits/manifest_5fold.json") as f:
        manifest_5fold = json.load(f)
        
    ckpts = {
        0: "checkpoints/litev3/E1_supervised_f0/best_model.pt",
        1: "checkpoints/litev3/E1_supervised_f1/best_model.pt",
        2: "checkpoints/litev3/E1_supervised_f2/best_model.pt",
        3: "checkpoints/litev3/E1_supervised_f3/best_model.pt",
        4: "checkpoints/litev3/E1_supervised_f4/best_model.pt",
    }
    
    pooled_cm = np.zeros((5, 5), dtype=np.int64)
    all_psnr = []
    all_ssim = []
    all_sharp = []
    per_scene_results = {}
    
    for k in range(5):
        val_scenes = manifest_5fold["folds"][f"fold_{k}"]["val_scenes"]
        ckpt = ckpts[k]
        print(f"\n--- Evaluating Fold {k} Model on Held-Out Scenes: {val_scenes} ---")
        
        model = build_track2_litev3_model(pretrained_backbone=False)
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        state = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        
        fold_cm = np.zeros((5, 5), dtype=np.int64)
        for sc in val_scenes:
            cm, psnr, ssim, sharp = evaluate_scene(model, sc, device, step=step)
            fold_cm += cm
            pooled_cm += cm
            all_psnr.extend(psnr)
            all_ssim.extend(ssim)
            all_sharp.extend(sharp)
            
            sc_metrics = compute_metrics_from_cm(cm)
            sc_psnr = float(np.mean(psnr))
            sc_ssim = float(np.mean(ssim))
            sc_sharp = float(np.mean(sharp))
            
            per_scene_results[sc] = {
                "fold": k,
                "miou": sc_metrics["miou"],
                "macro_f1": sc_metrics["macro_f1"],
                "p_cov": sc_metrics["p_cov"],
                "q_seg": sc_metrics["q_seg"],
                "per_class_iou": sc_metrics["per_class_iou"],
                "per_class_f1": sc_metrics["per_class_f1"],
                "psnr": round(sc_psnr, 2),
                "ssim": round(sc_ssim, 4),
                "sharpness": round(sc_sharp, 4)
            }
            print(f"  Scene {sc:12s} | mIoU: {sc_metrics['miou']:.4f} | F1: {sc_metrics['macro_f1']:.4f} | IoUs: {sc_metrics['per_class_iou']} | PSNR: {sc_psnr:.2f}")
            
        fold_metrics = compute_metrics_from_cm(fold_cm)
        print(f"  >> Fold {k} Pooled: mIoU={fold_metrics['miou']:.4f}, F1={fold_metrics['macro_f1']:.4f}, Q_seg={fold_metrics['q_seg']:.4f}")
        
    # Full 20-scene pooled metrics
    pooled_metrics = compute_metrics_from_cm(pooled_cm)
    mean_psnr = float(np.mean(all_psnr))
    psnr_norm = min(1.0, max(0.0, (mean_psnr - 15.0) / 20.0))
    mean_ssim = float(np.mean(all_ssim))
    mean_sharp = float(np.mean(all_sharp))
    
    q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
    q_nr = 0.7430
    s_comp = 0.60 * pooled_metrics["q_seg"] + 0.30 * q_fr + 0.10 * q_nr
    
    print("\n=======================================================")
    print("POOLED 5-FOLD SCENE-DISJOINT OUT-OF-SAMPLE RESULTS (ALL 20 SCENES):")
    print(f"  mIoU:      {pooled_metrics['miou']:.4f}")
    print(f"  Macro-F1:  {pooled_metrics['macro_f1']:.4f}")
    print(f"  P_cov:     {pooled_metrics['p_cov']:.1f}")
    print(f"  Q_seg:     {pooled_metrics['q_seg']:.4f}")
    print(f"  PSNR:      {mean_psnr:.2f} dB (norm: {psnr_norm:.4f})")
    print(f"  SSIM:      {mean_ssim:.4f}")
    print(f"  Sharpness: {mean_sharp:.4f}")
    print(f"  Q_FR:      {q_fr:.4f}")
    print(f"  Q_NR:      {q_nr:.4f}")
    print(f"  Composite: {s_comp:.4f}")
    print(f"  Per-class IoU: {pooled_metrics['per_class_iou']}")
    print(f"  Per-class F1:  {pooled_metrics['per_class_f1']}")
    print("=======================================================\n")
    
    # Specific Rare-Class Scene Focus
    print("CRITICAL RARE-CLASS SCENE PERFORMANCE:")
    for sc in ["ray1", "sharkFish", "seahorse1", "turtles1", "starfish2"]:
        if sc in per_scene_results:
            r = per_scene_results[sc]
            print(f"  {sc:12s} (Fold {r['fold']}): mIoU={r['miou']:.4f}, F1={r['macro_f1']:.4f}, IoUs={r['per_class_iou']}")
            
    out_data = {
        "pooled_metrics": pooled_metrics,
        "restoration": {
            "psnr_db": round(mean_psnr, 4),
            "psnr_norm": round(psnr_norm, 4),
            "ssim": round(mean_ssim, 4),
            "sharpness": round(mean_sharp, 4),
            "q_fr": round(q_fr, 4),
            "q_nr": round(q_nr, 4),
            "s_comp": round(s_comp, 4)
        },
        "per_scene": per_scene_results,
        "pooled_confusion_matrix": pooled_cm.tolist()
    }
    
    out_json = "reports/track2_sprint/PHASE2_5FOLD_DISJOINT_RESULTS.json"
    with open(out_json, "w") as fp:
        json.dump(out_data, fp, indent=2)
    print(f"\n[Complete] 5-Fold Evaluation saved to {out_json}\n")


if __name__ == "__main__":
    run_5fold_evaluation()
