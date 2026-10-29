"""
SAT-EURS Track 2 — Phase 1: Restoration Sharpening Filter Comparison Engine
Evaluates 4 sharpening variants on held-out out-of-sample scenes (Fold 0 and Fold 1):
1. Existing global unsharp (factor = 1.20)
2. No unsharp filtering (factor = 1.00)
3. Reduced global sharpening (factor = 1.10)
4. Edge-gated sharpening (base strength = 0.45, threshold = 0.08)

Measures:
- PSNR (dB), PSNR_norm [15-35 dB]
- SSIM
- Sharpness ratio against ground-truth clean image
- Q_FR = 0.45 * PSNR_norm + 0.45 * SSIM + 0.10 * Sharpness
- Downstream segmentation: mIoU, Macro-F1, P_cov, Q_seg = mIoU * Macro-F1 * P_cov
- Full composite score: S_comp = 0.60 * Q_seg + 0.30 * Q_FR + 0.10 * Q_NR
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
from metrics.segmentation_metrics import SegmentationEvaluator


class FilterVariantWrapper(nn.Module):
    """
    Wraps SATEURSLiteV3 model to test different sharpening variants dynamically.
    """
    def __init__(self, base_model: nn.Module, mode: str = "global_1.20"):
        super().__init__()
        self.base = base_model
        self.mode = mode
        
        # Sobel kernels for edge-gated sharpening
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        
        # 1. RSUIGM Optical Polynomial Reconstruction
        x_small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        r_feat = self.base.router(x_small).flatten(1)
        r_logits = self.base.router_fc(r_feat)
        r_logits_scaled = (r_logits.float() - r_logits.float().max(dim=-1, keepdim=True).values) / self.base.routing_temperature
        alpha = F.softmax(r_logits_scaled, dim=-1).to(r_logits.dtype)
        
        w_eff = torch.matmul(alpha, self.base.table_bank).view(b, 19, 3)
        phi = self.base._build_poly_features(x)
        phi_flat = phi.permute(0, 2, 3, 1).reshape(b, h * w, 19)
        res_flat = torch.bmm(phi_flat, w_eff)
        res_raw = res_flat.permute(0, 2, 1).reshape(b, 3, h, w).clamp(0.0, 1.0)
        
        # 2. Sharpening Filter Variants
        blur = F.conv2d(res_raw, self.base.unsharp_blur_kernel, padding=1, groups=3)
        
        if self.mode == "global_1.20":
            restored = torch.clamp(1.20 * res_raw - 0.20 * blur, 0.0, 1.0)
        elif self.mode == "none_1.00":
            restored = res_raw
        elif self.mode == "reduced_1.10":
            restored = torch.clamp(1.10 * res_raw - 0.10 * blur, 0.0, 1.0)
        elif self.mode == "edge_gated":
            # Grayscale luminance
            gray = 0.299 * res_raw[:, 0:1] + 0.587 * res_raw[:, 1:2] + 0.114 * res_raw[:, 2:3]
            gx = F.conv2d(gray, self.sobel_x, padding=1)
            gy = F.conv2d(gray, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
            
            # Base strength 0.45, threshold 0.08
            edge_thresh = 0.08
            edge_mask = torch.clamp((grad_mag - edge_thresh) / (0.30 - edge_thresh + 1e-6), 0.0, 1.0)
            unsharp_delta = res_raw - blur
            restored = torch.clamp(res_raw + 0.45 * edge_mask * unsharp_delta, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        # 3. Downstream Segmentation
        x_norm = (restored.detach() - self.base.norm_mean) / self.base.norm_std
        curr = x_norm
        f4, f8, f16 = None, None, None
        for i, blk in enumerate(self.base.backbone_blocks):
            curr = blk(curr)
            if i == 3:
                f4 = curr
            elif i == 6:
                f8 = curr
            elif i == 11:
                f16 = curr
                
        seg_logits = self.base.decoder(f4, f8, f16, (h, w))
        return restored, seg_logits


def evaluate_filter_on_fold(
    ckpt_path: str,
    val_scenes: List[str],
    mode: str,
    device: torch.device,
    step: int = 5
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    base_model = build_track2_litev3_model(pretrained_backbone=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    base_model.load_state_dict(state)
    
    wrapper = FilterVariantWrapper(base_model, mode=mode).to(device)
    wrapper.eval()
    
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    psnr_list, ssim_list, sharp_list = [], [], []
    
    with torch.no_grad():
        for sc in val_scenes:
            c_p = f"data/cached_1024/clean/{sc}.png"
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"
            
            c_im = cv2.imread(c_p)
            c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::step]
            
            for df in deg_files:
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                d_in = cv2.resize(d_rgb, (768, 384))
                x_th = torch.from_numpy(d_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                restored_t, logits_t = wrapper(x_th)
                
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
                # Sharpness measured strictly against clean ground-truth reference
                sh = compute_sharpness_ratio(res_1024, c_rgb)
                psnr_list.append(p)
                ssim_list.append(s)
                sharp_list.append(sh)
                
    return conf_matrix, psnr_list, ssim_list, sharp_list


def run_phase1_sharpening_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"SAT-EURS Track 2 Phase 1: Restoration Sharpening Sweep")
    print(f"Device: {device}")
    print(f"=======================================================\n")
    
    with open("splits/manifest_5fold.json") as f:
        manifest_5fold = json.load(f)
        
    val_scenes_f0 = manifest_5fold["folds"]["fold_0"]["val_scenes"]
    val_scenes_f1 = manifest_5fold["folds"]["fold_1"]["val_scenes"]
    ckpt_f0 = "checkpoints/litev3/E1_supervised_f0/best_model.pt"
    ckpt_f1 = "checkpoints/litev3/E1_supervised_f1/best_model.pt"
    
    variants = [
        ("global_1.20", "1. Existing Global Unsharp (gamma=1.20)"),
        ("none_1.00", "2. No Unsharp Filtering (gamma=1.00)"),
        ("reduced_1.10", "3. Reduced Global Sharpening (gamma=1.10)"),
        ("edge_gated", "4. Edge-Gated Sharpening (base=0.45, thresh=0.08)")
    ]
    
    results = {}
    q_nr_calibrated = 0.7430 # Calibrated from official leaderboard
    
    for mode, label in variants:
        t0 = time.time()
        print(f"\nEvaluating Variant: {label}...")
        c0, p0, s0, sh0 = evaluate_filter_on_fold(ckpt_f0, val_scenes_f0, mode, device)
        c1, p1, s1, sh1 = evaluate_filter_on_fold(ckpt_f1, val_scenes_f1, mode, device)
        
        pooled_cm = c0 + c1
        all_psnr = p0 + p1
        all_ssim = s0 + s1
        all_sharp = sh0 + sh1
        
        # Segmentation Metrics
        tp = np.diag(pooled_cm).astype(float)
        fp = pooled_cm.sum(axis=0) - tp
        fn = pooled_cm.sum(axis=1) - tp
        
        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        per_class_iou = [float(tp[c] / denom_iou[c]) if denom_iou[c] > 0 else 0.0 for c in range(5)]
        per_class_f1 = [float(2 * tp[c] / denom_f1[c]) if denom_f1[c] > 0 else 0.0 for c in range(5)]
        
        miou = float(np.mean(per_class_iou))
        macro_f1 = float(np.mean(per_class_f1))
        
        gt_present = pooled_cm.sum(axis=1) > 0
        p_cov = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0
        q_seg = miou * macro_f1 * p_cov
        
        # Restoration Metrics
        mean_psnr = float(np.mean(all_psnr))
        psnr_norm = min(1.0, max(0.0, (mean_psnr - 15.0) / 20.0))
        mean_ssim = float(np.mean(all_ssim))
        mean_sharp = float(np.mean(all_sharp))
        
        q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
        s_comp = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr_calibrated
        
        elapsed = time.time() - t0
        print(f"  Finished in {elapsed:.1f}s")
        print(f"  PSNR: {mean_psnr:.2f} dB (norm: {psnr_norm:.4f}) | SSIM: {mean_ssim:.4f} | Sharpness: {mean_sharp:.4f}")
        print(f"  Q_FR: {q_fr:.4f}")
        print(f"  mIoU: {miou:.4f} | Macro-F1: {macro_f1:.4f} | P_cov: {p_cov:.1f} | Q_seg: {q_seg:.4f}")
        print(f"  Official Composite S_comp: {s_comp:.4f}")
        print(f"  Per-class IoU: {[round(x, 4) for x in per_class_iou]}")
        print(f"  Per-class F1:  {[round(x, 4) for x in per_class_f1]}")
        
        results[mode] = {
            "label": label,
            "psnr_db": round(mean_psnr, 4),
            "psnr_norm": round(psnr_norm, 4),
            "ssim": round(mean_ssim, 4),
            "sharpness": round(mean_sharp, 4),
            "q_fr": round(q_fr, 4),
            "q_nr": round(q_nr_calibrated, 4),
            "miou": round(miou, 4),
            "macro_f1": round(macro_f1, 4),
            "p_cov": round(p_cov, 4),
            "q_seg": round(q_seg, 4),
            "s_comp": round(s_comp, 4),
            "per_class_iou": [round(x, 4) for x in per_class_iou],
            "per_class_f1": [round(x, 4) for x in per_class_f1]
        }
        
    out_path = "reports/track2_sprint/PHASE1_SHARPENING_RESULTS.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n[Complete] Results saved to {out_path}\n")


if __name__ == "__main__":
    run_phase1_sharpening_experiment()
