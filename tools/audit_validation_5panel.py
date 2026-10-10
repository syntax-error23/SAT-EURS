"""
Comprehensive 5-Panel Validation Diagnostic Audit for EURS-Net V2.
Saves [Degraded | Restored | Clean GT | GT Mask | Pred Mask] side-by-side plots.
Reports per-image and aggregate PSNR, SSIM, mean, std, min, max, and finite checks.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import torch
import numpy as np
from models.eurs_net_v2 import EURSNetV2
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single

def run_diagnostic_audit():
    out_dir = Path("experiments/plots/audit_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    model = EURSNetV2(num_classes=5).cuda().eval()

    PALETTE = np.array([
        [30, 30, 30],      # 0: Bg (Dark gray)
        [255, 128, 0],     # 1: Coral (Orange)
        [255, 255, 0],     # 2: Fish (Yellow)
        [0, 128, 255],     # 3: Shark/Ray (Cyan/Blue)
        [255, 0, 128],     # 4: Other (Magenta/Red)
    ], dtype=np.uint8)

    print("=" * 75)
    print("      5-PANEL VALIDATION DIAGNOSTIC AUDIT (UNTRAINED ZERO-INIT BASELINE)")
    print("=" * 75)

    metrics_log = []

    for s in val_scenes:
        clean = cv2.imread(f"data/clean/{s}.png")
        clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_rgb = cv2.resize(clean_rgb, (1024, 1024))

        mask = cv2.imread(f"data/masks/{s}.png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        deg_files = sorted(list(Path(f"data/degraded/{s}").glob("*.png")))
        deg_file = deg_files[0]
        deg = cv2.imread(str(deg_file))
        deg_rgb = cv2.cvtColor(deg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        deg_rgb = cv2.resize(deg_rgb, (1024, 1024))

        inp = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).cuda().float()
        with torch.no_grad():
            restored, seg_logits = model(inp)

        rest_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred_mask = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        psnr, psnr_norm = compute_psnr_and_norm(rest_np, clean_rgb)
        ssim = compute_ssim_single(rest_np, clean_rgb)

        is_finite = bool(np.isfinite(rest_np).all() and np.isfinite(seg_logits.cpu().numpy()).all())

        stats = {
            'scene': s,
            'psnr': psnr,
            'ssim': ssim,
            'mean': float(rest_np.mean()),
            'std': float(rest_np.std()),
            'min': float(rest_np.min()),
            'max': float(rest_np.max()),
            'finite': is_finite,
            'gt_classes': sorted(list(np.unique(mask))),
            'pred_classes': sorted(list(np.unique(pred_mask)))
        }
        metrics_log.append(stats)

        print(f"Scene: {s:10s} | PSNR: {psnr:5.2f} dB | SSIM: {ssim:5.4f} | "
              f"Mean: {stats['mean']:.3f} | Std: {stats['std']:.3f} | "
              f"Range: [{stats['min']:.3f}, {stats['max']:.3f}] | Finite: {is_finite} | "
              f"GT Classes: {stats['gt_classes']} | Pred Classes: {stats['pred_classes']}")

        # Render 5 panels: [Degraded | Restored | Clean GT | GT Mask | Pred Mask]
        deg_vis = (deg_rgb * 255).astype(np.uint8)
        rest_vis = (np.clip(rest_np, 0, 1) * 255).astype(np.uint8)
        clean_vis = (clean_rgb * 255).astype(np.uint8)
        gt_mask_rgb = PALETTE[mask]
        pred_mask_rgb = PALETTE[pred_mask]

        panel = np.hstack([deg_vis, rest_vis, clean_vis, gt_mask_rgb, pred_mask_rgb])
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

        w = 1024
        cv2.putText(panel_bgr, f"Degraded ({s})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(panel_bgr, f"Restored Output ({psnr:.1f} dB)", (w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(panel_bgr, "Clean GT", (2 * w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(panel_bgr, f"GT Mask {stats['gt_classes']}", (3 * w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(panel_bgr, f"Pred Mask (Init)", (4 * w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        save_path = out_dir / f"panel_5_{s}.png"
        cv2.imwrite(str(save_path), panel_bgr)

    # Aggregate metrics
    mean_psnr = np.mean([m['psnr'] for m in metrics_log])
    mean_ssim = np.mean([m['ssim'] for m in metrics_log])
    all_finite = all(m['finite'] for m in metrics_log)

    print("-" * 75)
    print(f"Aggregate Untrained Metrics:")
    print(f"  - Mean PSNR:    {mean_psnr:.2f} dB")
    print(f"  - Mean SSIM:    {mean_ssim:.4f}")
    print(f"  - 100% Finite:  {all_finite}")
    print(f"  - Saved plots:  {out_dir}/")
    print("=" * 75)

if __name__ == "__main__":
    run_diagnostic_audit()
