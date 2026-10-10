"""
Quick inference and diagnostic script to visually inspect model restoration quality on validation samples.
Calculates PSNR, SSIM, and channel statistics for degraded, restored, and clean GT images.
Saves comparison plots to experiments/plots/inference_inspect/
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import torch
from models import EURSNetV2
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single

def inspect_validation_samples(checkpoint_path="checkpoints/best_model.pt", num_samples=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INSPECT] Loading model on {device} from {checkpoint_path}...")
    
    model = EURSNetV2(num_classes=5)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Find 3 sample validation images
    val_scenes = ["ray1", "fish13", "sharkFish"]
    out_dir = Path("experiments/plots/inference_inspect")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("           VALIDATION RESTORATION DIAGNOSTIC AUDIT")
    print("=" * 70)

    for scene in val_scenes:
        clean_path = Path(f"data/clean/{scene}.png")
        deg_dir = Path(f"data/degraded/{scene}")
        if not clean_path.exists() or not deg_dir.exists():
            print(f"Skipping {scene}, files not found")
            continue

        deg_files = sorted([f for f in deg_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg')])
        if not deg_files:
            continue
        
        # Pick the first degraded image in this scene
        deg_path = deg_files[0]
        
        # Load degraded and clean
        deg_bgr = cv2.imread(str(deg_path))
        clean_bgr = cv2.imread(str(clean_path))
        
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if deg_rgb.shape[:2] != (1024, 1024):
            deg_rgb = cv2.resize(deg_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        if clean_rgb.shape[:2] != (1024, 1024):
            clean_rgb = cv2.resize(clean_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            restored_t, seg_logits = model(deg_t)
            
            # Extract intermediate white balance and residual delta
            wb_t = model.fwb(deg_t)
            stem, f256, f128, f64, f32 = model.backbone(wb_t)
            # Decoder delta
            proj128 = model.restoration_decoder.proj_f128(f128)
            up1 = torch.nn.functional.interpolate(proj128, scale_factor=2.0, mode='bilinear', align_corners=False)
            x = up1 + model.restoration_decoder.proj_f256(f256)
            x = model.restoration_decoder.dw_conv256(x)
            up2 = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
            x = up2 + model.restoration_decoder.proj_stem(stem)
            x = model.restoration_decoder.dw_conv512(x)
            delta_512 = model.restoration_decoder.head(x)
            delta_1024 = torch.nn.functional.interpolate(delta_512, size=(1024, 1024), mode='bilinear', align_corners=False)

        restored_np = restored_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        delta_np = delta_1024.squeeze(0).permute(1, 2, 0).cpu().numpy()
        wb_np = wb_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Compute metrics
        raw_psnr, raw_norm = compute_psnr_and_norm(deg_rgb, clean_rgb)
        raw_ssim = compute_ssim_single(deg_rgb, clean_rgb)
        
        rest_psnr, rest_norm = compute_psnr_and_norm(restored_np, clean_rgb)
        rest_ssim = compute_ssim_single(restored_np, clean_rgb)

        print(f"\nScene: {scene} | Sample: {deg_path.name}")
        print(f"  [Baseline: Raw Degraded vs Clean GT]")
        print(f"    PSNR: {raw_psnr:.2f} dB | SSIM: {raw_ssim:.4f}")
        print(f"    Degraded RGB Mean: [{deg_rgb[...,0].mean():.3f}, {deg_rgb[...,1].mean():.3f}, {deg_rgb[...,2].mean():.3f}]")
        print(f"    Clean GT RGB Mean: [{clean_rgb[...,0].mean():.3f}, {clean_rgb[...,1].mean():.3f}, {clean_rgb[...,2].mean():.3f}]")
        print(f"  [Model Output: Restored vs Clean GT]")
        print(f"    PSNR: {rest_psnr:.2f} dB | SSIM: {rest_ssim:.4f}")
        print(f"    Restored RGB Mean: [{restored_np[...,0].mean():.3f}, {restored_np[...,1].mean():.3f}, {restored_np[...,2].mean():.3f}]")
        print(f"    Residual Delta Mean: [{delta_np[...,0].mean():.4f}, {delta_np[...,1].mean():.4f}, {delta_np[...,2].mean():.4f}] (Std: {delta_np.std():.4f})")
        print(f"    FWB Alpha: {model.fwb.alpha.squeeze().tolist()} | Beta: {model.fwb.beta.squeeze().tolist()}")

        # Save side-by-side comparison: [Degraded | Restored | Clean GT]
        deg_vis = (np.clip(deg_rgb, 0, 1) * 255).astype(np.uint8)
        rest_vis = (np.clip(restored_np, 0, 1) * 255).astype(np.uint8)
        clean_vis = (np.clip(clean_rgb, 0, 1) * 255).astype(np.uint8)
        delta_vis = (np.clip(delta_np + 0.5, 0, 1) * 255).astype(np.uint8)

        # Concatenate horizontally: [Degraded | Restored | Delta | Clean GT]
        combined_rgb = np.hstack([deg_vis, rest_vis, delta_vis, clean_vis])
        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

        # Add text labels
        h, w = deg_vis.shape[:2]
        cv2.putText(combined_bgr, f"Degraded ({raw_psnr:.1f}dB)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(combined_bgr, f"Restored ({rest_psnr:.1f}dB)", (w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(combined_bgr, f"Delta Map (+0.5)", (2 * w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(combined_bgr, f"Clean GT", (3 * w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        fig_path = out_dir / f"inspect_{scene}_{deg_path.stem}.png"
        cv2.imwrite(str(fig_path), combined_bgr)
        print(f"  Saved visual inspection plot to: {fig_path}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    inspect_validation_samples()
