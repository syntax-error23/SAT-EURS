import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
import cv2
import numpy as np
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from utils.paths import DATA_DIR, FROZEN_MANIFEST_PATH
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio

def run_phase2():
    ckpt = torch.load('checkpoints/run2_50ep/latest.pt', map_location='cpu', weights_only=False)
    model = EURSNetV2(5)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ds = EURSDataset(DATA_DIR, 'val', FROZEN_MANIFEST_PATH)

    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    sample_indices = {}
    for idx in range(len(ds)):
        item = ds.samples[idx]
        s_name = item.get('scene_id')
        if s_name in scenes and s_name not in sample_indices:
            sample_indices[s_name] = idx
        if len(sample_indices) == len(scenes):
            break

    os.makedirs('audits/forensic_plots', exist_ok=True)

    print("=== PHASE 2: LIVE RESTORATION TENSOR INSPECTION ===")
    print(f"Checkpoint Epoch: {ckpt.get('epoch')}, Step: {ckpt.get('global_step')}")
    
    for scene_name, idx in sample_indices.items():
        deg_t, clean_t, mask_t, meta = ds[idx]
        x = deg_t.unsqueeze(0)
        
        with torch.no_grad():
            wb_x = model.fwb(x)
            stem, f256, f128, f64, f32 = model.backbone(wb_x)
            
            x128 = model.restoration_decoder.proj_f128(f128)
            import torch.nn.functional as F
            x256 = F.interpolate(x128, scale_factor=2.0, mode='bilinear', align_corners=False) + model.restoration_decoder.proj_f256(f256)
            x256 = model.restoration_decoder.dw_conv256(x256)
            x512 = F.interpolate(x256, scale_factor=2.0, mode='bilinear', align_corners=False) + model.restoration_decoder.proj_stem(stem)
            x512 = model.restoration_decoder.dw_conv512(x512)
            delta_512 = model.restoration_decoder.head(x512)
            delta_1024 = F.interpolate(delta_512, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            base = wb_x
            restored = torch.clamp(base + delta_1024, 0.0, 1.0)
        
        deg_np = deg_t.numpy()
        clean_np = clean_t.numpy()
        wb_np = wb_x.squeeze(0).numpy()
        delta_np = delta_1024.squeeze(0).numpy()
        rest_np = restored.squeeze(0).numpy()
        
        sample_id = meta.get('sample_id')
        print(f"\n--- Scene: {scene_name} (Index: {idx}, ID: {sample_id}) ---")
        tensors = {
            'Degraded Input': deg_np,
            'Clean Target': clean_np,
            'White-Balanced Baseline': wb_np,
            'Predicted Residual Delta': delta_np,
            'Final Restored Image': rest_np
        }
        
        for tname, tarr in tensors.items():
            is_finite = bool(np.isfinite(tarr).all())
            print(f"  {tname:24s}: min={tarr.min():.4f}, max={tarr.max():.4f}, mean={tarr.mean():.4f}, std={tarr.std():.4f}, all_finite={is_finite}")
        
        frac_out_1 = float(np.mean(np.abs(delta_np) > 1.0))
        frac_out_05 = float(np.mean(np.abs(delta_np) > 0.5))
        print(f"  Residual fraction |delta| > 1.0: {frac_out_1*100:.2f}% | |delta| > 0.5: {frac_out_05*100:.2f}%")
        
        for c, cname in enumerate(['Red', 'Green', 'Blue']):
            c_0 = float(np.mean(rest_np[c] == 0.0))
            c_1 = float(np.mean(rest_np[c] == 1.0))
            print(f"    Channel {cname:5s}: fraction at 0.0: {c_0*100:.2f}%, at 1.0: {c_1*100:.2f}% (Total saturated: {(c_0+c_1)*100:.2f}%)")
        
        total_sat = float(np.mean((rest_np == 0.0) | (rest_np == 1.0)))
        print(f"  Overall Pixels Clamped to exactly 0 or 1: {total_sat*100:.2f}%")
        
        deg_hwc = np.transpose(deg_np, (1, 2, 0))
        clean_hwc = np.transpose(clean_np, (1, 2, 0))
        wb_hwc = np.transpose(wb_np, (1, 2, 0))
        rest_hwc = np.transpose(rest_np, (1, 2, 0))
        
        psnr_base, _ = compute_psnr_and_norm(wb_hwc, clean_hwc)
        ssim_base = compute_ssim_single(wb_hwc, clean_hwc)
        
        psnr_deg, _ = compute_psnr_and_norm(deg_hwc, clean_hwc)
        ssim_deg = compute_ssim_single(deg_hwc, clean_hwc)
        
        psnr_rest, _ = compute_psnr_and_norm(rest_hwc, clean_hwc)
        ssim_rest = compute_ssim_single(rest_hwc, clean_hwc)
        
        print(f"  Metrics Comparison:")
        print(f"    Raw Degraded:   PSNR = {psnr_deg:.2f} dB, SSIM = {ssim_deg:.4f}")
        print(f"    WB Baseline:    PSNR = {psnr_base:.2f} dB, SSIM = {ssim_base:.4f}")
        print(f"    Final Restored: PSNR = {psnr_rest:.2f} dB, SSIM = {ssim_rest:.4f}")
        
        vis = np.hstack([
            (deg_hwc[:, :, ::-1] * 255).astype(np.uint8),
            (wb_hwc[:, :, ::-1] * 255).astype(np.uint8),
            (rest_hwc[:, :, ::-1] * 255).astype(np.uint8),
            (clean_hwc[:, :, ::-1] * 255).astype(np.uint8)
        ])
        vis_path = f"audits/forensic_plots/{scene_name}_comparison.png"
        cv2.imwrite(vis_path, vis)
        print(f"  Saved visual comparison -> {vis_path}")

if __name__ == '__main__':
    run_phase2()
