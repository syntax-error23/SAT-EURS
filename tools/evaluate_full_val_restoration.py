"""
Evaluate full validation set (800 samples) across:
A. Raw Degraded
B. FastWhiteBalance
C. Zero-Initialized Residual EURS-Net V2
D. Recovered EURS-Net V2 (bounded_recovery_ep5.pt)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from utils.paths import FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)
    print(f"Loaded validation set: {len(val_ds)} samples")
    
    src_ckpt = torch.load('checkpoints/run2_50ep/forensic_backup/best_model_preserved.pt', map_location='cpu', weights_only=False)
    src_sd = src_ckpt['model_state_dict']
    
    zero_model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)
    zero_model.load_state_dict(src_sd)
    nn.init.zeros_(zero_model.restoration_decoder.head.weight)
    nn.init.zeros_(zero_model.restoration_decoder.head.bias)
    zero_model.eval()
    
    rec_ckpt = torch.load('checkpoints/exp_bounded_recovery/bounded_recovery_ep5.pt', map_location='cpu', weights_only=False)
    rec_model = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)
    rec_model.load_state_dict(rec_ckpt['model_state_dict'])
    rec_model.eval()
    
    fwb_alpha = src_sd['fwb.alpha'].to(device)
    fwb_beta = src_sd['fwb.beta'].to(device)
    
    scenes = ['fish1', 'fish12', 'seahorse1', 'sharkFish']
    methods = ['Raw', 'FWB', 'ZeroInit', 'Recovered']
    res = {k: {s: {'psnr': [], 'ssim': [], 'clip': []} for s in scenes} for k in methods}
    
    print("Computing metrics across all 800 images...")
    with torch.no_grad():
        for idx in range(len(val_ds)):
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(val_ds)}...")
            item = val_ds.samples[idx]
            sid = item.get('scene_id')
            deg_t, clean_t, _, _ = val_ds[idx]
            deg_b = deg_t.unsqueeze(0).to(device)
            clean_np = clean_t.permute(1, 2, 0).numpy()
            deg_np = deg_t.permute(1, 2, 0).numpy()
            
            # A. Raw
            p_raw, _ = compute_psnr_and_norm(deg_np, clean_np)
            s_raw = compute_ssim_single(deg_np, clean_np)
            c_raw = float(np.mean((deg_np == 0.0) | (deg_np == 1.0))) * 100
            res['Raw'][sid]['psnr'].append(p_raw)
            res['Raw'][sid]['ssim'].append(s_raw)
            res['Raw'][sid]['clip'].append(c_raw)
            
            # B. FWB
            wb_b = torch.clamp(deg_b * fwb_alpha + fwb_beta, 0.0, 1.0)
            wb_np = wb_b.squeeze(0).permute(1, 2, 0).cpu().numpy()
            p_wb, _ = compute_psnr_and_norm(wb_np, clean_np)
            s_wb = compute_ssim_single(wb_np, clean_np)
            c_wb = float(np.mean((wb_np == 0.0) | (wb_np == 1.0))) * 100
            res['FWB'][sid]['psnr'].append(p_wb)
            res['FWB'][sid]['ssim'].append(s_wb)
            res['FWB'][sid]['clip'].append(c_wb)
            
            # C. ZeroInit
            rest_zero, _ = zero_model(deg_b)
            zero_np = rest_zero.squeeze(0).permute(1, 2, 0).cpu().numpy()
            p_z, _ = compute_psnr_and_norm(zero_np, clean_np)
            s_z = compute_ssim_single(zero_np, clean_np)
            c_z = float(np.mean((zero_np == 0.0) | (zero_np == 1.0))) * 100
            res['ZeroInit'][sid]['psnr'].append(p_z)
            res['ZeroInit'][sid]['ssim'].append(s_z)
            res['ZeroInit'][sid]['clip'].append(c_z)
            
            # D. Recovered
            rest_rec, _ = rec_model(deg_b)
            rec_np = rest_rec.squeeze(0).permute(1, 2, 0).cpu().numpy()
            p_r, _ = compute_psnr_and_norm(rec_np, clean_np)
            s_r = compute_ssim_single(rec_np, clean_np)
            c_r = float(np.mean((rec_np == 0.0) | (rec_np == 1.0))) * 100
            res['Recovered'][sid]['psnr'].append(p_r)
            res['Recovered'][sid]['ssim'].append(s_r)
            res['Recovered'][sid]['clip'].append(c_r)
            
    summary = {}
    for m_name in methods:
        summary[m_name] = {}
        all_p, all_s, all_c = [], [], []
        for s in scenes:
            ps = res[m_name][s]['psnr']
            ss = res[m_name][s]['ssim']
            cs = res[m_name][s]['clip']
            all_p.extend(ps)
            all_s.extend(ss)
            all_c.extend(cs)
            summary[m_name][s] = {
                'count': len(ps),
                'psnr_mean': float(np.mean(ps)),
                'ssim_mean': float(np.mean(ss)),
                'clip_pct_mean': float(np.mean(cs))
            }
        summary[m_name]['aggregate'] = {
            'count': len(all_p),
            'psnr_mean': float(np.mean(all_p)),
            'ssim_mean': float(np.mean(all_s)),
            'clip_pct_mean': float(np.mean(all_c))
        }

    os.makedirs('audits/final_forensic_audit', exist_ok=True)
    out_path = 'audits/final_forensic_audit/full_val_restoration_comparison.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("      FULL VALIDATION (800 SAMPLES) RESTORATION COMPARISON")
    print("=" * 75)
    print(f"{'Method':12s} | {'Scene':12s} | {'Count':6s} | {'PSNR (dB)':10s} | {'SSIM':8s} | {'Clipping %':10s}")
    print("-" * 75)
    for m_name in methods:
        for s in scenes:
            row = summary[m_name][s]
            print(f"{m_name:12s} | {s:12s} | {row['count']:6d} | {row['psnr_mean']:9.2f} dB | {row['ssim_mean']:7.4f} | {row['clip_pct_mean']:9.4f}%")
        agg = summary[m_name]['aggregate']
        print(f"{m_name:12s} | {'[ALL 800]':12s} | {agg['count']:6d} | {agg['psnr_mean']:9.2f} dB | {agg['ssim_mean']:7.4f} | {agg['clip_pct_mean']:9.4f}%")
        print("-" * 75)

if __name__ == '__main__':
    main()
