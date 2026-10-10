"""
Forensic Diagnostic for Scene fish12 Clipping Behavior
Inspects:
- Raw degraded pixel distribution
- FWB output pixel distribution
- Model output (Epoch 2 vs Epoch 3) on fish12
- Breakdown of clipping: low-end (0.0) vs high-end (1.0)
- Per-channel breakdown: R, G, B
- Spatial distribution and pixel intensity percentiles
"""

import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from utils.paths import FROZEN_MANIFEST_PATH
from models.eurs_net_v2 import EURSNetV2
from data.dataset import EURSDataset
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single

def diagnose_fish12():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("      FORENSIC DIAGNOSTIC: SCENE fish12 CLIPPING ANALYSIS")
    print(f"      Device: {device}")
    print("=" * 70)

    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    val_ds = EURSDataset(data_dir, split='val', manifest_path=FROZEN_MANIFEST_PATH)
    
    # Filter fish12 indices
    fish12_indices = [i for i, s in enumerate(val_ds.samples) if s['scene_id'] == 'fish12']
    print(f"Found {len(fish12_indices)} samples for scene 'fish12'.")

    # Load baseline, ep2, and ep3
    ckpt_base = torch.load('checkpoints/run2_50ep/best_model.pt', map_location='cpu', weights_only=False)
    ckpt_ep2 = torch.load('experiments/exp_pilot_restricted_restoration/pilot_restricted_ep2.pt', map_location='cpu', weights_only=False)
    ckpt_ep3 = torch.load('experiments/exp_pilot_restricted_restoration/pilot_restricted_ep3.pt', map_location='cpu', weights_only=False)

    m_ep2 = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)
    m_ep2.load_state_dict(ckpt_ep2['model_state_dict'])
    m_ep2.eval()

    m_ep3 = EURSNetV2(num_classes=5, bounded_restoration=True, bound_scale=0.20).to(device)
    m_ep3.load_state_dict(ckpt_ep3['model_state_dict'])
    m_ep3.eval()

    # Inspect FWB parameters
    fwb_alpha = ckpt_base['model_state_dict']['fwb.alpha'].to(device).squeeze() # [3]
    fwb_beta = ckpt_base['model_state_dict']['fwb.beta'].to(device).squeeze()   # [3]
    print(f"FWB Alpha (R, G, B): {fwb_alpha.tolist()}")
    print(f"FWB Beta  (R, G, B): {fwb_beta.tolist()}")

    # Statistics accumulators
    stats = {
        'ep2': {'clip_0': [], 'clip_1': [], 'clip_total': [], 'r_clip': [], 'g_clip': [], 'b_clip': []},
        'ep3': {'clip_0': [], 'clip_1': [], 'clip_total': [], 'r_clip': [], 'g_clip': [], 'b_clip': []},
        'raw': {'min': [], 'max': [], 'mean': [], 'clip_0': [], 'clip_1': []},
        'fwb': {'min': [], 'max': [], 'mean': [], 'clip_0': [], 'clip_1': []}
    }

    # Evaluate all 200 fish12 samples
    with torch.no_grad():
        for idx in fish12_indices:
            deg, clean, _, _ = val_ds[idx]
            deg = deg.unsqueeze(0).to(device)
            clean = clean.unsqueeze(0).to(device)

            # Raw
            raw_np = deg.squeeze(0).permute(1, 2, 0).cpu().numpy()
            stats['raw']['min'].append(float(raw_np.min()))
            stats['raw']['max'].append(float(raw_np.max()))
            stats['raw']['mean'].append(float(raw_np.mean()))
            stats['raw']['clip_0'].append(float(np.mean(raw_np == 0.0)) * 100)
            stats['raw']['clip_1'].append(float(np.mean(raw_np == 1.0)) * 100)

            # FWB
            fwb_x = torch.clamp(deg * fwb_alpha.view(1, 3, 1, 1) + fwb_beta.view(1, 3, 1, 1), 0.0, 1.0)
            fwb_np = fwb_x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            stats['fwb']['min'].append(float(fwb_np.min()))
            stats['fwb']['max'].append(float(fwb_np.max()))
            stats['fwb']['mean'].append(float(fwb_np.mean()))
            stats['fwb']['clip_0'].append(float(np.mean(fwb_np == 0.0)) * 100)
            stats['fwb']['clip_1'].append(float(np.mean(fwb_np == 1.0)) * 100)

            # Ep 2
            rest_ep2, _ = m_ep2(deg)
            ep2_np = rest_ep2.squeeze(0).permute(1, 2, 0).cpu().numpy()
            stats['ep2']['clip_0'].append(float(np.mean(ep2_np == 0.0)) * 100)
            stats['ep2']['clip_1'].append(float(np.mean(ep2_np == 1.0)) * 100)
            stats['ep2']['clip_total'].append(float(np.mean((ep2_np == 0.0) | (ep2_np == 1.0))) * 100)
            stats['ep2']['r_clip'].append(float(np.mean((ep2_np[:, :, 0] == 0.0) | (ep2_np[:, :, 0] == 1.0))) * 100)
            stats['ep2']['g_clip'].append(float(np.mean((ep2_np[:, :, 1] == 0.0) | (ep2_np[:, :, 1] == 1.0))) * 100)
            stats['ep2']['b_clip'].append(float(np.mean((ep2_np[:, :, 2] == 0.0) | (ep2_np[:, :, 2] == 1.0))) * 100)

            # Ep 3
            rest_ep3, _ = m_ep3(deg)
            ep3_np = rest_ep3.squeeze(0).permute(1, 2, 0).cpu().numpy()
            stats['ep3']['clip_0'].append(float(np.mean(ep3_np == 0.0)) * 100)
            stats['ep3']['clip_1'].append(float(np.mean(ep3_np == 1.0)) * 100)
            stats['ep3']['clip_total'].append(float(np.mean((ep3_np == 0.0) | (ep3_np == 1.0))) * 100)
            stats['ep3']['r_clip'].append(float(np.mean((ep3_np[:, :, 0] == 0.0) | (ep3_np[:, :, 0] == 1.0))) * 100)
            stats['ep3']['g_clip'].append(float(np.mean((ep3_np[:, :, 1] == 0.0) | (ep3_np[:, :, 1] == 1.0))) * 100)
            stats['ep3']['b_clip'].append(float(np.mean((ep3_np[:, :, 2] == 0.0) | (ep3_np[:, :, 2] == 1.0))) * 100)

    print("\n--- RESULTS: RAW DEGRADED & FWB BASELINE ON fish12 ---")
    print(f"Raw Input:  Mean={np.mean(stats['raw']['mean']):.4f}, Min={np.mean(stats['raw']['min']):.4f}, Max={np.mean(stats['raw']['max']):.4f}")
    print(f"Raw Input:  Clip at 0.0={np.mean(stats['raw']['clip_0']):.4f}%, Clip at 1.0={np.mean(stats['raw']['clip_1']):.4f}%")
    print(f"FWB Output: Mean={np.mean(stats['fwb']['mean']):.4f}, Min={np.mean(stats['fwb']['min']):.4f}, Max={np.mean(stats['fwb']['max']):.4f}")
    print(f"FWB Output: Clip at 0.0={np.mean(stats['fwb']['clip_0']):.4f}%, Clip at 1.0={np.mean(stats['fwb']['clip_1']):.4f}% (Total={np.mean(stats['fwb']['clip_0'])+np.mean(stats['fwb']['clip_1']):.4f}%)")

    print("\n--- RESULTS: EPOCH 2 vs EPOCH 3 CLIPPING BREAKDOWN ON fish12 ---")
    print(f"EPOCH 2 Total Clipping:   {np.mean(stats['ep2']['clip_total']):.4f}% (Max: {np.max(stats['ep2']['clip_total']):.4f}%)")
    print(f"  - Ep 2 Low-End (0.0):   {np.mean(stats['ep2']['clip_0']):.4f}%")
    print(f"  - Ep 2 High-End (1.0):  {np.mean(stats['ep2']['clip_1']):.4f}%")
    print(f"  - Ep 2 Channel R:       {np.mean(stats['ep2']['r_clip']):.4f}%")
    print(f"  - Ep 2 Channel G:       {np.mean(stats['ep2']['g_clip']):.4f}%")
    print(f"  - Ep 2 Channel B:       {np.mean(stats['ep2']['b_clip']):.4f}%")

    print(f"\nEPOCH 3 Total Clipping:   {np.mean(stats['ep3']['clip_total']):.4f}% (Max: {np.max(stats['ep3']['clip_total']):.4f}%)")
    print(f"  - Ep 3 Low-End (0.0):   {np.mean(stats['ep3']['clip_0']):.4f}%")
    print(f"  - Ep 3 High-End (1.0):  {np.mean(stats['ep3']['clip_1']):.4f}%")
    print(f"  - Ep 3 Channel R:       {np.mean(stats['ep3']['r_clip']):.4f}%")
    print(f"  - Ep 3 Channel G:       {np.mean(stats['ep3']['g_clip']):.4f}%")
    print(f"  - Ep 3 Channel B:       {np.mean(stats['ep3']['b_clip']):.4f}%")

    # Save diagnostic results
    diag_summary = {
        'scene': 'fish12',
        'sample_count': len(fish12_indices),
        'fwb_alpha': fwb_alpha.tolist(),
        'fwb_beta': fwb_beta.tolist(),
        'raw_clipping': {
            'clip_0': float(np.mean(stats['raw']['clip_0'])),
            'clip_1': float(np.mean(stats['raw']['clip_1'])),
        },
        'fwb_clipping': {
            'clip_0': float(np.mean(stats['fwb']['clip_0'])),
            'clip_1': float(np.mean(stats['fwb']['clip_1'])),
            'total': float(np.mean(stats['fwb']['clip_0']) + np.mean(stats['fwb']['clip_1']))
        },
        'ep2': {
            'total_clip_mean': float(np.mean(stats['ep2']['clip_total'])),
            'total_clip_max': float(np.max(stats['ep2']['clip_total'])),
            'clip_0_mean': float(np.mean(stats['ep2']['clip_0'])),
            'clip_1_mean': float(np.mean(stats['ep2']['clip_1'])),
            'r_clip_mean': float(np.mean(stats['ep2']['r_clip'])),
            'g_clip_mean': float(np.mean(stats['ep2']['g_clip'])),
            'b_clip_mean': float(np.mean(stats['ep2']['b_clip'])),
        },
        'ep3': {
            'total_clip_mean': float(np.mean(stats['ep3']['clip_total'])),
            'total_clip_max': float(np.max(stats['ep3']['clip_total'])),
            'clip_0_mean': float(np.mean(stats['ep3']['clip_0'])),
            'clip_1_mean': float(np.mean(stats['ep3']['clip_1'])),
            'r_clip_mean': float(np.mean(stats['ep3']['r_clip'])),
            'g_clip_mean': float(np.mean(stats['ep3']['g_clip'])),
            'b_clip_mean': float(np.mean(stats['ep3']['b_clip'])),
        }
    }
    os.makedirs('audits/diagnostics', exist_ok=True)
    with open('audits/diagnostics/fish12_clipping_diagnosis.json', 'w') as f:
        json.dump(diag_summary, f, indent=2)
    print("\nSaved diagnostic findings to: audits/diagnostics/fish12_clipping_diagnosis.json")

if __name__ == '__main__':
    diagnose_fish12()
