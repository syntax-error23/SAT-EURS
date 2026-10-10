import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import EURSDataset
from utils.paths import FROZEN_MANIFEST_PATH
from metrics.restoration_metrics import RestorationEvaluator, compute_psnr_and_norm, compute_ssim_single

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    ckpt_path = 'checkpoints/run2_50ep/forensic_backup/best_model_preserved.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    alpha = sd['fwb.alpha'].to(device)
    beta = sd['fwb.beta'].to(device)
    print("Best checkpoint FWB alpha:", alpha.squeeze().tolist())
    print("Best checkpoint FWB beta: ", beta.squeeze().tolist())

    data_dir = 'data/cached_1024' if os.path.exists('data/cached_1024') else 'data'
    ds = EURSDataset(data_dir, 'val', FROZEN_MANIFEST_PATH)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    eval_raw = RestorationEvaluator()
    eval_wb = RestorationEvaluator()

    with torch.no_grad():
        for degs, cleans, _, _ in loader:
            degs = degs.to(device)
            cleans = cleans.to(device)
            
            # 1. Raw Degraded baseline
            eval_raw.update(degs, cleans)
            
            # 2. FastWhiteBalance baseline
            wb = torch.clamp(degs * alpha + beta, 0.0, 1.0)
            eval_wb.update(wb, cleans)

    raw_res = eval_raw.compute()
    wb_res = eval_wb.compute()

    print("\n" + "=" * 65)
    print("     EXACT FORWARD-PASS BASELINE VALIDATION REPORT (800 IMAGES)")
    print("=" * 65)
    print(f"1. RAW DEGRADED INPUT vs CLEAN TARGET:")
    print(f"   - PSNR:            {raw_res['PSNR_dB']:.4f} dB")
    print(f"   - PSNR_norm:       {raw_res['PSNR_norm']:.4f}")
    print(f"   - SSIM:            {raw_res['SSIM']:.4f}")
    print(f"   - Sharpness Ratio: {raw_res['sharpness_ratio']:.4f}")
    print(f"   - Q_rest:          {raw_res['Q_rest']:.4f}")
    print("-----------------------------------------------------------------")
    print(f"2. FASTWHITEBALANCE (FWB) BASELINE vs CLEAN TARGET:")
    print(f"   - PSNR:            {wb_res['PSNR_dB']:.4f} dB")
    print(f"   - PSNR_norm:       {wb_res['PSNR_norm']:.4f}")
    print(f"   - SSIM:            {wb_res['SSIM']:.4f}")
    print(f"   - Sharpness Ratio: {wb_res['sharpness_ratio']:.4f}")
    print(f"   - Q_rest:          {wb_res['Q_rest']:.4f}")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
