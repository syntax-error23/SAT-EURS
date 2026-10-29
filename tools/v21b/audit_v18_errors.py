import os
import os.path as osp
import json
import torch
import cv2
import numpy as np

import sys
sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('splits/manifest_5fold.json') as f:
    manifest = json.load(f)

print("=" * 80)
print("PHASE 0: V18 REFINER RESIDUAL ERROR & CLIPPING AUDIT")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
print("=" * 80)

for fold in [0, 2]:
    f_key = f"fold_{fold}"
    val_scenes = manifest['folds'][f_key]['val_scenes']
    tbl_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
    tbl_fold = torch.load(tbl_path, map_location=device)
    ref_path = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    ref_fold = build_v18_restoration_refiner(rsuigm_table=tbl_fold, use_neural_refiner=True).to(device)
    ref_ckpt = torch.load(ref_path, map_location=device)
    ref_fold.load_state_dict(ref_ckpt['state_dict'] if 'state_dict' in ref_ckpt else ref_ckpt)
    ref_fold.eval()

    psnrs = {'shallow': [], 'medium': [], 'deep': []}
    err_r, err_g, err_b = [], [], []
    clipped_pixels_count = 0
    total_pixels_count = 0

    with torch.no_grad():
        for sc in val_scenes:
            clean_p = f"data/cached_1024/clean/{sc}.png"
            clean = cv2.cvtColor(cv2.imread(clean_p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clean_t = torch.from_numpy(clean).permute(2, 0, 1).unsqueeze(0).to(device)

            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    deg_p = f"data/cached_1024/degraded/{sc}/{sc}_{wt}_{depth}.png"
                    deg = cv2.cvtColor(cv2.imread(deg_p), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    d_t = torch.from_numpy(deg).permute(2, 0, 1).unsqueeze(0).to(device)
                    wt_t = torch.tensor([wt], device=device)
                    dp_t = torch.tensor([depth], device=device)

                    out = ref_fold(d_t, wt_t, dp_t)['refined_image']
                    diff = (out - clean_t).squeeze(0).cpu().numpy()  # [3, H, W]

                    mse = float(np.mean(diff ** 2))
                    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))

                    if depth <= 30:
                        psnrs['shallow'].append(psnr)
                    elif depth <= 60:
                        psnrs['medium'].append(psnr)
                    else:
                        psnrs['deep'].append(psnr)

                    err_r.append(float(np.mean(np.abs(diff[0]))))
                    err_g.append(float(np.mean(np.abs(diff[1]))))
                    err_b.append(float(np.mean(np.abs(diff[2]))))

                    out_np = out.squeeze(0).cpu().numpy()
                    clip_mask = (out_np <= 0.001) | (out_np >= 0.999)
                    clipped_pixels_count += int(np.sum(clip_mask))
                    total_pixels_count += out_np.size

    print(f"\n--- FOLD {fold} V18 REFINER RESIDUAL ERROR BREAKDOWN ---")
    print(f"Shallow (5-30m)  Mean PSNR: {np.mean(psnrs['shallow']):.2f} dB (N={len(psnrs['shallow'])})")
    print(f"Medium (35-60m)  Mean PSNR: {np.mean(psnrs['medium']):.2f} dB (N={len(psnrs['medium'])})")
    print(f"Deep (65-100m)   Mean PSNR: {np.mean(psnrs['deep']):.2f} dB (N={len(psnrs['deep'])})")
    print(f"Per-Channel Mean Absolute Error: Red={np.mean(err_r):.4f}, Green={np.mean(err_g):.4f}, Blue={np.mean(err_b):.4f}")
    print(f"Clipped Pixel Fraction (<=0.001 or >=0.999): {clipped_pixels_count / total_pixels_count * 100:.3f}% ({clipped_pixels_count:,} / {total_pixels_count:,} px)")
