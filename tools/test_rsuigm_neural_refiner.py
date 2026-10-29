"""
Controlled Experiment: Testing Neural Refinement on top of RSUIGM Prior
Hypothesis: A lightweight NAFNet refiner trained on top of RSUIGM estimates
can either improve or degrade the 24.54 dB baseline.
Tests 2 fast epochs with lr=1e-4.
"""

import os, sys, time, json
import os.path as osp
import cv2, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")

from models.nafnet import NAFBlock
from tools.train_track1_screening import apply_rsuigm_batch_gpu
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


class ShallowNAFRefiner(nn.Module):
    """Shallow 4-block NAFNet refiner with tiny residual scaling."""
    def __init__(self, channels=32):
        super().__init__()
        self.intro = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(channels) for _ in range(4)])
        self.ending = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, x):
        feat = self.intro(x)
        feat = self.blocks(feat)
        res = self.ending(feat)
        return torch.clamp(x + res, 0.0, 1.0)


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tbl16 = torch.load("checkpoints/track1_screening/tbl19_16sc_fold0.pt", map_location=device)

    # Load 4 held-out validation scenes
    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    val_items = []
    clean_cache = {}
    for sc in val_scenes:
        c_bgr = cv2.imread(f"data/cached_1024/clean/{sc}.png")
        clean_cache[sc] = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sc_dir = f"data/cached_1024/degraded/{sc}"
        files = sorted(os.listdir(sc_dir))[::10] # 80 images
        for f in files:
            parts = osp.splitext(f)[0].split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            dp = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
            val_items.append((sc, osp.join(sc_dir, f), wt, dp))

    print(f"Evaluating RSUIGM baseline on {len(val_items)} validation images...")
    base_psnrs, base_ssims, base_sharps = [], [], []
    rsu_cache = []

    with torch.no_grad():
        for sc, deg_p, wt, dp in val_items:
            deg_bgr = cv2.imread(deg_p)
            deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

            rsu_t = apply_rsuigm_batch_gpu(deg_t, torch.tensor([wt], device=device), torch.tensor([dp], device=device), tbl16)
            rsu_np = np.clip(rsu_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            rsu_cache.append((rsu_t.cpu(), clean_cache[sc]))

            p, _ = compute_psnr_and_norm(rsu_np, clean_cache[sc])
            s = compute_ssim_single(rsu_np, clean_cache[sc])
            sh = compute_sharpness_ratio(rsu_np, clean_cache[sc])
            base_psnrs.append(p)
            base_ssims.append(s)
            base_sharps.append(sh)

    mean_base_p = np.mean(base_psnrs)
    mean_base_s = np.mean(base_ssims)
    mean_base_sh = np.mean(base_sharps)
    p_norm = np.clip((mean_base_p - 15.0) / 20.0, 0.0, 1.0)
    base_q_fr = 0.45 * p_norm + 0.45 * mean_base_s + 0.10 * mean_base_sh

    print(f"RSUIGM Base: PSNR={mean_base_p:.2f} dB | SSIM={mean_base_s:.4f} | Sharp={mean_base_sh:.4f} | Q_FR={base_q_fr:.4f}")

    # Build shallow refiner
    model = ShallowNAFRefiner(channels=32).to(device)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Shallow NAF Refiner: {total_p:.2f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Quick train on 320 training pairs
    train_scenes = ["fish13", "fish14", "fish15", "fish17", "fish18", "fish2", "fish20", "fish21"]
    train_items = []
    tr_clean = {}
    for sc in train_scenes:
        c_bgr = cv2.imread(f"data/cached_1024/clean/{sc}.png")
        tr_clean[sc] = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sc_dir = f"data/cached_1024/degraded/{sc}"
        files = sorted(os.listdir(sc_dir))[::5] # 40 per scene x 8 scenes = 320 images
        for f in files:
            parts = osp.splitext(f)[0].split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            dp = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
            train_items.append((sc, osp.join(sc_dir, f), wt, dp))

    print(f"\nTraining on {len(train_items)} images for 2 epochs (lr=1e-4)...")
    for epoch in range(2):
        model.train()
        losses = []
        for sc, deg_p, wt, dp in train_items:
            deg_bgr = cv2.imread(deg_p)
            deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
            clean_t = torch.from_numpy(tr_clean[sc].transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                rsu_t = apply_rsuigm_batch_gpu(deg_t, torch.tensor([wt], device=device), torch.tensor([dp], device=device), tbl16)

            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_t = model(rsu_t)
                loss = F.l1_loss(pred_t, clean_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"  Epoch {epoch+1}: Train L1 Loss = {np.mean(losses):.4f}")

    # Validate on held-out scenes
    model.eval()
    ref_psnrs, ref_ssims, ref_sharps = [], [], []
    with torch.no_grad():
        for rsu_t, clean_np in rsu_cache:
            rsu_t = rsu_t.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_t = model(rsu_t)
            pred_np = np.clip(pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
            p, _ = compute_psnr_and_norm(pred_np, clean_np)
            s = compute_ssim_single(pred_np, clean_np)
            sh = compute_sharpness_ratio(pred_np, clean_np)
            ref_psnrs.append(p)
            ref_ssims.append(s)
            ref_sharps.append(sh)

    mean_ref_p = np.mean(ref_psnrs)
    mean_ref_s = np.mean(ref_ssims)
    mean_ref_sh = np.mean(ref_sharps)
    p_norm_ref = np.clip((mean_ref_p - 15.0) / 20.0, 0.0, 1.0)
    ref_q_fr = 0.45 * p_norm_ref + 0.45 * mean_ref_s + 0.10 * mean_ref_sh

    print("\n" + "=" * 80)
    print("EXPERIMENTAL VERDICT: RSUIGM PRIOR vs NEURAL REFINER")
    print("=" * 80)
    print(f"RSUIGM Base:    PSNR = {mean_base_p:.2f} dB | SSIM = {mean_base_s:.4f} | Sharp = {mean_base_sh:.4f} | Q_FR = {base_q_fr:.4f}")
    print(f"Neural Refiner: PSNR = {mean_ref_p:.2f} dB | SSIM = {mean_ref_s:.4f} | Sharp = {mean_ref_sh:.4f} | Q_FR = {ref_q_fr:.4f}")
    delta_p = mean_ref_p - mean_base_p
    delta_s = mean_ref_s - mean_base_s
    delta_q = ref_q_fr - base_q_fr
    print(f"Delta:          PSNR = {delta_p:+.2f} dB | SSIM = {delta_s:+.4f} | Q_FR = {delta_q:+.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
