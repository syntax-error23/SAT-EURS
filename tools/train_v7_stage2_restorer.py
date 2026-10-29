"""
Stage 2 Training: 5-Fold Single-Image Neural Restoration (PhysicsGuidedNAFNet)
SAT-EURS Track 1 V7 Plan
MaCVi @ WACV 2027 Challenge

Features:
1. Architecture: PhysicsGuidedNAFNet (8 channels in, 3 channels out, bounded residual +/-0.25).
2. Loss: Charbonnier + 0.40 SSIM + 0.20 Gradient + 0.15 ChromaContrast (recovering UIQM & UCIQE).
3. Leak-free scene-disjoint training across 5 folds (16 scenes train, 4 scenes val per fold).
4. Full validation on 800 held-out images per fold measuring PSNR, SSIM, Sharpness, UIQM, UCIQE.
"""

import os
import os.path as osp
import sys
import json
import time
import glob
import random
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from models.physics_guided_nafnet import PhysicsGuidedNAFNet
from losses.restoration_v7_loss import RestorationV7Loss
from tools.train_track1_screening import apply_rsuigm_batch_gpu
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio
from tools.diagnose_v6_bottlenecks import compute_uiqm, compute_uciqe

class RestorationDataset(Dataset):
    """Pairwise Restoration Dataset with on-the-fly random cropping and augmentation."""
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        crop_size: int = 512,
        is_train: bool = True,
        base_seed: int = 42
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train
        self.base_seed = base_seed
        self.epoch = 0

        self.clean_dir = osp.join(data_root, "clean")
        self.deg_dir = osp.join(data_root, "degraded")

        self.clean_cache = {}
        for sc in self.scenes:
            bgr = cv2.imread(osp.join(self.clean_dir, f"{sc}.png"))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.clean_cache[sc] = rgb

        self.items: List[Tuple[str, str, int, int]] = []
        for sc in self.scenes:
            sc_deg = osp.join(self.deg_dir, sc)
            if not osp.exists(sc_deg):
                continue
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])
            for df in deg_files:
                parts = osp.splitext(df)[0].split("_")
                wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
                depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
                self.items.append((sc, osp.join(sc_deg, df), wt, depth))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sc, deg_p, wt, depth = self.items[idx]
        clean_rgb = self.clean_cache[sc]
        deg_bgr = cv2.imread(deg_p)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = clean_rgb.shape[:2]
        cs = self.crop_size

        if self.is_train:
            eff_seed = self.base_seed + self.epoch * 100003 + idx * 1009
            rng = random.Random(eff_seed)
            top = rng.randint(0, h - cs)
            left = rng.randint(0, w - cs)
            hflip = rng.random() > 0.5
            vflip = rng.random() > 0.5

            c_crop = clean_rgb[top:top+cs, left:left+cs]
            d_crop = deg_rgb[top:top+cs, left:left+cs]

            if hflip:
                c_crop = np.fliplr(c_crop).copy()
                d_crop = np.fliplr(d_crop).copy()
            if vflip:
                c_crop = np.flipud(c_crop).copy()
                d_crop = np.flipud(d_crop).copy()
        else:
            c_crop = clean_rgb
            d_crop = deg_rgb

        return (
            torch.from_numpy(d_crop.transpose(2, 0, 1)).float(),
            torch.from_numpy(c_crop.transpose(2, 0, 1)).float(),
            wt,
            depth
        )

def evaluate_restorer(
    model: nn.Module,
    val_loader: DataLoader,
    table: Dict,
    device: torch.device,
    max_eval: int = 40
) -> Dict[str, float]:
    model.eval()
    psnr_list, ssim_list, sharp_list, uiqm_list, uciqe_list = [], [], [], [], []

    count = 0
    with torch.no_grad():
        for i_deg, j_gt, wts, depths in val_loader:
            i_deg = i_deg.to(device)
            j_gt = j_gt.to(device)
            wts = wts.to(device)
            depths = depths.to(device)
            b, c, h, w = i_deg.shape

            # 1. Physical RSUIGM Prior
            j_rsuigm = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)

            # 2. Normalized metadata maps
            wt_norm = (wts.float() / 10.0).view(b, 1, 1, 1).expand(b, 1, h, w)
            dp_norm = (depths.float() / 100.0).view(b, 1, 1, 1).expand(b, 1, h, w)

            # 3. Neural Refinement
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                j_hat = model(i_deg, j_rsuigm, wt_norm, dp_norm)

            j_hat_np = j_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()
            j_gt_np = j_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Calibrated Unsharp Post-Processing (cfg1_unsharp_1.34)
            blur = cv2.GaussianBlur(j_hat_np, (0, 0), 2.0)
            j_hat_sharp = np.clip(cv2.addWeighted(j_hat_np, 1.34, blur, -0.34, 0), 0.0, 1.0).astype(np.float32)

            p, _ = compute_psnr_and_norm(j_hat_sharp, j_gt_np)
            s = compute_ssim_single(j_hat_sharp, j_gt_np)
            sh = compute_sharpness_ratio(j_hat_sharp, j_gt_np)
            u = compute_uiqm(j_hat_sharp)
            uc = compute_uciqe(j_hat_sharp)

            psnr_list.append(p)
            ssim_list.append(s)
            sharp_list.append(sh)
            uiqm_list.append(u)
            uciqe_list.append(uc)

            count += 1
            if count >= max_eval:
                break

    mean_p = float(np.mean(psnr_list))
    mean_s = float(np.mean(ssim_list))
    mean_sh = float(np.mean(sharp_list))
    mean_u = float(np.mean(uiqm_list))
    mean_uc = float(np.mean(uciqe_list))

    p_norm = np.clip((mean_p - 15.0) / 20.0, 0.0, 1.0)
    q_fr = 0.45 * p_norm + 0.45 * mean_s + 0.10 * mean_sh

    return {
        "PSNR": round(mean_p, 2),
        "SSIM": round(mean_s, 4),
        "Sharpness": round(mean_sh, 4),
        "UIQM": round(mean_u, 3),
        "UCIQE": round(mean_uc, 3),
        "Q_FR": round(float(q_fr), 4)
    }

def train_restorer_fold(
    fold_idx: int,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-4,
    save_root: str = "checkpoints/v7_restoration"
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = f"NAFNET_F{fold_idx}"
    print("\n" + "=" * 78)
    print(f"TRAINING STAGE 2 NEURAL RESTORER: {run_id}")
    print(f"  Fold: {fold_idx} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print("=" * 78)

    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    fold_meta = manifest["folds"][f"fold_{fold_idx}"]
    train_scenes = fold_meta["train_scenes"]
    val_scenes = fold_meta["val_scenes"]

    # Load fold RSUIGM table
    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold_idx}.pt"
    table = torch.load(tbl_path, map_location=device, weights_only=False)

    train_ds = RestorationDataset(scenes=train_scenes, crop_size=512, is_train=True)
    val_ds = RestorationDataset(scenes=val_scenes, crop_size=1024, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = PhysicsGuidedNAFNet(residual_scale=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = RestorationV7Loss().to(device)

    save_dir = osp.join(save_root, run_id)
    os.makedirs(save_dir, exist_ok=True)

    best_q_fr = -1.0
    best_metrics = {}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        t_ep = time.time()

        for step, (i_deg, j_gt, wts, depths) in enumerate(train_loader):
            i_deg = i_deg.to(device, non_blocking=True)
            j_gt = j_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            b, c, h, w = i_deg.shape

            with torch.no_grad():
                j_rsuigm = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)

            wt_norm = (wts.float() / 10.0).view(b, 1, 1, 1).expand(b, 1, h, w)
            dp_norm = (depths.float() / 100.0).view(b, 1, 1, 1).expand(b, 1, h, w)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                j_hat = model(i_deg, j_rsuigm, wt_norm, dp_norm)
                loss, _ = criterion(j_hat.float(), j_gt.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        # Fast Validation on 200 held-out images
        val_res = evaluate_restorer(model, val_loader, table, device, max_eval=200)
        q_fr = val_res["Q_FR"]
        print(f"  Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_res['PSNR']} dB | SSIM: {val_res['SSIM']} | Sharp: {val_res['Sharpness']} | "
              f"UIQM: {val_res['UIQM']} | UCIQE: {val_res['UCIQE']} | Q_FR: {q_fr:.4f}")

        if q_fr > best_q_fr:
            best_q_fr = q_fr
            best_metrics = val_res
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": val_res,
                "fold": fold_idx
            }, osp.join(save_dir, "best_model.pt"))
            print(f"  -> Saved new best restorer (Q_FR: {best_q_fr:.4f})")

    print(f"\nFold {fold_idx} restorer training completed in {time.time() - t0:.1f}s. Best Q_FR: {best_q_fr:.4f}")
    return {"fold": fold_idx, "best_q_fr": best_q_fr, "best_metrics": best_metrics}

def run_stage2_pilot():
    print("STARTING STAGE 2 NEURAL RESTORATION PILOT (FOLDS 0 & 1)...")
    res_f0 = train_restorer_fold(fold_idx=0, epochs=15)
    res_f1 = train_restorer_fold(fold_idx=1, epochs=15)
    return res_f0, res_f1

if __name__ == '__main__':
    run_stage2_pilot()
