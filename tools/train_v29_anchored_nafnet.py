"""
SAT-EURS Track 1 — V29 Anchored PhysicsGuidedNAFNet Training Engine
Experiment: EXP-01 (High-Capacity Paired Underwater Restoration)

Key Architectural Properties:
1. PhysicsGuidedNAFNet (21.4M parameters)
2. Anchored on fold-specific RSUIGM physical inversion prior (zero cross-scene leakage)
3. Step 0 starts smoothly from the physical solution (residual initialized near zero)
4. Loss: 0.50 Charbonnier + 0.30 MS-SSIM + 0.10 Laplacian + 0.10 ResidualMagnitude
5. Strict held-out evaluation on unseen validation scenes after every epoch
"""

import os
import os.path as osp
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

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
from losses.v18.restoration_losses import CharbonnierLoss, MSSSIMLoss, LaplacianLoss, ResidualMagnitudeLoss
from tools.train_track1_screening import apply_rsuigm_batch_gpu
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


class EURSPairedRestorationDataset(Dataset):
    """
    Dataset of paired degraded and clean images for Track 1 restoration.
    Loads clean images into RAM for fast zero-overhead training.
    """
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

        # Preload clean target images into RAM
        self.clean_cache: Dict[str, np.ndarray] = {}
        for sc in self.scenes:
            clean_path = osp.join(self.clean_dir, f"{sc}.png")
            if not osp.exists(clean_path):
                raise FileNotFoundError(f"Missing clean scene image: {clean_path}")
            bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.clean_cache[sc] = rgb

        # Build list of degraded image items
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
        deg_bgr = cv2.imread(deg_p, cv2.IMREAD_COLOR)
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
            # Deterministic center crop for validation
            if (h, w) != (cs, cs):
                top = (h - cs) // 2
                left = (w - cs) // 2
                c_crop = clean_rgb[top:top+cs, left:left+cs]
                d_crop = deg_rgb[top:top+cs, left:left+cs]
            else:
                c_crop = clean_rgb
                d_crop = deg_rgb

        return (
            torch.from_numpy(d_crop.transpose(2, 0, 1)).float(),
            torch.from_numpy(c_crop.transpose(2, 0, 1)).float(),
            wt,
            depth
        )


def evaluate_nafnet_model(
    model: nn.Module,
    val_loader: DataLoader,
    table: Dict,
    device: torch.device,
    max_eval: int = 160
) -> Dict[str, float]:
    """Evaluates NAFNet on held-out validation scenes."""
    model.eval()
    psnr_list, ssim_list, sharp_list = [], [], []

    count = 0
    with torch.no_grad():
        for i_deg, j_gt, wts, depths in val_loader:
            i_deg = i_deg.to(device, non_blocking=True)
            j_gt = j_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            b, c, h, w = i_deg.shape

            # 1. Physical RSUIGM Prior
            j_rsuigm = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)

            # 2. Normalized metadata
            wt_norm = (wts.float() / 10.0).view(b, 1, 1, 1).expand(b, 1, h, w)
            dp_norm = (depths.float() / 100.0).view(b, 1, 1, 1).expand(b, 1, h, w)

            # 3. Model forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                j_hat = model(i_deg, j_rsuigm, wt_norm, dp_norm)

            for i in range(b):
                p_np = j_hat[i].permute(1, 2, 0).cpu().numpy()
                g_np = j_gt[i].permute(1, 2, 0).cpu().numpy()

                p, _ = compute_psnr_and_norm(p_np, g_np)
                s = compute_ssim_single(p_np, g_np)
                sh = compute_sharpness_ratio(p_np, g_np)

                psnr_list.append(p)
                ssim_list.append(s)
                sharp_list.append(sh)

                count += 1
                if count >= max_eval:
                    break
            if count >= max_eval:
                break

    mean_p = float(np.mean(psnr_list))
    mean_s = float(np.mean(ssim_list))
    mean_sh = float(np.mean(sharp_list))
    p_norm = float(np.clip((mean_p - 15.0) / 20.0, 0.0, 1.0))
    q_fr = float(0.45 * p_norm + 0.45 * mean_s + 0.10 * mean_sh)

    return {
        "PSNR": round(mean_p, 2),
        "PSNR_norm": round(p_norm, 4),
        "SSIM": round(mean_s, 4),
        "Sharpness": round(mean_sh, 4),
        "Q_FR": round(q_fr, 4),
        "eval_count": count
    }


def train_nafnet_fold(
    fold: int,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    save_dir: str = "checkpoints/v29"
) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING V29 ANCHORED PHYSICS-GUIDED NAFNET (FOLD {fold})")
    print(f"  Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 80)

    # 1. Manifest and Splits
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    f_data = manifest["folds"][f"fold_{fold}"]
    train_scenes = sorted(f_data["train_scenes"])
    val_scenes = sorted(f_data["val_scenes"])

    print(f"  Train Scenes ({len(train_scenes)}): {train_scenes}")
    print(f"  Val Scenes   ({len(val_scenes)}): {val_scenes}")

    # 2. Fold-specific RSUIGM table (strict zero leakage)
    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold}.pt"
    if not osp.exists(tbl_path):
        raise FileNotFoundError(f"Missing fold table: {tbl_path}")
    table = torch.load(tbl_path, map_location=device, weights_only=False)

    # 3. Datasets and Loaders
    train_ds = EURSPairedRestorationDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=42 + fold)
    val_ds = EURSPairedRestorationDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=42 + fold)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train samples: {len(train_ds)} ({len(train_loader)} batches)")
    print(f"  Val samples:   {len(val_ds)} ({len(val_loader)} batches)")

    # 4. Model Setup
    model = PhysicsGuidedNAFNet(residual_scale=0.20).to(device)
    # Initialize ending layer with small weights so step 0 starts at RSUIGM baseline while providing gradients
    nn.init.normal_(model.ending.weight, std=1e-4)
    nn.init.zeros_(model.ending.bias)

    # 5. Losses
    loss_charb = CharbonnierLoss().to(device)
    loss_msssim = MSSSIMLoss().to(device)
    loss_lap = LaplacianLoss().to(device)
    loss_res = ResidualMagnitudeLoss().to(device)

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(save_dir, exist_ok=True)
    out_ckpt = osp.join(save_dir, f"nafnet_fold_{fold}_best.pt")

    # 7. Baseline Evaluation at Epoch 0 (Starting Prior Check)
    print("\n--- Evaluating Starting Baseline Prior (Epoch 0) ---")
    base_metrics = evaluate_nafnet_model(model, val_loader, table, device, max_eval=160)
    print(f"  Epoch  0 [Baseline Prior] | PSNR: {base_metrics['PSNR']} dB | SSIM: {base_metrics['SSIM']} | Sharp: {base_metrics['Sharpness']} | Q_FR: {base_metrics['Q_FR']:.4f}")

    best_q_fr = base_metrics["Q_FR"]
    best_metrics = base_metrics
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
                j_hat, residual = model(i_deg, j_rsuigm, wt_norm, dp_norm, return_residual=True)

                l_c = loss_charb(j_hat, j_gt)
                l_s = loss_msssim(j_hat, j_gt)
                l_l = loss_lap(j_hat, j_gt)
                l_r = loss_res(residual)

                loss = 0.50 * l_c + 0.30 * l_s + 0.10 * l_l + 0.10 * l_r

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        # Evaluation on held-out scenes
        val_metrics = evaluate_nafnet_model(model, val_loader, table, device, max_eval=160)
        q_fr = val_metrics["Q_FR"]
        delta_p = val_metrics["PSNR"] - base_metrics["PSNR"]
        delta_q = q_fr - base_metrics["Q_FR"]

        print(f"  Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val PSNR: {val_metrics['PSNR']} dB ({delta_p:+.2f}) | SSIM: {val_metrics['SSIM']} | "
              f"Sharp: {val_metrics['Sharpness']} | Q_FR: {q_fr:.4f} ({delta_q:+.4f})", flush=True)

        if q_fr > best_q_fr:
            best_q_fr = q_fr
            best_metrics = val_metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "base_metrics": base_metrics,
                "fold": fold,
                "train_scenes": train_scenes,
                "val_scenes": val_scenes,
                "hyperparams": {"lr": lr, "batch_size": batch_size, "residual_scale": 0.20}
            }, out_ckpt)
            print(f"  -> Saved new best model checkpoint (Q_FR: {best_q_fr:.4f}) to {out_ckpt}", flush=True)

    print("\n" + "=" * 80)
    print(f"Fold {fold} Training Complete in {time.time() - t0:.1f}s!")
    print(f"  Starting Base Q_FR: {base_metrics['Q_FR']:.4f} (PSNR: {base_metrics['PSNR']} dB, SSIM: {base_metrics['SSIM']})")
    print(f"  Best Held-Out Q_FR: {best_metrics['Q_FR']:.4f} (PSNR: {best_metrics['PSNR']} dB, SSIM: {best_metrics['SSIM']})")
    print(f"  Net Delta Q_FR:     {best_metrics['Q_FR'] - base_metrics['Q_FR']:+.4f}")
    print("=" * 80)

    return {
        "fold": fold,
        "base_metrics": base_metrics,
        "best_metrics": best_metrics,
        "checkpoint": out_ckpt
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_nafnet_fold(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
