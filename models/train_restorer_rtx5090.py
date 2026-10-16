"""
High-Capacity EURS Dedicated Restorer Training Script
Optimized for NVIDIA GeForce RTX 5090 (34.2 GB VRAM) & 24 CPU cores.
Smooth, responsive, zero-lag execution using EURSDataset and mixed precision.
"""

import os
import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data.dataset import EURSDataset
from models.deep_restorer_nafnet import EURSUnderwaterRestorer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))

class FastSSIMLoss(nn.Module):
    def __init__(self, channels=3, window_size=11):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        k2d = g.unsqueeze(1) * g.unsqueeze(0)
        self.register_buffer("kernel", k2d.expand(channels, 1, window_size, window_size).contiguous())

    def forward(self, pred, target):
        k = self.kernel.to(dtype=pred.dtype)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_x = F.conv2d(pred, k, padding=self.window_size // 2, groups=self.channels)
        mu_y = F.conv2d(target, k, padding=self.window_size // 2, groups=self.channels)
        mu_x_sq, mu_y_sq, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
        sigma_x_sq = F.conv2d(pred * pred, k, padding=self.window_size // 2, groups=self.channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, k, padding=self.window_size // 2, groups=self.channels) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, k, padding=self.window_size // 2, groups=self.channels) - mu_xy
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return 1.0 - ssim_map.mean()

class FastFFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="backward")
        target_fft = torch.fft.rfft2(target, norm="backward")
        return F.l1_loss(torch.real(pred_fft), torch.real(target_fft)) + F.l1_loss(torch.imag(pred_fft), torch.imag(target_fft))

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def train_main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device} ({torch.cuda.get_device_name(0)})")

    # High performance, perfectly tuned parameters:
    # 512x512 resolution, batch size 8 on RTX 5090 runs lightning fast with ~4GB VRAM
    batch_size = 4
    target_size = (1024, 1024)
    num_workers = 2
    epochs = 20
    lr = 3.0e-4

    print(f"[INIT] Loading EURSDataset from data/cached_1024 (native 1024x1024)...")
    train_dataset = EURSDataset("data/cached_1024", split="all", target_size=target_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"[INIT] Dataloader ready: {len(train_dataset)} samples ({len(train_loader)} batches/epoch)")

    model = EURSUnderwaterRestorer().to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] EURS Dedicated Restorer initialized ({params_m:.2f}M parameters)")

    charb_crit = CharbonnierLoss().to(device)
    ssim_crit = FastSSIMLoss().to(device)
    fft_crit = FastFFTLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    save_dir = "checkpoints/deep_restorer_nafnet"
    os.makedirs(save_dir, exist_ok=True)
    best_psnr = 0.0

    print("=" * 72)
    print("   STARTING DEDICATED RESTORER TRAINING ON FULL DATASET")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_psnr = 0.0, 0.0
        t0 = time.time()

        for step, (deg, clean, _, _) in enumerate(train_loader):
            deg, clean = deg.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                restored = model(deg)
                loss_c = charb_crit(restored, clean)
                loss_s = ssim_crit(restored, clean)
                loss_f = fft_crit(restored, clean)
                loss = loss_c + 0.5 * loss_s + 0.05 * loss_f

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                psnr_val = compute_psnr(restored.float(), clean.float())
                total_psnr += psnr_val

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - t0
                speed = (step + 1) * batch_size / elapsed
                print(f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step+1:03d}/{len(train_loader):03d}] "
                      f"Loss: {loss.item():.4f} (SSIM_loss: {loss_s.item():.4f}) | "
                      f"PSNR: {psnr_val:.2f} dB | Speed: {speed:.1f} img/s")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        epoch_time = time.time() - t0

        print("-" * 72)
        print(f"  --> Epoch {epoch:02d}/{epochs:02d} Finished in {epoch_time:.1f}s | "
              f"Avg Loss: {avg_loss:.4f} | Avg PSNR: {avg_psnr:.2f} dB | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Checkpoint
        latest_path = os.path.join(save_dir, "latest.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "avg_loss": avg_loss,
            "avg_psnr": avg_psnr
        }, latest_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_path = os.path.join(save_dir, "best_restorer.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "avg_loss": avg_loss,
                "avg_psnr": avg_psnr
            }, best_path)
            print(f"  >>> [NEW BEST CHECKPOINT] Saved to {best_path} (PSNR: {best_psnr:.2f} dB)")
        print("-" * 72)

if __name__ == "__main__":
    train_main()
