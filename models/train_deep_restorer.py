import os
import glob
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, ".")
from models.deep_restorer_nafnet import EURSUnderwaterRestorer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PairedRestorationDataset(Dataset):
    def __init__(self, data_root="data/train", crop_size=512, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train
        self.pairs = []

        deg_scenes = sorted(glob.glob(os.path.join(data_root, "degraded", "*")))
        for scene_dir in deg_scenes:
            scene_name = os.path.basename(scene_dir)
            clean_f = os.path.join(data_root, "clean", f"{scene_name}.png")
            if not os.path.exists(clean_f):
                continue
            deg_files = sorted(glob.glob(os.path.join(scene_dir, "*.png")))
            for deg_f in deg_files:
                self.pairs.append((deg_f, clean_f))

        print(f"Loaded {len(self.pairs)} image pairs from {data_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg_img = Image.open(deg_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if self.is_train:
            # Random Crop
            w, h = deg_img.size
            if w > self.crop_size and h > self.crop_size:
                i = random.randint(0, h - self.crop_size)
                j = random.randint(0, w - self.crop_size)
                deg_img = TF.crop(deg_img, i, j, self.crop_size, self.crop_size)
                clean_img = TF.crop(clean_img, i, j, self.crop_size, self.crop_size)
            else:
                deg_img = TF.resize(deg_img, (self.crop_size, self.crop_size))
                clean_img = TF.resize(clean_img, (self.crop_size, self.crop_size))

            # Random Horizontal and Vertical Flip
            if random.random() > 0.5:
                deg_img = TF.hflip(deg_img)
                clean_img = TF.hflip(clean_img)
            if random.random() > 0.5:
                deg_img = TF.vflip(deg_img)
                clean_img = TF.vflip(clean_img)
        else:
            deg_img = TF.resize(deg_img, (self.crop_size, self.crop_size))
            clean_img = TF.resize(clean_img, (self.crop_size, self.crop_size))

        deg_t = TF.to_tensor(deg_img)
        clean_t = TF.to_tensor(clean_img)
        return deg_t, clean_t

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps2))
        return loss

class MultiScaleSSIMLoss(nn.Module):
    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.gaussian_kernel = self._create_kernel(window_size)

    def _create_kernel(self, size, sigma=1.5):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel_4d = kernel_2d.expand(self.channels, 1, size, size)
        return kernel_4d

    def forward(self, pred, target):
        kernel = self.gaussian_kernel.to(pred.device, dtype=pred.dtype)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=self.channels)
        mu_y = F.conv2d(target, kernel, padding=self.window_size // 2, groups=self.channels)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(pred * pred, kernel, padding=self.window_size // 2, groups=self.channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, kernel, padding=self.window_size // 2, groups=self.channels) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, kernel, padding=self.window_size // 2, groups=self.channels) - mu_xy

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        return 1.0 - ssim_map.mean()

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="backward")
        target_fft = torch.fft.rfft2(target, norm="backward")
        loss = F.l1_loss(torch.real(pred_fft), torch.real(target_fft)) + \
               F.l1_loss(torch.imag(pred_fft), torch.imag(target_fft))
        return loss

class LaplacianEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = lap.repeat(3, 1, 1, 1)

    def forward(self, pred, target):
        k = self.kernel.to(pred.device, dtype=pred.dtype)
        pred_edge = F.conv2d(pred, k, padding=1, groups=3)
        target_edge = F.conv2d(target, k, padding=1, groups=3)
        return F.l1_loss(pred_edge, target_edge)

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def train_restorer():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0)})")

    batch_size = 8
    num_workers = 0
    epochs = 25
    lr = 3e-4

    dataset = PairedRestorationDataset("data/train", crop_size=512, is_train=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    model = EURSUnderwaterRestorer().to(device)
    print(f"EURS Deep Restorer initialized: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    charb_crit = CharbonnierLoss().to(device)
    ssim_crit = MultiScaleSSIMLoss().to(device)
    fft_crit = FFTLoss().to(device)
    edge_crit = LaplacianEdgeLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    os.makedirs("checkpoints/deep_restorer_nafnet", exist_ok=True)
    best_loss = 1e9

    print(f"Beginning deep restoration training across {len(dataset)} pairs for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_psnr = 0.0
        t0 = time.time()

        for step, (deg, clean) in enumerate(train_loader):
            deg, clean = deg.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                restored = model(deg)
                loss_c = charb_crit(restored, clean)
                loss_s = ssim_crit(restored, clean)
                loss_f = fft_crit(restored, clean)
                loss_e = edge_crit(restored, clean)

                loss = loss_c + 0.5 * loss_s + 0.1 * loss_f + 0.05 * loss_e

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                total_psnr += compute_psnr(restored, clean)

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Step [{step+1:03d}/{len(train_loader):03d}] "
                      f"Loss: {loss.item():.4f} (Charb: {loss_c.item():.4f}, SSIM: {loss_s.item():.4f}, FFT: {loss_f.item():.4f}) "
                      f"PSNR: {compute_psnr(restored, clean):.2f} dB")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} Complete in {dt:.1f}s | Avg Loss: {avg_loss:.4f} | Avg PSNR: {avg_psnr:.2f} dB | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "avg_loss": avg_loss,
            "avg_psnr": avg_psnr
        }, f"checkpoints/deep_restorer_nafnet/latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "avg_loss": avg_loss,
                "avg_psnr": avg_psnr
            }, f"checkpoints/deep_restorer_nafnet/best_restorer.pt")
            print(f"--> Saved new BEST restoration checkpoint: Avg Loss {avg_loss:.4f}, PSNR {avg_psnr:.2f} dB")

if __name__ == "__main__":
    train_restorer()
