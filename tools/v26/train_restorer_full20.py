"""
SAT-EURS Track 1 — V26 Deep Bilateral Restorer Training Pipeline (Full-20 SOTA)
Trains DeepBilateralRestorerV26 on all 20 EURS training scenes (4,000 images) using RTX 5090.
Target: Breach 33.0 dB PSNR, 0.93 SSIM, 0.98 Sharpness Ratio.
"""

import os
import os.path as osp
import sys
import time
import math
import random
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")

from models.v26.deep_bilateral_restorer import DeepBilateralRestorerV26
from losses.v26.restoration_loss import RestorationV26Loss
from metrics.restoration_metrics import RestorationEvaluator


class Full20RestorationDataset(Dataset):
    """
    Loads all 20 scenes (4,000 paired images) with fast in-memory clean caching
    and random 512x512 crops + horizontal flips for training.
    """
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        crop_size: int = 512,
        is_train: bool = True
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train

        self.clean_dir = osp.join(data_root, "clean")
        self.deg_dir = osp.join(data_root, "degraded")

        self.items: List[Tuple[str, int, int]] = []
        for sc in self.scenes:
            for wt in range(1, 11):
                for depth in range(5, 105, 5):
                    self.items.append((sc, wt, depth))

        # Cache clean images in float32 RGB [0, 1]
        self.clean_cache: Dict[str, np.ndarray] = {}
        for sc in self.scenes:
            p = osp.join(self.clean_dir, f"{sc}.png")
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.clean_cache[sc] = rgb

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sc, wt, depth = self.items[idx]
        deg_path = osp.join(self.deg_dir, sc, f"{sc}_{wt}_{depth}.png")
        deg_bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_rgb = self.clean_cache[sc]

        H, W, _ = deg_rgb.shape

        if self.is_train and self.crop_size < min(H, W):
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            deg_patch = deg_rgb[top:top+self.crop_size, left:left+self.crop_size]
            clean_patch = clean_rgb[top:top+self.crop_size, left:left+self.crop_size]

            if random.random() > 0.5:
                deg_patch = np.fliplr(deg_patch).copy()
                clean_patch = np.fliplr(clean_patch).copy()
        else:
            deg_patch = deg_rgb
            clean_patch = clean_rgb

        deg_t = torch.from_numpy(deg_patch).permute(2, 0, 1)
        clean_t = torch.from_numpy(clean_patch).permute(2, 0, 1)

        return {
            "degraded": deg_t,
            "clean": clean_t,
            "wt": torch.tensor(wt, dtype=torch.long),
            "depth": torch.tensor(depth, dtype=torch.float32),
            "scene": sc
        }


def train_restorer_full20(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 4e-4,
    device_id: int = 0,
    out_dir: str = "checkpoints/v26"
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[V26 RESTORER] Training DeepBilateralRestorerV26 on {device} ({torch.cuda.get_device_name(device)})")
    print(f"Hyperparams: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # 1. Load Base Physical Table
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    if not osp.exists(table_path):
        raise FileNotFoundError(f"Missing base table: {table_path}")
    base_table = torch.load(table_path, map_location=device)

    # 2. Build Model & Loss
    model = DeepBilateralRestorerV26(rsuigm_table=base_table).to(device)
    loss_fn = RestorationV26Loss().to(device)

    # 3. Setup Dataset on all 20 scenes
    all_scenes = [
        'fish1', 'fish11', 'fish12', 'fish13', 'fish14', 'fish15', 'fish17', 'fish18',
        'fish2', 'fish20', 'fish21', 'fish4', 'fish5', 'fish7', 'fish9',
        'ray1', 'seahorse1', 'sharkFish', 'starfish2', 'turtles1'
    ]

    train_ds = Full20RestorationDataset(scenes=all_scenes, crop_size=512, is_train=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"Dataset loaded: {len(train_ds)} samples ({len(train_loader)} batches/epoch).")

    best_loss = 1e9
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            deg = batch["degraded"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            wt = batch["wt"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(deg, wt, depth)
                loss_dict = loss_fn(out, clean)
                loss = loss_dict["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % 125 == 0 or (step + 1) == len(train_loader):
                print(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step+1:03d}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} (Charb: {loss_dict['charb']:.4f}, SSIM: {loss_dict['msssim']:.4f}, "
                    f"Lap: {loss_dict['lap']:.4f}, Aux: {loss_dict['aux']:.4f})",
                    flush=True
                )

        scheduler.step()
        mean_epoch_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        print(f"--> Epoch {epoch:02d} complete in {elapsed:.1f}s | Mean Loss: {mean_epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        ckpt_path = osp.join(out_dir, "restorer_full20_latest.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "mean_loss": mean_epoch_loss,
            "hyperparams": {"epochs": epochs, "batch_size": batch_size, "lr": lr}
        }, ckpt_path)

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            best_path = osp.join(out_dir, "restorer_full20_best.pt")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "hyperparams": {"epochs": epochs, "batch_size": batch_size, "lr": lr}
            }, best_path)
            print(f"  *** New best restorer checkpoint saved to {best_path} (Loss: {best_loss:.4f}) ***")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes! Best loss: {best_loss:.4f}")
    return best_path


if __name__ == "__main__":
    train_restorer_full20(epochs=15, batch_size=8, lr=4e-4)
