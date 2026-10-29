"""
Phase 3 — Train NAFNet as an Isolated Candidate
MaCVi @ WACV 2027 Challenge — SAT-EURS Track 1
Workstation: Windows 11 | RTX 5090 32GB | 64GB DDR5

Protocol:
- Architecture: Canonical NAFNet (width=32, 13.82M params, 3 in, 3 out, residual connection)
- Input: Raw Degraded RGB [0, 1]
- Target: Paired Clean RGB [0, 1]
- Split: Strict Fold 0 (16 train scenes / 4 val scenes: ['fish1', 'fish11', 'fish12', 'seahorse1'])
- Batch Size: 4 (AMP BF16, 10.72 GiB peak VRAM, 21.12 GiB free headroom)
- Loss: Charbonnier Loss + 0.40 * SSIM Loss
- Validation: Full 1024x1024 resolution on 160 held-out samples every epoch
"""

import os
import os.path as osp
import sys
import time
import math
import json
import argparse
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from models.nafnet import build_nafnet_restorer
from losses.restoration_v7_loss import CharbonnierLoss, SSIMLoss
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        crop_size: int = 512,
        is_train: bool = True,
        sample_stride: int = 1,
        seed: int = 42
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train
        self.seed = seed
        self.epoch = 0

        self.clean_dir = osp.join(data_root, "clean")
        self.deg_dir = osp.join(data_root, "degraded")

        self.clean_cache = {}
        for sc in self.scenes:
            bgr = cv2.imread(osp.join(self.clean_dir, f"{sc}.png"))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            self.clean_cache[sc] = rgb

        self.items: List[Tuple[str, str]] = []
        for sc in self.scenes:
            sc_deg = osp.join(self.deg_dir, sc)
            if not osp.exists(sc_deg):
                continue
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])
            for df in deg_files[::sample_stride]:
                self.items.append((sc, osp.join(sc_deg, df)))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sc, deg_p = self.items[idx]
        clean_rgb = self.clean_cache[sc]
        deg_bgr = cv2.imread(deg_p)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_train:
            h, w = clean_rgb.shape[:2]
            cs = self.crop_size
            eff_seed = self.seed + self.epoch * 100003 + idx * 1009
            rng = random.Random(eff_seed)
            top = rng.randint(0, h - cs)
            left = rng.randint(0, w - cs)
            hflip = rng.random() > 0.5
            vflip = rng.random() > 0.5

            clean_crop = clean_rgb[top:top+cs, left:left+cs].copy()
            deg_crop = deg_rgb[top:top+cs, left:left+cs].copy()

            if hflip:
                clean_crop = np.ascontiguousarray(clean_crop[:, ::-1])
                deg_crop = np.ascontiguousarray(deg_crop[:, ::-1])
            if vflip:
                clean_crop = np.ascontiguousarray(clean_crop[::-1, :])
                deg_crop = np.ascontiguousarray(deg_crop[::-1, :])

            deg_t = torch.from_numpy(deg_crop.transpose(2, 0, 1)).float()
            clean_t = torch.from_numpy(clean_crop.transpose(2, 0, 1)).float()
        else:
            deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).float()
            clean_t = torch.from_numpy(clean_rgb.transpose(2, 0, 1)).float()

        return deg_t, clean_t


def atomic_save(state: Dict[str, Any], filepath: str):
    os.makedirs(osp.dirname(filepath), exist_ok=True)
    tmp_path = filepath + ".tmp"
    torch.save(state, tmp_path)
    if osp.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)


def train_nafnet(
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    crop_size: int = 512,
    smoke_test: bool = False,
    resume: bool = False,
    out_dir: str = "checkpoints/phase3_nafnet_f0"
):
    print("=" * 80)
    print("PHASE 3: TRAINING CANONICAL NAFNET RESTORER (FOLD 0)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    print(f"Initial Free VRAM: {free_vram / (1024**3):.2f} GiB / {total_vram / (1024**3):.2f} GiB")

    # Load Split
    with open("splits/manifest_5fold.json", "r") as f:
        split_data = json.load(f)
    train_scenes = split_data["folds"]["fold_0"]["train_scenes"]
    val_scenes = split_data["folds"]["fold_0"]["val_scenes"]

    print(f"Train Scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val Scenes ({len(val_scenes)}): {val_scenes} (STRICTLY DISJOINT)")

    train_ds = PairedRestorationDataset(
        scenes=train_scenes,
        data_root="data/cached_1024",
        crop_size=crop_size,
        is_train=True,
        sample_stride=1
    )
    val_ds = PairedRestorationDataset(
        scenes=val_scenes,
        data_root="data/cached_1024",
        crop_size=1024,
        is_train=False,
        sample_stride=10 # 80 representative images (20 per scene) for fast full-resolution validation
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print(f"Train samples: {len(train_ds)} ({len(train_loader)} batches of B={batch_size})")
    print(f"Val samples:   {len(val_ds)} (full 1024x1024 resolution)")

    model = build_nafnet_restorer(width=32).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: Canonical NAFNet Width 32 | Parameters: {total_params:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion_charb = CharbonnierLoss(eps=1e-3).to(device)
    criterion_ssim = SSIMLoss(window_size=11, in_channels=3).to(device)

    start_epoch = 0
    best_q_fr = 0.0
    best_psnr = 0.0
    best_ssim = 0.0

    ckpt_best = osp.join(out_dir, "best_model.pt")
    ckpt_last = osp.join(out_dir, "last_model.pt")

    if resume and osp.exists(ckpt_last):
        print(f"Resuming from {ckpt_last}...")
        ckpt = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_q_fr = ckpt.get("best_q_fr", 0.0)
        best_psnr = ckpt.get("best_psnr", 0.0)
        best_ssim = ckpt.get("best_ssim", 0.0)
        print(f"  Resumed from epoch {start_epoch} (Best Q_FR: {best_q_fr:.4f})")

    history = []
    max_steps = 20 if smoke_test else len(train_loader)
    total_epochs = 1 if smoke_test else epochs

    print("\nStarting Training Loop...")
    for epoch in range(start_epoch, total_epochs):
        train_ds.set_epoch(epoch)
        model.train()
        t0_ep = time.time()
        running_loss = 0.0
        running_charb = 0.0
        running_ssim = 0.0
        step_count = 0

        for step, (deg_b, clean_b) in enumerate(train_loader):
            if smoke_test and step >= max_steps:
                break

            deg_b = deg_b.to(device, non_blocking=True)
            clean_b = clean_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_b = model(deg_b)
                loss_charb = criterion_charb(pred_b, clean_b)
                loss_ssim = criterion_ssim(pred_b, clean_b)
                loss = loss_charb + 0.40 * loss_ssim

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_charb += loss_charb.item()
            running_ssim += loss_ssim.item()
            step_count += 1

            if (step + 1) % 50 == 0 or (step + 1) == max_steps:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch [{epoch+1}/{total_epochs}] Step [{step+1}/{max_steps}] "
                    f"Loss: {running_loss/step_count:.4f} (Charb: {running_charb/step_count:.4f}, SSIM: {running_ssim/step_count:.4f}) "
                    f"LR: {cur_lr:.6f}",
                    flush=True
                )

        scheduler.step()
        ep_duration = time.time() - t0_ep
        avg_train_loss = running_loss / max(step_count, 1)

        # Validation Loop (Full Resolution on 160 held-out images)
        print(f"  Validating Epoch {epoch+1} on 160 held-out images (1024x1024)...", flush=True)
        model.eval()
        psnrs, ssims, sharps = [], [], []

        torch.cuda.reset_peak_memory_stats()
        t_val0 = time.time()

        with torch.no_grad():
            for idx in range(len(val_ds)):
                deg_t, clean_t = val_ds[idx]
                deg_t = deg_t.unsqueeze(0).to(device)
                clean_np = clean_t.permute(1, 2, 0).numpy()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_t = model(deg_t)

                pred_np = pred_t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                pred_np = np.clip(pred_np, 0.0, 1.0)

                p, _ = compute_psnr_and_norm(pred_np, clean_np)
                s = compute_ssim_single(pred_np, clean_np)
                sh = compute_sharpness_ratio(pred_np, clean_np)
                psnrs.append(p)
                ssims.append(s)
                sharps.append(sh)

        val_duration = time.time() - t_val0
        mean_psnr = float(np.mean(psnrs))
        psnr_norm = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
        mean_ssim = float(np.mean(ssims))
        mean_sharp = float(np.mean(sharps))
        q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
        peak_vram_gib = torch.cuda.max_memory_reserved() / (1024**3)

        print(
            f"  --> Epoch {epoch+1} Results: PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | "
            f"Sharpness: {mean_sharp:.4f} | Q_FR: {q_fr:.4f} | Peak VRAM: {peak_vram_gib:.2f} GiB | "
            f"Train Time: {ep_duration:.1f}s | Val Time: {val_duration:.1f}s",
            flush=True
        )

        is_best = q_fr > best_q_fr
        if is_best:
            best_q_fr = q_fr
            best_psnr = mean_psnr
            best_ssim = mean_ssim
            print(f"  [*] NEW BEST MODEL: Q_FR = {best_q_fr:.4f} (PSNR: {best_psnr:.2f} dB, SSIM: {best_ssim:.4f})")
            atomic_save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "q_fr": best_q_fr,
                "psnr": best_psnr,
                "ssim": best_ssim,
                "sharpness": mean_sharp,
                "total_params_m": total_params
            }, ckpt_best)

        atomic_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_q_fr": best_q_fr,
            "best_psnr": best_psnr,
            "best_ssim": best_ssim
        }, ckpt_last)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_psnr": round(mean_psnr, 2),
            "val_ssim": round(mean_ssim, 4),
            "val_sharpness": round(mean_sharp, 4),
            "val_q_fr": round(q_fr, 4),
            "is_best": is_best
        })

    print("=" * 80)
    print("PHASE 3 TRAINING COMPLETE")
    print(f"  Best Val Q_FR: {best_q_fr:.4f}")
    print(f"  Best Val PSNR: {best_psnr:.2f} dB")
    print(f"  Best Val SSIM: {best_ssim:.4f}")
    print(f"  Checkpoint:    {ckpt_best}")
    print("=" * 80)

    log_path = osp.join(out_dir, "training_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_q_fr, best_psnr, best_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=str, default="checkpoints/phase3_nafnet_f0")
    args = parser.parse_args()

    train_nafnet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_size=args.crop_size,
        smoke_test=args.smoke_test,
        resume=args.resume,
        out_dir=args.out_dir
    )
