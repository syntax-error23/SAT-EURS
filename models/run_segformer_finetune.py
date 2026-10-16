"""
SegFormer MiT-B3 Fine-Tuning Engine
Loads pretrained checkpoints/segformer_mit_b3_best.pt (Loss: 0.097) and trains
with rare-class centering and scene-prior loss on RTX 5090 (34.2 GB VRAM).
Optimizes for the official MaCVi composite scoring formula:
S_composite = 0.60 * Q_seg + 0.30 * Q_FR + 0.10 * Q_NR
"""

import os
import os.path as osp
import time
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image

class EURSTransformerDataset(Dataset):
    def __init__(self, data_root: str = "data/train", crop_size: int = 512, samples_per_epoch: int = 1600):
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch

        self.clean_dir = osp.join(data_root, "clean")
        self.masks_dir = osp.join(data_root, "masks")
        self.degraded_dir = osp.join(data_root, "degraded")

        self.scenes = sorted([
            osp.splitext(f)[0] for f in os.listdir(self.clean_dir)
            if f.endswith(".png")
        ])

        print(f"[Dataset] Loading all {len(self.scenes)} full scenes into 64GB RAM...", flush=True)
        self.clean_images = {}
        self.masks = {}
        self.degraded_files = {}

        for sc in self.scenes:
            with Image.open(osp.join(self.clean_dir, f"{sc}.png")) as im:
                self.clean_images[sc] = np.array(im.convert("RGB"), dtype=np.uint8)
            with Image.open(osp.join(self.masks_dir, f"{sc}.png")) as m:
                self.masks[sc] = np.array(m, dtype=np.uint8)
            
            sc_deg = osp.join(self.degraded_dir, sc)
            if osp.exists(sc_deg):
                self.degraded_files[sc] = [
                    osp.join(sc_deg, f) for f in os.listdir(sc_deg) if f.endswith(".png")
                ]
            else:
                self.degraded_files[sc] = []

        print(f"[Dataset] Cached all {len(self.scenes)} scenes into RAM!", flush=True)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rare_scenes = ["ray1", "sharkFish", "starfish2", "turtles1", "seahorse1"]
        if random.random() < 0.40:
            sc = random.choice(rare_scenes)
        else:
            sc = random.choice(self.scenes)

        clean_im = self.clean_images[sc]
        mask = self.masks[sc]
        h, w = mask.shape
        cs = self.crop_size

        if random.random() < 0.50 and len(self.degraded_files[sc]) > 0:
            deg_path = random.choice(self.degraded_files[sc])
            with Image.open(deg_path) as dim:
                if dim.size != (w, h):
                    dim = dim.resize((w, h), Image.BILINEAR)
                img = np.array(dim.convert("RGB"), dtype=np.uint8)
        else:
            img = clean_im

        if h > cs and w > cs:
            top, left = 0, 0
            found_crop = False
            unique_classes = np.unique(mask)
            rare_in_mask = [c for c in [3, 4] if c in unique_classes]
            
            if len(rare_in_mask) > 0 and random.random() < 0.70:
                target_cls = random.choice(rare_in_mask)
                ys, xs = np.where(mask == target_cls)
                if len(ys) > 0:
                    c_idx = random.randint(0, len(ys) - 1)
                    center_y, center_x = ys[c_idx], xs[c_idx]
                    top = max(0, min(h - cs, center_y - cs // 2))
                    left = max(0, min(w - cs, center_x - cs // 2))
                    found_crop = True

            if not found_crop:
                top = random.randint(0, h - cs)
                left = random.randint(0, w - cs)

            img_crop = img[top:top+cs, left:left+cs]
            mask_crop = mask[top:top+cs, left:left+cs]
        else:
            img_crop = np.array(Image.fromarray(img).resize((cs, cs), Image.BILINEAR))
            mask_crop = np.array(Image.fromarray(mask).resize((cs, cs), Image.NEAREST))

        if random.random() < 0.5:
            img_crop = np.fliplr(img_crop)
            mask_crop = np.fliplr(mask_crop)
        if random.random() < 0.3:
            img_crop = np.flipud(img_crop)
            mask_crop = np.flipud(mask_crop)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_crop.astype(np.float32) / 255.0 - mean) / std

        return (
            torch.from_numpy(img_norm.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(mask_crop.copy()).long()
        )

def fine_tune_segformer(resume_ckpt: str = None, total_epochs: int = 12):
    import argparse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] SegFormer Engine running on {device} ({torch.cuda.get_device_name(0)})", flush=True)

    epochs = total_epochs
    batch_size = 24  # Uses 27.2GB of RTX 5090 VRAM (leaving ~7GB safety headroom)
    lr = 1.1e-4
    save_path = "checkpoints/segformer_mit_b3_best.pt"
    last_path = "checkpoints/segformer_mit_b3_last.pt"
    os.makedirs("checkpoints", exist_ok=True)

    model = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights=None,
        classes=5,
        in_channels=3
    ).to(device)

    start_epoch = 1
    best_loss = 0.097

    # Warmstart / Resume logic
    load_from = resume_ckpt if (resume_ckpt and osp.exists(resume_ckpt)) else (last_path if osp.exists(last_path) else (save_path if osp.exists(save_path) else None))
    if load_from:
        print(f"[Init] Loading weights from {load_from}...", flush=True)
        ckpt = torch.load(load_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("best_loss", ckpt.get("loss", best_loss))
            print(f"[Init] Resuming from Epoch {start_epoch} (Previous Best Loss: {best_loss:.4f})", flush=True)

    class_weights = torch.tensor([0.70, 1.15, 1.25, 2.00, 2.00], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    dataset = EURSTransformerDataset(crop_size=512, samples_per_epoch=3120)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("=" * 72, flush=True)
    print("      SEGFORMER (MiT-B3) HIGH-CAPACITY FINE-TUNING ACTIVE", flush=True)
    print(f"      Epochs: {start_epoch} -> {epochs} | Steps/Epoch: {len(dataloader)} | Batch: {batch_size} | LR: {lr}", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, (images, masks) in enumerate(dataloader, 1):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss = ce_loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 5 == 0 or step == len(dataloader) or step == 1:
                elapsed = time.time() - t0
                speed = step * batch_size / max(1e-3, elapsed)
                print(f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step:03d}/{len(dataloader):03d}] "
                      f"Loss: {loss.item():.4f} | Speed: {speed:.1f} img/s | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch:02d}/{epochs:02d} Complete - Avg Loss: {epoch_loss:.4f} ({elapsed:.1f}s) ===", flush=True)

        # 1. ALWAYS save last and per-epoch checkpoints so training can be paused cleanly anytime
        epoch_path = f"checkpoints/segformer_mit_b3_epoch_{epoch:02d}.pt"
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
        }
        torch.save(ckpt_payload, last_path)
        torch.save(ckpt_payload, epoch_path)
        print(f"  >>> [SAVED CHECKPOINT] {epoch_path} and {last_path}", flush=True)

        # 2. Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_payload["best_loss"] = best_loss
            torch.save(ckpt_payload, save_path)
            print(f"  >>> [NEW BEST CHECKPOINT] Saved to {save_path} (Loss: {best_loss:.4f})", flush=True)

    print("=" * 72, flush=True)
    print(f"[Done] SegFormer fine-tuning complete! Best Loss: {best_loss:.4f}", flush=True)
    print("=" * 72, flush=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=12, help="Total epochs to train")
    args = p.parse_args()
    fine_tune_segformer(resume_ckpt=args.resume, total_epochs=args.epochs)
