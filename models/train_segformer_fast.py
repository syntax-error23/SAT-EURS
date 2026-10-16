"""
Ultra-Fast Pure In-Memory SegFormer MiT-B3 Training Engine.
Engineered for 32GB RTX 5090 GPU + 64GB RAM.
Pre-caches object bounding centers for rare classes so crop generation takes < 1 ms.
Zero disk latency during training.
"""

import os
import os.path as osp
import time
import random

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image


def train_segformer_fast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] SegFormer MiT-B3 DeepLabV3Plus on {device} ({torch.cuda.get_device_name(0)})...")

    clean_dir = "data/train/clean"
    masks_dir = "data/train/masks"
    scene_names = sorted([osp.splitext(f)[0] for f in os.listdir(clean_dir) if f.endswith(".png")])
    print(f"[Data] Preloading {len(scene_names)} high-res clean images and masks into RAM...")

    clean_images = {}
    masks = {}
    rare_centers = {}  # {sc: {cls: list_of_(y, x)}}
    rare_scenes = ["ray1", "sharkFish", "starfish2", "turtles1", "seahorse1"]

    for sc in scene_names:
        with Image.open(osp.join(clean_dir, f"{sc}.png")) as im:
            clean_images[sc] = np.array(im.convert("RGB"), dtype=np.uint8)
        with Image.open(osp.join(masks_dir, f"{sc}.png")) as m:
            m_arr = np.array(m, dtype=np.uint8)
            masks[sc] = m_arr

            # Pre-compute object centers for rare classes (3 and 4)
            rare_centers[sc] = {}
            for c in [3, 4]:
                ys, xs = np.where(m_arr == c)
                if len(ys) > 0:
                    # Subsample up to 200 coordinates
                    step_c = max(1, len(ys) // 200)
                    rare_centers[sc][c] = list(zip(ys[::step_c], xs[::step_c]))

    print(f"[Data] Preloaded all 20 scenes and pre-indexed rare marine object centers into RAM!")

    model = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        classes=5,
        in_channels=3
    ).to(device)

    # Balanced class weights to eliminate background waterbody dominance
    class_weights = torch.tensor([0.70, 1.15, 1.25, 2.20, 2.20], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    epochs = 8
    steps_per_epoch = 150
    batch_size = 8
    crop_size = 512
    lr = 1.2e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    save_path = "checkpoints/segformer_mit_b3_best.pt"
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    print("=" * 72)
    print("      SEGFORMER (MiT-B3) HIGH-CAPACITY UNDERWATER SEGMENTATION ENGINE")
    print(f"      Epochs: {epochs} | Steps/Epoch: {steps_per_epoch} | Batch: {batch_size} | Crop: {crop_size}x{crop_size}")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step in range(1, steps_per_epoch + 1):
            batch_imgs = []
            batch_masks = []

            for _ in range(batch_size):
                if random.random() < 0.45:
                    sc = random.choice(rare_scenes)
                else:
                    sc = random.choice(scene_names)

                img_full = clean_images[sc]
                mask_full = masks[sc]
                h, w = mask_full.shape

                # Fast crop using pre-indexed object coordinates if rare class present
                if h > crop_size and w > crop_size:
                    found = False
                    if sc in rare_centers:
                        available_cls = [c for c in [3, 4] if c in rare_centers[sc] and len(rare_centers[sc][c]) > 0]
                        if len(available_cls) > 0 and random.random() < 0.75:
                            c_target = random.choice(available_cls)
                            center_y, center_x = random.choice(rare_centers[sc][c_target])
                            top = max(0, min(h - crop_size, center_y - crop_size // 2))
                            left = max(0, min(w - crop_size, center_x - crop_size // 2))
                            found = True

                    if not found:
                        top = random.randint(0, h - crop_size)
                        left = random.randint(0, w - crop_size)

                    img_c = img_full[top:top+crop_size, left:left+crop_size]
                    mask_c = mask_full[top:top+crop_size, left:left+crop_size]
                else:
                    img_c = np.array(Image.fromarray(img_full).resize((crop_size, crop_size), Image.BILINEAR))
                    mask_c = np.array(Image.fromarray(mask_full).resize((crop_size, crop_size), Image.NEAREST))

                # Geometric augmentations
                if random.random() < 0.5:
                    img_c = np.fliplr(img_c)
                    mask_c = np.fliplr(mask_c)
                if random.random() < 0.3:
                    img_c = np.flipud(img_c)
                    mask_c = np.flipud(mask_c)

                img_norm = (img_c.astype(np.float32) / 255.0 - mean) / std
                batch_imgs.append(img_norm.transpose(2, 0, 1))
                batch_masks.append(mask_c)

            x = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device, non_blocking=True)
            y = torch.from_numpy(np.stack(batch_masks, axis=0)).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(x)
                loss = ce_loss_fn(outputs, y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 30 == 0 or step == steps_per_epoch:
                print(f"  Epoch [{epoch}/{epochs}] Step [{step}/{steps_per_epoch}] - Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()
        epoch_loss = running_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch}/{epochs} Complete - Avg Loss: {epoch_loss:.4f} ({elapsed:.1f}s) ===")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, save_path)
            print(f"  >>> Checkpoint saved to {save_path} (Loss: {best_loss:.4f})")

    print("=" * 72)
    print(f"[Done] SegFormer MiT-B3 training complete! Best Loss: {best_loss:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    train_segformer_fast()
