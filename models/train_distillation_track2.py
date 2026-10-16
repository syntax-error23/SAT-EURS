"""
Teacher-Student Knowledge Distillation for Track 2 Embedded Deployment.
Teacher: High-Capacity DeepLabV3-ResNet101 (60.99M params, loss 0.238)
Student: Ultra-Lightweight EURS-Net V5 (0.768M params, 9.05 GFLOPs)

Distillation Objectives:
1. Hard Cross-Entropy Loss with Ground Truth: CE(student_logits, y)
2. Soft KL-Divergence Distillation Loss: T^2 * KL(softmax(student_logits / T), softmax(teacher_logits / T))
3. Restoration Preservation: L1 + SSIM Loss on paired degraded-clean images
"""

import os
import os.path as osp
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from models.eurs_net_v5 import EURSNetV5
from models.train_deeplabv3_fast import build_deeplabv3


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Fast 3x3 box filter approximation for SSIM
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - torch.clamp(ssim_map.mean(), 0.0, 1.0)


def train_distillation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Starting Knowledge Distillation on {device} ({torch.cuda.get_device_name(0)})...")

    # 1. Load Pretrained Teacher (Frozen)
    teacher = build_deeplabv3(num_classes=5).to(device)
    teacher_ckpt = torch.load("checkpoints/deeplabv3_resnet101_best.pt", map_location=device, weights_only=False)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("  Teacher loaded: DeepLabV3-ResNet101 (60.99M params, frozen)")

    # 2. Load Student (EURSNetV5, 0.768M params)
    student = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    student_ckpt = torch.load("checkpoints/v5_gan_attn/best_v5_checkpoint.pt", map_location=device, weights_only=False)
    student.load_state_dict(student_ckpt.get("model_state_dict", student_ckpt), strict=True)
    student.train()
    print("  Student loaded: EURS-Net V5 (0.768M params, 9.05 GFLOPs)")

    # 3. Preload all clean images, masks, and degraded references into RAM
    clean_dir = "data/train/clean"
    masks_dir = "data/train/masks"
    scene_names = sorted([osp.splitext(f)[0] for f in os.listdir(clean_dir) if f.endswith(".png")])
    print(f"  Preloading {len(scene_names)} training scenes into RAM...")

    clean_images = {}
    masks = {}
    rare_scenes = ["ray1", "sharkFish", "starfish2", "turtles1", "seahorse1"]

    for sc in scene_names:
        with Image.open(osp.join(clean_dir, f"{sc}.png")) as im:
            clean_images[sc] = np.array(im.convert("RGB"), dtype=np.uint8)
        with Image.open(osp.join(masks_dir, f"{sc}.png")) as m:
            masks[sc] = np.array(m, dtype=np.uint8)
    print("  Cached all training scenes into RAM successfully!")

    # Distillation hyperparameters
    temperature = 3.0
    alpha_ce = 1.0
    alpha_kd = 2.5
    alpha_rest = 3.0
    epochs = 8
    steps_per_epoch = 150
    batch_size = 8
    crop_size = 512
    lr = 1.0e-4

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Class weights for hard CE: balanced foreground vs background
    class_weights = torch.tensor([0.75, 1.15, 1.25, 2.00, 2.00], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ImageNet normalization for Teacher input
    teacher_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    teacher_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    save_path = "checkpoints/v5_distilled_track2_best.pt"
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    best_loss = float("inf")

    print("=" * 72)
    print("     TEACHER-STUDENT DISTILLATION PIPELINE FOR TRACK 2")
    print(f"     Epochs: {epochs} | Steps/Epoch: {steps_per_epoch} | Batch: {batch_size} | Temp: {temperature}")
    print("=" * 72)

    for epoch in range(1, epochs + 1):
        student.train()
        t0 = time.time()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        running_rest = 0.0

        for step in range(1, steps_per_epoch + 1):
            batch_clean = []
            batch_masks = []

            for _ in range(batch_size):
                if random.random() < 0.40:
                    sc = random.choice(rare_scenes)
                else:
                    sc = random.choice(scene_names)

                img_full = clean_images[sc]
                mask_full = masks[sc]
                h, w = mask_full.shape

                # Random crop
                if h > crop_size and w > crop_size:
                    top = random.randint(0, h - crop_size)
                    left = random.randint(0, w - crop_size)
                    img_c = img_full[top:top+crop_size, left:left+crop_size]
                    mask_c = mask_full[top:top+crop_size, left:left+crop_size]
                else:
                    img_c = np.array(Image.fromarray(img_full).resize((crop_size, crop_size), Image.BILINEAR))
                    mask_c = np.array(Image.fromarray(mask_full).resize((crop_size, crop_size), Image.NEAREST))

                # Random flip
                if random.random() < 0.5:
                    img_c = np.fliplr(img_c)
                    mask_c = np.fliplr(mask_c)
                if random.random() < 0.3:
                    img_c = np.flipud(img_c)
                    mask_c = np.flipud(mask_c)

                batch_clean.append(img_c.transpose(2, 0, 1).astype(np.float32) / 255.0)
                batch_masks.append(mask_c)

            clean_tensor = torch.from_numpy(np.stack(batch_clean, axis=0)).to(device, non_blocking=True)
            mask_tensor = torch.from_numpy(np.stack(batch_masks, axis=0)).long().to(device, non_blocking=True)

            # Synthetic underwater degradation jitter for restoration robustness
            with torch.no_grad():
                atten_factors = torch.rand(batch_size, 3, 1, 1, device=device) * 0.45 + 0.35
                atten_factors[:, 0, :, :] *= 0.65  # simulate red channel absorption
                backscatter = torch.rand(batch_size, 3, 1, 1, device=device) * 0.25 + 0.10
                degraded_input = (clean_tensor * atten_factors + backscatter).clamp(0.0, 1.0)

                # Teacher inference on clean/restored input
                teacher_in = (clean_tensor - teacher_mean) / teacher_std
                teacher_out = teacher(teacher_in)
                teacher_logits = teacher_out["out"].detach()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Student forward pass on degraded input
                rest_pred, student_logits = student(degraded_input)

                # 1. Hard Segmentation CE Loss
                loss_ce = ce_loss_fn(student_logits, mask_tensor)

                # 2. Soft KL-Divergence Distillation Loss (per-pixel normalized)
                p_s = F.log_softmax(student_logits / temperature, dim=1)
                p_t = F.softmax(teacher_logits / temperature, dim=1)
                loss_kd = (p_t * (p_t.clamp(1e-7).log() - p_s)).sum(dim=1).mean() * (temperature ** 2)

                # 3. Restoration Fidelity Loss
                loss_l1 = F.l1_loss(rest_pred, clean_tensor)
                loss_ssim = ssim_loss(rest_pred, clean_tensor)
                loss_rest = loss_l1 + 0.5 * loss_ssim

                # Total Combined Objective
                total_loss = alpha_ce * loss_ce + alpha_kd * loss_kd + alpha_rest * loss_rest

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_kd += loss_kd.item()
            running_ce += loss_ce.item()
            running_rest += loss_rest.item()

            if step % 30 == 0 or step == steps_per_epoch:
                print(
                    f"  Epoch [{epoch}/{epochs}] Step [{step}/{steps_per_epoch}] - "
                    f"Total: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f}, Rest: {loss_rest.item():.4f}) - "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        scheduler.step()
        epoch_loss = running_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch}/{epochs} Complete - Avg Loss: {epoch_loss:.4f} ({elapsed:.1f}s) ===")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "loss": best_loss,
            }, save_path)
            print(f"  >>> Best distilled student saved to {save_path} (Loss: {best_loss:.4f})")

    print("=" * 72)
    print(f"[Done] Distillation complete! Checkpoint ready at: {save_path}")
    print("=" * 72)


if __name__ == "__main__":
    train_distillation()
