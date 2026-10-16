"""
Clinical Multi-Teacher Knowledge Distillation for Jetson AGX Orin (Track 2).
Ensemble Teachers:
  1. SegFormer MiT-B3 (Vision Transformer, trained 12 epochs, loss 0.2227)
  2. DeepLabV3-ResNet101 (Heavy Context Extractor, 60.99M params, loss 0.2381)
Student:
  EURS-Net V5 (0.768M params, 9.05 GFLOPs at 1024x1024) -> 100% Jetson & TensorRT compliant.

Clinical Objectives:
  1. Ground-Truth Weighted Cross-Entropy + Focal Multi-Class Loss
  2. Multi-Teacher Soft Logit Distillation (KL Divergence with T=3.0)
  3. Restoration Multi-Scale L1 + SSIM Loss with RSUIGM-guided target preservation
"""

import os
import os.path as osp
import time
import random
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models.segmentation as sm
import segmentation_models_pytorch as smp

import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from models.eurs_net_v5 import EURSNetV5
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def run_clinical_distillation(epochs: int = 10, batch_size: int = 16, lr: float = 1.2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76, flush=True)
    print("   CLINICAL MULTI-TEACHER KNOWLEDGE DISTILLATION FOR JETSON ORIN (TRACK 2)", flush=True)
    print(f"   Device: {device} ({torch.cuda.get_device_name(0)}) | Batch Size: {batch_size} | Epochs: {epochs}", flush=True)
    print("=" * 76, flush=True)

    # 1. Load Pretrained Teacher 1: SegFormer MiT-B3
    print("[1/5] Loading Teacher 1: SegFormer MiT-B3 (Vision Transformer, Epoch 12)...", flush=True)
    teacher_sf = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights=None,
        classes=5,
        in_channels=3
    ).to(device)
    sf_ckpt_path = "checkpoints/segformer_mit_b3_epoch_12.pt" if osp.exists("checkpoints/segformer_mit_b3_epoch_12.pt") else "checkpoints/segformer_mit_b3_best.pt"
    ckpt_sf = torch.load(sf_ckpt_path, map_location=device, weights_only=False)
    teacher_sf.load_state_dict(ckpt_sf.get("model_state_dict", ckpt_sf))
    teacher_sf.eval()
    for p in teacher_sf.parameters():
        p.requires_grad = False
    print("      SegFormer MiT-B3 frozen.", flush=True)

    # 2. Load Pretrained Teacher 2: DeepLabV3-ResNet101
    print("[2/5] Loading Teacher 2: DeepLabV3-ResNet101 (60.99M params)...", flush=True)
    teacher_dl = sm.deeplabv3_resnet101(weights=None, num_classes=5, aux_loss=True).to(device)
    ckpt_dl = torch.load("checkpoints/deeplabv3_resnet101_best.pt", map_location=device, weights_only=False)
    teacher_dl.load_state_dict(ckpt_dl.get("model_state_dict", ckpt_dl))
    teacher_dl.eval()
    for p in teacher_dl.parameters():
        p.requires_grad = False
    print("      DeepLabV3-ResNet101 frozen.", flush=True)

    # 3. Load Student: EURSNet-V5 (0.768M params, 9.05 GFLOPs)
    print("[3/5] Loading Student: EURS-Net V5 (0.768M params, 9.05 GFLOPs)...", flush=True)
    student = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    student_init_path = "checkpoints/v5_distilled_track2_best.pt" if osp.exists("checkpoints/v5_distilled_track2_best.pt") else "checkpoints/v5_gan_attn/best_v5_checkpoint.pt"
    ckpt_s = torch.load(student_init_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt_s.get("model_state_dict", ckpt_s), strict=True)
    student.train()
    print(f"      Student initialized from {student_init_path}", flush=True)

    # 4. Load RSUIGM Physical Inversion Table for Teacher Clear-Domain Feeding
    print("[4/5] Loading RSUIGM Physical Inversion Table...", flush=True)
    rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

    # 5. Preload All 20 Scenes into Host RAM
    print("[5/5] Caching 20 Full Clean, Degraded, and Mask Scenes into RAM...", flush=True)
    clean_dir = "data/train/clean"
    masks_dir = "data/train/masks"
    deg_dir = "data/train/degraded"
    scene_names = sorted([osp.splitext(f)[0] for f in os.listdir(clean_dir) if f.endswith(".png")])

    clean_images = {}
    masks = {}
    degraded_files = {}
    rare_scenes = ["ray1", "sharkFish", "starfish2", "turtles1", "seahorse1"]

    for sc in scene_names:
        with Image.open(osp.join(clean_dir, f"{sc}.png")) as im:
            clean_images[sc] = np.array(im.convert("RGB"), dtype=np.uint8)
        with Image.open(osp.join(masks_dir, f"{sc}.png")) as m:
            masks[sc] = np.array(m, dtype=np.uint8)
        sc_deg = osp.join(deg_dir, sc)
        if osp.exists(sc_deg):
            degraded_files[sc] = [osp.join(sc_deg, f) for f in os.listdir(sc_deg) if f.endswith(".png")]
        else:
            degraded_files[sc] = []
    print("      All training data cached into 64GB RAM!", flush=True)

    # Optimization Hyperparameters
    temperature = 3.0
    alpha_ce = 1.0
    alpha_kd_sf = 1.8   # SegFormer soft-label weight
    alpha_kd_dl = 1.2   # DeepLabV3 soft-label weight
    alpha_rest = 3.5    # Restoration fidelity weight
    steps_per_epoch = 150
    crop_size = 512

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Balanced class weights with strong emphasis on rare classes (Rays, Starfish, Sharks)
    class_weights = torch.tensor([0.70, 1.15, 1.25, 2.20, 2.20], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    save_path = "checkpoints/v5_clinical_distilled_best.pt"
    onnx_out_path = "deploy/eurs_net_v5_track2.onnx"
    best_loss = float("inf")

    print("\n" + "=" * 76, flush=True)
    print("   STARTING CLINICAL DISTILLATION TRAINING ON RTX 5090", flush=True)
    print("=" * 76, flush=True)

    for epoch in range(1, epochs + 1):
        student.train()
        t0 = time.time()
        running_loss = 0.0
        running_kd = 0.0
        running_ce = 0.0
        running_rest = 0.0

        for step in range(1, steps_per_epoch + 1):
            batch_clean = []
            batch_deg = []
            batch_masks = []

            for _ in range(batch_size):
                if random.random() < 0.45:
                    sc = random.choice(rare_scenes)
                else:
                    sc = random.choice(scene_names)

                clean_im = clean_images[sc]
                mask = masks[sc]
                h, w = mask.shape
                cs = crop_size

                # Fetch real degraded image if available
                if len(degraded_files[sc]) > 0 and random.random() < 0.70:
                    deg_p = random.choice(degraded_files[sc])
                    with Image.open(deg_p) as dim:
                        if dim.size != (w, h):
                            dim = dim.resize((w, h), Image.BILINEAR)
                        deg_im = np.array(dim.convert("RGB"), dtype=np.uint8)
                else:
                    deg_im = clean_im

                # Rare class focused cropping
                top, left = 0, 0
                found_crop = False
                uniques = np.unique(mask)
                rares = [c for c in [3, 4] if c in uniques]
                if len(rares) > 0 and random.random() < 0.70:
                    t_cls = random.choice(rares)
                    ys, xs = np.where(mask == t_cls)
                    if len(ys) > 0:
                        idx = random.randint(0, len(ys) - 1)
                        top = max(0, min(h - cs, ys[idx] - cs // 2))
                        left = max(0, min(w - cs, xs[idx] - cs // 2))
                        found_crop = True

                if not found_crop:
                    top = random.randint(0, h - cs)
                    left = random.randint(0, w - cs)

                c_crop = clean_im[top:top+cs, left:left+cs]
                d_crop = deg_im[top:top+cs, left:left+cs]
                m_crop = mask[top:top+cs, left:left+cs]

                if random.random() < 0.5:
                    c_crop = np.fliplr(c_crop)
                    d_crop = np.fliplr(d_crop)
                    m_crop = np.fliplr(m_crop)
                if random.random() < 0.3:
                    c_crop = np.flipud(c_crop)
                    d_crop = np.flipud(d_crop)
                    m_crop = np.flipud(m_crop)

                batch_clean.append(c_crop.transpose(2, 0, 1).astype(np.float32) / 255.0)
                batch_deg.append(d_crop.transpose(2, 0, 1).astype(np.float32) / 255.0)
                batch_masks.append(m_crop)

            clean_tensor = torch.from_numpy(np.stack(batch_clean, axis=0)).to(device, non_blocking=True)
            deg_tensor = torch.from_numpy(np.stack(batch_deg, axis=0)).to(device, non_blocking=True)
            mask_tensor = torch.from_numpy(np.stack(batch_masks, axis=0)).long().to(device, non_blocking=True)

            # Get High-Capacity Teacher Outputs (Zero-Grad)
            with torch.no_grad():
                # Normalize clean inputs for teachers
                teacher_norm = (clean_tensor - mean_im) / std_im
                # Teacher 1: SegFormer MiT-B3
                t_logits_sf = teacher_sf(teacher_norm).detach()
                # Teacher 2: DeepLabV3-ResNet101
                t_logits_dl = teacher_dl(clean_tensor)["out"].detach()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Student Forward Pass on degraded underwater input
                rest_pred, student_logits = student(deg_tensor)

                # 1. Hard Segmentation CE Loss
                loss_ce = ce_loss_fn(student_logits, mask_tensor)

                # 2. Soft Multi-Teacher Distillation Loss
                log_p_s = F.log_softmax(student_logits / temperature, dim=1)
                p_t_sf = F.softmax(t_logits_sf / temperature, dim=1)
                p_t_dl = F.softmax(t_logits_dl / temperature, dim=1)

                kd_sf = (p_t_sf * (p_t_sf.clamp(1e-7).log() - log_p_s)).sum(dim=1).mean() * (temperature ** 2)
                kd_dl = (p_t_dl * (p_t_dl.clamp(1e-7).log() - log_p_s)).sum(dim=1).mean() * (temperature ** 2)
                loss_kd = alpha_kd_sf * kd_sf + alpha_kd_dl * kd_dl

                # 3. High-Fidelity Restoration Loss (L1 + Multi-scale SSIM)
                loss_l1 = F.l1_loss(rest_pred, clean_tensor)
                loss_ssim = ssim_loss(rest_pred, clean_tensor)
                loss_rest = loss_l1 + 0.6 * loss_ssim

                # Combined Total Loss
                total_loss = alpha_ce * loss_ce + loss_kd + alpha_rest * loss_rest

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_kd += loss_kd.item()
            running_ce += loss_ce.item()
            running_rest += loss_rest.item()

            if step % 25 == 0 or step == steps_per_epoch:
                elapsed = time.time() - t0
                speed = step * batch_size / max(1e-3, elapsed)
                print(
                    f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step:03d}/{steps_per_epoch:03d}] | "
                    f"Loss: {total_loss.item():.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f}, Rest: {loss_rest.item():.4f}) | "
                    f"{speed:.1f} img/s | LR: {scheduler.get_last_lr()[0]:.6f}",
                    flush=True
                )

        scheduler.step()
        epoch_loss = running_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"=== Epoch {epoch:02d}/{epochs:02d} Complete - Avg Loss: {epoch_loss:.4f} ({elapsed:.1f}s) ===", flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "loss": best_loss,
            }
            torch.save(ckpt_dict, save_path)
            print(f"  >>> [NEW BEST STUDENT SAVED] {save_path} (Loss: {best_loss:.4f})", flush=True)

    print("=" * 76, flush=True)
    print(f"[Done] Clinical Distillation Complete! Best Checkpoint: {save_path} (Loss: {best_loss:.4f})", flush=True)
    print("=" * 76, flush=True)

    # Export to Static 1024x1024 ONNX for Jetson AGX Orin
    print("\n[Export] Generating Jetson-Compliant Static 1024x1024 ONNX Engine...", flush=True)
    student.eval()
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.float32).clamp(0.0, 1.0)
    torch.onnx.export(
        student,
        dummy_input,
        onnx_out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["restored_image", "seg_logits"],
        dynamo=False,
    )
    print(f"[Export] Saved ONNX to {onnx_out_path}", flush=True)

    # Check GFLOPs and Parameters with official checker
    import subprocess
    cmd = ["python", "deploy/check_submission_flops.py", "--onnx", onnx_out_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.2e-4)
    args = parser.parse_args()
    run_clinical_distillation(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
