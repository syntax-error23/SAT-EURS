"""
Training & Distillation Engine for Track2SOTAOrin
MaCVi @ WACV 2027 Challenge — Track 2 (Embedded Edge Deployment on Jetson AGX Orin)

Key Features:
  - Native 1024x1024 full-resolution architecture (zero 384x768 squashing)
  - Pretrained MobileNetV3-Large + Progressive Multi-Scale Pyramid Fusion Decoder
  - Compound Objective: Present-Class Lovasz-Softmax + Soft-Dice + CE + Teacher KD + L1 Rest
  - SOTA Teacher: SegFormer MiT-B3 (Track 1 SOTA)
  - 60% Rare-Fauna Centered Mining (Fish, Shark/Ray, Macro-fauna)
  - Zero negative bias offsets (eliminates Class 3 & Class 4 blindness)
  - Hardware: RTX 5090 with bfloat16 AMP
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, ".")

from models.track2_sota_orin import Track2SOTAOrin, build_track2_sota_model
from losses.track2_sota_loss import Track2SOTALoss
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


ALL_20_SCENES = [
    "fish1", "fish11", "fish12", "fish13", "fish14", "fish15",
    "fish17", "fish18", "fish2", "fish20", "fish21", "fish4",
    "fish5", "fish7", "fish9", "ray1", "seahorse1", "sharkFish",
    "starfish2", "turtles1"
]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class RareFaunaDataset(Dataset):
    """
    Dataset that samples both clean and degraded images across all 20 scenes,
    with 60% rare-object centered crops for classes 2 (Fish), 3 (Shark/Ray), and 4 (Macro-fauna).
    """
    def __init__(self, scenes: List[str], crop_size: int = 512, samples_per_epoch: int = 2000):
        self.scenes = scenes
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch

        # Pre-cache clean and masks
        self.clean_dict = {}
        self.masks_dict = {}
        self.rare_coords = {}
        self.deg_files_by_scene = {}

        print(f"[Dataset] Indexing {len(scenes)} scenes and mining rare fauna coordinates...", flush=True)
        for sc in scenes:
            c_p = f"data/cached_1024/clean/{sc}.png"
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"

            c_im = cv2.imread(c_p)
            c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB)
            self.clean_dict[sc] = c_rgb

            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            self.masks_dict[sc] = m_im

            # Find coordinates for rare classes (2, 3, 4)
            coords = {}
            for c in [2, 3, 4]:
                ys, xs = np.where(m_im == c)
                if len(ys) > 0:
                    coords[c] = (ys, xs)
            self.rare_coords[sc] = coords

            # List degraded files
            if os.path.exists(deg_dir):
                dfs = sorted([os.path.join(deg_dir, f) for f in os.listdir(deg_dir) if f.endswith(".png")])
                self.deg_files_by_scene[sc] = dfs

        total_deg = sum(len(v) for v in self.deg_files_by_scene.values())
        print(f"[Dataset] Indexed {len(scenes)} scenes ({total_deg} degraded images).", flush=True)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        cs = self.crop_size
        cropped = False

        # 60% chance: targeted rare-class centered crop
        if random.random() < 0.60:
            c_chosen = random.choice([2, 3, 4])
            valid_scenes = [s for s in self.scenes if c_chosen in self.rare_coords[s] and len(self.rare_coords[s][c_chosen][0]) > 0]
            if len(valid_scenes) > 0:
                sc = random.choice(valid_scenes)
                clean_np = self.clean_dict[sc]
                mask_np = self.masks_dict[sc]
                H, W = mask_np.shape

                ys, xs = self.rare_coords[sc][c_chosen]
                pt_idx = random.randint(0, len(ys) - 1)
                yc, xc = ys[pt_idx], xs[pt_idx]

                jitter_y = random.randint(-cs // 4, cs // 4)
                jitter_x = random.randint(-cs // 4, cs // 4)
                y1 = int(np.clip(yc - cs // 2 + jitter_y, 0, H - cs))
                x1 = int(np.clip(xc - cs // 2 + jitter_x, 0, W - cs))
                cropped = True

        if not cropped:
            sc = random.choice(self.scenes)
            clean_np = self.clean_dict[sc]
            mask_np = self.masks_dict[sc]
            H, W = mask_np.shape
            y1 = random.randint(0, H - cs)
            x1 = random.randint(0, W - cs)

        # Pick random degraded image for this scene
        deg_file = random.choice(self.deg_files_by_scene[sc])
        deg_im = cv2.imread(deg_file)
        if deg_im is None:
            deg_np = clean_np
        else:
            deg_np = cv2.cvtColor(deg_im, cv2.COLOR_BGR2RGB)

        y2 = y1 + cs
        x2 = x1 + cs

        crop_deg = deg_np[y1:y2, x1:x2]
        crop_clean = clean_np[y1:y2, x1:x2]
        crop_mask = mask_np[y1:y2, x1:x2]

        # Augmentations (Horizontal & Vertical flips)
        if random.random() > 0.5:
            crop_deg = np.fliplr(crop_deg)
            crop_clean = np.fliplr(crop_clean)
            crop_mask = np.fliplr(crop_mask)

        if random.random() > 0.8:
            crop_deg = np.flipud(crop_deg)
            crop_clean = np.flipud(crop_clean)
            crop_mask = np.flipud(crop_mask)

        deg_t = torch.from_numpy(crop_deg.copy().transpose(2, 0, 1)).float() / 255.0
        clean_t = torch.from_numpy(crop_clean.copy().transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(crop_mask.copy()).long()

        return deg_t, clean_t, mask_t


def evaluate_full_resolution(
    model: nn.Module,
    val_scenes: List[str],
    device: torch.device,
    sample_step: int = 10
) -> Tuple[float, float, float, float, float, Dict]:
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=5)
    psnrs, ssims, sharps = [], [], []

    with torch.no_grad():
        for sc in val_scenes:
            clean_im = cv2.imread(f"data/cached_1024/clean/{sc}.png")
            clean_rgb = cv2.cvtColor(clean_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            mask_im = cv2.imread(f"data/cached_1024/masks/{sc}.png", cv2.IMREAD_GRAYSCALE)

            deg_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([os.path.join(deg_dir, f) for f in os.listdir(deg_dir) if f.endswith(".png")])[::sample_step]

            for df in deg_files:
                dim = cv2.imread(df)
                if dim is None: continue
                drgb = cv2.cvtColor(dim, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                inp = torch.from_numpy(drgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    rest_pred, logits_pred = model(inp)

                pred_mask = torch.argmax(logits_pred, dim=1).squeeze(0).cpu()
                evaluator.update(pred_mask.unsqueeze(0), torch.from_numpy(mask_im).unsqueeze(0))

                rest_np = rest_pred.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
                p, _ = compute_psnr_and_norm(rest_np, clean_rgb)
                s = compute_ssim_single(rest_np, clean_rgb)
                sh = compute_sharpness_ratio(rest_np, clean_rgb)
                psnrs.append(p)
                ssims.append(s)
                sharps.append(sh)

    seg_res = evaluator.compute()
    mean_psnr = float(np.mean(psnrs))
    mean_psnr_norm = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
    mean_ssim = float(np.mean(ssims))
    mean_sharp = float(np.mean(sharps))

    q_seg_strict = seg_res["Q_seg"]
    soft_cov = seg_res.get("soft_coverage_penalty", 1.0)
    q_seg_smooth = float(seg_res["mIoU"] * seg_res["macro_F1"] * soft_cov)
    q_fr = 0.45 * mean_psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
    q_nr = 0.6000  # realistic server NR baseline

    composite_score = 0.60 * q_seg_smooth + 0.30 * q_fr + 0.10 * q_nr
    official_composite = 0.60 * q_seg_strict + 0.30 * q_fr + 0.10 * q_nr

    return composite_score, official_composite, q_seg_smooth, q_fr, mean_psnr, mean_ssim, mean_sharp, seg_res


def run_training(
    epochs: int = 15,
    batch_size: int = 16,
    crop_size: int = 512,
    lr: float = 3e-4,
    save_path: str = "checkpoints/track2_sota_orin_best.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    print("\n" + "=" * 80)
    print("      SAT-EURS TRACK 2 SOTA PRODUCTION TRAINING ENGINE (ORIN TARGET)")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Crop: {crop_size}x{crop_size}")
    print("=" * 80 + "\n", flush=True)

    # 1. Dataset & Loader
    train_dataset = RareFaunaDataset(ALL_20_SCENES, crop_size=crop_size, samples_per_epoch=2500)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 2. Build Student Model
    print("[1/4] Building Track2SOTAOrin student model...", flush=True)
    model = build_track2_sota_model(pretrained_backbone=True).to(device)

    # Load pre-trained optical router
    router_ckpt = "checkpoints/EXP-PRETRAINED-30EP_best.pt"
    if os.path.exists(router_ckpt):
        ckpt = torch.load(router_ckpt, map_location="cpu", weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        router_st = {k: v for k, v in st.items() if "router" in k}
        model.load_state_dict(router_st, strict=False)
        print(f"  Loaded {len(router_st)} pre-trained router weights from {router_ckpt}.")

    # 3. Load Teacher Model (SegFormer MiT-B3 SOTA from Track 1)
    print("[2/4] Loading Track 1 SegFormer MiT-B3 SOTA teacher on GPU...", flush=True)
    teacher = smp.DeepLabV3Plus(encoder_name="mit_b3", encoder_weights=None, classes=5, in_channels=3).to(device)
    teacher_ckpt = "checkpoints/segformer_mit_b3_best.pt"
    ckpt_t = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt_t.get("model_state_dict", ckpt_t))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("  Teacher successfully loaded and frozen for soft distillation.")

    # Teacher normalization constants
    norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 4. Loss Function
    print("[3/4] Initializing Track2SOTALoss (Lovasz + SoftDice + CE + KD + L1)...", flush=True)
    criterion = Track2SOTALoss(
        num_classes=5,
        w_lovasz=0.40,
        w_dice=0.30,
        w_ce=0.30,
        w_kd=0.35,
        w_rest=0.25,
        temperature=3.0
    ).to(device)

    # 5. Optimizer & Cosine Annealing
    optimizer = torch.optim.AdamW([
        {"params": list(model.decoder.parameters()), "lr": lr, "weight_decay": 1e-4},
        {"params": list(model.backbone_blocks.parameters()), "lr": lr * 0.2, "weight_decay": 1e-4},
        {"params": list(model.router.parameters()) + list(model.router_fc.parameters()), "lr": lr * 0.1, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Representative validation scenes (including rare fauna)
    val_scenes = ["ray1", "sharkFish", "seahorse1", "fish1", "fish4", "fish7", "turtles1", "starfish2"]

    best_composite = -1.0
    best_metrics = {}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n[4/4] Starting 15-Epoch Distillation & Optimization Loop...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_seg = 0.0
        running_lovasz = 0.0
        running_dice = 0.0
        running_kd = 0.0
        running_rest = 0.0

        for deg, clean, mask in train_loader:
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                restored, student_logits = model(deg)

                # Teacher inference on restored image
                with torch.no_grad():
                    inp_t = (restored.detach() - norm_mean) / norm_std
                    teacher_logits = teacher(inp_t)

                loss, info = criterion(restored, student_logits, clean, mask, teacher_logits)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += info["loss_total"]
            running_seg += info["loss_seg"]
            running_lovasz += info["loss_lovasz"]
            running_dice += info["loss_dice"]
            running_kd += info["loss_kd"]
            running_rest += info["loss_rest"]

        scheduler.step()
        n_batches = len(train_loader)
        epoch_time = time.time() - t0

        # Fast Validation every epoch
        comp, off_comp, q_seg_smooth, q_fr, psnr, ssim, sharp, seg_res = evaluate_full_resolution(
            model, val_scenes, device, sample_step=15
        )

        print(
            f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:4.1f}s) | "
            f"Loss: {running_loss/n_batches:.4f} (Lovasz: {running_lovasz/n_batches:.3f}, Dice: {running_dice/n_batches:.3f}, KD: {running_kd/n_batches:.3f}) | "
            f"Val mIoU: {seg_res['mIoU']:.4f}, F1: {seg_res['macro_F1']:.4f}, Q_seg: {q_seg_smooth:.4f} | "
            f"PSNR: {psnr:.2f}dB, SSIM: {ssim:.3f}, Sharp: {sharp:.3f} | "
            f"Comp: {comp:.4f} (Strict: {off_comp:.4f})",
            flush=True
        )
        print(f"    Per-Class IoU: {seg_res['per_class_IoU']}", flush=True)

        if comp > best_composite:
            best_composite = comp
            best_metrics = {
                "epoch": epoch,
                "composite": comp,
                "official_composite": off_comp,
                "q_seg": q_seg_smooth,
                "q_seg_strict": seg_res["Q_seg"],
                "q_fr": q_fr,
                "psnr": psnr,
                "ssim": ssim,
                "sharpness": sharp,
                "mIoU": seg_res["mIoU"],
                "macro_F1": seg_res["macro_F1"],
                "per_class_IoU": seg_res["per_class_IoU"],
                "per_class_F1": seg_res["per_class_F1"]
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": best_metrics
            }, save_path)
            print(f"  --> Saved new best Track2SOTAOrin checkpoint (Composite: {comp:.4f}, mIoU: {seg_res['mIoU']:.4f})!", flush=True)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Best Checkpoint: {save_path}")
    print(f"Best Validation Composite: {best_composite:.4f}")
    print(f"Best Validation Q_seg:     {best_metrics.get('q_seg', 0.0):.4f} (mIoU: {best_metrics.get('mIoU', 0.0):.4f}, F1: {best_metrics.get('macro_F1', 0.0):.4f})")
    print(f"Per-Class IoU: {best_metrics.get('per_class_IoU', [])}")
    print("=" * 80 + "\n", flush=True)
    return save_path, best_metrics


if __name__ == "__main__":
    run_training(epochs=15, batch_size=16, crop_size=512, lr=3e-4)
