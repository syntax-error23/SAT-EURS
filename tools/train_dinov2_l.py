"""
High-Capacity DINOv2-L/14 Semantic Segmentation
MaCVi WACV 2027 EURS Challenge (Track 1 High-Capacity Benchmark)

Key Architectural & Training Principles:
1. Foundation Vision Backbone:
   - Pretrained DINOv2-L/14 encoder (304M parameters).
   - Multi-scale intermediate feature extraction from blocks (5, 11, 17, 23).
   - Lightweight multi-scale projection and fusion decoder (total ~308M parameters).
2. Resolution & Patch Alignment:
   - Input resolution: 518x518 (multiple of 14, yielding 37x37 token grid).
   - Full validation resolution resized to nearest multiple of 14 for inference.
3. Strict 16/4 Scene-Disjoint Split (manifest_full.json):
   - Train scenes (16): fish11, fish13, fish14, fish15, fish17, fish18, fish2,
                        fish20, fish21, fish4, fish5, fish7, fish9, ray1,
                        starfish2, turtles1
   - Val scenes (4):    fish1, fish12, seahorse1, sharkFish
   - Zero scene overlap; zero validation scenes seen during training.
4. Preprocessing & Input Compatibility:
   - Closed-form per-image RSUIGM physical inversion on GPU.
   - ImageNet normalization: (x - mean) / std.
5. Preserved Geometry Augmentation:
   - Rare-class centered crop (p=0.60 for Class 3 and 4)
   - Random horizontal flip (p=0.50)
   - Zero rotation, zero scale jitter.
6. Differential Optimization:
   - Backbone learning rate: 3e-5 (fine-tuning foundation representation)
   - Decoder learning rate: 1e-4
   - Batch size 4 with gradient accumulation 2 (effective batch size 8).
7. Storage & Reproducibility:
   - Directory: checkpoints/track1_high_capacity/exp_dinov2_l/
"""

import os
import os.path as osp
import sys
import time
import json
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu
from metrics.segmentation_metrics import SegmentationEvaluator

# Canonical Scene-Disjoint Split Definition (splits/manifest_full.json)
TRAIN_SCENES = [
    "fish11", "fish13", "fish14", "fish15", "fish17", "fish18", "fish2",
    "fish20", "fish21", "fish4", "fish5", "fish7", "fish9",
    "ray1", "starfish2", "turtles1"
]

VAL_SCENES = [
    "fish1", "fish12", "seahorse1", "sharkFish"
]


class DINOv2L14Segmenter(nn.Module):
    def __init__(self, num_classes: int = 5, intermediate_blocks: Tuple[int, ...] = (5, 11, 17, 23)):
        super().__init__()
        self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
        self.num_classes = num_classes
        self.intermediate_blocks = intermediate_blocks
        embed_dim = self.encoder.embed_dim  # 1024

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in intermediate_blocks
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(256 * len(intermediate_blocks), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Input must be multiple of 14
        features = self.encoder.get_intermediate_layers(x, n=self.intermediate_blocks, return_class_token=False)
        h_feat, w_feat = H // 14, W // 14

        projected = []
        for feat, p in zip(features, self.proj):
            f_map = feat.permute(0, 2, 1).reshape(B, -1, h_feat, w_feat)
            projected.append(p(f_map))

        cat_feat = torch.cat(projected, dim=1)
        logits_low = self.fusion(cat_feat)
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


class HighCapacityDINOv2Dataset(Dataset):
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/train",
        crop_size: int = 518,  # Multiple of 14 (37 * 14 = 518)
        is_train: bool = True,
        samples_per_epoch: int = 3200,
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch

        self.clean_dir = osp.join(data_root, "clean")
        self.masks_dir = osp.join(data_root, "masks")
        self.degraded_dir = osp.join(data_root, "degraded")

        self.items: List[Tuple[str, str, int, int]] = []
        for sc in self.scenes:
            sc_deg = osp.join(self.degraded_dir, sc)
            if not osp.exists(sc_deg):
                continue
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])
            for df in deg_files:
                stem = osp.splitext(df)[0]
                parts = stem.split("_")
                wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
                depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
                self.items.append((sc, osp.join(sc_deg, df), wt, depth))

        print(f"[{'TRAIN' if is_train else 'VAL'}] Loaded {len(self.items)} degraded images across {len(self.scenes)} scenes: {self.scenes}")

        self.masks: Dict[str, np.ndarray] = {}
        self.rare_in_mask: Dict[str, List[int]] = {}
        for sc in self.scenes:
            mp = osp.join(self.masks_dir, f"{sc}.png")
            with Image.open(mp) as mim:
                m_arr = np.array(mim, dtype=np.int64)
                self.masks[sc] = m_arr
                self.rare_in_mask[sc] = [c for c in [3, 4] if (m_arr == c).any()]

    def __len__(self) -> int:
        return self.samples_per_epoch if self.is_train else len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.is_train:
            item = random.choice(self.items)
        else:
            item = self.items[idx]

        sc, deg_path, wt, depth = item
        mask = self.masks[sc]
        h, w = mask.shape
        cs = self.crop_size

        with Image.open(deg_path) as dim:
            if dim.size != (w, h):
                dim = dim.resize((w, h), Image.BILINEAR)
            raw_rgb = np.array(dim.convert("RGB"), dtype=np.float32) / 255.0

        if self.is_train:
            # Object-centered cropping for rare marine classes
            if h > cs and w > cs:
                top, left = 0, 0
                found_crop = False
                rares = self.rare_in_mask.get(sc, [])
                if len(rares) > 0 and random.random() < 0.60:
                    target_cls = random.choice(rares)
                    ys, xs = np.where(mask == target_cls)
                    if len(ys) > 0:
                        c_idx = random.randint(0, len(ys) - 1)
                        top = max(0, min(h - cs, ys[c_idx] - cs // 2))
                        left = max(0, min(w - cs, xs[c_idx] - cs // 2))
                        found_crop = True

                if not found_crop:
                    top = random.randint(0, h - cs)
                    left = random.randint(0, w - cs)

                raw_crop = raw_rgb[top:top+cs, left:left+cs]
                mask_crop = mask[top:top+cs, left:left+cs]
            else:
                raw_crop = cv2.resize(raw_rgb, (cs, cs), interpolation=cv2.INTER_LINEAR)
                mask_crop = cv2.resize(mask.astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(np.int64)

            # Random Horizontal Flip (preserving realistic object geometry)
            if random.random() < 0.5:
                raw_crop = np.fliplr(raw_crop)
                mask_crop = np.fliplr(mask_crop)
        else:
            # Full validation resolution (scaled to 1022x1022, nearest multiple of 14)
            val_dim = 1022  # 73 * 14 = 1022
            raw_crop = cv2.resize(raw_rgb, (val_dim, val_dim), interpolation=cv2.INTER_LINEAR)
            mask_crop = cv2.resize(mask.astype(np.uint8), (val_dim, val_dim), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        return (
            torch.from_numpy(raw_crop.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(mask_crop.copy()).long(),
            wt,
            depth
        )


def train_dinov2_l(
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    encoder_lr: float = 3.0e-5,
    decoder_lr: float = 1.0e-4,
    crop_size: int = 518,
    seed: int = 42,
    samples_per_epoch: int = 3200,
    save_dir: str = "checkpoints/track1_high_capacity/exp_dinov2_l",
    data_root: str = "data/cached_1024" if osp.exists("data/cached_1024") else "data/train"
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "experiment_name": "exp_dinov2_l",
        "description": "High-Capacity DINOv2-L/14 Semantic Segmentation (308.1M parameters)",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "encoder_lr": encoder_lr,
        "decoder_lr": decoder_lr,
        "crop_size": crop_size,
        "encoder": "dinov2_vitl14",
        "num_classes": 5,
        "class_weights": [0.80, 1.00, 1.20, 2.00, 2.00],
        "train_scenes": TRAIN_SCENES,
        "val_scenes": VAL_SCENES,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

    with open(osp.join(save_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    print("=" * 76)
    print("   STAGE 2 PRIMARY CANDIDATE: DINOv2-L/14 (308.1M PARAMS)")
    print(f"   Directory:   {save_dir}")
    print(f"   Hardware:    {device} ({torch.cuda.get_device_name(0)})")
    print(f"   Train Scenes ({len(TRAIN_SCENES)}): {TRAIN_SCENES}")
    print(f"   Val Scenes   ({len(VAL_SCENES)}): {VAL_SCENES}")
    print(f"   Epochs: {epochs} | Batch: {batch_size}x{grad_accum_steps} | LR: {encoder_lr}/{decoder_lr} | Crop: {crop_size}")
    print("=" * 76)

    # 1. Closed-form RSUIGM table on GPU
    rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

    # 2. Datasets & Loaders
    train_dataset = HighCapacityDINOv2Dataset(
        scenes=TRAIN_SCENES,
        data_root=data_root,
        crop_size=crop_size,
        is_train=True,
        samples_per_epoch=samples_per_epoch
    )
    val_dataset = HighCapacityDINOv2Dataset(
        scenes=VAL_SCENES,
        data_root=data_root,
        crop_size=crop_size,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model: DINOv2-L/14 Segmenter
    model = DINOv2L14Segmenter(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] DINOv2-L/14 Segmenter initialized with {total_params:,} parameters.")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    class_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Differential learning rates: smaller for encoder, larger for decoder
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.proj.parameters(), "lr": decoder_lr},
        {"params": model.fusion.parameters(), "lr": decoder_lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_q_seg = -1.0
    history = []

    def _apply_rsuigm_batch(x_raw: torch.Tensor, wts: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_raw.shape
        restored = []
        for i in range(B):
            xi = x_raw[i:i+1]
            wt_i = int(wts[i].item())
            d_i = int(depths[i].item())
            mat = rsuigm_table.get((wt_i, d_i), rsuigm_table[(1, 10)])
            feats = build_rsuigm_poly_features_gpu(xi)
            phys = torch.clamp(torch.matmul(feats, mat).reshape(1, H, W, 3).permute(0, 3, 1, 2), 0.0, 1.0)
            restored.append(phys)
        return torch.cat(restored, dim=0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()
        optimizer.zero_grad()

        for step, (x_raw, y_mask, wts, depths) in enumerate(train_loader):
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            with torch.no_grad():
                x_rest = _apply_rsuigm_batch(x_raw, wts, depths)
                x_norm = (x_rest - mean_im) / std_im

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)
                loss = loss_fn(logits, y_mask) / grad_accum_steps

            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # Validation on held-out scenes
        model.eval()
        evaluator = SegmentationEvaluator(num_classes=5)

        with torch.no_grad():
            for x_raw, y_mask, wts, depths in val_loader:
                x_raw = x_raw.to(device, non_blocking=True)
                y_mask = y_mask.to(device, non_blocking=True)

                x_rest = _apply_rsuigm_batch(x_raw, wts, depths)
                x_norm = (x_rest - mean_im) / std_im

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x_norm)

                evaluator.update(logits, y_mask)

        metrics = evaluator.compute()
        elapsed = time.time() - t_epoch
        q_seg = metrics["Q_seg"]

        print(
            f"Epoch {epoch:2d}/{epochs:2d} ({elapsed:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val mIoU: {metrics['mIoU']:.4f} | "
            f"Macro F1: {metrics['macro_F1']:.4f} | "
            f"Cov: {metrics['coverage_penalty']:.1f} | "
            f"Q_seg: {q_seg:.4f} | "
            f"Cls IoU: {metrics['per_class_IoU']}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": metrics,
            "elapsed_sec": elapsed
        }
        history.append(record)

        if q_seg > best_q_seg:
            best_q_seg = q_seg
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": metrics,
                "config": config,
                "train_scenes": TRAIN_SCENES,
                "val_scenes": VAL_SCENES
            }, osp.join(save_dir, "best_model.pt"))

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_metrics": metrics,
            "config": config,
            "train_scenes": TRAIN_SCENES,
            "val_scenes": VAL_SCENES
        }, osp.join(save_dir, "final_model.pt"))

    with open(osp.join(save_dir, "training_history.json"), "w") as fp:
        json.dump(history, fp, indent=2)

    print(f"\n[DONE] DINOv2-L/14 training complete! Best Q_seg: {best_q_seg:.4f}")
    return history


if __name__ == "__main__":
    train_dinov2_l()
