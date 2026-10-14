"""
Controlled Single-Stream SegFormer MiT-B3 with Mild Rare-Class Loss Weighting
MaCVi WACV 2027 EURS Challenge (Track 1)

Controlled Experiment Design:
1. Strict 16/4 Scene-Disjoint Split (manifest_full.json):
   - Train scenes (16): fish11, fish13, fish14, fish15, fish17, fish18, fish2,
                        fish20, fish21, fish4, fish5, fish7, fish9, ray1,
                        starfish2, turtles1
   - Val scenes (4):    fish1, fish12, seahorse1, sharkFish
   - Zero scene overlap; zero validation scenes seen during training.
2. Architecture & Backbone Parity:
   - Identical to baseline: SegFormer MiT-B3 encoder with DeepLabV3Plus head.
   - Evaluates closed-form per-image RSUIGM physical inversion on GPU.
   - Standard ImageNet normalization: (x - mean) / std.
3. Augmentation Parity:
   - EXACT BASELINE AUGMENTATION UNCHANGED:
     * Rare-class centering (p=0.60 for Class 3 and 4)
     * Random horizontal flip (p=0.50)
     * ZERO rotation, ZERO scale jitter, ZERO photometric distortion.
4. Single Controlled Variable:
   - Mild Rare-Class Weighted CrossEntropyLoss:
     Baseline weights: [0.80, 1.00, 1.20, 2.00, 2.00]
     Mild loss weights: [0.75, 1.00, 1.20, 2.00, 3.50]
     * Class 3 (Shark/Ray) kept at 2.00 to preserve its proven 0.5831 IoU.
     * Class 4 (Other) mildly boosted from 2.00 to 3.50 to test whether extra gradient
       recovers seahorse boundary features without triggering false-positive collapse.
5. Storage:
   - All artifacts saved to checkpoints/exp_segformer_mild_loss/
   - Baseline checkpoints/exp_baseline_segformer/ remains strictly immutable.
"""

import os
import os.path as osp
import sys
import time
import json
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

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


class LeakFreeSegFormerDataset(Dataset):
    """
    Exact baseline dataset partitioning and augmentation:
    Applies per-image RSUIGM physical inversion + ImageNet normalization.
    Augmentation: Rare-class centered crop + horizontal flip only.
    """
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/train",
        crop_size: int = 512,
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

        # Preload clean masks into RAM
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
            # Baseline Crop with Object-Centering for Rare Marine Classes (Classes 3, 4)
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
                raw_crop = np.array(Image.fromarray((raw_rgb * 255).astype(np.uint8)).resize((cs, cs), Image.BILINEAR)).astype(np.float32) / 255.0
                mask_crop = np.array(Image.fromarray(mask.astype(np.uint8)).resize((cs, cs), Image.NEAREST)).astype(np.int64)

            # Random Horizontal Flip (Exact Baseline Augmentation)
            if random.random() < 0.5:
                raw_crop = np.fliplr(raw_crop)
                mask_crop = np.fliplr(mask_crop)
        else:
            # Full validation resolution (scaled to 1024x1024 max side)
            raw_crop = raw_rgb
            mask_crop = mask

        return (
            torch.from_numpy(raw_crop.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(mask_crop.copy()).long(),
            wt,
            depth
        )


def train_mild_loss_experiment(
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1.0e-4,
    crop_size: int = 512,
    seed: int = 42,
    samples_per_epoch: int = 3200,
    save_dir: str = "checkpoints/exp_segformer_mild_loss",
    data_root: str = "data/cached_1024" if osp.exists("data/cached_1024") else "data/train"
):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Mild rare-class weighting vector:
    # Baseline was: [0.80, 1.00, 1.20, 2.00, 2.00]
    # Mild variant: [0.75, 1.00, 1.20, 2.00, 3.50]
    class_weights_list = [0.75, 1.00, 1.20, 2.00, 3.50]
    class_weights = torch.tensor(class_weights_list, device=device)

    config = {
        "experiment_name": "exp_segformer_mild_loss",
        "description": "Controlled Single-Stream SegFormer with Mild Rare-Class Weighted CrossEntropyLoss and Baseline Augmentation Unchanged",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "crop_size": crop_size,
        "optimizer": "AdamW (lr=1e-4, weight_decay=1e-4)",
        "lr_scheduler": "CosineAnnealingLR (T_max=10, eta_min=1e-6)",
        "loss": {
            "type": "CrossEntropyLoss",
            "class_weights": class_weights_list,
            "baseline_weights": [0.80, 1.00, 1.20, 2.00, 2.00],
            "controlled_change": "Class 4 weight boosted from 2.00 to 3.50; Class 0 downweighted from 0.80 to 0.75; Class 3 (Shark) preserved at 2.00."
        },
        "augmentation": {
            "type": "Baseline Unchanged",
            "horizontal_flip_p": 0.50,
            "rare_class_centering_p": 0.60,
            "rotation": "None",
            "scale_jitter": "None",
            "photometric_jitter": "None"
        },
        "train_scenes": TRAIN_SCENES,
        "val_scenes": VAL_SCENES,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

    with open(osp.join(save_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    print("=" * 76)
    print("   CONTROLLED EXPERIMENT: MILD RARE-CLASS LOSS (AUGMENTATION UNCHANGED)")
    print(f"   Directory:   {save_dir}")
    print(f"   Hardware:    {device} ({torch.cuda.get_device_name(0)})")
    print(f"   Train Scenes ({len(TRAIN_SCENES)}): {TRAIN_SCENES}")
    print(f"   Val Scenes   ({len(VAL_SCENES)}): {VAL_SCENES}")
    print(f"   Class Weights: {class_weights_list}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Seed: {seed}")
    print("=" * 76)

    # 1. Physical Inversion Table on GPU
    rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

    # 2. Datasets & Loaders
    train_dataset = LeakFreeSegFormerDataset(
        scenes=TRAIN_SCENES,
        data_root=data_root,
        crop_size=crop_size,
        is_train=True,
        samples_per_epoch=samples_per_epoch
    )
    val_dataset = LeakFreeSegFormerDataset(
        scenes=VAL_SCENES,
        data_root=data_root,
        crop_size=crop_size,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model: SegFormer MiT-B3 with 3 input channels (restored domain)
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        classes=5,
        in_channels=3
    ).to(device)

    # ImageNet normalization tensors
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_q_seg = -1.0
    history = []

    def _apply_rsuigm_batch(x_raw: torch.Tensor, wts: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """Apply closed-form physical inversion on GPU per item in batch."""
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

        for step, (x_raw, y_mask, wts, depths) in enumerate(train_loader):
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            with torch.no_grad():
                x_rest = _apply_rsuigm_batch(x_raw, wts, depths)
                x_norm = (x_rest - mean_im) / std_im

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)
                loss = loss_fn(logits, y_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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

                B, C, H, W = x_norm.shape
                if H != 1024 or W != 1024:
                    x_in = F.interpolate(x_norm, size=(1024, 1024), mode="bilinear", align_corners=False)
                else:
                    x_in = x_norm

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits_sq = model(x_in)

                if H != 1024 or W != 1024:
                    logits = F.interpolate(logits_sq, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    logits = logits_sq

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

    print(f"\n[DONE] Mild Loss experiment complete! Best Q_seg: {best_q_seg:.4f}")
    return history


if __name__ == "__main__":
    train_mild_loss_experiment()
