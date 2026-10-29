"""
SAT-EURS Track 1 — V29 SegFormer-B5 Transfer & Fine-Tuning Engine
Experiment: EXP-02 (Coralscapes-Pretrained SegFormer-B5 Fine-Tuning)

Key Architectural & Training Properties:
1. Backbone: MiT-B5 pretrained on CoralscapesV2 (EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024)
2. Decoupled 5-Class Head for EURS benthic reef classification:
   Class 0: Background
   Class 1: Coral
   Class 2: Fish
   Class 3: Elasmobranchs (Shark/Ray)
   Class 4: MacroFauna (Invertebrates)
3. Input: Physical-Inversion Restored RGB with ImageNet normalization
4. Loss: Class-Weighted Cross-Entropy ([0.8, 1.0, 1.2, 2.0, 2.0]) + PresentClassLovaszSoftmaxLoss
5. Rare-Class Centered Cropping: Prioritizes crops containing rare Class 3 and Class 4
6. Strict held-out evaluation measuring mIoU, Macro F1, and Q_seg = mIoU * Macro_F1
"""

import os
import os.path as osp
import sys
import time
import json
import random
import argparse
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from tools.train_track1_screening import apply_rsuigm_batch_gpu
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from metrics.segmentation_metrics import SegmentationEvaluator


class EURSSegmentationDataset(Dataset):
    """
    Segmentation dataset with RSUIGM restoration on GPU,
    rare-class centered cropping, and deterministic evaluation indexing.
    """
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        crop_size: int = 512,
        is_train: bool = True,
        base_seed: int = 42
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train
        self.base_seed = base_seed
        self.epoch = 0

        self.clean_dir = osp.join(data_root, "clean")
        self.masks_dir = osp.join(data_root, "masks")
        self.deg_dir = osp.join(data_root, "degraded")

        # Preload masks into RAM
        self.masks: Dict[str, np.ndarray] = {}
        self.rare_coords: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
        for sc in self.scenes:
            mp = osp.join(self.masks_dir, f"{sc}.png")
            if not osp.exists(mp):
                raise FileNotFoundError(f"Missing mask: {mp}")
            m_arr = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            self.masks[sc] = m_arr

            # Index rare class coordinates (classes 3 and 4)
            self.rare_coords[sc] = {3: [], 4: []}
            for c in [3, 4]:
                ys, xs = np.where(m_arr == c)
                if len(ys) > 0:
                    # Subsample up to 200 coordinates
                    stride = max(1, len(ys) // 200)
                    self.rare_coords[sc][c] = list(zip(ys[::stride], xs[::stride]))

        # Build items list
        self.items: List[Tuple[str, str, int, int]] = []
        for sc in self.scenes:
            sc_deg = osp.join(self.deg_dir, sc)
            if not osp.exists(sc_deg):
                continue
            deg_files = sorted([f for f in os.listdir(sc_deg) if f.endswith(".png")])
            for df in deg_files:
                parts = osp.splitext(df)[0].split("_")
                wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
                depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10
                self.items.append((sc, osp.join(sc_deg, df), wt, depth))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sc, deg_p, wt, depth = self.items[idx]
        mask = self.masks[sc]
        deg_bgr = cv2.imread(deg_p, cv2.IMREAD_COLOR)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = mask.shape
        cs = self.crop_size

        if self.is_train:
            eff_seed = self.base_seed + self.epoch * 100003 + idx * 1009
            rng = random.Random(eff_seed)

            # Check if scene contains rare classes (3 or 4)
            chosen_rare = None
            for c in [4, 3]:  # prioritize MacroFauna (4) then Shark/Ray (3)
                if self.rare_coords[sc][c] and rng.random() < 0.65:
                    chosen_rare = c
                    break

            if chosen_rare is not None and self.rare_coords[sc][chosen_rare]:
                cy, cx = rng.choice(self.rare_coords[sc][chosen_rare])
                top = max(0, min(h - cs, cy - cs // 2 + rng.randint(-30, 30)))
                left = max(0, min(w - cs, cx - cs // 2 + rng.randint(-30, 30)))
            else:
                top = rng.randint(0, h - cs)
                left = rng.randint(0, w - cs)

            hflip = rng.random() > 0.5
            d_crop = deg_rgb[top:top+cs, left:left+cs]
            m_crop = mask[top:top+cs, left:left+cs]

            if hflip:
                d_crop = np.fliplr(d_crop).copy()
                m_crop = np.fliplr(m_crop).copy()
        else:
            if (h, w) != (cs, cs):
                top = (h - cs) // 2
                left = (w - cs) // 2
                d_crop = deg_rgb[top:top+cs, left:left+cs]
                m_crop = mask[top:top+cs, left:left+cs]
            else:
                d_crop = deg_rgb
                m_crop = mask

        return (
            torch.from_numpy(d_crop.transpose(2, 0, 1)).float(),
            torch.from_numpy(m_crop.astype(np.int64)).long(),
            wt,
            depth
        )


def evaluate_segformer_model(
    model: nn.Module,
    val_loader: DataLoader,
    table: Dict,
    device: torch.device,
    im_mean: torch.Tensor,
    im_std: torch.Tensor,
    max_eval: int = 160
) -> Dict[str, Any]:
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=5)

    count = 0
    with torch.no_grad():
        for i_deg, y_gt, wts, depths in val_loader:
            i_deg = i_deg.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            # 1. RSUIGM restoration
            rest_rgb = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)

            # 2. ImageNet normalization
            x_norm = (rest_rgb - im_mean) / im_std

            # 3. Model forward pass
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=x_norm)
                logits = outputs.logits
                logits = F.interpolate(logits, size=y_gt.shape[-2:], mode="bilinear", align_corners=False)

            preds = torch.argmax(logits, dim=1)
            evaluator.update(preds, y_gt)
            count += preds.shape[0]
            if count >= max_eval:
                break

    metrics = evaluator.compute()
    miou = metrics.get("mIoU", 0.0)
    f1 = metrics.get("macro_F1", 0.0)
    q_seg = miou * f1

    return {
        "mIoU": round(miou, 4),
        "macro_F1": round(f1, 4),
        "Q_seg": round(q_seg, 4),
        "per_class_IoU": [round(x, 4) for x in metrics.get("per_class_IoU", [])],
        "per_class_F1": [round(x, 4) for x in metrics.get("per_class_F1", [])],
        "eval_count": count
    }


def train_segformer_fold(
    fold: int,
    epochs: int = 12,
    batch_size: int = 8,
    lr: float = 6e-5,
    save_dir: str = "checkpoints/v29"
) -> Dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TRAINING V29 CORALSCAPES SEGFORMER-B5 (FOLD {fold})")
    print(f"  Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 80)

    # 1. Manifest
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    f_data = manifest["folds"][f"fold_{fold}"]
    train_scenes = sorted(f_data["train_scenes"])
    val_scenes = sorted(f_data["val_scenes"])

    print(f"  Train Scenes ({len(train_scenes)}): {train_scenes}")
    print(f"  Val Scenes   ({len(val_scenes)}): {val_scenes}")

    # 2. Table
    tbl_path = f"checkpoints/track1_screening/tbl19_16sc_fold{fold}.pt"
    table = torch.load(tbl_path, map_location=device, weights_only=False)

    # 3. Datasets
    train_ds = EURSSegmentationDataset(scenes=train_scenes, crop_size=512, is_train=True, base_seed=42 + fold)
    val_ds = EURSSegmentationDataset(scenes=val_scenes, crop_size=512, is_train=False, base_seed=42 + fold)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Model: Load Coralscapes Pretrained SegFormer-B5
    model_id = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024"
    print(f"  Loading SegFormer-B5 from local cache: {model_id}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=5,
        ignore_mismatched_sizes=True
    ).to(device)

    # Normalization constants
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 5. Losses: Class-Weighted CE + Lovasz
    ce_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    loss_ce = nn.CrossEntropyLoss(weight=ce_weights)
    loss_lovasz = PresentClassLovaszSoftmaxLoss().to(device)

    # 6. Optimizer: AdamW with warmup and Cosine Annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(save_dir, exist_ok=True)
    out_ckpt = osp.join(save_dir, f"segformer_b5_fold_{fold}_best.pt")

    # Evaluate zero-shot head at Epoch 0
    print("\n--- Evaluating Initial Head (Epoch 0) ---")
    base_metrics = evaluate_segformer_model(model, val_loader, table, device, im_mean, im_std, max_eval=160)
    print(f"  Epoch  0 [Init Head] | mIoU: {base_metrics['mIoU']} | F1: {base_metrics['macro_F1']} | Q_seg: {base_metrics['Q_seg']} | IoU per class: {base_metrics['per_class_IoU']}")

    best_q_seg = -1.0
    best_metrics = {}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        t_ep = time.time()

        for step, (i_deg, y_gt, wts, depths) in enumerate(train_loader):
            i_deg = i_deg.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            with torch.no_grad():
                rest_rgb = apply_rsuigm_batch_gpu(i_deg, wts, depths, table)

            x_norm = (rest_rgb - im_mean) / im_std

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(pixel_values=x_norm)
                logits = outputs.logits
                logits = F.interpolate(logits, size=y_gt.shape[-2:], mode="bilinear", align_corners=False)

                l_c = loss_ce(logits, y_gt)
                l_v = loss_lovasz(logits, y_gt)
                loss = l_c + 0.50 * l_v

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        val_metrics = evaluate_segformer_model(model, val_loader, table, device, im_mean, im_std, max_eval=160)
        q_seg = val_metrics["Q_seg"]

        print(f"  Epoch {epoch:2d}/{epochs} ({ep_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val mIoU: {val_metrics['mIoU']} | F1: {val_metrics['macro_F1']} | Q_seg: {q_seg:.4f} | "
              f"IoU: {val_metrics['per_class_IoU']}", flush=True)

        if q_seg > best_q_seg:
            best_q_seg = q_seg
            best_metrics = val_metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "fold": fold,
                "train_scenes": train_scenes,
                "val_scenes": val_scenes,
                "hyperparams": {"lr": lr, "batch_size": batch_size}
            }, out_ckpt)
            print(f"  -> Saved new best SegFormer-B5 checkpoint (Q_seg: {best_q_seg:.4f}) to {out_ckpt}", flush=True)

    print("\n" + "=" * 80)
    print(f"Fold {fold} SegFormer-B5 Training Complete in {time.time() - t0:.1f}s!")
    print(f"  Best Held-Out Q_seg: {best_metrics.get('Q_seg', 0.0):.4f}")
    print(f"  mIoU: {best_metrics.get('mIoU', 0.0):.4f} | Macro F1: {best_metrics.get('macro_F1', 0.0):.4f}")
    print(f"  Per-class IoU: {best_metrics.get('per_class_IoU', [])}")
    print("=" * 80)

    return {
        "fold": fold,
        "best_metrics": best_metrics,
        "checkpoint": out_ckpt
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    train_segformer_fold(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
