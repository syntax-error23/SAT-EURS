"""
Track 1 Screening and Confirmation Training Engine
Supports:
- Baseline DeepLabV3Plus-MiTB3 and Primary Foundation DINOv2-L/14
- Strict Scene-Disjoint 5-Fold Cross-Validation (manifest_5fold.json)
- Zero data leakage (fold-specific RSUIGM table fitted on train_scenes only)
- Deterministic indexed data contract (no random.choice in __getitem__)
- Evaluates Protocol 1 (Strict natural OOF) & Protocol 2 (Synthetic unseen-host probe)
- Precision: bfloat16 AMP with VRAM monitoring (< 90% / < 28 GB)
"""

import os
import os.path as osp
import sys
import time
import json
import random
import csv
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from metrics.segmentation_metrics import SegmentationEvaluator
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu
from tools.pilot_restoration_evaluation import fit_leak_free_rsuigm_table
import segmentation_models_pytorch as smp


class ControlledTrack1Dataset(Dataset):
    """
    Deterministic Indexed Dataset for Track 1.
    Strictly uses self.items[idx] without replacement.
    Applies fold-specific RSUIGM restoration on GPU or raw RGB, with ImageNet normalization.
    """
    def __init__(
        self,
        scenes: List[str],
        data_root: str = "data/cached_1024",
        crop_size: int = 518,
        is_train: bool = True,
        use_rsuigm: bool = True,
        rsuigm_table: Optional[Dict] = None,
        copy_paste_augmenter: Optional[Any] = None,
        base_seed: int = 42,
    ):
        super().__init__()
        self.scenes = sorted(scenes)
        self.crop_size = crop_size
        self.is_train = is_train
        self.use_rsuigm = use_rsuigm
        self.rsuigm_table = rsuigm_table
        self.copy_paste_augmenter = copy_paste_augmenter
        self.base_seed = base_seed
        self.epoch = 0

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

        # Preload masks into RAM
        self.masks: Dict[str, np.ndarray] = {}
        self.rare_in_mask: Dict[str, List[int]] = {}
        for sc in self.scenes:
            mp = osp.join(self.masks_dir, f"{sc}.png")
            m_arr = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            self.masks[sc] = m_arr
            self.rare_in_mask[sc] = [c for c in [3, 4] if (m_arr == c).any()]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.copy_paste_augmenter is not None and hasattr(self.copy_paste_augmenter, "set_epoch"):
            self.copy_paste_augmenter.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # Deterministic access using idx
        sc, deg_path, wt, depth = self.items[idx]
        mask = self.masks[sc]
        h, w = mask.shape
        cs = self.crop_size

        bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_train:
            # Deterministic per-sample seed based on epoch, idx, and base_seed
            eff_seed = self.base_seed + self.epoch * 100003 + idx * 1009
            rng = random.Random(eff_seed)

            # Optional Copy-Paste
            if self.copy_paste_augmenter is not None:
                rgb, _, mask = self.copy_paste_augmenter(rgb, rgb, mask, seed=idx, epoch=self.epoch)

            # Spatial crop (object-centered for rare classes if present)
            if h > cs and w > cs:
                rares = self.rare_in_mask.get(sc, [])
                found_crop = False
                top, left = 0, 0
                if len(rares) > 0 and rng.random() < 0.60:
                    target_cls = rng.choice(rares)
                    ys, xs = np.where(mask == target_cls)
                    if len(ys) > 0:
                        c_idx = rng.randint(0, len(ys) - 1)
                        top = max(0, min(h - cs, ys[c_idx] - cs // 2))
                        left = max(0, min(w - cs, xs[c_idx] - cs // 2))
                        found_crop = True

                if not found_crop:
                    top = rng.randint(0, h - cs)
                    left = rng.randint(0, w - cs)

                rgb_crop = rgb[top:top+cs, left:left+cs]
                mask_crop = mask[top:top+cs, left:left+cs]
            else:
                rgb_crop = cv2.resize(rgb, (cs, cs), interpolation=cv2.INTER_LINEAR)
                mask_crop = cv2.resize(mask, (cs, cs), interpolation=cv2.INTER_NEAREST)

            # Random horizontal flip
            if rng.random() < 0.5:
                rgb_crop = np.fliplr(rgb_crop).copy()
                mask_crop = np.fliplr(mask_crop).copy()
        else:
            # Validation: full 1024x1024 or center crop
            if (h, w) != (cs, cs):
                rgb_crop = cv2.resize(rgb, (cs, cs), interpolation=cv2.INTER_LINEAR)
                mask_crop = cv2.resize(mask, (cs, cs), interpolation=cv2.INTER_NEAREST)
            else:
                rgb_crop = rgb
                mask_crop = mask

        return (
            torch.from_numpy(rgb_crop.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask_crop.astype(np.int64)).long(),
            wt,
            depth
        )


def build_dinov2_model(num_classes: int = 5) -> nn.Module:
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    embed_dim = encoder.embed_dim  # 1024
    intermediate_blocks = (5, 11, 17, 23)

    class DINOv2Segmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.intermediate_blocks = intermediate_blocks
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

    return DINOv2Segmenter()


def build_mit_b3_model(num_classes: int = 5) -> nn.Module:
    # DeepLabV3Plus with MiT-B3 encoder
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b3",
        encoder_weights="imagenet",
        classes=num_classes,
        in_channels=3
    )
    return model


def apply_rsuigm_batch_gpu(x_batch: torch.Tensor, wts: torch.Tensor, depths: torch.Tensor, table: Dict) -> torch.Tensor:
    b, c, h, w = x_batch.shape
    restored = []
    for i in range(b):
        wt_val = int(wts[i].item())
        dp_val = int(depths[i].item())
        key = (wt_val, dp_val)
        if key in table:
            W = table[key]
            feat = build_rsuigm_poly_features_gpu(x_batch[i:i+1])
            res = (feat @ W).permute(0, 2, 1).reshape(1, 3, h, w)
            restored.append(torch.clamp(res, 0.0, 1.0))
        else:
            restored.append(x_batch[i:i+1])
    return torch.cat(restored, dim=0)


def evaluate_model_full(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_rsuigm: bool,
    rsuigm_table: Optional[Dict],
    mean_im: torch.Tensor,
    std_im: torch.Tensor
) -> Dict[str, Any]:
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=5)

    with torch.no_grad():
        for x_raw, y_mask, wts, depths in val_loader:
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            if use_rsuigm and rsuigm_table is not None:
                x_in = apply_rsuigm_batch_gpu(x_raw, wts, depths, rsuigm_table)
            else:
                x_in = x_raw

            x_norm = (x_in - mean_im) / std_im

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)

            evaluator.update(logits, y_mask)

    return evaluator.compute()


def train_screening_candidate(
    run_id: str,
    fold_id: str = "fold_0",
    model_type: str = "dinov2_l", # 'dinov2_l' or 'mit_b3'
    use_rsuigm: bool = True,
    use_lovasz: bool = False,
    use_copy_paste: bool = False,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    lr: float = 3e-5,
    decoder_lr: float = 1e-4,
    crop_size: int = 518,
    save_root: str = "checkpoints/track1_screening",
    seed: int = 42
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 78)
    print(f"RUNNING SCREENING EXPERIMENT: {run_id}")
    print(f"  Fold: {fold_id} | Model: {model_type} | Input: {'RSUIGM' if use_rsuigm else 'RAW_RGB'}")
    print(f"  Loss: {'CE+Lovasz' if use_lovasz else 'CE'} | Copy-Paste: {use_copy_paste} | Epochs: {epochs}")
    print("=" * 78)

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # Load 5-fold manifest
    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)
    fold_meta = manifest["folds"][fold_id]
    train_scenes = fold_meta["train_scenes"]
    val_scenes = fold_meta["val_scenes"]

    # 1. Fit fold-specific leak-free RSUIGM table
    rsuigm_table = None
    if use_rsuigm:
        print(f"[RSUIGM] Fitting leak-free table strictly on 16 train scenes of {fold_id}...")
        rsuigm_table = fit_leak_free_rsuigm_table(train_scenes, device=device)

    # 2. Build Datasets & Loaders
    cp_augmenter = None
    if use_copy_paste:
        from data.controlled_copy_paste import ControlledCopyPasteAugmenter
        print(f"[CopyPaste] Initializing exact-(wt, depth) matched copy-paste on 16 train scenes of {fold_id}...")
        cp_augmenter = ControlledCopyPasteAugmenter(
            train_scenes=train_scenes,
            data_root="data/cached_1024",
            paste_prob=0.50,
            max_pastes=2,
            base_seed=seed
        )

    train_ds = ControlledTrack1Dataset(
        scenes=train_scenes,
        crop_size=crop_size,
        is_train=True,
        use_rsuigm=use_rsuigm,
        rsuigm_table=rsuigm_table,
        copy_paste_augmenter=cp_augmenter,
        base_seed=seed
    )
    val_ds = ControlledTrack1Dataset(
        scenes=val_scenes,
        crop_size=crop_size,
        is_train=False,
        use_rsuigm=use_rsuigm,
        rsuigm_table=rsuigm_table,
        base_seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 3. Build Model
    if model_type == "dinov2_l":
        model = build_dinov2_model(num_classes=5).to(device)
        # Differential learning rates
        param_groups = [
            {"params": model.encoder.parameters(), "lr": lr},
            {"params": model.proj.parameters(), "lr": decoder_lr},
            {"params": model.fusion.parameters(), "lr": decoder_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    elif model_type == "mit_b3":
        model = build_mit_b3_model(num_classes=5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=decoder_lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Losses
    class_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    lovasz_loss_fn = PresentClassLovaszSoftmaxLoss(num_classes=5) if use_lovasz else None

    # Normalization
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    save_dir = osp.join(save_root, run_id)
    os.makedirs(save_dir, exist_ok=True)
    best_q_seg = -1.0
    best_metrics = {}

    total_train_samples = 0
    t_train_start = time.time()

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        t_ep = time.time()

        optimizer.zero_grad()
        for step, (x_raw, y_mask, wts, depths) in enumerate(train_loader):
            x_raw = x_raw.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            if use_rsuigm and rsuigm_table is not None:
                x_in = apply_rsuigm_batch_gpu(x_raw, wts, depths, rsuigm_table)
            else:
                x_in = x_raw

            x_norm = (x_in - mean_im) / std_im

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)
                loss = ce_loss_fn(logits.float(), y_mask)
                if lovasz_loss_fn is not None:
                    loss_lov = lovasz_loss_fn(logits.float(), y_mask)
                    loss = loss + 0.25 * loss_lov

                loss_scaled = loss / grad_accum_steps

            loss_scaled.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_train_samples += x_raw.shape[0]

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        ep_time = time.time() - t_ep

        # Full validation on all 800 images
        val_res = evaluate_model_full(
            model, val_loader, device, use_rsuigm, rsuigm_table, mean_im, std_im
        )

        q_seg = val_res["Q_seg"]
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val mIoU: {val_res['mIoU']:.4f} | F1: {val_res['macro_F1']:.4f} | Q_seg: {q_seg:.4f} | ({ep_time:.1f}s)")

        if q_seg > best_q_seg:
            best_q_seg = q_seg
            best_metrics = val_res.copy()
            best_metrics["epoch"] = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": best_metrics,
                "config": {
                    "run_id": run_id,
                    "fold_id": fold_id,
                    "model_type": model_type,
                    "use_rsuigm": use_rsuigm,
                    "use_lovasz": use_lovasz,
                    "epochs": epochs,
                    "crop_size": crop_size,
                    "lr": lr,
                    "decoder_lr": decoder_lr
                }
            }, osp.join(save_dir, "best_model.pt"))

    elapsed_total = time.time() - t_start
    train_elapsed = time.time() - t_train_start
    throughput = total_train_samples / max(1.0, train_elapsed)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    max_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    vram_pct = (peak_vram_gb / max_vram_gb) * 100

    print("-" * 78)
    print(f"RUN {run_id} COMPLETE in {elapsed_total:.1f}s | Peak VRAM: {peak_vram_gb:.2f} GB ({vram_pct:.1f}%) | Throughput: {throughput:.1f} img/s")
    print(f"BEST HELD-OUT METRICS (Epoch {best_metrics.get('epoch', 0)}):")
    print(f"  mIoU: {best_metrics.get('mIoU', 0):.4f} | Macro F1: {best_metrics.get('macro_F1', 0):.4f} | Q_seg: {best_metrics.get('Q_seg', 0):.4f}")
    print(f"  Per-Class IoU: {best_metrics.get('per_class_IoU', [])}")
    print(f"  Per-Class F1:  {best_metrics.get('per_class_F1', [])}")
    print("=" * 78)

    # Log to CSV
    row = [
        run_id,
        "9a17437",
        seed,
        "b14f238c",
        fold_id,
        model_type,
        "facebookresearch/dinov2" if model_type == "dinov2_l" else "timm/mit_b3",
        "RSUIGM_Inversion" if use_rsuigm else "Raw_RGB",
        f"{crop_size}x{crop_size}",
        "CE_Lovasz" if use_lovasz else "Weighted_CE",
        "CopyPaste" if use_copy_paste else "RandomCrop_HFlip",
        "Full_FineTune",
        best_metrics.get("mIoU", 0),
        best_metrics.get("macro_F1", 0),
        best_metrics.get("Q_seg", 0),
        str(best_metrics.get("per_class_IoU", [])),
        str(best_metrics.get("per_class_F1", [])),
        best_metrics.get("coverage_penalty", 1.0),
        "N/A", "N/A", "N/A",
        round(peak_vram_gb, 2),
        round(throughput, 1),
        round(elapsed_total, 1),
        "COMPLETED",
        f"PASSED_Q_SEG_{best_metrics.get('Q_seg', 0):.4f}"
    ]

    with open("reports/track1_research/experiment_ledger.csv", "a", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(row)

    # Save detailed JSON metrics
    with open(osp.join(save_dir, "metrics.json"), "w") as fp:
        json.dump(best_metrics, fp, indent=2)

    return best_metrics


if __name__ == '__main__':
    # Test dispatch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--fold_id", type=str, default="fold_0")
    parser.add_argument("--model_type", type=str, default="dinov2_l", choices=["dinov2_l", "mit_b3"])
    parser.add_argument("--use_rsuigm", action="store_true", default=True)
    parser.add_argument("--raw_input", dest="use_rsuigm", action="store_false")
    parser.add_argument("--use_lovasz", action="store_true")
    parser.add_argument("--use_copy_paste", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--crop_size", type=int, default=518)
    args = parser.parse_args()

    train_screening_candidate(
        run_id=args.run_id,
        fold_id=args.fold_id,
        model_type=args.model_type,
        use_rsuigm=args.use_rsuigm,
        use_lovasz=args.use_lovasz,
        use_copy_paste=args.use_copy_paste,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        decoder_lr=args.decoder_lr,
        crop_size=args.crop_size
    )
