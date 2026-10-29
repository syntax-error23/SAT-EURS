"""
SAT-EURS Track 2 — LiteV3 Training & Ablation Engine
Ablation matrix:
- E1: Supervised baseline (Weighted CE + L1 Restoration)
- E2: Standard Distillation (WCE + T=2.5 KL against cached DINOv2 targets + L1 Restoration)
- E3: Class-Aware Distillation (WCE + Rare-Class-Weighted Distillation + L1 Restoration)
- E4: Boundary-Band Loss Suite (SegmentationV7Loss + Distillation + L1 Restoration)

Hardware Target: Embedded Jetson AGX Orin (Static [1, 3, 384, 768], <=1.0M params, <=12.0 GFLOPs)
Zero leakage: S0 trains on non-Fold0 scenes; S1 trains on non-Fold1 scenes.
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

sys.path.insert(0, ".")
from models.track2_litev3 import SATEURSLiteV3, build_track2_litev3_model
from losses.segmentation_v7_loss import SegmentationV7Loss
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class LiteV3Dataset(Dataset):
    """
    In-memory preloaded dataset for zero-disk-IO training iterations.
    """
    def __init__(self, scenes: List[str], fold: int, load_targets: bool = True, max_per_scene: int = 100):
        self.scenes = scenes
        self.fold = fold
        self.load_targets = load_targets
        
        print(f"[Dataset F{fold}] Preloading {len(scenes)} scenes into memory...", flush=True)
        self.samples = [] # list of (deg_img, clean_img, mask, teacher_target)
        
        for sc in scenes:
            c_p = f"data/cached_1024/clean/{sc}.png"
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"
            tgt_dir = f"data/cached_teacher_targets/fold_{fold}/{sc}"
            
            c_im = cv2.imread(c_p)
            c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB)
            c_res = cv2.resize(c_rgb, (768, 384), interpolation=cv2.INTER_LINEAR)
            clean_t = torch.from_numpy(c_res.transpose(2, 0, 1)).float() / 255.0
            
            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            m_res = cv2.resize(m_im, (768, 384), interpolation=cv2.INTER_NEAREST)
            mask_t = torch.from_numpy(m_res).long()
            
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])
            deg_files = deg_files[::2][:max_per_scene]
            
            for df in deg_files:
                stem = os.path.splitext(df)[0]
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB)
                d_res = cv2.resize(d_rgb, (768, 384), interpolation=cv2.INTER_LINEAR)
                deg_t = torch.from_numpy(d_res.transpose(2, 0, 1)).float() / 255.0
                
                target_t = None
                if load_targets:
                    tgt_p = os.path.join(tgt_dir, f"{stem}.pt")
                    if os.path.exists(tgt_p):
                        target_t = torch.load(tgt_p, map_location="cpu", weights_only=False).float()
                    else:
                        target_t = torch.zeros((5, 192, 384), dtype=torch.float32)
                
                self.samples.append((deg_t, clean_t, mask_t, target_t))
                
        print(f"[Dataset F{fold}] Loaded {len(self.samples)} samples into RAM!", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        deg, clean, mask, tgt = self.samples[idx]
        
        # Random horizontal flip
        if random.random() > 0.5:
            deg = torch.flip(deg, dims=[2])
            clean = torch.flip(clean, dims=[2])
            mask = torch.flip(mask, dims=[1])
            if tgt is not None:
                tgt = torch.flip(tgt, dims=[2])
                
        # Random vertical flip
        if random.random() > 0.8:
            deg = torch.flip(deg, dims=[1])
            clean = torch.flip(clean, dims=[1])
            mask = torch.flip(mask, dims=[0])
            if tgt is not None:
                tgt = torch.flip(tgt, dims=[1])
                
        if tgt is None:
            tgt = torch.zeros(1)
            
        return deg, clean, mask, tgt


def evaluate_heldout(model: nn.Module, val_scenes: List[str], device: torch.device, is_full: bool = False) -> Dict[str, float]:
    """
    Evaluates model on held-out validation scenes.
    Computes per-class IoU, present-class mIoU, PSNR, SSIM, Sharpness.
    is_full=True samples 40 images per scene (160 total).
    is_full=False samples 10 images per scene (40 total) for rapid epoch monitoring.
    """
    model.eval()
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    psnr_list, ssim_list, sharp_list = [], [], []
    step = 5 if is_full else 20
    
    with torch.no_grad():
        for sc in val_scenes:
            c_p = f"data/cached_1024/clean/{sc}.png"
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"
            
            c_im = cv2.imread(c_p)
            c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::step]
            
            for df in deg_files:
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Resize to model resolution [384, 768]
                d_in = cv2.resize(d_rgb, (768, 384))
                x_th = torch.from_numpy(d_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                restored_t, logits_t = model(x_th)
                
                # Resize restored back to 1024x1024 for metric computation
                res_np = restored_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                res_1024 = cv2.resize(res_np, (1024, 1024))
                
                # Resize logits back to 1024x1024 for mask prediction
                logits_1024 = F.interpolate(logits_t, size=(1024, 1024), mode="bilinear", align_corners=False)
                pred_mask = logits_1024.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)
                
                # Update confusion matrix
                valid = (m_im >= 0) & (m_im < num_classes)
                bins = num_classes * m_im[valid].astype(np.int64) + pred_mask[valid].astype(np.int64)
                counts = np.bincount(bins, minlength=num_classes * num_classes)
                conf_matrix += counts.reshape(num_classes, num_classes)
                
                # Restoration metrics
                p, _ = compute_psnr_and_norm(res_1024, c_rgb)
                s = compute_ssim_single(res_1024, c_rgb)
                sh = compute_sharpness_ratio(res_1024, d_rgb)
                psnr_list.append(p)
                ssim_list.append(s)
                sharp_list.append(sh)
                
    # Calculate IoU and Recalls
    ious = []
    recalls = []
    present_ious = []
    
    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fn = conf_matrix[c, :].sum() - tp
        fp = conf_matrix[:, c].sum() - tp
        denom = tp + fn + fp
        gt_c = conf_matrix[c, :].sum()
        
        iou_c = float(tp / denom) if denom > 0 else 0.0
        rec_c = float(tp / gt_c) if gt_c > 0 else 0.0
        
        ious.append(iou_c)
        recalls.append(rec_c)
        if gt_c > 0:
            present_ious.append(iou_c)
            
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_sharp = float(np.mean(sharp_list))
    
    q_fr = (
        0.40 * np.clip((mean_psnr - 10.0) / 25.0, 0.0, 1.0) +
        0.35 * np.clip(mean_ssim, 0.0, 1.0) +
        0.25 * np.clip(mean_sharp / 1.50, 0.0, 1.0)
    )
    
    present_miou = float(np.mean(present_ious)) if present_ious else 0.0
    macro_5_iou = float(np.mean(ious))
    
    s_local = 0.60 * present_miou + 0.30 * q_fr
    
    return {
        "present_miou": present_miou,
        "macro_5_iou": macro_5_iou,
        "ious": [round(x, 4) for x in ious],
        "recalls": [round(x, 4) for x in recalls],
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "sharpness": mean_sharp,
        "q_fr": q_fr,
        "s_local": s_local,
        "conf_matrix": conf_matrix.tolist()
    }


def train_litev3(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"SAT-EURS Track 2 LiteV3 Training: Exp={args.exp}, Fold={args.fold}")
    print(f"Device: {device}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"=======================================================\n")
    
    with open("splits/manifest_5fold.json") as f:
        manifest_5fold = json.load(f)
        
    fold_key = f"fold_{args.fold}"
    train_scenes = manifest_5fold["folds"][fold_key]["train_scenes"]
    val_scenes = manifest_5fold["folds"][fold_key]["val_scenes"]
    
    print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Val scenes ({len(val_scenes)}): {val_scenes}")
    
    load_targets = (args.exp != "E1_supervised")
    train_dataset = LiteV3Dataset(train_scenes, fold=args.fold, load_targets=load_targets, max_per_scene=100)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize Model
    model = build_track2_litev3_model(pretrained_backbone=True, unsharp_factor=1.20).to(device)
    
    # Load calibrated router weights if available
    router_ckpt = "checkpoints/EXP-PRETRAINED-30EP_best.pt"
    if os.path.exists(router_ckpt):
        ckpt = torch.load(router_ckpt, map_location="cpu", weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        r_st = {k: v for k, v in st.items() if "router" in k}
        res = model.load_state_dict(r_st, strict=False)
        print(f"[Model Init] Loaded {len(r_st)} pre-calibrated optical router weights.")
        
    # Optimizer with differential learning rates
    backbone_params = list(model.backbone_blocks.parameters())
    decoder_params = list(model.decoder.parameters())
    router_params = list(model.router.parameters()) + list(model.router_fc.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.2, "weight_decay": 1e-4},
        {"params": decoder_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": router_params, "lr": args.lr * 0.1, "weight_decay": 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss criteria
    class_weights = torch.tensor([0.80, 1.00, 1.20, 1.80, 2.00], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    v7_loss_fn = SegmentationV7Loss(num_classes=5, class_weights=class_weights).to(device)
    
    kd_temperature = 2.5
    
    # Output directory
    out_dir = os.path.join(args.save_dir, f"{args.exp}_f{args.fold}")
    os.makedirs(out_dir, exist_ok=True)
    
    best_s_local = -1.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_seg_losses = []
        epoch_rest_losses = []
        epoch_kd_losses = []
        
        t0 = time.time()
        for deg, clean, mask, tgt in train_loader:
            deg = deg.to(device)
            clean = clean.to(device)
            mask = mask.to(device)
            if load_targets:
                tgt = tgt.to(device)
                
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                restored, logits = model(deg)
                
                # Restoration L1 loss
                loss_rest = F.l1_loss(restored, clean)
                
                if args.exp == "E1_supervised":
                    loss_seg = ce_loss_fn(logits, mask)
                    total_loss = 0.70 * loss_seg + 0.30 * loss_rest
                    loss_kd = torch.tensor(0.0, device=device)
                    
                elif args.exp == "E2_distill_kl":
                    loss_seg = ce_loss_fn(logits, mask)
                    
                    # KD loss at stride 2 [192, 384] normalized spatially per pixel
                    logits_s2 = F.interpolate(logits, size=(192, 384), mode="bilinear", align_corners=False).float()
                    s_log_prob = F.log_softmax(logits_s2 / kd_temperature, dim=1)
                    t_prob = F.softmax(tgt.float() / kd_temperature, dim=1)
                    b, c, h, w = logits_s2.shape
                    loss_kd = (F.kl_div(s_log_prob, t_prob, reduction="sum") / (b * h * w)) * (kd_temperature * kd_temperature)
                    
                    total_loss = 0.50 * loss_seg + 0.50 * loss_kd + 0.20 * loss_rest
                    
                elif args.exp == "E3_class_aware_kd":
                    loss_seg = ce_loss_fn(logits, mask)
                    
                    # Class-weighted KD loss normalized spatially per pixel
                    logits_s2 = F.interpolate(logits, size=(192, 384), mode="bilinear", align_corners=False).float()
                    s_log_prob = F.log_softmax(logits_s2 / kd_temperature, dim=1)
                    t_prob = F.softmax(tgt.float() / kd_temperature, dim=1)
                    kl_per_class = F.kl_div(s_log_prob, t_prob, reduction="none").mean(dim=[2, 3]) # [B, 5]
                    weighted_kl = (kl_per_class * class_weights.view(1, 5)).sum(dim=1).mean()
                    loss_kd = weighted_kl * (kd_temperature * kd_temperature)
                    
                    total_loss = 0.40 * loss_seg + 0.50 * loss_kd + 0.20 * loss_rest
                    
                elif args.exp == "E4_boundary_v7":
                    loss_seg, _ = v7_loss_fn(logits, mask)
                    
                    logits_s2 = F.interpolate(logits, size=(192, 384), mode="bilinear", align_corners=False).float()
                    s_log_prob = F.log_softmax(logits_s2 / kd_temperature, dim=1)
                    t_prob = F.softmax(tgt.float() / kd_temperature, dim=1)
                    kl_per_class = F.kl_div(s_log_prob, t_prob, reduction="none").mean(dim=[2, 3])
                    weighted_kl = (kl_per_class * class_weights.view(1, 5)).sum(dim=1).mean()
                    loss_kd = weighted_kl * (kd_temperature * kd_temperature)
                    
                    total_loss = 0.50 * loss_seg + 0.40 * loss_kd + 0.20 * loss_rest
                    
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            epoch_seg_losses.append(loss_seg.item())
            epoch_rest_losses.append(loss_rest.item())
            epoch_kd_losses.append(loss_kd.item())
            
        scheduler.step()
        train_time = time.time() - t0
        
        # Validation
        is_full = (epoch == 1 or epoch % 5 == 0 or epoch == args.epochs)
        val_res = evaluate_heldout(model, val_scenes, device, is_full=is_full)
        
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "seg_loss": float(np.mean(epoch_seg_losses)),
            "rest_loss": float(np.mean(epoch_rest_losses)),
            "kd_loss": float(np.mean(epoch_kd_losses)),
            "train_time_s": train_time,
            **val_res
        }
        history.append(rec)
        
        print(
            f"Ep {epoch:02d}/{args.epochs:02d} | "
            f"Loss: {rec['train_loss']:.4f} (Seg:{rec['seg_loss']:.4f}, KD:{rec['kd_loss']:.4f}, R:{rec['rest_loss']:.4f}) | "
            f"Val mIoU(pres): {rec['present_miou']:.4f} | "
            f"IoUs: {rec['ious']} | "
            f"PSNR: {rec['psnr']:.2f} | "
            f"S_local: {rec['s_local']:.4f} | "
            f"Time: {train_time:.1f}s",
            flush=True
        )
        
        # Save best model
        if rec["s_local"] > best_s_local:
            best_s_local = rec["s_local"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_res": val_res,
                "args": vars(args)
            }, os.path.join(out_dir, "best_model.pt"))
            print(f"  >>> Best model saved! S_local={best_s_local:.4f}", flush=True)
            
    # Save last model & history
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args)
    }, os.path.join(out_dir, "last_model.pt"))
    
    with open(os.path.join(out_dir, "history.json"), "w") as fp:
        json.dump(history, fp, indent=2)
        
    print(f"\nTraining complete for {args.exp}_f{args.fold}! Best S_local: {best_s_local:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["E1_supervised", "E2_distill_kl", "E3_class_aware_kd", "E4_boundary_v7"])
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/litev3")
    args = parser.parse_args()
    
    train_litev3(args)
