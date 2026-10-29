"""
SAT-EURS Track 2 — Phase 5: Full-20 Scene Production Training Engine
Trains champion SAT-EURS-LiteV3 on all 20 scenes (4,000 degraded images).
Features:
- Integrated native Edge-Gated Sharpening (base=0.45, thresh=0.08)
- Champion Phase 3 Class-Weighted Cross-Entropy [0.70, 0.90, 1.20, 3.00, 3.50]
- Physical Optical Polynomial Router + L1 Restoration
- Pre-calibrated RSUIGM table and router initialization
- Differential AdamW learning rates + Cosine Annealing (20 epochs)
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
from models.track2_litev3 import build_track2_litev3_model
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


ALL_20_SCENES = [
    "fish1", "fish11", "fish12", "fish13", "fish14", "fish15",
    "fish17", "fish18", "fish2", "fish20", "fish21", "fish4",
    "fish5", "fish7", "fish9", "ray1", "seahorse1", "sharkFish",
    "starfish2", "turtles1"
]


class Full20Dataset(Dataset):
    """
    Preloads all 4,000 degraded images across all 20 scenes into RAM for zero-disk-IO training.
    """
    def __init__(self, scenes: List[str]):
        self.scenes = scenes
        self.samples = []
        print(f"[Dataset Full-20] Preloading all {len(scenes)} scenes into system RAM...", flush=True)
        t0 = time.time()
        
        for sc in scenes:
            c_p = f"data/cached_1024/clean/{sc}.png"
            m_p = f"data/cached_1024/masks/{sc}.png"
            deg_dir = f"data/cached_1024/degraded/{sc}"
            
            c_im = cv2.imread(c_p)
            c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB)
            c_res = cv2.resize(c_rgb, (768, 384), interpolation=cv2.INTER_LINEAR)
            clean_t = torch.from_numpy(c_res.transpose(2, 0, 1)).float() / 255.0
            
            m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            m_res = cv2.resize(m_im, (768, 384), interpolation=cv2.INTER_NEAREST)
            mask_t = torch.from_numpy(m_res).long()
            
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])
            for df in deg_files:
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB)
                d_res = cv2.resize(d_rgb, (768, 384), interpolation=cv2.INTER_LINEAR)
                deg_t = torch.from_numpy(d_res.transpose(2, 0, 1)).float() / 255.0
                self.samples.append((deg_t, clean_t, mask_t))
                
        print(f"[Dataset Full-20] Loaded {len(self.samples)} images into RAM in {time.time() - t0:.1f}s!", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        deg, clean, mask = self.samples[idx]
        if random.random() > 0.5:
            deg = torch.flip(deg, dims=[2])
            clean = torch.flip(clean, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() > 0.8:
            deg = torch.flip(deg, dims=[1])
            clean = torch.flip(clean, dims=[1])
            mask = torch.flip(mask, dims=[0])
        return deg, clean, mask


def evaluate_dataset(model: nn.Module, scenes: List[str], device: torch.device, step: int = 10) -> Tuple[np.ndarray, float, float, float]:
    model.eval()
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    psnr_list, ssim_list, sharp_list = [], [], []
    
    with torch.no_grad():
        for sc in scenes:
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
                d_in = cv2.resize(d_rgb, (768, 384))
                x_th = torch.from_numpy(d_in.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                restored_t, logits_t = model(x_th)
                res_np = restored_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                res_1024 = cv2.resize(res_np, (1024, 1024))
                
                logits_1024 = F.interpolate(logits_t, size=(1024, 1024), mode="bilinear", align_corners=False)
                pred_mask = logits_1024.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)
                
                valid = (m_im >= 0) & (m_im < num_classes)
                bins = num_classes * m_im[valid].astype(np.int64) + pred_mask[valid].astype(np.int64)
                counts = np.bincount(bins, minlength=num_classes * num_classes)
                conf_matrix += counts.reshape(num_classes, num_classes)
                
                p, _ = compute_psnr_and_norm(res_1024, c_rgb)
                s = compute_ssim_single(res_1024, c_rgb)
                sh = compute_sharpness_ratio(res_1024, c_rgb)
                psnr_list.append(p)
                ssim_list.append(s)
                sharp_list.append(sh)
                
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_sharp = float(np.mean(sharp_list))
    return conf_matrix, mean_psnr, mean_ssim, mean_sharp


def train_production_run(epochs: int = 20, batch_size: int = 16, lr: float = 5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    
    print("\n=======================================================")
    print("SAT-EURS Track 2: Production Full-20 Scene Training Run")
    print(f"Device: {device}, Epochs: {epochs}, Batch Size: {batch_size}")
    print("=======================================================\n")
    
    # 1. Dataset & Loader
    train_dataset = Full20Dataset(ALL_20_SCENES)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    # 2. Model Initialization
    model = build_track2_litev3_model(pretrained_backbone=True, unsharp_factor=1.20, use_edge_gated=True).to(device)
    
    router_ckpt = "checkpoints/EXP-PRETRAINED-30EP_best.pt"
    if os.path.exists(router_ckpt):
        ckpt = torch.load(router_ckpt, map_location="cpu", weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        r_st = {k: v for k, v in st.items() if "router" in k}
        model.load_state_dict(r_st, strict=False)
        print(f"[Init] Loaded {len(r_st)} pre-calibrated optical router weights.")
        
    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone_blocks.parameters()), "lr": lr * 0.2, "weight_decay": 1e-4},
        {"params": list(model.decoder.parameters()), "lr": lr, "weight_decay": 1e-4},
        {"params": list(model.router.parameters()) + list(model.router_fc.parameters()), "lr": lr * 0.1, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 4. Champion Loss Recipe
    class_weights = torch.tensor([0.70, 0.90, 1.20, 3.00, 3.50], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    out_dir = "checkpoints/production"
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, "litev3_production_full20_best.pt")
    
    best_composite = -1.0
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_seg = 0.0
        running_rest = 0.0
        
        for deg, clean, mask in train_loader:
            deg, clean, mask = deg.to(device), clean.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            restored, logits = model(deg)
            loss_rest = F.l1_loss(restored, clean)
            loss_seg = ce_loss_fn(logits, mask)
            loss = loss_seg + 0.15 * loss_rest
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_seg += loss_seg.item()
            running_rest += loss_rest.item()
            
        scheduler.step()
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_seg = running_seg / n_batches
        avg_rest = running_rest / n_batches
        
        # Periodic evaluation (every 2 epochs or final)
        if epoch % 2 == 0 or epoch == epochs:
            cm, mean_p, mean_s, mean_sh = evaluate_dataset(model, ALL_20_SCENES, device, step=10)
            tp = np.diag(cm).astype(float)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            denom_iou = tp + fp + fn
            per_class_iou = [float(tp[c] / denom_iou[c]) if denom_iou[c] > 0 else 0.0 for c in range(5)]
            miou = float(np.mean(per_class_iou))
            denom_f1 = 2 * tp + fp + fn
            macro_f1 = float(np.mean([float(2 * tp[c] / denom_f1[c]) if denom_f1[c] > 0 else 0.0 for c in range(5)]))
            p_cov = 1.0 if bool(np.all(tp[cm.sum(axis=1) > 0] > 0)) else 0.0
            q_seg = miou * macro_f1 * p_cov
            
            psnr_norm = min(1.0, max(0.0, (mean_p - 15.0) / 20.0))
            q_fr = 0.45 * psnr_norm + 0.45 * mean_s + 0.10 * mean_sh
            q_nr = 0.7430
            s_comp = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr
            
            elapsed = time.time() - t0
            print(f"Ep {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} (Seg: {avg_seg:.4f}, Rest: {avg_rest:.4f}) | mIoU: {miou:.4f} | F1: {macro_f1:.4f} | Q_seg: {q_seg:.4f} | PSNR: {mean_p:.2f} | S_comp: {s_comp:.4f} | Time: {elapsed:.1f}s", flush=True)
            
            if s_comp > best_composite:
                best_composite = s_comp
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": {
                        "miou": miou, "macro_f1": macro_f1, "q_seg": q_seg,
                        "psnr": mean_p, "ssim": mean_s, "sharpness": mean_sh,
                        "q_fr": q_fr, "s_comp": s_comp, "per_class_iou": per_class_iou
                    }
                }, best_ckpt_path)
                print(f"  >>> Best Production Checkpoint Saved! S_comp: {best_composite:.4f}", flush=True)
                
            history.append({
                "epoch": epoch, "loss": avg_loss, "mIoU": miou, "macro_f1": macro_f1,
                "q_seg": q_seg, "psnr": mean_p, "ssim": mean_s, "sharpness": mean_sh,
                "s_comp": s_comp
            })
        else:
            elapsed = time.time() - t0
            print(f"Ep {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} (Seg: {avg_seg:.4f}, Rest: {avg_rest:.4f}) | Time: {elapsed:.1f}s", flush=True)
            
    with open("checkpoints/production/history.json", "w") as fp:
        json.dump(history, fp, indent=2)
        
    print(f"\n[Complete] Production Training Complete! Best S_comp: {best_composite:.4f}")
    print(f"Checkpoint: {best_ckpt_path}\n")
    return best_ckpt_path


if __name__ == "__main__":
    train_production_run()
