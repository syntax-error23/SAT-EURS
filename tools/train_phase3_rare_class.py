"""
SAT-EURS Track 2 — Phase 3: Controlled Rare-Class Ablation Engine
Evaluates controlled rare-class interventions against the E1 control baseline:
1. scene_balanced: 1.75x sampling boost on rare scenes [ray1, sharkFish, seahorse1, turtles1, starfish2]
2. class_weighted: Class weights boosted to [0.70, 0.90, 1.20, 3.00, 3.50] with uniform sampling
3. combined: scene-balanced + class-weighted
4. dice_present: CE + 0.30 * Present-Class Dice loss

Hardware: RTX 5090 (20 epochs, batch size 16, Cosine Annealing LR 5e-4)
Evaluated on Fold 0 and Fold 1 held-out validation scenes (320 images).
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


class PresentClassDiceLoss(nn.Module):
    """
    Computes soft Dice loss strictly across classes present in each ground-truth batch.
    Prevents empty-class zero-division artifacts and boundary inflation.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], target: [B, H, W]
        probs = F.softmax(logits, dim=1)
        b, c, h, w = probs.shape
        
        target_1hot = F.one_hot(target, num_classes=c).permute(0, 3, 1, 2).float() # [B, C, H, W]
        
        dice_losses = []
        for i in range(b):
            present_classes = torch.unique(target[i])
            for cls in present_classes:
                cls_idx = cls.item()
                p_c = probs[i, cls_idx]
                t_c = target_1hot[i, cls_idx]
                
                intersection = (p_c * t_c).sum()
                cardinality = (p_c * p_c).sum() + (t_c * t_c).sum()
                dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
                dice_losses.append(1.0 - dice)
                
        if dice_losses:
            return torch.stack(dice_losses).mean()
        return torch.tensor(0.0, device=logits.device)


class RareClassAblationDataset(Dataset):
    def __init__(self, scenes: List[str], fold: int, rare_boost: float = 1.0):
        self.scenes = scenes
        self.fold = fold
        self.rare_scenes = {"ray1", "sharkFish", "seahorse1", "turtles1", "starfish2"}
        
        print(f"[Dataset F{fold}] Preloading scenes (rare_boost={rare_boost})...", flush=True)
        self.samples = []
        
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
            
            # Determine sample count for this scene
            if sc in self.rare_scenes and rare_boost > 1.0:
                max_samples = int(round(100 * rare_boost))
                deg_files = deg_files[:max_samples]
            else:
                deg_files = deg_files[::2][:100]
                
            for df in deg_files:
                d_p = os.path.join(deg_dir, df)
                d_im = cv2.imread(d_p)
                if d_im is None:
                    continue
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB)
                d_res = cv2.resize(d_rgb, (768, 384), interpolation=cv2.INTER_LINEAR)
                deg_t = torch.from_numpy(d_res.transpose(2, 0, 1)).float() / 255.0
                self.samples.append((deg_t, clean_t, mask_t))
                
        print(f"[Dataset F{fold}] Preloaded {len(self.samples)} total samples into RAM!", flush=True)

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


def evaluate_heldout(model: nn.Module, val_scenes: List[str], device: torch.device, step: int = 5) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    model.eval()
    num_classes = 5
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    psnr_list, ssim_list, sharp_list = [], [], []
    
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
                
    return conf_matrix, psnr_list, ssim_list, sharp_list


def compute_official_metrics(cm: np.ndarray, psnr_list: List[float], ssim_list: List[float], sharp_list: List[float]) -> Dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    denom_iou = tp + fp + fn
    denom_f1 = 2 * tp + fp + fn
    
    per_class_iou = [float(tp[c] / denom_iou[c]) if denom_iou[c] > 0 else 0.0 for c in range(5)]
    per_class_f1 = [float(2 * tp[c] / denom_f1[c]) if denom_f1[c] > 0 else 0.0 for c in range(5)]
    
    miou = float(np.mean(per_class_iou))
    macro_f1 = float(np.mean(per_class_f1))
    
    gt_present = cm.sum(axis=1) > 0
    p_cov = 1.0 if bool(np.all(tp[gt_present] > 0)) else 0.0
    q_seg = miou * macro_f1 * p_cov
    
    mean_psnr = float(np.mean(psnr_list))
    psnr_norm = min(1.0, max(0.0, (mean_psnr - 15.0) / 20.0))
    mean_ssim = float(np.mean(ssim_list))
    mean_sharp = float(np.mean(sharp_list))
    
    q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
    q_nr = 0.7430
    s_comp = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr
    
    return {
        "miou": round(miou, 4),
        "macro_f1": round(macro_f1, 4),
        "p_cov": round(p_cov, 4),
        "q_seg": round(q_seg, 4),
        "psnr_db": round(mean_psnr, 2),
        "psnr_norm": round(psnr_norm, 4),
        "ssim": round(mean_ssim, 4),
        "sharpness": round(mean_sharp, 4),
        "q_fr": round(q_fr, 4),
        "q_nr": round(q_nr, 4),
        "s_comp": round(s_comp, 4),
        "per_class_iou": [round(x, 4) for x in per_class_iou],
        "per_class_f1": [round(x, 4) for x in per_class_f1]
    }


def train_single_fold(variant: str, fold: int, epochs: int = 20, lr: float = 5e-4, batch_size: int = 16) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42 + fold)
    
    with open("splits/manifest_5fold.json") as f:
        manifest_5fold = json.load(f)
        
    train_scenes = manifest_5fold["folds"][f"fold_{fold}"]["train_scenes"]
    val_scenes = manifest_5fold["folds"][f"fold_{fold}"]["val_scenes"]
    
    rare_boost = 1.75 if "scene_balanced" in variant or "combined" in variant else 1.0
    train_dataset = RareClassAblationDataset(train_scenes, fold=fold, rare_boost=rare_boost)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    model = build_track2_litev3_model(pretrained_backbone=True, unsharp_factor=1.20).to(device)
    
    # Load calibrated router
    router_ckpt = "checkpoints/EXP-PRETRAINED-30EP_best.pt"
    if os.path.exists(router_ckpt):
        ckpt = torch.load(router_ckpt, map_location="cpu", weights_only=False)
        st = ckpt.get("model_state_dict", ckpt)
        r_st = {k: v for k, v in st.items() if "router" in k}
        model.load_state_dict(r_st, strict=False)
        
    optimizer = torch.optim.AdamW([
        {"params": list(model.backbone_blocks.parameters()), "lr": lr * 0.2, "weight_decay": 1e-4},
        {"params": list(model.decoder.parameters()), "lr": lr, "weight_decay": 1e-4},
        {"params": list(model.router.parameters()) + list(model.router_fc.parameters()), "lr": lr * 0.1, "weight_decay": 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Configure weights and losses
    if "class_weighted" in variant or "combined" in variant:
        class_weights = torch.tensor([0.70, 0.90, 1.20, 3.00, 3.50], device=device)
    else:
        class_weights = torch.tensor([0.80, 1.00, 1.20, 1.80, 2.00], device=device)
        
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_fn = PresentClassDiceLoss() if "dice_present" in variant else None
    
    out_dir = f"checkpoints/phase3/{variant}_f{fold}"
    os.makedirs(out_dir, exist_ok=True)
    
    best_q_seg = -1.0
    best_ckpt_path = os.path.join(out_dir, "best_model.pt")
    
    for epoch in range(1, epochs + 1):
        model.train()
        for deg, clean, mask in train_loader:
            deg, clean, mask = deg.to(device), clean.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            restored, logits = model(deg)
            loss_rest = F.l1_loss(restored, clean)
            loss_ce = ce_loss_fn(logits, mask)
            
            loss = loss_ce + 0.15 * loss_rest
            if dice_loss_fn is not None:
                loss_dice = dice_loss_fn(logits, mask)
                loss += 0.30 * loss_dice
                
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        # Monitor validation
        cm, p, s, sh = evaluate_heldout(model, val_scenes, device, step=20)
        tp = np.diag(cm).astype(float)
        denom_iou = tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
        per_class_iou = [float(tp[c] / denom_iou[c]) if denom_iou[c] > 0 else 0.0 for c in range(5)]
        miou = float(np.mean(per_class_iou))
        
        denom_f1 = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
        macro_f1 = float(np.mean([float(2 * tp[c] / denom_f1[c]) if denom_f1[c] > 0 else 0.0 for c in range(5)]))
        q_seg_ep = miou * macro_f1
        
        if q_seg_ep > best_q_seg:
            best_q_seg = q_seg_ep
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, best_ckpt_path)
            
    print(f"[{variant}_f{fold}] Training done! Best Q_seg={best_q_seg:.4f}")
    return best_ckpt_path


def run_ablation(variant: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Running Phase 3 Ablation: {variant}")
    print(f"Device: {device}")
    print(f"=======================================================\n")
    
    # Train Fold 0 and Fold 1
    ckpt_f0 = train_single_fold(variant, fold=0)
    ckpt_f1 = train_single_fold(variant, fold=1)
    
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
        
    val_scenes_f0 = manifest["folds"]["fold_0"]["val_scenes"]
    val_scenes_f1 = manifest["folds"]["fold_1"]["val_scenes"]
    
    # Full out-of-fold evaluation on 320 images (step=5)
    m0 = build_track2_litev3_model(pretrained_backbone=False).to(device)
    m0.load_state_dict(torch.load(ckpt_f0, map_location=device)["model_state_dict"])
    c0, p0, s0, sh0 = evaluate_heldout(m0, val_scenes_f0, device, step=5)
    
    m1 = build_track2_litev3_model(pretrained_backbone=False).to(device)
    m1.load_state_dict(torch.load(ckpt_f1, map_location=device)["model_state_dict"])
    c1, p1, s1, sh1 = evaluate_heldout(m1, val_scenes_f1, device, step=5)
    
    pooled_cm = c0 + c1
    all_psnr = p0 + p1
    all_ssim = s0 + s1
    all_sharp = sh0 + sh1
    
    metrics = compute_official_metrics(pooled_cm, all_psnr, all_ssim, all_sharp)
    print(f"\n[{variant} Results]")
    print(f"  mIoU:      {metrics['miou']:.4f}")
    print(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
    print(f"  P_cov:     {metrics['p_cov']:.1f}")
    print(f"  Q_seg:     {metrics['q_seg']:.4f}")
    print(f"  PSNR:      {metrics['psnr_db']} dB | SSIM: {metrics['ssim']:.4f} | Sharpness: {metrics['sharpness']:.4f}")
    print(f"  Q_FR:      {metrics['q_fr']:.4f}")
    print(f"  Composite: {metrics['s_comp']:.4f}")
    print(f"  Per-class IoU: {metrics['per_class_iou']}")
    print(f"  Per-class F1:  {metrics['per_class_f1']}")
    
    out_file = f"reports/track2_sprint/PHASE3_{variant}_RESULTS.json"
    with open(out_file, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved to {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=["scene_balanced", "class_weighted", "combined", "dice_present"])
    args = parser.parse_args()
    
    run_ablation(args.variant)
