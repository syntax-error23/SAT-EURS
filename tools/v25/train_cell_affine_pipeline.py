"""
SAT-EURS Track 1 — V25 Conditioned Cell-Affine Restoration Training & Evaluation Pipeline
Protocol Compliance: Strict Single-Image Inference
Evaluator: Exact tools.v18.evaluator.V18OOFEvaluator
"""

import os
import os.path as osp
import sys
import time
import json
import csv
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from models.v25.cell_affine_restorer import (
    build_r1_cell_affine,
    build_r2_cell_affine_refiner,
    ConditionedCellAffineRestorer
)
from losses.v25.restoration_loss import RestorationLossV25
from data.v18.dataset import V18EURSInvarianceDataset
from tools.v18.evaluator import V18OOFEvaluator
from models.v18.segmenter import build_v18_segmenter


EXPECTED_V18_ZIP_SHA256 = "e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52"
V18_ZIP_PATH = "deploy/v18/submission_v18_dual_encoder_sota.zip"


def verify_baseline():
    import hashlib
    if not osp.exists(V18_ZIP_PATH):
        raise FileNotFoundError(f"Baseline ZIP missing: {V18_ZIP_PATH}")
    h = hashlib.sha256()
    with open(V18_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha = h.hexdigest().lower()
    assert sha == EXPECTED_V18_ZIP_SHA256, f"BASELINE CORRUPTED: expected {EXPECTED_V18_ZIP_SHA256}, got {sha}"
    print(f"[PREFLIGHT] Baseline verified intact (SHA-256: {sha})")


def log_experiment_result(csv_path: str, row_dict: Dict[str, Any]):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    file_exists = osp.exists(csv_path)
    fieldnames = [
        "timestamp", "experiment_id", "model_type", "fold", "epochs", "batch_size",
        "lr", "train_loss", "val_psnr_db", "val_ssim", "val_sharpness",
        "val_q_fr", "val_miou", "val_macro_f1", "val_q_seg", "val_comp_exact_90",
        "val_comp_est", "delta_vs_v18_comp", "checkpoint_path"
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def preload_val_cache(val_scenes: List[str], data_root: str) -> List[Dict[str, Any]]:
    """Preloads the 800 validation images and masks into RAM once at startup."""
    print(f"Preloading {len(val_scenes)} validation scenes (800 images) into RAM...")
    t0 = time.time()
    cache = []
    for sc in val_scenes:
        clean_path = osp.join(data_root, "clean", f"{sc}.png")
        clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_path = osp.join(data_root, "masks", f"{sc}.png")
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for wt in range(1, 11):
            for depth in range(5, 105, 5):
                deg_path = osp.join(data_root, "degraded", sc, f"{sc}_{wt}_{depth}.png")
                deg_bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
                deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                cache.append({
                    "scene": sc,
                    "wt": wt,
                    "depth": depth,
                    "deg_rgb": deg_rgb,
                    "clean_rgb": clean_rgb,
                    "gt_mask": gt_mask
                })
    print(f"Preloaded {len(cache)} validation items into RAM in {time.time() - t0:.1f}s.")
    return cache


def preload_train_cache(train_scenes: List[str], data_root: str = "data/cached_1024") -> Dict[str, Any]:
    print(f"Preloading {len(train_scenes)} training scenes (3,200 degraded images) into RAM...")
    t0 = time.time()
    cache = {"clean": {}, "degraded": {}}
    for sc in train_scenes:
        clean_path = osp.join(data_root, "clean", f"{sc}.png")
        if osp.exists(clean_path):
            clean_bgr = cv2.imread(clean_path, cv2.IMREAD_COLOR)
            cache["clean"][sc] = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for wt in range(1, 11):
            for depth in range(5, 105, 5):
                deg_path = osp.join(data_root, "degraded", sc, f"{sc}_{wt}_{depth}.png")
                deg_bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
                cache["degraded"][(sc, wt, depth)] = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB)
    print(f"Preloaded 3,200 training images into RAM in {time.time() - t0:.1f}s.")
    return cache


def evaluate_restoration_full(
    model: nn.Module,
    val_cache: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
    save_samples_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates restoration model on all 800 held-out images using V18OOFEvaluator.
    Strict single-image forward pass, batched for GPU throughput.
    """
    model.eval()
    evaluator = V18OOFEvaluator(num_classes=5)

    if save_samples_dir:
        os.makedirs(save_samples_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(0, len(val_cache), batch_size):
            batch_items = val_cache[i:i + batch_size]
            deg_tensors = [torch.from_numpy(it["deg_rgb"]).permute(2, 0, 1) for it in batch_items]
            deg_batch = torch.stack(deg_tensors, dim=0).to(device)
            wt_batch = torch.tensor([it["wt"] for it in batch_items], device=device)
            dp_batch = torch.tensor([it["depth"] for it in batch_items], device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(deg_batch, wt_batch, dp_batch)
                rest_batch = out["restored_image"].float()

            rest_np_batch = rest_batch.permute(0, 2, 3, 1).cpu().numpy()

            for b_idx, it in enumerate(batch_items):
                rest_np = np.clip(rest_np_batch[b_idx], 0.0, 1.0)
                evaluator.update_restoration(rest_np, it["clean_rgb"], scene=it["scene"])

                # Save sample
                if save_samples_dir and it["wt"] in [1, 5, 10] and it["depth"] in [10, 50, 100]:
                    sample_name = f"{it['scene']}_wt{it['wt']}_dp{it['depth']}.jpg"
                    sample_bgr = (np.clip(rest_np * 255.0, 0, 255)).astype(np.uint8)
                    sample_bgr = cv2.cvtColor(sample_bgr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(osp.join(save_samples_dir, sample_name), sample_bgr)

    res = evaluator.compute()
    return res["restoration"]


def evaluate_downstream_segmentation(
    restoration_model: nn.Module,
    segmenter_model: nn.Module,
    val_cache: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Evaluates downstream segmentation when fed with the newly restored images.
    Strict single-image forward pass!
    """
    restoration_model.eval()
    segmenter_model.eval()
    evaluator = V18OOFEvaluator(num_classes=5)

    with torch.no_grad():
        for i in range(0, len(val_cache), batch_size):
            batch_items = val_cache[i:i + batch_size]
            deg_tensors = [torch.from_numpy(it["deg_rgb"]).permute(2, 0, 1) for it in batch_items]
            deg_batch = torch.stack(deg_tensors, dim=0).to(device)
            wt_batch = torch.tensor([it["wt"] for it in batch_items], device=device)
            dp_batch = torch.tensor([it["depth"] for it in batch_items], device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_rest = restoration_model(deg_batch, wt_batch, dp_batch)
                rest_batch = out_rest["restored_image"].float()

                # Segmenter expects (rest, deg, wt, depth)
                # Unpadded size is 1024x1024
                out_seg = segmenter_model(rest_batch, deg_batch, wt_batch, dp_batch, unpadded_size=(1024, 1024))
                logits = out_seg["logits"].float()

            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            rest_np_batch = rest_batch.permute(0, 2, 3, 1).cpu().numpy()

            for b_idx, it in enumerate(batch_items):
                rest_np = np.clip(rest_np_batch[b_idx], 0.0, 1.0)
                pred_mask = preds[b_idx]
                evaluator.update_restoration(rest_np, it["clean_rgb"], scene=it["scene"])
                evaluator.update_segmentation(pred_mask, it["gt_mask"], scene=it["scene"])

    return evaluator.compute()


def train_candidate(
    model_type: str = "R1",
    fold: int = 0,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 5e-4,
    crop_size: int = 512,
    device_id: int = 0
):
    print("=" * 80)
    print(f"SAT-EURS V25 TRAINING: {model_type} (Fold {fold})")
    print(f"Protocol: Strict Single-Image Inference | Hardware: RTX 5090")
    print("=" * 80)

    verify_baseline()

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    # Load fold splits
    with open("splits/manifest_5fold.json") as f:
        manifest = json.load(f)
    train_scenes = manifest["folds"][f"fold_{fold}"]["train_scenes"]
    val_scenes = manifest["folds"][f"fold_{fold}"]["val_scenes"]

    print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Held-out validation scenes ({len(val_scenes)}): {val_scenes}")

    # Build model
    if model_type == "R1":
        model = build_r1_cell_affine(grid_size=(32, 32)).to(device)
    elif model_type == "R2":
        model = build_r2_cell_affine_refiner(grid_size=(32, 32), refiner_channels=32, refiner_blocks=4).to(device)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_type} | Parameters: {total_params:.2f}M")

    # Build Dataset and Loader
    data_root = "data/cached_1024"
    train_cache = preload_train_cache(train_scenes, data_root)
    train_ds = V18EURSInvarianceDataset(
        scenes=train_scenes,
        data_root=data_root,
        crop_size=crop_size,
        is_train=True,
        rare_crop_prob=0.50,
        base_seed=42 + fold,
        cache_dict=train_cache
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    criterion = RestorationLossV25(w_charb=1.0, w_ssim=0.20, w_lap=0.05, w_tv=1e-4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ckpt_dir = f"checkpoints/v25/{model_type}_fold_{fold}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = osp.join(ckpt_dir, "best_model.pt")

    best_q_fr = -1.0
    best_rest_metrics = {}
    best_epoch = -1

    # Preload validation cache into RAM for ultra-fast GPU-batched validation
    val_cache = preload_val_cache(val_scenes, data_root)

    t_train_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_ds.set_epoch(ep)
        t_ep0 = time.time()
        loss_accum = 0.0

        for b_idx, batch in enumerate(train_loader):
            d_img = batch["deg_rgb"].to(device, non_blocking=True)
            c_img = batch["clean_rgb"].to(device, non_blocking=True)
            wt = batch["wt"].to(device, non_blocking=True)
            dp = batch["depth"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(d_img, wt, dp)
                pred_img = out["restored_image"]
                grid = out.get("affine_grid", None)
                loss, loss_dict = criterion(pred_img, c_img, grid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_accum += loss.item()
            if (b_idx + 1) % 100 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Ep {ep:02d} [{b_idx+1:03d}/{len(train_loader):03d}] Loss: {loss.item():.4f} (charb: {loss_dict['loss_charb']:.4f}, ssim: {loss_dict['loss_ssim']:.4f})", flush=True)

        scheduler.step()
        train_loss = loss_accum / len(train_loader)
        ep_duration = time.time() - t_ep0

        # Evaluate on all 800 held-out images
        t_v0 = time.time()
        rest_metrics = evaluate_restoration_full(
            model=model,
            val_cache=val_cache,
            device=device,
            batch_size=8,
            save_samples_dir=f"reports/v25/visuals/{model_type}_fold_{fold}_ep{ep}" if ep == epochs else None
        )
        val_time = time.time() - t_v0

        q_fr = rest_metrics["Q_FR_dataset"]
        psnr_val = rest_metrics["PSNR_dB"]
        ssim_val = rest_metrics["SSIM"]
        sharp_val = rest_metrics["sharpness_ratio"]

        is_best = q_fr > best_q_fr
        if is_best:
            best_q_fr = q_fr
            best_epoch = ep
            best_rest_metrics = rest_metrics
            # Atomic save best checkpoint
            torch.save({
                "model_type": model_type,
                "fold": fold,
                "epoch": ep,
                "state_dict": model.state_dict(),
                "metrics": rest_metrics,
                "val_scenes": val_scenes,
                "train_scenes": train_scenes,
                "total_params_m": total_params
            }, best_ckpt_path)

        star = " [*BEST*]" if is_best else ""
        print(
            f"Epoch [{ep:02d}/{epochs:02d}] ({ep_duration:.1f}s train, {val_time:.1f}s val) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val PSNR: {psnr_val:.2f} dB | "
            f"SSIM: {ssim_val:.4f} | "
            f"Sharp: {sharp_val:.4f} | "
            f"Q_FR: {q_fr:.4f}{star}",
            flush=True
        )

    total_train_time = time.time() - t_train_start
    print(f"\nTraining completed in {total_train_time:.1f}s. Best Epoch: {best_epoch} (Q_FR: {best_q_fr:.4f})")

    # Load best checkpoint for full downstream evaluation with frozen V18 segmenter
    print(f"\nLoading best checkpoint for downstream joint evaluation...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    # Load V18 Segmenter
    seg_ckpt_path = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
    segmenter = build_v18_segmenter(num_classes=5).to(device)
    seg_ckpt = torch.load(seg_ckpt_path, map_location=device)
    segmenter.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))
    segmenter.eval()

    print(f"Running Joint Restoration + Segmentation Evaluation on Fold {fold} (800 held-out images)...")
    joint_results = evaluate_downstream_segmentation(
        restoration_model=model,
        segmenter_model=segmenter,
        val_cache=val_cache,
        device=device,
        batch_size=4
    )

    seg_m = joint_results["segmentation"]
    rest_m = joint_results["restoration"]
    q_seg = seg_m["Q_seg"]
    q_fr = rest_m["Q_FR_dataset"]
    comp_exact_90 = 0.60 * q_seg + 0.30 * q_fr
    comp_est = comp_exact_90 + 0.10 * 0.737

    # V18 baseline reference values
    v18_ref_comp = 0.4889 if fold == 0 else 0.6445
    delta_comp = comp_est - v18_ref_comp

    print("=" * 80)
    print(f"FINAL EVALUATION REPORT: {model_type} (Fold {fold})")
    print("=" * 80)
    print(f"Restoration:")
    print(f"  PSNR: {rest_m['PSNR_dB']:.2f} dB, SSIM: {rest_m['SSIM']:.4f}, Sharpness: {rest_m['sharpness_ratio']:.4f} -> Q_FR = {q_fr:.4f}")
    print(f"Segmentation:")
    print(f"  mIoU: {seg_m['mIoU']:.4f}, Macro F1: {seg_m['Macro_F1']:.4f}, Coverage: {seg_m['coverage_penalty']} -> Q_seg = {q_seg:.4f}")
    print(f"Composite Scores:")
    print(f"  Exact 90% (0.60*Q_seg + 0.30*Q_FR): {comp_exact_90:.4f}")
    print(f"  Composite Est (with Q_NR=0.737):     {comp_est:.4f} (Delta vs V18: {delta_comp:+.4f})")
    print("=" * 80)

    # Log to CSV
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": f"{model_type}_fold_{fold}",
        "model_type": model_type,
        "fold": fold,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_loss": round(train_loss, 4),
        "val_psnr_db": rest_m["PSNR_dB"],
        "val_ssim": rest_m["SSIM"],
        "val_sharpness": rest_m["sharpness_ratio"],
        "val_q_fr": q_fr,
        "val_miou": seg_m["mIoU"],
        "val_macro_f1": seg_m["Macro_F1"],
        "val_q_seg": q_seg,
        "val_comp_exact_90": round(comp_exact_90, 4),
        "val_comp_est": round(comp_est, 4),
        "delta_vs_v18_comp": round(delta_comp, 4),
        "checkpoint_path": best_ckpt_path
    }
    log_experiment_result("reports/v25/experiment_registry.csv", row)

    # Save detailed JSON
    json_path = f"reports/v25/{model_type}_fold_{fold}_metrics.json"
    with open(json_path, "w") as fp:
        json.dump(joint_results, fp, indent=2)
    print(f"Saved metrics JSON to {json_path}")

    # Re-verify baseline
    verify_baseline()
    return joint_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="R1", choices=["R1", "R2"])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train_candidate(
        model_type=args.model_type,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
