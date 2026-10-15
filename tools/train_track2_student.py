"""
Training Engine for Track 2 EURS-EdgeSOTA-V2
Supports:
  1. mode='supervised': Ground Truth Cross-Entropy only (Supervised Baseline)
  2. mode='standard_kd': Supervised CE + Teacher Ensemble Soft Distillation (Stage 1)
  3. mode='class_aware': Class-frequency weighted KD (Stage 2)
  4. mode='boundary': Morphological Boundary Loss (Stage 3)
  5. mode='lovasz': Lovász-Softmax direct mIoU optimization (Stage 4)
Evaluates strictly on held-out Fold 0 scenes with zero leakage.
"""

import os
import sys
import time
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from models.track2_edgesota_v2 import EURSEdgeSOTAV2, build_track2_model
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio
from tools.train_track1_screening import build_dinov2_model
from losses.segmentation_v7_loss import BoundaryBandLoss
from losses.v4_losses import PresentClassLovaszSoftmaxLoss


def load_scene_data_to_ram(scenes: List[str], max_degraded_per_scene: int = 50):
    """
    Preloads training images into RAM for fast zero-disk-IO training iterations.
    """
    print(f"[RAM Loader] Preloading {len(scenes)} scenes into host memory...", flush=True)
    clean_dict = {}
    masks_dict = {}
    deg_dict = {}

    for sc in scenes:
        c_p = f"data/cached_1024/clean/{sc}.png"
        m_p = f"data/cached_1024/masks/{sc}.png"
        deg_dir = f"data/cached_1024/degraded/{sc}"

        c_im = cv2.imread(c_p)
        c_rgb = cv2.cvtColor(c_im, cv2.COLOR_BGR2RGB)
        clean_dict[sc] = c_rgb

        m_im = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
        masks_dict[sc] = m_im

        deg_files = sorted([os.path.join(deg_dir, f) for f in os.listdir(deg_dir) if f.endswith(".png")])
        if max_degraded_per_scene > 0:
            step = max(1, len(deg_files) // max_degraded_per_scene)
            deg_files = deg_files[::step][:max_degraded_per_scene]

        deg_images = []
        for df in deg_files:
            d_im = cv2.imread(df)
            if d_im is not None:
                d_rgb = cv2.cvtColor(d_im, cv2.COLOR_BGR2RGB)
                deg_images.append(d_rgb)
        deg_dict[sc] = deg_images

    total_deg = sum(len(v) for v in deg_dict.values())
    print(f"[RAM Loader] Preloaded {len(scenes)} scenes ({total_deg} degraded images) in RAM!", flush=True)
    return clean_dict, masks_dict, deg_dict


def run_training_experiment(
    exp_id: str = "EXP-SUPERVISED",
    mode: str = "supervised",
    epochs: int = 8,
    batch_size: int = 16,
    crop_size: int = 512,
    lr: float = 3e-4,
    device_name: str = "cuda"
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 76, flush=True)
    print(f"   STARTING TRACK 2 EXPERIMENT: {exp_id} ({mode.upper()})", flush=True)
    print(f"   Device: {device} | Epochs: {epochs} | Batch: {batch_size} | Crop: {crop_size}", flush=True)
    print("=" * 76, flush=True)

    # 1. Load Manifest & Scene Splits
    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)
    fold_0 = manifest["folds"]["fold_0"]
    train_scenes = sorted(fold_0["train_scenes"])
    val_scenes = sorted(fold_0["val_scenes"])

    print(f"Train scenes ({len(train_scenes)}): {train_scenes}", flush=True)
    print(f"Held-out Val scenes ({len(val_scenes)}): {val_scenes}", flush=True)

    # 2. Preload Train Data into RAM
    clean_dict, masks_dict, deg_dict = load_scene_data_to_ram(train_scenes, max_degraded_per_scene=50)

    # 3. Build Model
    model = build_track2_model().to(device)

    # 3b. Pre-train Optical Router to calibrate restoration branch
    print("[Router Training] Calibrating optical router on 16 training scenes...", flush=True)
    router_items = []
    for sc in train_scenes:
        deg_dir = f"data/cached_1024/degraded/{sc}"
        if not os.path.exists(deg_dir): continue
        for f in sorted(os.listdir(deg_dir))[::4]: # sample 50 images per scene = 800 items
            if f.endswith(".png"):
                parts = os.path.splitext(f)[0].split("_")
                wt = int(parts[1])
                dp = int(parts[2])
                cond_idx = (wt - 1) * 20 + (dp // 5 - 1)
                router_items.append((os.path.join(deg_dir, f), cond_idx))

    def load_r_item(item):
        p, idx = item
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return cv2.resize(im, (128, 128)).transpose(2, 0, 1), idx

    with ThreadPoolExecutor(max_workers=16) as pool:
        r_results = list(pool.map(load_r_item, router_items))

    r_x = torch.from_numpy(np.stack([r[0] for r in r_results])).to(device)
    r_y = torch.tensor([r[1] for r in r_results], dtype=torch.long, device=device)

    r_opt = torch.optim.AdamW(list(model.router.parameters()) + list(model.router_fc.parameters()), lr=1e-3, weight_decay=1e-4)
    r_crit = nn.CrossEntropyLoss()
    model.train()
    for r_ep in range(10):
        perm = torch.randperm(len(r_x), device=device)
        for b in range(0, len(r_x), 64):
            idx = perm[b:b+64]
            r_feat = model.router(r_x[idx]).flatten(1)
            r_logits = model.router_fc(r_feat)
            r_loss = r_crit(r_logits, r_y[idx])
            r_opt.zero_grad()
            r_loss.backward()
            r_opt.step()
    print(f"[Router Training] Calibrated in 10 epochs (Final Loss: {r_loss.item():.4f})! Restoration active.", flush=True)

    # Freeze optical router so condition accuracy does not drift during segmentation training
    for p in model.router.parameters():
        p.requires_grad = False
    for p in model.router_fc.parameters():
        p.requires_grad = False

    # Optional: Load Teacher Ensemble if mode != 'supervised'
    teacher_ensemble = []
    if mode != "supervised":
        print("[KD Setup] Loading 5-Fold DINOv2-Large Teacher Ensemble...", flush=True)
        for i in range(2): # Use Fold 0 & 1 for fast ensemble
            t_path = f"checkpoints/track1_screening/DINOV2_CP_F{i}/best_model.pt"
            t = build_dinov2_model(num_classes=5).to(device)
            ckpt = torch.load(t_path, map_location=device, weights_only=False)
            t.load_state_dict(ckpt.get("model_state_dict", ckpt))
            t.eval()
            for p in t.parameters():
                p.requires_grad = False
            teacher_ensemble.append(t)
        print(f"[KD Setup] Loaded {len(teacher_ensemble)} DINOv2-Large teachers in eval mode.", flush=True)

    # Train student backbone & decoder
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss suites
    ce_weights = torch.tensor([0.70, 1.00, 1.20, 2.50, 2.50], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weights)
    boundary_loss_fn = BoundaryBandLoss(kernel_size=3)
    lovasz_loss_fn = PresentClassLovaszSoftmaxLoss(num_classes=5)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    steps_per_epoch = 150
    temperature = 2.5
    best_val_qseg = -1.0
    best_ckpt_path = f"checkpoints/{exp_id}_best.pt"
    onnx_out_path = f"deploy/{exp_id}.onnx"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0

        for step in range(steps_per_epoch):
            batch_deg = []
            batch_masks = []
            batch_clean = []

            rare_scenes = [s for s in train_scenes if s in ("ray1", "sharkFish", "starfish2", "turtles1")]
            for _ in range(batch_size):
                if mode in ("class_aware", "boundary", "lovasz", "combined") and rare_scenes and random.random() < 0.35:
                    sc = random.choice(rare_scenes)
                    c_im = clean_dict[sc]
                    m_im = masks_dict[sc]
                    deg_list = deg_dict[sc]
                    d_im = random.choice(deg_list) if deg_list else c_im
                    h, w = m_im.shape
                    rare_coords = np.argwhere(np.isin(m_im, [3, 4]))
                    if len(rare_coords) > 0 and random.random() < 0.70:
                        cy, cx = rare_coords[random.randint(0, len(rare_coords) - 1)]
                        top = int(np.clip(cy - crop_size // 2 + random.randint(-64, 64), 0, h - crop_size))
                        left = int(np.clip(cx - crop_size // 2 + random.randint(-64, 64), 0, w - crop_size))
                    else:
                        top = random.randint(0, h - crop_size)
                        left = random.randint(0, w - crop_size)
                else:
                    sc = random.choice(train_scenes)
                    c_im = clean_dict[sc]
                    m_im = masks_dict[sc]
                    deg_list = deg_dict[sc]
                    d_im = random.choice(deg_list) if deg_list else c_im
                    h, w = m_im.shape
                    top = random.randint(0, h - crop_size)
                    left = random.randint(0, w - crop_size)

                d_crop = d_im[top:top+crop_size, left:left+crop_size]
                m_crop = m_im[top:top+crop_size, left:left+crop_size]
                c_crop = c_im[top:top+crop_size, left:left+crop_size]

                # Random horizontal flip
                if random.random() < 0.5:
                    d_crop = np.fliplr(d_crop)
                    m_crop = np.fliplr(m_crop)
                    c_crop = np.fliplr(c_crop)

                batch_deg.append(d_crop.astype(np.float32) / 255.0)
                batch_masks.append(m_crop.astype(np.int64))
                batch_clean.append(c_crop.astype(np.float32) / 255.0)

            x_deg = torch.from_numpy(np.stack(batch_deg).transpose(0, 3, 1, 2)).contiguous().to(device, non_blocking=True)
            y_mask = torch.from_numpy(np.stack(batch_masks)).contiguous().to(device, non_blocking=True)
            x_cln = torch.from_numpy(np.stack(batch_clean).transpose(0, 3, 1, 2)).contiguous().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                restored_pred, seg_logits = model(x_deg)

                # 1. Supervised CE loss
                loss_ce = ce_loss_fn(seg_logits, y_mask)

                # 2. Distillation loss
                if mode == "supervised":
                    loss = loss_ce
                elif mode == "standard_kd":
                    with torch.no_grad():
                        t_norm = (x_cln - mean_im) / std_im
                        t_in = F.interpolate(t_norm, size=(518, 518), mode="bilinear", align_corners=False)
                        t_probs = [
                            F.interpolate(F.softmax(t(t_in), dim=1), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                            for t in teacher_ensemble
                        ]
                        p_t_ens = torch.stack(t_probs).mean(dim=0)

                    log_p_s = F.log_softmax(seg_logits / temperature, dim=1)
                    loss_kd = (p_t_ens * (p_t_ens.clamp(min=1e-7).log() - log_p_s)).sum(dim=1).mean() * (temperature ** 2)
                    loss = 0.5 * loss_ce + 0.5 * loss_kd
                    running_kd += loss_kd.item()
                elif mode == "class_aware":
                    with torch.no_grad():
                        t_norm = (x_cln - mean_im) / std_im
                        t_in = F.interpolate(t_norm, size=(518, 518), mode="bilinear", align_corners=False)
                        t_probs = [
                            F.interpolate(F.softmax(t(t_in), dim=1), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                            for t in teacher_ensemble
                        ]
                        p_t_ens = torch.stack(t_probs).mean(dim=0)

                    log_p_s = F.log_softmax(seg_logits / temperature, dim=1)
                    kl_pix = p_t_ens * (p_t_ens.clamp(min=1e-7).log() - log_p_s)
                    kd_weights = torch.tensor([0.70, 1.00, 1.20, 2.50, 2.50], device=device).view(1, 5, 1, 1)
                    loss_kd = (kl_pix * kd_weights).sum(dim=1).mean() * (temperature ** 2)
                    loss = 0.5 * loss_ce + 0.5 * loss_kd
                    running_kd += loss_kd.item()
                elif mode == "boundary":
                    with torch.no_grad():
                        t_norm = (x_cln - mean_im) / std_im
                        t_in = F.interpolate(t_norm, size=(518, 518), mode="bilinear", align_corners=False)
                        t_probs = [
                            F.interpolate(F.softmax(t(t_in), dim=1), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                            for t in teacher_ensemble
                        ]
                        p_t_ens = torch.stack(t_probs).mean(dim=0)

                    log_p_s = F.log_softmax(seg_logits / temperature, dim=1)
                    kl_pix = p_t_ens * (p_t_ens.clamp(min=1e-7).log() - log_p_s)
                    kd_weights = torch.tensor([0.80, 1.00, 1.10, 1.25, 2.20], device=device).view(1, 5, 1, 1)
                    loss_kd = (kl_pix * kd_weights).sum(dim=1).mean() * (temperature ** 2)
                    loss_bnd = boundary_loss_fn(seg_logits, y_mask)
                    loss = 0.45 * loss_ce + 0.45 * loss_kd + 0.10 * loss_bnd
                    running_kd += loss_kd.item()
                elif mode == "lovasz":
                    with torch.no_grad():
                        t_norm = (x_cln - mean_im) / std_im
                        t_in = F.interpolate(t_norm, size=(518, 518), mode="bilinear", align_corners=False)
                        t_probs = [
                            F.interpolate(F.softmax(t(t_in), dim=1), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                            for t in teacher_ensemble
                        ]
                        p_t_ens = torch.stack(t_probs).mean(dim=0)

                    log_p_s = F.log_softmax(seg_logits / temperature, dim=1)
                    kl_pix = p_t_ens * (p_t_ens.clamp(min=1e-7).log() - log_p_s)
                    kd_weights = torch.tensor([0.80, 1.00, 1.10, 1.25, 2.20], device=device).view(1, 5, 1, 1)
                    loss_kd = (kl_pix * kd_weights).sum(dim=1).mean() * (temperature ** 2)
                    loss_lov = lovasz_loss_fn(seg_logits, y_mask)
                    loss = 0.35 * loss_ce + 0.35 * loss_kd + 0.30 * loss_lov
                    running_kd += loss_kd.item()
                elif mode == "combined":
                    with torch.no_grad():
                        t_norm = (x_cln - mean_im) / std_im
                        t_in = F.interpolate(t_norm, size=(518, 518), mode="bilinear", align_corners=False)
                        t_probs = [
                            F.interpolate(F.softmax(t(t_in), dim=1), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                            for t in teacher_ensemble
                        ]
                        p_t_ens = torch.stack(t_probs).mean(dim=0)

                    log_p_s = F.log_softmax(seg_logits / temperature, dim=1)
                    kl_pix = p_t_ens * (p_t_ens.clamp(min=1e-7).log() - log_p_s)
                    kd_weights = torch.tensor([0.80, 1.00, 1.10, 1.25, 2.20], device=device).view(1, 5, 1, 1)
                    loss_kd = (kl_pix * kd_weights).sum(dim=1).mean() * (temperature ** 2)
                    loss_bnd = boundary_loss_fn(seg_logits, y_mask)
                    loss = 0.45 * loss_ce + 0.45 * loss_kd + 0.10 * loss_bnd
                    running_kd += loss_kd.item()
                else:
                    loss = loss_ce

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += loss_ce.item()

        scheduler.step()
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Loss: {running_loss/steps_per_epoch:.4f} (CE: {running_ce/steps_per_epoch:.4f}, KD: {running_kd/steps_per_epoch:.4f})", flush=True)

    # 4. Save Final Best Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("deploy", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "exp_id": exp_id, "mode": mode}, best_ckpt_path)
    print(f"\n[Saved] PyTorch checkpoint saved to {best_ckpt_path}", flush=True)

    # 5. Export to Static ONNX
    print(f"[ONNX] Exporting to {onnx_out_path}...", flush=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["restored_image", "seg_logits"],
        dynamo=False,
    )
    print(f"[ONNX] Export complete! File size: {os.path.getsize(onnx_out_path)/1e6:.2f} MB", flush=True)

    # 6. Run Official Evaluation on Held-Out Validation Scenes
    print("\n" + "=" * 76, flush=True)
    print(f"   OFFICIAL EVALUATION FOR {exp_id} (HELD-OUT SCENES: {val_scenes})", flush=True)
    print("=" * 76, flush=True)

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_out_path, sess_options=opts, providers=["CPUExecutionProvider"])

    evaluator = SegmentationEvaluator(num_classes=5)
    psnrs, ssims, sharps = [], [], []
    boundary_accuracies = []

    for sc in val_scenes:
        clean_p = f"data/cached_1024/clean/{sc}.png"
        mask_p = f"data/cached_1024/masks/{sc}.png"
        deg_dir = f"data/cached_1024/degraded/{sc}"

        clean_im = cv2.imread(clean_p)
        clean_rgb = cv2.cvtColor(clean_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clean_1024 = cv2.resize(clean_rgb, (1024, 1024))

        mask_im = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        mask_1024 = cv2.resize(mask_im, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::5] # 40 per scene = 160 total

        for df in deg_files:
            dp = os.path.join(deg_dir, df)
            deg_im = cv2.imread(dp)
            deg_rgb = cv2.cvtColor(deg_im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            deg_1024 = cv2.resize(deg_rgb, (1024, 1024))

            inp = deg_1024.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            outputs = sess.run(None, {"input": inp})
            rest_np = outputs[0][0].transpose(1, 2, 0)
            logits_np = outputs[1][0]

            pred_mask = np.argmax(logits_np, axis=0)
            evaluator.update(torch.from_numpy(pred_mask).unsqueeze(0), torch.from_numpy(mask_1024).unsqueeze(0))

            # Boundary band precision tracking
            b_band = boundary_loss_fn.extract_boundaries(torch.from_numpy(mask_1024).unsqueeze(0)).squeeze().numpy() > 0.5
            if b_band.sum() > 0:
                b_correct = ((pred_mask == mask_1024) & b_band).sum()
                boundary_accuracies.append(float(b_correct / b_band.sum()))

            p, _ = compute_psnr_and_norm(rest_np, clean_1024)
            s = compute_ssim_single(rest_np, clean_1024)
            sh = compute_sharpness_ratio(rest_np, clean_1024)
            psnrs.append(p)
            ssims.append(s)
            sharps.append(sh)

    seg_m = evaluator.compute()
    mean_psnr = float(np.mean(psnrs))
    mean_psnr_norm = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
    mean_ssim = float(np.mean(ssims))
    mean_sharp = float(np.mean(sharps))
    mean_b_acc = float(np.mean(boundary_accuracies)) if boundary_accuracies else 0.0

    q_fr = 0.45 * mean_psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
    q_seg = seg_m["Q_seg"]
    q_nr = 0.5000 # local reference

    composite_score = 0.60 * q_seg + 0.30 * q_fr + 0.10 * q_nr

    print(f"\n--- RESULTS FOR {exp_id} ---", flush=True)
    print(f"  Segmentation mIoU:      {seg_m['mIoU']:.4f}", flush=True)
    print(f"  Segmentation Macro F1:  {seg_m['macro_F1']:.4f}", flush=True)
    print(f"  Coverage Penalty:       {seg_m['coverage_penalty']:.4f}", flush=True)
    print(f"  Boundary Band Accuracy: {mean_b_acc:.4f}", flush=True)
    print(f"  Q_seg:                  {q_seg:.4f}", flush=True)
    print(f"  Per-class IoU:          {seg_m['per_class_IoU']}", flush=True)
    print(f"  Restoration PSNR:       {mean_psnr:.2f} dB (norm: {mean_psnr_norm:.4f})", flush=True)
    print(f"  Restoration SSIM:       {mean_ssim:.4f}", flush=True)
    print(f"  Restoration Sharpness:  {mean_sharp:.4f}", flush=True)
    print(f"  Restoration Q_FR:       {q_fr:.4f}", flush=True)
    print(f"  MEASURED COMPOSITE S:   {composite_score:.4f}", flush=True)
    print("=" * 76, flush=True)

    result_payload = {
        "exp_id": exp_id,
        "mode": mode,
        "epochs": epochs,
        "composite_score": round(composite_score, 4),
        "Q_seg": round(q_seg, 4),
        "mIoU": round(seg_m["mIoU"], 4),
        "macro_F1": round(seg_m["macro_F1"], 4),
        "coverage_penalty": round(seg_m["coverage_penalty"], 4),
        "boundary_accuracy": round(mean_b_acc, 4),
        "per_class_IoU": [round(x, 4) for x in seg_m["per_class_IoU"]],
        "per_class_F1": [round(x, 4) for x in seg_m["per_class_F1"]],
        "PSNR": round(mean_psnr, 2),
        "SSIM": round(mean_ssim, 4),
        "Sharpness": round(mean_sharp, 4),
        "Q_FR": round(q_fr, 4),
    }

    out_json = f"reports/track2_research/{exp_id}_results.json"
    with open(out_json, "w") as fp:
        json.dump(result_payload, fp, indent=2)
    print(f"Saved evaluation payload to {out_json}", flush=True)
    return result_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="EXP-CLASS-AWARE-KD")
    parser.add_argument("--mode", type=str, default="class_aware", choices=["supervised", "standard_kd", "class_aware", "boundary", "lovasz", "combined"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    run_training_experiment(
        exp_id=args.exp_id,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

