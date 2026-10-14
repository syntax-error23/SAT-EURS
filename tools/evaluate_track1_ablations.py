"""
Track 1 Inference Pipeline & Controlled Ablation Benchmark (Exp-03, Exp-04)
Evaluates:
1. Restoration methods on validation images:
   - EURS-Net V5 Restorer
   - RSUIGM GPU Physical Inversion
   - Hybrid Blend
2. Segmentation Models (fed with their optimal input domains):
   - SegFormer MiT-B3 (Teacher)
   - DeepLabV3-ResNet101 (Teacher)
   - EURSNet-V4
   - EURSNet-V5 Student (calibrated vs uncalibrated)
3. TTA Ablations: Single-pass vs Horizontal Flip
4. Ensemble Blending Optimization:
   - Grid search over model weights
   - Verifies Q_seg and full Composite Score
"""

import os
import os.path as osp
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.models.segmentation as sm
import segmentation_models_pytorch as smp

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.eurs_net_v5 import EURSNetV5
from models.eurs_net_v4 import EURSNetV4
from deploy.export_v5_and_track1 import fit_or_load_rsuigm_wt_depth_table_gpu, build_rsuigm_poly_features_gpu
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def run_track1_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    data_clean_dir = "data/cached_1024/clean"
    data_masks_dir = "data/cached_1024/masks"
    data_deg_dir = "data/cached_1024/degraded"

    print("=" * 72)
    print("   RUNNING TRACK 1 CONTROLLED EXPERIMENT ABLATIONS (EXP-03 & EXP-04)")
    print("=" * 72)

    # Load models
    print("[1/5] Loading EURS-Net V5 Calibrated Student...", flush=True)
    model_v5 = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt_v5 = torch.load("checkpoints/v5_calibrated_track2_best.pt", map_location=device, weights_only=False)
    model_v5.load_state_dict(ckpt_v5.get("model_state_dict", ckpt_v5), strict=True)
    model_v5.eval()

    print("[2/5] Loading EURS-Net V4 Full20...", flush=True)
    model_v4 = EURSNetV4(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt_v4 = torch.load("checkpoints/v4_full20/best_v4_checkpoint.pt", map_location=device, weights_only=False)
    model_v4.load_state_dict(ckpt_v4.get("model_state_dict", ckpt_v4), strict=False)
    model_v4.eval()

    print("[3/5] Loading DeepLabV3-ResNet101...", flush=True)
    model_dl = sm.deeplabv3_resnet101(weights=None, num_classes=5, aux_loss=True).to(device)
    ckpt_dl = torch.load("checkpoints/deeplabv3_resnet101_best.pt", map_location=device, weights_only=False)
    model_dl.load_state_dict(ckpt_dl.get("model_state_dict", ckpt_dl))
    model_dl.eval()

    print("[4/5] Loading SegFormer MiT-B3...", flush=True)
    model_sf = smp.DeepLabV3Plus(encoder_name="mit_b3", encoder_weights=None, classes=5, in_channels=3).to(device)
    sf_ckpt = "checkpoints/segformer_mit_b3_epoch_12.pt" if osp.exists("checkpoints/segformer_mit_b3_epoch_12.pt") else "checkpoints/segformer_mit_b3_best.pt"
    ckpt_sf = torch.load(sf_ckpt, map_location=device, weights_only=False)
    model_sf.load_state_dict(ckpt_sf.get("model_state_dict", ckpt_sf))
    model_sf.eval()

    print("[5/5] Loading RSUIGM Physical Inversion Table...", flush=True)
    rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

    # Mean & Std for ImageNet normalization
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Collect validation data
    val_items = []
    for sc in val_scenes:
        clean_p = osp.join(data_clean_dir, f"{sc}.png")
        mask_p = osp.join(data_masks_dir, f"{sc}.png")
        deg_folder = osp.join(data_deg_dir, sc)

        with Image.open(clean_p) as c_im:
            clean_np = np.asarray(c_im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mask_p) as m_im:
            mask_np = np.asarray(m_im, dtype=np.int64)

        deg_files = sorted([f for f in os.listdir(deg_folder) if f.endswith(".png")])
        sampled = deg_files[::10]  # 20 per scene = 80 total

        for df in sampled:
            # Parse water type and depth from filename if available, else default (1, 10)
            stem = osp.splitext(df)[0]
            parts = stem.split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10

            deg_p = osp.join(deg_folder, df)
            with Image.open(deg_p) as d_im:
                deg_np = np.asarray(d_im.convert("RGB"), dtype=np.float32) / 255.0

            val_items.append({
                "scene": sc,
                "stem": stem,
                "wt": wt,
                "depth": depth,
                "deg_np": deg_np,
                "clean_np": clean_np,
                "mask_np": mask_np,
            })

    print(f"Loaded {len(val_items)} validation samples.\n")

    # -------------------------------------------------------------
    # PART 1: RESTORATION BENCHMARK
    # -------------------------------------------------------------
    print("-" * 72)
    print("   PART 1: RESTORATION ABLATIONS")
    print("-" * 72)

    psnr_v5, ssim_v5, sharp_v5 = [], [], []
    psnr_phys, ssim_phys, sharp_phys = [], [], []
    psnr_hybrid, ssim_hybrid, sharp_hybrid = [], [], []

    # Store tensors for segmentation
    cached_tensors = []

    for item in val_items:
        clean_np = item["clean_np"]
        deg_tensor = torch.from_numpy(item["deg_np"].transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            # 1. EURS-Net V5 restoration
            rest_v5, logits_v5 = model_v5(deg_tensor)

            # 2. RSUIGM Physical Inversion
            W = rsuigm_table.get((item["wt"], item["depth"]), rsuigm_table[(1, 10)])
            feats = build_rsuigm_poly_features_gpu(deg_tensor)
            rest_phys = torch.clamp(torch.matmul(feats, W).reshape(1, 1024, 1024, 3).permute(0, 3, 1, 2), 0.0, 1.0)

            # 3. Hybrid blend
            alpha = 0.85 if item["depth"] <= 60 else 0.55
            rest_hybrid = torch.clamp(alpha * rest_phys + (1.0 - alpha) * rest_v5, 0.0, 1.0)

        # Compute metrics
        r_v5_np = rest_v5[0].cpu().numpy().transpose(1, 2, 0)
        p, _ = compute_psnr_and_norm(r_v5_np, clean_np)
        psnr_v5.append(p)
        ssim_v5.append(compute_ssim_single(r_v5_np, clean_np))
        sharp_v5.append(compute_sharpness_ratio(r_v5_np, clean_np))

        r_phys_np = rest_phys[0].cpu().numpy().transpose(1, 2, 0)
        p, _ = compute_psnr_and_norm(r_phys_np, clean_np)
        psnr_phys.append(p)
        ssim_phys.append(compute_ssim_single(r_phys_np, clean_np))
        sharp_phys.append(compute_sharpness_ratio(r_phys_np, clean_np))

        r_hyb_np = rest_hybrid[0].cpu().numpy().transpose(1, 2, 0)
        p, _ = compute_psnr_and_norm(r_hyb_np, clean_np)
        psnr_hybrid.append(p)
        ssim_hybrid.append(compute_ssim_single(r_hyb_np, clean_np))
        sharp_hybrid.append(compute_sharpness_ratio(r_hyb_np, clean_np))

        cached_tensors.append({
            "deg_tensor": deg_tensor,
            "rest_phys": rest_phys,
            "rest_hybrid": rest_hybrid,
            "mask_tensor": torch.from_numpy(item["mask_np"]).unsqueeze(0).to(device)
        })

    def summarize_rest(name, psnrs, ssims, sharps):
        m_p = float(np.mean(psnrs))
        m_s = float(np.mean(ssims))
        m_sh = float(np.mean(sharps))
        p_n = min(1.0, max(0.0, (m_p - 15.0) / 20.0))
        q_fr = 0.45 * p_n + 0.45 * m_s + 0.10 * m_sh
        print(f"  {name:<25}: PSNR = {m_p:6.2f} dB | SSIM = {m_s:.4f} | Sharpness = {m_sh:.4f} | Q_FR = {q_fr:.4f}")
        return m_p, m_s, m_sh, q_fr

    res_v5 = summarize_rest("EURS-Net V5 Restorer", psnr_v5, ssim_v5, sharp_v5)
    res_phys = summarize_rest("RSUIGM Physical Inv", psnr_phys, ssim_phys, sharp_phys)
    res_hyb = summarize_rest("Hybrid Physical+V5", psnr_hybrid, ssim_hybrid, sharp_hybrid)

    # -------------------------------------------------------------
    # PART 2: SEGMENTATION & TTA ABLATIONS
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("   PART 2: SEGMENTATION MODELS & TTA ABLATIONS")
    print("-" * 72)

    # Precompute model predictions across all samples
    preds_sf_notta, preds_sf_tta = [], []
    preds_dl_notta, preds_dl_tta = [], []
    preds_v4_notta, preds_v4_tta = [], []
    preds_v5_notta, preds_v5_tta = [], []
    targets = []

    for item in cached_tensors:
        x_raw = item["deg_tensor"]
        x_rest = item["rest_phys"]
        target = item["mask_tensor"]
        targets.append(target.cpu())

        # Normalize restored image for SF and DL
        x_rest_norm = (x_rest - mean_im) / std_im

        with torch.no_grad():
            # 1. SegFormer
            out_sf = model_sf(x_rest_norm)
            out_sf_flip = model_sf(torch.flip(x_rest_norm, dims=[3]))
            out_sf_tta = 0.5 * (out_sf + torch.flip(out_sf_flip, dims=[3]))
            preds_sf_notta.append(out_sf.cpu())
            preds_sf_tta.append(out_sf_tta.cpu())

            # 2. DeepLabV3
            out_dl = model_dl(x_rest)["out"]
            out_dl_flip = model_dl(torch.flip(x_rest, dims=[3]))["out"]
            out_dl_tta = 0.5 * (out_dl + torch.flip(out_dl_flip, dims=[3]))
            preds_dl_notta.append(out_dl.cpu())
            preds_dl_tta.append(out_dl_tta.cpu())

            # 3. EURS-Net V4
            _, out_v4 = model_v4(x_raw)
            _, out_v4_flip = model_v4(torch.flip(x_raw, dims=[3]))
            out_v4_tta = 0.5 * (out_v4 + torch.flip(out_v4_flip, dims=[3]))
            preds_v4_notta.append(out_v4.cpu())
            preds_v4_tta.append(out_v4_tta.cpu())

            # 4. EURS-Net V5 (Calibrated)
            _, out_v5 = model_v5(x_raw)
            _, out_v5_flip = model_v5(torch.flip(x_raw, dims=[3]))
            out_v5_tta = 0.5 * (out_v5 + torch.flip(out_v5_flip, dims=[3]))
            preds_v5_notta.append(out_v5.cpu())
            preds_v5_tta.append(out_v5_tta.cpu())

    evaluator = SegmentationEvaluator(num_classes=5)

    def eval_logits(logits_list, name):
        evaluator.reset()
        for log, tgt in zip(logits_list, targets):
            evaluator.update(log, tgt)
        res = evaluator.compute()
        print(f"  {name:<30}: mIoU = {res['mIoU']:.4f} | F1 = {res['macro_F1']:.4f} | Q_seg = {res['Q_seg']:.4f}")
        return res

    print("\n[A] Single Models (No-TTA vs H-Flip TTA):")
    eval_logits(preds_sf_notta, "SegFormer MiT-B3 (No TTA)")
    r_sf_tta = eval_logits(preds_sf_tta, "SegFormer MiT-B3 (H-Flip TTA)")

    eval_logits(preds_dl_notta, "DeepLabV3-ResNet101 (No TTA)")
    r_dl_tta = eval_logits(preds_dl_tta, "DeepLabV3-ResNet101 (H-Flip TTA)")

    eval_logits(preds_v4_notta, "EURS-Net V4 (No TTA)")
    r_v4_tta = eval_logits(preds_v4_tta, "EURS-Net V4 (H-Flip TTA)")

    eval_logits(preds_v5_notta, "EURS-Net V5 Calibrated (No TTA)")
    r_v5_tta = eval_logits(preds_v5_tta, "EURS-Net V5 Calibrated (H-Flip TTA)")

    # -------------------------------------------------------------
    # PART 3: ENSEMBLE WEIGHT OPTIMIZATION
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("   PART 3: ENSEMBLE COMBINATION SEARCH (WITH TTA)")
    print("-" * 72)

    # Test candidate ensembles
    candidate_weights = [
        ("SF 50% + DL 50%", [0.50, 0.50, 0.00, 0.00]),
        ("SF 40% + DL 35% + V4 15% + V5 10% (Baseline #18668)", [0.40, 0.35, 0.15, 0.10]),
        ("SF 40% + DL 30% + V4 15% + V5 15%", [0.40, 0.30, 0.15, 0.15]),
        ("SF 30% + DL 30% + V4 20% + V5 20%", [0.30, 0.30, 0.20, 0.20]),
        ("SF 25% + DL 25% + V4 25% + V5 25% (Equal)", [0.25, 0.25, 0.25, 0.25]),
        ("SF 30% + DL 20% + V4 10% + V5 40% (Student Heavy)", [0.30, 0.20, 0.10, 0.40]),
        ("SF 20% + DL 10% + V4 10% + V5 60% (Calibrated V5 Focus)", [0.20, 0.10, 0.10, 0.60]),
        ("SF 10% + DL 10% + V4 10% + V5 70%", [0.10, 0.10, 0.10, 0.70]),
    ]

    best_q_seg = 0.0
    best_ens_name = ""
    best_ens_res = None

    for name, (w_sf, w_dl, w_v4, w_v5) in candidate_weights:
        ens_logits = []
        for l_sf, l_dl, l_v4, l_v5 in zip(preds_sf_tta, preds_dl_tta, preds_v4_tta, preds_v5_tta):
            l_ens = w_sf * l_sf + w_dl * l_dl + w_v4 * l_v4 + w_v5 * l_v5
            ens_logits.append(l_ens)
        res = eval_logits(ens_logits, name)
        if res["Q_seg"] > best_q_seg:
            best_q_seg = res["Q_seg"]
            best_ens_name = name
            best_ens_res = res

    print("\n" + "=" * 72)
    print(f"Optimal Track 1 Ensemble: {best_ens_name}")
    print(f"Optimal Q_seg: {best_q_seg:.4f} (mIoU = {best_ens_res['mIoU']:.4f}, Macro F1 = {best_ens_res['macro_F1']:.4f})")
    
    # Calculate Track 1 Composite with RSUIGM Physical Inversion
    # Q_FR from Physical Inversion = 0.6443 (or measured 0.640+)
    q_fr = res_phys[3]
    # Assumed Q_NR baseline from leaderboard = 0.441
    q_nr = 0.441
    track1_comp = 0.60 * best_q_seg + 0.30 * q_fr + 0.10 * q_nr
    print(f"Projected Track 1 Composite Score: {track1_comp:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    run_track1_ablations()
