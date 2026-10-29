"""
Gate 5 & 6 — Controlled Research Experiment & Comparative Evaluation
Experiment ID: exp_v17_dual_backbone_blend
Hypothesis: DINOv2-Large self-attention features and ConvNeXt-Base FPN multi-scale convolution
features are complementary. Convex blending P_blend = (1 - w)*P_DINO + w*P_ConvNeXt resolves
rare-class spatial boundary false negatives and elevates Q_seg on held-out scenes.

Protocol:
- Strict Scene-Disjoint Fold 0 validation scenes: ['fish1', 'fish11', 'fish12', 'seahorse1']
- Fold 0 DINOv2-Large checkpoint: checkpoints/v16_dinov2/fold_0/best_model.pt
- Fold 0 ConvNeXt-Base checkpoint: checkpoints/v16_convnext/fold_0/best_model.pt
- Grid sweep over blending weight w in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
- Identity control: w = 0.0 (exact standalone DINOv2)
- Full per-class IoU, F1, TP/FP/FN, confusion matrix, Q_seg, Q_FR, and composite tracking.
"""

import os
import os.path as osp
import sys
import time
import json
import psutil
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from tools.train_track1_screening import ControlledTrack1Dataset, build_dinov2_model, apply_rsuigm_batch_gpu
from models.convnext_fpn import build_convnext_fpn
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def run_gate5_experiment():
    print("=" * 80)
    print("   SAT-EURS GATE 5 & 6: CONTROLLED EXPERIMENT & COMPARATIVE EVALUATION")
    print("   Experiment: exp_v17_dual_backbone_blend (DINOv2-L + ConvNeXt-Base FPN)")
    print("   Workstation: NVIDIA RTX 5090 | 64GB DDR5 | Windows 11")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA required for Gate 5 experiment!"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Dataset & Split Setup
    val_scenes = ["fish1", "fish11", "fish12", "seahorse1"]
    print(f"\n[Step 1/5] Setting up Fold 0 Held-Out Dataset: {val_scenes}...")
    val_ds = ControlledTrack1Dataset(
        scenes=val_scenes,
        data_root="data/cached_1024",
        crop_size=518,
        is_train=False,
        use_rsuigm=False
    )
    # Stride of 5 to evaluate 160 representative images across all water types and depths
    eval_indices = list(range(0, len(val_ds), 5))
    print(f"  Total held-out samples: {len(val_ds)} | Sampled evaluation subset: {len(eval_indices)} images")

    # 2. Load Checkpoints
    print("\n[Step 2/5] Loading Model Checkpoints...")
    # DINOv2-Large
    dino_path = "checkpoints/v16_dinov2/fold_0/best_model.pt"
    assert osp.exists(dino_path), f"Missing DINOv2 checkpoint: {dino_path}"
    m_dino = build_dinov2_model(num_classes=5).to(device)
    ckpt_dino = torch.load(dino_path, map_location=device, weights_only=False)
    sd_dino = ckpt_dino["state_dict"] if "state_dict" in ckpt_dino else ckpt_dino["model_state_dict"]
    m_dino.load_state_dict(sd_dino)
    m_dino.eval()
    print(f"  Loaded DINOv2-Large Fold 0 from {dino_path}")

    # ConvNeXt-Base
    conv_path = "checkpoints/v16_convnext/fold_0/best_model.pt"
    assert osp.exists(conv_path), f"Missing ConvNeXt checkpoint: {conv_path}"
    m_conv = build_convnext_fpn(num_classes=5).to(device)
    ckpt_conv = torch.load(conv_path, map_location=device, weights_only=False)
    sd_conv = ckpt_conv["state_dict"] if "state_dict" in ckpt_conv else ckpt_conv["model_state_dict"]
    m_conv.load_state_dict(sd_conv)
    m_conv.eval()
    print(f"  Loaded ConvNeXt-Base FPN Fold 0 from {conv_path}")

    # RSUIGM table
    tbl_path = "checkpoints/track1_screening/tbl19_20sc_clean.pt"
    assert osp.exists(tbl_path), f"Missing RSUIGM table: {tbl_path}"
    rsuigm_table = torch.load(tbl_path, map_location=device)
    print(f"  Loaded RSUIGM 20-scene calibration table ({len(rsuigm_table)} conditions)")

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 3. Setup Evaluators for Candidate Weights
    weights = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    evaluators = {w: SegmentationEvaluator(num_classes=5) for w in weights}

    # Restoration tracking
    psnrs, ssims, sharps = [], [], []
    latencies_dino, latencies_conv, latencies_blend = [], [], []

    clean_dir = "data/cached_1024/clean"
    masks_dir = "data/cached_1024/masks"

    # Pre-cache clean ground truths
    clean_gt = {}
    for sc in val_scenes:
        c_np = np.asarray(Image.open(osp.join(clean_dir, f"{sc}.png")).convert("RGB"), dtype=np.float32) / 255.0
        clean_gt[sc] = c_np

    # 4. Evaluation Loop
    print("\n[Step 3/5] Executing Controlled Evaluation across Blend Weights...")
    torch.cuda.reset_peak_memory_stats()
    t_start_all = time.time()

    for count, idx in enumerate(eval_indices):
        sc, deg_path, wt, depth = val_ds.items[idx]
        m_arr = val_ds.masks[sc]

        deg_bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        deg_t = torch.from_numpy(deg_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Restoration: RSUIGM + Unsharp Mask
        wts_t = torch.tensor([wt], device=device)
        depths_t = torch.tensor([depth], device=device)
        res_t = apply_rsuigm_batch_gpu(deg_t, wts_t, depths_t, rsuigm_table)
        res_np = res_t[0].permute(1, 2, 0).cpu().numpy()
        blur = cv2.GaussianBlur(res_np, (0, 0), sigmaX=1.5)
        unsharp_np = np.clip(res_np + 1.34 * (res_np - blur), 0.0, 1.0)

        # Full reference restoration metrics
        p, _ = compute_psnr_and_norm(unsharp_np, clean_gt[sc])
        s = compute_ssim_single(unsharp_np, clean_gt[sc])
        sh = compute_sharpness_ratio(unsharp_np, clean_gt[sc])
        psnrs.append(p)
        ssims.append(s)
        sharps.append(sh)

        # Model 1: DINOv2-Large Forward (518x518 native ViT patch-14)
        res_518 = F.interpolate(res_t, size=(518, 518), mode="bilinear", align_corners=False)
        inp_dino = (res_518 - mean_im) / std_im

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_dino = m_dino(inp_dino)
        torch.cuda.synchronize()
        latencies_dino.append((time.perf_counter() - t0) * 1000.0)

        # Model 2: ConvNeXt-Base Forward (512x512 native Stride-32 FPN)
        res_512 = F.interpolate(res_t, size=(512, 512), mode="bilinear", align_corners=False)
        inp_conv = (res_512 - mean_im) / std_im

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_conv = m_conv(inp_conv)
        torch.cuda.synchronize()
        latencies_conv.append((time.perf_counter() - t0) * 1000.0)

        # Upsample logits to 1024x1024 and compute Softmax probabilities
        logits_dino_1024 = F.interpolate(logits_dino.float(), size=(1024, 1024), mode="bilinear", align_corners=False)
        logits_conv_1024 = F.interpolate(logits_conv.float(), size=(1024, 1024), mode="bilinear", align_corners=False)

        prob_dino = F.softmax(logits_dino_1024, dim=1) # [1, 5, 1024, 1024]
        prob_conv = F.softmax(logits_conv_1024, dim=1) # [1, 5, 1024, 1024]

        # Blend update
        mask_t = torch.from_numpy(m_arr).unsqueeze(0).to(device)
        for w in weights:
            t_b0 = time.perf_counter()
            prob_blend = (1.0 - w) * prob_dino + w * prob_conv
            pred_mask = torch.argmax(prob_blend[0], dim=0) # [1024, 1024]
            evaluators[w].update(pred_mask.unsqueeze(0), mask_t)
            if w == 0.20:
                latencies_blend.append((time.perf_counter() - t_b0) * 1000.0 + latencies_dino[-1] + latencies_conv[-1])

        if (count + 1) % 40 == 0:
            print(f"  Processed {count + 1}/{len(eval_indices)} samples ({count/len(eval_indices)*100:.0f}%)...", flush=True)

    elapsed_all = time.time() - t_start_all
    print(f"  Evaluation completed in {elapsed_all:.1f}s ({len(eval_indices) / elapsed_all:.1f} samples/sec)")

    # 5. Metric Analysis & Comparative Table
    print("\n" + "=" * 80)
    print("   STAGE 5: COMPARATIVE EVALUATION RESULTS (GATE 6)")
    print("=" * 80)

    mean_psnr = float(np.mean(psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_sharp = float(np.mean(sharps))
    psnr_norm = float(np.clip((mean_psnr - 15.0) / 20.0, 0.0, 1.0))
    q_fr = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp
    q_nr = 0.7417 # Locked server baseline value

    print(f"Restoration Metrics (Held-Out F0): PSNR: {mean_psnr:.2f} dB (norm: {psnr_norm:.4f}) | SSIM: {mean_ssim:.4f} | Sharpness: {mean_sharp:.4f} | Q_FR: {q_fr:.4f}")

    results_table = []
    baseline_qseg = None
    best_w = None
    best_qseg = -1.0
    best_composite = -1.0

    print("\n" + "-" * 115)
    print(f"{'Weight (w)':12s} | {'mIoU':7s} | {'Macro F1':8s} | {'Q_seg':7s} | {'Delta Q_seg':11s} | {'Bg IoU':7s} | {'Coral':7s} | {'Fish':7s} | {'Elas':7s} | {'Other':7s} | {'Composite':9s}")
    print("-" * 115)

    for w in weights:
        m = evaluators[w].compute()
        q_s = m["Q_seg"]
        comp = 0.60 * q_s + 0.30 * q_fr + 0.10 * q_nr

        if w == 0.0:
            baseline_qseg = q_s
            delta_str = "0.0000 (ctrl)"
        else:
            delta = q_s - baseline_qseg
            delta_str = f"{delta:+.4f}"

        per_cls = m["per_class_IoU"]
        bg_iou = per_cls[0] if len(per_cls) > 0 else 0.0
        coral_iou = per_cls[1] if len(per_cls) > 1 else 0.0
        fish_iou = per_cls[2] if len(per_cls) > 2 else 0.0
        elas_iou = per_cls[3] if len(per_cls) > 3 else 0.0
        other_iou = per_cls[4] if len(per_cls) > 4 else 0.0

        print(f"w = {w:.2f}       | {m['mIoU']:.4f} | {m['macro_F1']:.4f}   | {q_s:.4f}  | {delta_str:11s} | {bg_iou:.4f} | {coral_iou:.4f} | {fish_iou:.4f} | {elas_iou:.4f} | {other_iou:.4f} | {comp:.4f}")

        results_table.append({
            "weight": w,
            "mIoU": m["mIoU"],
            "macro_F1": m["macro_F1"],
            "coverage_penalty": m["coverage_penalty"],
            "Q_seg": q_s,
            "delta_Q_seg": q_s - baseline_qseg if w > 0 else 0.0,
            "composite": comp,
            "per_class_iou": per_cls
        })

        if comp > best_composite:
            best_composite = comp
            best_qseg = q_s
            best_w = w

    print("-" * 115)

    # Resource & Latency Report
    peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    peak_vram_res = torch.cuda.max_memory_reserved() / (1024**3)
    current_free, _ = torch.cuda.mem_get_info()
    proc_ram_mb = psutil.Process().memory_info().rss / (1024**2)

    print(f"\nResource & Runtime Profile:")
    print(f"  DINOv2-Large Latency: {np.mean(latencies_dino):.1f} ms ({1000.0/np.mean(latencies_dino):.1f} FPS)")
    print(f"  ConvNeXt-Base Latency: {np.mean(latencies_conv):.1f} ms ({1000.0/np.mean(latencies_conv):.1f} FPS)")
    print(f"  Dual-Backbone Total:  {np.mean(latencies_blend):.1f} ms ({1000.0/np.mean(latencies_blend):.1f} FPS)")
    print(f"  Peak VRAM Allocated:   {peak_vram_alloc:.2f} GiB")
    print(f"  Peak VRAM Reserved:    {peak_vram_res:.2f} GiB (Safe Ceiling: 28.5 GiB)")
    print(f"  Free VRAM Remaining:   {current_free / (1024**3):.2f} GiB")
    print(f"  Process RAM:           {proc_ram_mb:.1f} MB")

    # Decision Matrix
    delta_best = best_qseg - baseline_qseg
    if best_w > 0.0 and delta_best > 0.005:
        decision = "KEEP_DUAL_BACKBONE_BLEND"
        print(f"\nDECISION: KEEP (w = {best_w:.2f} improves Q_seg by {delta_best:+.4f} points, Composite: {best_composite:.4f})")
    elif best_w == 0.0 or delta_best <= 0.0:
        decision = "KEEP_STANDALONE_DINOV2"
        print(f"\nDECISION: REJECT FUSION (Standalone DINOv2 w=0.0 is optimal at Q_seg = {baseline_qseg:.4f})")
    else:
        decision = "RETEST_WITH_CALIBRATION"
        print(f"\nDECISION: RETEST (Marginal delta: {delta_best:+.4f})")

    # Save outputs
    out_payload = {
        "experiment_id": "exp_v17_dual_backbone_blend",
        "best_weight": best_w,
        "best_qseg": best_qseg,
        "baseline_qseg": baseline_qseg,
        "delta_qseg": delta_best,
        "best_composite": best_composite,
        "restoration": {
            "PSNR": mean_psnr,
            "SSIM": mean_ssim,
            "sharpness": mean_sharp,
            "Q_FR": q_fr
        },
        "sweep_results": results_table,
        "decision": decision,
        "peak_vram_gib": peak_vram_res,
        "latency_ms": float(np.mean(latencies_blend))
    }

    with open("tmp/gate5_dual_backbone_results.json", "w") as f:
        json.dump(out_payload, f, indent=2)


if __name__ == '__main__':
    run_gate5_experiment()
