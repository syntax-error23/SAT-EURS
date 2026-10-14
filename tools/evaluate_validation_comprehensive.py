"""
Comprehensive Validation Split Evaluator (Forensic Error Analysis)
Evaluates on the official scene-disjoint validation split: ['fish1', 'fish12', 'seahorse1', 'sharkFish'].
Measures per-class IoU, Precision, Recall, F1, Confusion Matrix, PSNR, SSIM, and Sharpness.
"""

import os
import os.path as osp
import time
import json
from typing import Dict, Any, List

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
from metrics.segmentation_metrics import SegmentationEvaluator
from metrics.restoration_metrics import compute_psnr_and_norm, compute_ssim_single, compute_sharpness_ratio


def evaluate_val_scenes(sample_stride: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    data_clean_dir = "data/cached_1024/clean"
    data_masks_dir = "data/cached_1024/masks"
    data_deg_dir = "data/cached_1024/degraded"

    print("=" * 72)
    print("   RUNNING INDEPENDENT FORENSIC EVALUATION ON VALIDATION SPLIT")
    print(f"   Scenes: {val_scenes} | Stride: {sample_stride} (1 in {sample_stride} images)")
    print("=" * 72)

    # 1. Load Student: EURSNet-V5 Distilled
    student = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt_s = torch.load("checkpoints/v5_clinical_distilled_best.pt", map_location=device, weights_only=False)
    student.load_state_dict(ckpt_s.get("model_state_dict", ckpt_s), strict=True)
    student.eval()

    # 2. Load SegFormer MiT-B3 Teacher
    model_sf = smp.DeepLabV3Plus(encoder_name="mit_b3", encoder_weights=None, classes=5, in_channels=3).to(device)
    sf_ckpt = "checkpoints/segformer_mit_b3_epoch_12.pt" if osp.exists("checkpoints/segformer_mit_b3_epoch_12.pt") else "checkpoints/segformer_mit_b3_best.pt"
    ckpt_sf = torch.load(sf_ckpt, map_location=device, weights_only=False)
    model_sf.load_state_dict(ckpt_sf.get("model_state_dict", ckpt_sf))
    model_sf.eval()

    # 3. Load DeepLabV3-ResNet101 Teacher
    model_dl = sm.deeplabv3_resnet101(weights=None, num_classes=5, aux_loss=True).to(device)
    ckpt_dl = torch.load("checkpoints/deeplabv3_resnet101_best.pt", map_location=device, weights_only=False)
    model_dl.load_state_dict(ckpt_dl.get("model_state_dict", ckpt_dl))
    model_dl.eval()

    # Evaluators
    eval_student_seg = SegmentationEvaluator(num_classes=5)
    eval_sf_seg = SegmentationEvaluator(num_classes=5)
    eval_dl_seg = SegmentationEvaluator(num_classes=5)
    eval_ens_seg = SegmentationEvaluator(num_classes=5)

    student_psnrs, student_ssims, student_sharps = [], [], []

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    total_evaluated = 0
    t0 = time.time()

    for sc in val_scenes:
        clean_p = osp.join(data_clean_dir, f"{sc}.png")
        mask_p = osp.join(data_masks_dir, f"{sc}.png")
        deg_folder = osp.join(data_deg_dir, sc)

        with Image.open(clean_p) as c_im:
            clean_np = np.asarray(c_im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(mask_p) as m_im:
            mask_np = np.asarray(m_im, dtype=np.int64)

        deg_files = sorted([osp.join(deg_folder, f) for f in os.listdir(deg_folder) if f.endswith(".png")])
        sampled_files = deg_files[::sample_stride]

        clean_tensor = torch.from_numpy(clean_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device)

        for df in sampled_files:
            total_evaluated += 1
            with Image.open(df) as d_im:
                deg_np = np.asarray(d_im.convert("RGB"), dtype=np.float32) / 255.0
            deg_tensor = torch.from_numpy(deg_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                # A. Student (EURS-Net V5)
                rest_s, logits_s = student(deg_tensor)
                eval_student_seg.update(logits_s, mask_tensor)

                # Student Restoration Metrics
                rest_s_np = rest_s[0].cpu().numpy().transpose(1, 2, 0)
                psnr, _ = compute_psnr_and_norm(rest_s_np, clean_np)
                ssim = compute_ssim_single(rest_s_np, clean_np)
                sharp = compute_sharpness_ratio(rest_s_np, clean_np)
                student_psnrs.append(psnr)
                student_ssims.append(ssim)
                student_sharps.append(sharp)

                # B. SegFormer MiT-B3
                deg_norm = (deg_tensor - mean_im) / std_im
                logits_sf = model_sf(deg_norm)
                eval_sf_seg.update(logits_sf, mask_tensor)

                # C. DeepLabV3-ResNet101
                logits_dl = model_dl(deg_tensor)["out"]
                eval_dl_seg.update(logits_dl, mask_tensor)

                # D. Multi-Model Ensemble (45% SF + 40% DL + 15% Student)
                logits_ens = 0.45 * logits_sf + 0.40 * logits_dl + 0.15 * logits_s
                eval_ens_seg.update(logits_ens, mask_tensor)

    t_eval = time.time() - t0
    print(f"\nCompleted evaluation of {total_evaluated} images across 4 scenes in {t_eval:.1f}s!\n")

    res_student = eval_student_seg.compute()
    res_sf = eval_sf_seg.compute()
    res_dl = eval_dl_seg.compute()
    res_ens = eval_ens_seg.compute()

    mean_psnr = float(np.mean(student_psnrs))
    mean_ssim = float(np.mean(student_ssims))
    mean_sharp = float(np.mean(student_sharps))
    psnr_norm = min(1.0, max(0.0, (mean_psnr - 15.0) / 20.0))
    q_fr_student = 0.45 * psnr_norm + 0.45 * mean_ssim + 0.10 * mean_sharp

    print("=" * 72)
    print("   EMPIRICAL VALIDATION METRIC COMPARISON")
    print("=" * 72)
    print(f"{'Model / Pipeline':<28} | {'mIoU':<7} | {'Macro F1':<8} | {'Q_seg':<7} | {'PSNR':<7} | {'SSIM':<6}")
    print("-" * 72)
    print(f"{'EURS-Net V5 Distilled (Track 2)':<28} | {res_student['mIoU']:<7.4f} | {res_student['macro_F1']:<8.4f} | {res_student['Q_seg']:<7.4f} | {mean_psnr:<7.2f} | {mean_ssim:<6.4f}")
    print(f"{'SegFormer MiT-B3 (Teacher)':<28} | {res_sf['mIoU']:<7.4f} | {res_sf['macro_F1']:<8.4f} | {res_sf['Q_seg']:<7.4f} | {'—':<7} | {'—':<6}")
    print(f"{'DeepLabV3-ResNet101 (Teacher)':<28} | {res_dl['mIoU']:<7.4f} | {res_dl['macro_F1']:<8.4f} | {res_dl['Q_seg']:<7.4f} | {'—':<7} | {'—':<6}")
    print(f"{'Multi-Model Ensemble':<28} | {res_ens['mIoU']:<7.4f} | {res_ens['macro_F1']:<8.4f} | {res_ens['Q_seg']:<7.4f} | {'—':<7} | {'—':<6}")
    print("=" * 72)

    # Save detailed JSON artifact
    out_dict = {
        'total_evaluated': total_evaluated,
        'val_scenes': val_scenes,
        'student_metrics': res_student,
        'student_restoration': {
            'mean_psnr': mean_psnr,
            'mean_ssim': mean_ssim,
            'mean_sharpness': mean_sharp,
            'psnr_norm': psnr_norm,
            'Q_FR': q_fr_student
        },
        'segformer_metrics': res_sf,
        'deeplabv3_metrics': res_dl,
        'ensemble_metrics': res_ens,
    }
    with open("reports/val_evaluation_results.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print("Saved detailed metrics to reports/val_evaluation_results.json")


if __name__ == "__main__":
    evaluate_val_scenes(sample_stride=10)
