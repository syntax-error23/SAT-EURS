"""
EURS-Net Phase 2 & 3 Comprehensive Diagnostic Engine
Rigorous, leak-free evaluation of:
1. SegFormer MiT-B3 (Raw vs Restored)
2. DeepLabV3-ResNet101 (Raw vs Restored with CORRECT ImageNet standardization)
3. EURS-Net V4 (Full20)
4. EURS-Net V5 (Distilled Student)
5. Multi-model combinations (SF + DL + V4 + V5)
6. TTA impact (No-TTA vs H-Flip TTA)
7. Per-class Confusion Matrices, IoU, Precision, Recall, and F1 across all 5 classes.
"""

import os
import os.path as osp
import time
import json
from typing import Dict, List, Tuple

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


def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    data_clean_dir = "data/cached_1024/clean"
    data_masks_dir = "data/cached_1024/masks"
    data_deg_dir = "data/cached_1024/degraded"

    print("=" * 72)
    print("   RUNNING PHASE 2 & 3 DIAGNOSTIC BENCHMARK ON RTX 5090")
    print("=" * 72)

    # 1. Load Models
    print("[1/4] Loading SegFormer MiT-B3 Teacher...", flush=True)
    model_sf = smp.DeepLabV3Plus(encoder_name="mit_b3", encoder_weights=None, classes=5, in_channels=3).to(device)
    sf_ckpt = "checkpoints/segformer_mit_b3_epoch_12.pt" if osp.exists("checkpoints/segformer_mit_b3_epoch_12.pt") else "checkpoints/segformer_mit_b3_best.pt"
    ckpt_sf = torch.load(sf_ckpt, map_location=device, weights_only=False)
    model_sf.load_state_dict(ckpt_sf.get("model_state_dict", ckpt_sf))
    model_sf.eval()

    print("[2/4] Loading DeepLabV3-ResNet101 Teacher...", flush=True)
    model_dl = sm.deeplabv3_resnet101(weights=None, num_classes=5, aux_loss=True).to(device)
    ckpt_dl = torch.load("checkpoints/deeplabv3_resnet101_best.pt", map_location=device, weights_only=False)
    model_dl.load_state_dict(ckpt_dl.get("model_state_dict", ckpt_dl))
    model_dl.eval()

    print("[3/4] Loading EURS-Net V4 Full20...", flush=True)
    model_v4 = EURSNetV4(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt_v4 = torch.load("checkpoints/v4_full20/best_v4_checkpoint.pt", map_location=device, weights_only=False)
    model_v4.load_state_dict(ckpt_v4.get("model_state_dict", ckpt_v4), strict=False)
    model_v4.eval()

    print("[4/4] Loading EURS-Net V5 Distilled Student...", flush=True)
    model_v5 = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt_v5 = torch.load("checkpoints/v5_clinical_distilled_best.pt", map_location=device, weights_only=False)
    model_v5.load_state_dict(ckpt_v5.get("model_state_dict", ckpt_v5), strict=True)
    model_v5.eval()

    # Physical RSUIGM table
    rsuigm_table = fit_or_load_rsuigm_wt_depth_table_gpu(device=device)

    # ImageNet standard normalization
    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 2. Ingest validation data
    val_samples = []
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
            stem = osp.splitext(df)[0]
            parts = stem.split("_")
            wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
            depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10

            deg_p = osp.join(deg_folder, df)
            with Image.open(deg_p) as d_im:
                deg_np = np.asarray(d_im.convert("RGB"), dtype=np.float32) / 255.0

            val_samples.append({
                "scene": sc,
                "stem": stem,
                "wt": wt,
                "depth": depth,
                "deg_np": deg_np,
                "clean_np": clean_np,
                "mask_np": mask_np,
            })

    print(f"Loaded {len(val_samples)} validation samples across 4 scenes.\n")

    # 3. Precompute representations and predictions
    targets = []
    
    # Store predictions for each model
    preds = {
        "SF_raw": [],
        "SF_rest": [],
        "SF_rest_tta": [],
        "DL_raw_buggy": [],     # Unnormalized (the bug)
        "DL_rest_fixed": [],    # Normalized (the fix)
        "DL_rest_fixed_tta": [],
        "V4_raw": [],
        "V4_raw_tta": [],
        "V5_raw": [],
        "V5_raw_tta": [],
        "V5_calibrated_tta": [], # With Delta3 = -1.5
    }

    t0 = time.time()
    for s_idx, item in enumerate(val_samples):
        clean_np = item["clean_np"]
        mask_t = torch.from_numpy(item["mask_np"]).unsqueeze(0)
        targets.append(mask_t)

        x_raw = torch.from_numpy(item["deg_np"].transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            # Physical Inversion
            W = rsuigm_table.get((item["wt"], item["depth"]), rsuigm_table[(1, 10)])
            feats = build_rsuigm_poly_features_gpu(x_raw)
            rest_phys = torch.clamp(torch.matmul(feats, W).reshape(1, 1024, 1024, 3).permute(0, 3, 1, 2), 0.0, 1.0)

            # Normalization variants
            x_raw_norm = (x_raw - mean_im) / std_im
            rest_norm = (rest_phys - mean_im) / std_im

            # Flips for TTA
            x_raw_flip = torch.flip(x_raw, dims=[3])
            rest_norm_flip = torch.flip(rest_norm, dims=[3])

            # A. SegFormer
            out_sf_raw = model_sf(x_raw_norm)
            out_sf_rest = model_sf(rest_norm)
            out_sf_rest_flip = model_sf(rest_norm_flip)
            out_sf_rest_tta = 0.5 * (out_sf_rest + torch.flip(out_sf_rest_flip, dims=[3]))

            preds["SF_raw"].append(out_sf_raw.cpu())
            preds["SF_rest"].append(out_sf_rest.cpu())
            preds["SF_rest_tta"].append(out_sf_rest_tta.cpu())

            # B. DeepLabV3 (Buggy unnormalized vs Fixed normalized)
            out_dl_buggy = model_dl(rest_phys)["out"]  # Fed [0, 1] without norm!
            out_dl_fixed = model_dl(rest_norm)["out"]  # Fed normalized!
            out_dl_fixed_flip = model_dl(rest_norm_flip)["out"]
            out_dl_fixed_tta = 0.5 * (out_dl_fixed + torch.flip(out_dl_fixed_flip, dims=[3]))

            preds["DL_raw_buggy"].append(out_dl_buggy.cpu())
            preds["DL_rest_fixed"].append(out_dl_fixed.cpu())
            preds["DL_rest_fixed_tta"].append(out_dl_fixed_tta.cpu())

            # C. EURSNet-V4
            _, out_v4 = model_v4(x_raw)
            _, out_v4_flip = model_v4(x_raw_flip)
            out_v4_tta = 0.5 * (out_v4 + torch.flip(out_v4_flip, dims=[3]))

            preds["V4_raw"].append(out_v4.cpu())
            preds["V4_raw_tta"].append(out_v4_tta.cpu())

            # D. EURSNet-V5
            _, out_v5 = model_v5(x_raw)
            _, out_v5_flip = model_v5(x_raw_flip)
            out_v5_tta = 0.5 * (out_v5 + torch.flip(out_v5_flip, dims=[3]))

            # Calibrated V5: Offset Delta3 = -1.5
            out_v5_cal = out_v5_tta.clone()
            out_v5_cal[:, 3, :, :] -= 1.5

            preds["V5_raw"].append(out_v5.cpu())
            preds["V5_raw_tta"].append(out_v5_tta.cpu())
            preds["V5_calibrated_tta"].append(out_v5_cal.cpu())

    print(f"Precomputed all representations in {time.time() - t0:.1f}s!\n")

    # 4. Evaluation Function
    evaluator = SegmentationEvaluator(num_classes=5)

    def evaluate_model_pipeline(pred_list: List[torch.Tensor], name: str) -> Dict:
        evaluator.reset()
        for p, t in zip(pred_list, targets):
            evaluator.update(p, t)
        res = evaluator.compute()
        return res

    results = {}
    print("=" * 88)
    print(f"{'Pipeline Configuration':<36} | {'mIoU':<7} | {'Macro F1':<8} | {'Q_seg':<7} | {'C3 F1':<7} | {'C4 F1':<7}")
    print("-" * 88)

    for k, v in preds.items():
        res = evaluate_model_pipeline(v, k)
        results[k] = res
        c3_f1 = res['per_class_F1'][3]
        c4_f1 = res['per_class_F1'][4]
        print(f"{k:<36} | {res['mIoU']:<7.4f} | {res['macro_F1']:<8.4f} | {res['Q_seg']:<7.4f} | {c3_f1:<7.4f} | {c4_f1:<7.4f}")

    # 5. Multi-Model Ensembles
    print("\n" + "-" * 88)
    print("   MULTI-MODEL ENSEMBLE COMPARISONS (WITH FIXED NORMALIZATION)")
    print("-" * 88)

    ensemble_configs = [
        ("Ens: SF_rest + DL_buggy (Old Buggy)", [0.50, 0.50, 0.00, 0.00], preds["SF_rest_tta"], preds["DL_raw_buggy"], preds["V4_raw_tta"], preds["V5_calibrated_tta"]),
        ("Ens: SF_rest + DL_fixed (Fixed 2-Way)", [0.50, 0.50, 0.00, 0.00], preds["SF_rest_tta"], preds["DL_rest_fixed_tta"], preds["V4_raw_tta"], preds["V5_calibrated_tta"]),
        ("Ens: SF + DL + V4 + V5 (Equal 4-Way Fixed)", [0.25, 0.25, 0.25, 0.25], preds["SF_rest_tta"], preds["DL_rest_fixed_tta"], preds["V4_raw_tta"], preds["V5_calibrated_tta"]),
        ("Ens: SF 40% + DL 30% + V4 15% + V5 15%", [0.40, 0.30, 0.15, 0.15], preds["SF_rest_tta"], preds["DL_rest_fixed_tta"], preds["V4_raw_tta"], preds["V5_calibrated_tta"]),
        ("Ens: SF 45% + DL 45% + V5 10% (No V4)", [0.45, 0.45, 0.00, 0.10], preds["SF_rest_tta"], preds["DL_rest_fixed_tta"], preds["V4_raw_tta"], preds["V5_calibrated_tta"]),
        ("Ens: SF 50% + V4 25% + V5 25%", [0.50, 0.00, 0.25, 0.25], preds["SF_rest_tta"], preds["DL_rest_fixed_tta"], preds["V4_raw_tta"], preds["V5_calibrated_tta"]),
    ]

    for name, weights, p_sf, p_dl, p_v4, p_v5 in ensemble_configs:
        w_sf, w_dl, w_v4, w_v5 = weights
        ens_preds = []
        for l_sf, l_dl, l_v4, l_v5 in zip(p_sf, p_dl, p_v4, p_v5):
            ens_log = w_sf * l_sf + w_dl * l_dl + w_v4 * l_v4 + w_v5 * l_v5
            ens_preds.append(ens_log)
        res = evaluate_model_pipeline(ens_preds, name)
        results[name] = res
        c3_f1 = res['per_class_F1'][3]
        c4_f1 = res['per_class_F1'][4]
        print(f"{name:<36} | {res['mIoU']:<7.4f} | {res['macro_F1']:<8.4f} | {res['Q_seg']:<7.4f} | {c3_f1:<7.4f} | {c4_f1:<7.4f}")

    print("=" * 88)

    # 6. Save detailed JSON artifact
    # Convert numpy arrays to lists for JSON serialization
    serializable_res = {}
    for k, v in results.items():
        serializable_res[k] = {
            "mIoU": float(v["mIoU"]),
            "macro_F1": float(v["macro_F1"]),
            "Q_seg": float(v["Q_seg"]),
            "per_class_IoU": [float(x) for x in v["per_class_IoU"]],
            "per_class_F1": [float(x) for x in v["per_class_F1"]],
            "per_class_Precision": [float(x) for x in v["per_class_precision"]],
            "per_class_Recall": [float(x) for x in v["per_class_recall"]],
            "confusion_matrix": v["confusion_matrix"]
        }

    with open("reports/phase2_diagnostics_results.json", "w") as fp:
        json.dump(serializable_res, fp, indent=2)
    print("\nSaved full forensic evaluation results to reports/phase2_diagnostics_results.json!")


if __name__ == "__main__":
    run_diagnostics()
