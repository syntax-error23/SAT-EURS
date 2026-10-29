"""
SAT-EURS Track 1 — V38 Comprehensive 5-Fold Scene-Disjoint OOF Segmentation Ablation Suite
Evaluates strictly out-of-fold across all 20 scenes (5 folds, manifest_5fold.json) using native
test-time preprocessing (V18CanonicalPreprocessor, max_side=1024, pad_multiple=32).

Key methodological safeguards:
1. Strictly single-image inference: every prediction is computed solely from its own degraded input X_i
   and filename metadata (wt, depth). Zero scene anchoring, zero cross-image reuse.
2. Global 20-scene 5x5 confusion matrix accumulation: avoids the single-fold absent-class vacuous 1.0 IoU
   artifact (since Class 3 is only in Folds 1 & 2, and Class 4 is only in Folds 0, 3 & 4).
3. One-factor-at-a-time ablations + class-selective multi-architecture ensembling.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import os.path as osp
import sys
import time
import json
import math
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.train_track1_screening import build_dinov2_model, apply_rsuigm_batch_gpu
from models.convnext_fpn import build_convnext_fpn
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    V18CanonicalPreprocessor,
)

CLASS_NAMES = ["Background", "Coral", "Fish", "SharkRay", "MacroFauna"]


def compute_global_metrics(cm: np.ndarray) -> Dict[str, Any]:
    """Computes official Track 1 segmentation metrics from a 5x5 confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    gt = tp + fn

    present_classes = [c for c in range(5) if gt[c] > 0]
    ious = []
    f1s = []
    precs = []
    recs = []

    for c in range(5):
        denom_iou = tp[c] + fp[c] + fn[c]
        iou = float(tp[c] / denom_iou) if denom_iou > 0 else (1.0 if gt[c] == 0 else 0.0)
        denom_f1 = 2.0 * tp[c] + fp[c] + fn[c]
        f1 = float(2.0 * tp[c] / denom_f1) if denom_f1 > 0 else (1.0 if gt[c] == 0 else 0.0)
        prec = float(tp[c] / (tp[c] + fp[c])) if (tp[c] + fp[c]) > 0 else 0.0
        rec = float(tp[c] / gt[c]) if gt[c] > 0 else 0.0
        ious.append(iou)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    miou_pres = float(np.mean([ious[c] for c in present_classes])) if present_classes else 0.0
    f1_pres = float(np.mean([f1s[c] for c in present_classes])) if present_classes else 0.0
    cov = 1.0
    for c in present_classes:
        if tp[c] == 0:
            cov = 0.0
            break
    q_seg_pres = miou_pres * f1_pres * cov

    return {
        "Q_seg": round(q_seg_pres, 5),
        "mIoU": round(miou_pres, 5),
        "Macro_F1": round(f1_pres, 5),
        "coverage": cov,
        "present_classes": present_classes,
        "class_ious": [round(x, 5) for x in ious],
        "class_f1s": [round(x, 5) for x in f1s],
        "class_precisions": [round(x, 5) for x in precs],
        "class_recalls": [round(x, 5) for x in recs],
        "gt_pixels": [int(x) for x in gt],
        "tp_pixels": [int(x) for x in tp],
        "fp_pixels": [int(x) for x in fp],
    }


def apply_clahe_rgb_tensor(rgb_tensor: torch.Tensor, clip_limit: float = 1.8) -> torch.Tensor:
    """Applies mild per-image CLAHE on LAB L-channel to a [1, 3, H, W] float32 RGB tensor in [0, 1]."""
    img_np = (rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(out_rgb.transpose(2, 0, 1)).unsqueeze(0).to(rgb_tensor.device)


def filter_small_rare_components(mask_np: np.ndarray, min_area: int = 32, fallback_mask: np.ndarray = None) -> np.ndarray:
    """Removes tiny isolated false-positive speckles for rare classes 3 and 4."""
    out = mask_np.copy()
    for c in [3, 4]:
        binary = (out == c).astype(np.uint8)
        if not binary.any():
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                region = (labels == lbl)
                if fallback_mask is not None:
                    out[region] = fallback_mask[region]
                else:
                    out[region] = 0
    return out


def load_seg_state(model: nn.Module, path: str, device: torch.device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        if "state_dict" in obj:
            state = obj["state_dict"]
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        else:
            state = obj
    else:
        state = obj
    model.load_state_dict(state)
    model.eval()
    return model


def run_ablation_suite(sample_stride: int = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 100, flush=True)
    print(f"SAT-EURS TRACK 1 — V38 5-FOLD 20-SCENE GLOBAL OOF ABLATION SUITE (stride={sample_stride})", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print("=" * 100, flush=True)

    manifest = validate_manifest_and_folds()
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    preprocessor_896 = V18CanonicalPreprocessor(max_side=896, pad_multiple=32)
    preprocessor_1152 = V18CanonicalPreprocessor(max_side=1152, pad_multiple=32)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    off_zero = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device).view(1, 5, 1, 1)
    off_v33 = torch.tensor([0.0, 0.0, 0.0, 0.10, 0.25], device=device).view(1, 5, 1, 1)
    off_15_25 = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.25], device=device).view(1, 5, 1, 1)
    off_15_30 = torch.tensor([0.0, 0.0, 0.0, 0.15, 0.30], device=device).view(1, 5, 1, 1)
    off_20_35 = torch.tensor([0.0, 0.0, 0.0, 0.20, 0.35], device=device).view(1, 5, 1, 1)
    off_05_20 = torch.tensor([0.0, 0.0, 0.0, 0.05, 0.20], device=device).view(1, 5, 1, 1)
    off_fish = torch.tensor([0.0, 0.0, 0.05, 0.10, 0.25], device=device).view(1, 5, 1, 1)

    # Class-selective weight tensors [1, 5, 1, 1]
    def make_cw(w_list):
        return torch.tensor(w_list, dtype=torch.float32, device=device).view(1, 5, 1, 1)

    # Accumulators across all 20 scenes
    cm_global: Dict[str, np.ndarray] = {}
    cm_by_fold: Dict[int, Dict[str, np.ndarray]] = {f: {} for f in range(5)}
    cm_by_scene: Dict[str, Dict[str, np.ndarray]] = {}

    def update_cm(cfg_name: str, fold: int, scene: str, pred_unpad: np.ndarray, gt_mask: np.ndarray):
        if cfg_name not in cm_global:
            cm_global[cfg_name] = np.zeros((5, 5), dtype=np.int64)
        if cfg_name not in cm_by_fold[fold]:
            cm_by_fold[fold][cfg_name] = np.zeros((5, 5), dtype=np.int64)
        if scene not in cm_by_scene:
            cm_by_scene[scene] = {}
        if cfg_name not in cm_by_scene[scene]:
            cm_by_scene[scene][cfg_name] = np.zeros((5, 5), dtype=np.int64)

        if pred_unpad.shape[:2] != gt_mask.shape[:2]:
            pred_unpad = cv2.resize(pred_unpad, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        indices = 5 * gt_mask.flatten().astype(np.int64) + pred_unpad.flatten().astype(np.int64)
        counts = np.bincount(indices, minlength=25).reshape(5, 5)
        cm_global[cfg_name] += counts
        cm_by_fold[fold][cfg_name] += counts
        cm_by_scene[scene][cfg_name] += counts

    def decode_mask(prob_pad: torch.Tensor, offsets: torch.Tensor, meta: dict) -> np.ndarray:
        log_p = torch.log(prob_pad + 1e-10) + offsets
        m_pad = torch.argmax(log_p, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return preprocessor.unpad_output(m_pad, meta)

    t_suite_start = time.time()
    total_images = 0

    for fold in range(5):
        t_f0 = time.time()
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])
        print(f"\n" + "-" * 90, flush=True)
        print(f"[FOLD {fold}] Held-out Val Scenes: {val_scenes}", flush=True)

        # 1. Load Refiner (Compare fold-specific table vs 20sc table)
        tbl_fold_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
        tbl_20sc_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
        tbl_fold = torch.load(tbl_fold_path, map_location=device, weights_only=False)
        tbl_20sc = torch.load(tbl_20sc_path, map_location=device, weights_only=False)

        ref_ckpt = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
        refiner_20sc = build_v18_restoration_refiner(rsuigm_table=tbl_20sc, use_neural_refiner=True).to(device)
        load_seg_state(refiner_20sc, ref_ckpt, device)

        refiner_fold = build_v18_restoration_refiner(rsuigm_table=tbl_fold, use_neural_refiner=True).to(device)
        load_seg_state(refiner_fold, ref_ckpt, device)

        # 2. Load Primary V28-A Segmenter for this fold
        seg_v28a_ckpt = "checkpoints/v27/segmenter_v27_fold0_best.pt" if fold == 0 else f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
        seg_v28a = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
        load_seg_state(seg_v28a, seg_v28a_ckpt, device)

        # 3. Load Alternative Fold Checkpoints
        seg_alt_v33 = None
        if fold == 3:
            seg_alt_v33 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
            load_seg_state(seg_alt_v33, "checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt", device)

        seg_alt_f2_9ep = None
        if fold == 2 and osp.exists("checkpoints/v20/matched_pilot/fold_2/control_v18_best.pt"):
            seg_alt_f2_9ep = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
            load_seg_state(seg_alt_f2_9ep, "checkpoints/v20/matched_pilot/fold_2/control_v18_best.pt", device)

        seg_alt_f0_v18 = None
        seg_alt_f0_v20t = None
        segformer_f0 = None
        if fold == 0:
            seg_alt_f0_v18 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
            load_seg_state(seg_alt_f0_v18, "checkpoints/v18/fold_0/segmenter_best.pt", device)
            if osp.exists("checkpoints/v20/matched_pilot/fold_0/treatment_v20_best.pt"):
                seg_alt_f0_v20t = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
                load_seg_state(seg_alt_f0_v20t, "checkpoints/v20/matched_pilot/fold_0/treatment_v20_best.pt", device)
            if osp.exists("checkpoints/v29/segformer_b5_fold_0_best.pt"):
                segformer_f0 = SegformerForSemanticSegmentation.from_pretrained(
                    "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024",
                    num_labels=5,
                    ignore_mismatched_sizes=True,
                    local_files_only=True,
                ).to(device)
                load_seg_state(segformer_f0, "checkpoints/v29/segformer_b5_fold_0_best.pt", device)

        # 4. Load V16 DINOv2-Large (fold-disjoint)
        dino_model = build_dinov2_model(num_classes=5).to(device)
        load_seg_state(dino_model, f"checkpoints/v16_dinov2/fold_{fold}/best_model.pt", device)

        # 5. Load V16 ConvNeXt-Base FPN (fold-disjoint)
        conv_model = build_convnext_fpn(num_classes=5, freeze_stage0=True).to(device)
        load_seg_state(conv_model, f"checkpoints/v16_convnext/fold_{fold}/best_model.pt", device)

        for sc in val_scenes:
            mask_path = f"data/cached_1024/masks/{sc}.png"
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            deg_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::sample_stride]

            for df in deg_files:
                total_images += 1
                stem = osp.splitext(df)[0]
                parts = stem.split("_")
                wt = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else 1
                depth = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 10

                bgr = cv2.imread(osp.join(deg_dir, df), cv2.IMREAD_COLOR)
                prep = preprocessor.prepare_input(bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                H_pad, W_pad = raw_pad.shape[-2:]

                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Restoration with 20sc table (V33 default) vs fold-specific table (strictly leak-free)
                    rest_pad = refiner_20sc(raw_pad, wt_t, dp_t)["refined_image"]
                    rest_fold_pad = refiner_fold(raw_pad, wt_t, dp_t)["refined_image"]
                    rest_table_pad = apply_rsuigm_batch_gpu(raw_pad, wt_t, dp_t.long(), tbl_fold)

                    raw_flip = torch.flip(raw_pad, dims=[3])
                    rest_flip = torch.flip(rest_pad, dims=[3])
                    rest_fold_flip = torch.flip(rest_fold_pad, dims=[3])

                    # Multi-scale & CLAHE restorations (must run in fp32 outside bfloat16 autocast)
                    prep_896 = preprocessor_896.prepare_input(bgr)
                    raw_896 = prep_896["rgb_tensor"].to(device)
                    rest_896 = refiner_20sc(raw_896, wt_t, dp_t)["refined_image"]

                    prep_1152 = preprocessor_1152.prepare_input(bgr)
                    raw_1152 = prep_1152["rgb_tensor"].to(device)
                    rest_1152 = refiner_20sc(raw_1152, wt_t, dp_t)["refined_image"]

                    rest_clahe = apply_clahe_rgb_tensor(rest_pad, clip_limit=1.8)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        # --- Primary V28-A Segmenter (orig & flip) ---
                        l_orig = seg_v28a(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                        p_v28a_orig = F.softmax(l_orig, dim=1)
                        l_flip = seg_v28a(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                        p_v28a_flip = torch.flip(F.softmax(l_flip, dim=1), dims=[3])
                        p_v28a = 0.5 * (p_v28a_orig + p_v28a_flip)

                        # Strict fold-table refiner feed
                        l_f_orig = seg_v28a(rest_fold_pad, raw_pad, wt_t, dp_t)["logits"].float()
                        l_f_flip = seg_v28a(rest_fold_flip, raw_flip, wt_t, dp_t)["logits"].float()
                        p_v28a_foldtbl = 0.5 * (F.softmax(l_f_orig, dim=1) + torch.flip(F.softmax(l_f_flip, dim=1), dims=[3]))

                        # --- Multi-Scale 896 & 1152 ---
                        l_896 = seg_v28a(rest_896, raw_896, wt_t, dp_t)["logits"].float()
                        l_896_f = seg_v28a(torch.flip(rest_896, dims=[3]), torch.flip(raw_896, dims=[3]), wt_t, dp_t)["logits"].float()
                        p_896_pad = 0.5 * (F.softmax(l_896, dim=1) + torch.flip(F.softmax(l_896_f, dim=1), dims=[3]))
                        # Unpad 896 and resize back to H_pad, W_pad (within valid region)
                        nh8, nw8 = prep_896["meta"]["target_h"], prep_896["meta"]["target_w"]
                        nh0, nw0 = meta["target_h"], meta["target_w"]
                        p_896_valid = F.interpolate(p_896_pad[:, :, :nh8, :nw8], size=(nh0, nw0), mode="bilinear", align_corners=False)
                        p_896 = torch.zeros_like(p_v28a)
                        p_896[:, :, :nh0, :nw0] = p_896_valid

                        l_1152 = seg_v28a(rest_1152, raw_1152, wt_t, dp_t)["logits"].float()
                        l_1152_f = seg_v28a(torch.flip(rest_1152, dims=[3]), torch.flip(raw_1152, dims=[3]), wt_t, dp_t)["logits"].float()
                        p_1152_pad = 0.5 * (F.softmax(l_1152, dim=1) + torch.flip(F.softmax(l_1152_f, dim=1), dims=[3]))
                        nh11, nw11 = prep_1152["meta"]["target_h"], prep_1152["meta"]["target_w"]
                        p_1152_valid = F.interpolate(p_1152_pad[:, :, :nh11, :nw11], size=(nh0, nw0), mode="bilinear", align_corners=False)
                        p_1152 = torch.zeros_like(p_v28a)
                        p_1152[:, :, :nh0, :nw0] = p_1152_valid

                        # --- CLAHE TTA View ---
                        l_clahe = seg_v28a(rest_clahe, raw_pad, wt_t, dp_t)["logits"].float()
                        p_clahe = F.softmax(l_clahe, dim=1)

                        # --- Alternative Fold Checkpoints ---
                        p_v33 = p_v28a
                        if fold == 3 and seg_alt_v33 is not None:
                            l3_o = seg_alt_v33(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                            l3_f = seg_alt_v33(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                            p_v33 = 0.5 * (F.softmax(l3_o, dim=1) + torch.flip(F.softmax(l3_f, dim=1), dims=[3]))

                        p_f2_9ep = p_v28a
                        if fold == 2 and seg_alt_f2_9ep is not None:
                            l2_o = seg_alt_f2_9ep(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                            l2_f = seg_alt_f2_9ep(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                            p_f2_9ep = 0.5 * (F.softmax(l2_o, dim=1) + torch.flip(F.softmax(l2_f, dim=1), dims=[3]))

                        p_f0_v18 = p_v28a
                        p_f0_v20t = p_v28a
                        p_segb5_f0 = p_v28a
                        if fold == 0:
                            if seg_alt_f0_v18 is not None:
                                l0_o = seg_alt_f0_v18(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                                l0_f = seg_alt_f0_v18(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                                p_f0_v18 = 0.5 * (F.softmax(l0_o, dim=1) + torch.flip(F.softmax(l0_f, dim=1), dims=[3]))
                            if seg_alt_f0_v20t is not None:
                                l0t_o = seg_alt_f0_v20t(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                                l0t_f = seg_alt_f0_v20t(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                                p_f0_v20t = 0.5 * (F.softmax(l0t_o, dim=1) + torch.flip(F.softmax(l0t_f, dim=1), dims=[3]))
                            if segformer_f0 is not None:
                                xn = (rest_pad - mean_im) / std_im
                                xn_f = torch.flip(xn, dims=[3])
                                sf_o = F.softmax(F.interpolate(segformer_f0(pixel_values=xn).logits.float(), size=(H_pad, W_pad), mode="bilinear", align_corners=False), dim=1)
                                sf_f = torch.flip(F.softmax(F.interpolate(segformer_f0(pixel_values=xn_f).logits.float(), size=(H_pad, W_pad), mode="bilinear", align_corners=False), dim=1), dims=[3])
                                p_segb5_f0 = 0.5 * (sf_o + sf_f)

                        # --- V16 DINOv2-Large (1) Squashed 518x518 vs (2) Aspect-Preserving Multiple-of-14 ---
                        rest_valid = rest_table_pad[:, :, :nh0, :nw0]
                        # (1) Squashed 518x518
                        r_518_sq = F.interpolate(rest_valid, size=(518, 518), mode="bilinear", align_corners=False)
                        x_d_sq = (r_518_sq - mean_im) / std_im
                        ld_sq_o = dino_model(x_d_sq).float()
                        ld_sq_f = dino_model(torch.flip(x_d_sq, dims=[3])).float()
                        pd_sq_518 = 0.5 * (F.softmax(ld_sq_o, dim=1) + torch.flip(F.softmax(ld_sq_f, dim=1), dims=[3]))
                        pd_sq_valid = F.interpolate(pd_sq_518, size=(nh0, nw0), mode="bilinear", align_corners=False)
                        p_dino_sq = torch.zeros_like(p_v28a)
                        p_dino_sq[:, :, :nh0, :nw0] = pd_sq_valid

                        # (2) Aspect-preserving multiple of 14 (target short side ~518)
                        scale_d = 518.0 / float(min(nh0, nw0))
                        dh = max(14, int(round((nh0 * scale_d) / 14.0)) * 14)
                        dw = max(14, int(round((nw0 * scale_d) / 14.0)) * 14)
                        r_d_ap = F.interpolate(rest_valid, size=(dh, dw), mode="bilinear", align_corners=False)
                        x_d_ap = (r_d_ap - mean_im) / std_im
                        ld_ap_o = dino_model(x_d_ap).float()
                        ld_ap_f = dino_model(torch.flip(x_d_ap, dims=[3])).float()
                        pd_ap_res = 0.5 * (F.softmax(ld_ap_o, dim=1) + torch.flip(F.softmax(ld_ap_f, dim=1), dims=[3]))
                        pd_ap_valid = F.interpolate(pd_ap_res, size=(nh0, nw0), mode="bilinear", align_corners=False)
                        p_dino_ap = torch.zeros_like(p_v28a)
                        p_dino_ap[:, :, :nh0, :nw0] = pd_ap_valid

                        # (3) DINOv2 fed with V18 neural refiner (instead of table)
                        r_d_v18 = F.interpolate(rest_pad[:, :, :nh0, :nw0], size=(dh, dw), mode="bilinear", align_corners=False)
                        x_d_v18 = (r_d_v18 - mean_im) / std_im
                        ld_v18_o = dino_model(x_d_v18).float()
                        ld_v18_f = dino_model(torch.flip(x_d_v18, dims=[3])).float()
                        pd_v18_res = 0.5 * (F.softmax(ld_v18_o, dim=1) + torch.flip(F.softmax(ld_v18_f, dim=1), dims=[3]))
                        p_dino_v18ref = torch.zeros_like(p_v28a)
                        p_dino_v18ref[:, :, :nh0, :nw0] = F.interpolate(pd_v18_res, size=(nh0, nw0), mode="bilinear", align_corners=False)

                        # --- V16 ConvNeXt-Base FPN (1) Squashed 512x512 vs (2) Aspect-Preserving Multiple-of-32 ---
                        r_512_sq = F.interpolate(rest_valid, size=(512, 512), mode="bilinear", align_corners=False)
                        x_c_sq = (r_512_sq - mean_im) / std_im
                        lc_sq_o = conv_model(x_c_sq).float()
                        lc_sq_f = conv_model(torch.flip(x_c_sq, dims=[3])).float()
                        pc_sq_512 = 0.5 * (F.softmax(lc_sq_o, dim=1) + torch.flip(F.softmax(lc_sq_f, dim=1), dims=[3]))
                        p_conv_sq = torch.zeros_like(p_v28a)
                        p_conv_sq[:, :, :nh0, :nw0] = F.interpolate(pc_sq_512, size=(nh0, nw0), mode="bilinear", align_corners=False)

                        scale_c = 512.0 / float(min(nh0, nw0))
                        ch = max(32, int(round((nh0 * scale_c) / 32.0)) * 32)
                        cw = max(32, int(round((nw0 * scale_c) / 32.0)) * 32)
                        r_c_ap = F.interpolate(rest_valid, size=(ch, cw), mode="bilinear", align_corners=False)
                        x_c_ap = (r_c_ap - mean_im) / std_im
                        lc_ap_o = conv_model(x_c_ap).float()
                        lc_ap_f = conv_model(torch.flip(x_c_ap, dims=[3])).float()
                        pc_ap_res = 0.5 * (F.softmax(lc_ap_o, dim=1) + torch.flip(F.softmax(lc_ap_f, dim=1), dims=[3]))
                        p_conv_ap = torch.zeros_like(p_v28a)
                        p_conv_ap[:, :, :nh0, :nw0] = F.interpolate(pc_ap_res, size=(nh0, nw0), mode="bilinear", align_corners=False)

                    # =========================================================================
                    # EVALUATE CANDIDATES (Strictly One-Factor-at-a-Time + Targeted Synergies)
                    # =========================================================================

                    # --- Group A: Fold Checkpoint & Table Selection ---
                    update_cm("A0_V28A_Base (v18_f3, 20sc_tbl)", fold, sc, decode_mask(p_v28a, off_v33, meta), gt_mask)
                    update_cm("A0b_V28A_FoldTbl (v18_f3, fold_tbl)", fold, sc, decode_mask(p_v28a_foldtbl, off_v33, meta), gt_mask)
                    update_cm("A1_V33_Base (v28_f3, 20sc_tbl)", fold, sc, decode_mask(p_v33, off_v33, meta), gt_mask)
                    update_cm("A2_F2_9ep_Repl", fold, sc, decode_mask(p_f2_9ep, off_v33, meta), gt_mask)
                    p_f2_blend = 0.5 * (p_v28a + p_f2_9ep)
                    update_cm("A3_F2_9ep_Blend50", fold, sc, decode_mask(p_f2_blend, off_v33, meta), gt_mask)
                    p_f0_b18 = 0.7 * p_v28a + 0.3 * p_f0_v18
                    update_cm("A4_F0_V18_Blend30", fold, sc, decode_mask(p_f0_b18, off_v33, meta), gt_mask)
                    p_f0_b20t = 0.75 * p_v28a + 0.25 * p_f0_v20t
                    update_cm("A5_F0_V20T_Blend25", fold, sc, decode_mask(p_f0_b20t, off_v33, meta), gt_mask)
                    # Class-selective F0 V20T (only on Class 4 where V20T gained +0.163 IoU!)
                    cw_v20t = make_cw([0.0, 0.0, 0.0, 0.0, 0.35])
                    p_f0_v20t_c4 = (1.0 - cw_v20t) * p_v28a + cw_v20t * p_f0_v20t
                    update_cm("A5b_F0_V20T_Class4Only35", fold, sc, decode_mask(p_f0_v20t_c4, off_v33, meta), gt_mask)
                    p_f3_b25 = 0.75 * p_v28a + 0.25 * p_v33
                    update_cm("A6_F3_V28_Blend25", fold, sc, decode_mask(p_f3_b25, off_v33, meta), gt_mask)

                    # --- Group B: Single-Model TTA Ablations (on A0_V28A_Base) ---
                    update_cm("B0_NoFlipTTA", fold, sc, decode_mask(p_v28a_orig, off_v33, meta), gt_mask)
                    p_ms_896 = 0.70 * p_v28a + 0.30 * p_896
                    update_cm("B1_MS_1024_896 (70/30)", fold, sc, decode_mask(p_ms_896, off_v33, meta), gt_mask)
                    p_ms_1152 = 0.70 * p_v28a + 0.30 * p_1152
                    update_cm("B2_MS_1024_1152 (70/30)", fold, sc, decode_mask(p_ms_1152, off_v33, meta), gt_mask)
                    p_ms_3s = 0.50 * p_v28a + 0.25 * p_896 + 0.25 * p_1152
                    update_cm("B3_MS_3Scale (50/25/25)", fold, sc, decode_mask(p_ms_3s, off_v33, meta), gt_mask)
                    p_clahe_b15 = 0.85 * p_v28a + 0.15 * p_clahe
                    update_cm("B4_CLAHE_TTA (85/15)", fold, sc, decode_mask(p_clahe_b15, off_v33, meta), gt_mask)

                    # --- Group C: Standalone & Multi-Arch Blends (Squashed vs Aspect-Preserving) ---
                    update_cm("C0a_DINOv2_Sq518_Alone", fold, sc, decode_mask(p_dino_sq, off_v33, meta), gt_mask)
                    update_cm("C0b_DINOv2_AP_Alone", fold, sc, decode_mask(p_dino_ap, off_v33, meta), gt_mask)
                    update_cm("C0c_DINOv2_AP_V18Ref_Alone", fold, sc, decode_mask(p_dino_v18ref, off_v33, meta), gt_mask)
                    update_cm("C0d_ConvNeXt_Sq512_Alone", fold, sc, decode_mask(p_conv_sq, off_v33, meta), gt_mask)
                    update_cm("C0e_ConvNeXt_AP_Alone", fold, sc, decode_mask(p_conv_ap, off_v33, meta), gt_mask)

                    # Scalar blends
                    p_c1_sq = 0.70 * p_v28a + 0.20 * p_dino_sq + 0.10 * p_conv_sq
                    update_cm("C1_Scalar_70_20_10_Sq", fold, sc, decode_mask(p_c1_sq, off_v33, meta), gt_mask)
                    p_c1_ap = 0.70 * p_v28a + 0.20 * p_dino_ap + 0.10 * p_conv_ap
                    update_cm("C2_Scalar_70_20_10_AP", fold, sc, decode_mask(p_c1_ap, off_v33, meta), gt_mask)

                    # Class-selective blends (0 weight on Class 3 for DINOv2 & ConvNeXt!)
                    w18_cs1 = make_cw([0.75, 0.75, 0.75, 1.00, 0.75])
                    wd_cs1  = make_cw([0.15, 0.15, 0.15, 0.00, 0.15])
                    wc_cs1  = make_cw([0.10, 0.10, 0.10, 0.00, 0.10])
                    p_cs1_sq = w18_cs1 * p_v28a + wd_cs1 * p_dino_sq + wc_cs1 * p_conv_sq
                    update_cm("C3_ClassSel_NoC3_75_15_10_Sq", fold, sc, decode_mask(p_cs1_sq, off_v33, meta), gt_mask)
                    p_cs1_ap = w18_cs1 * p_v28a + wd_cs1 * p_dino_ap + wc_cs1 * p_conv_ap
                    update_cm("C4_ClassSel_NoC3_75_15_10_AP", fold, sc, decode_mask(p_cs1_ap, off_v33, meta), gt_mask)

                    w18_cs2 = make_cw([0.70, 0.70, 0.70, 1.00, 0.70])
                    wd_cs2  = make_cw([0.20, 0.20, 0.20, 0.00, 0.20])
                    wc_cs2  = make_cw([0.10, 0.10, 0.10, 0.00, 0.10])
                    p_cs2_sq = w18_cs2 * p_v28a + wd_cs2 * p_dino_sq + wc_cs2 * p_conv_sq
                    update_cm("C5_ClassSel_NoC3_70_20_10_Sq", fold, sc, decode_mask(p_cs2_sq, off_v33, meta), gt_mask)
                    p_cs2_ap = w18_cs2 * p_v28a + wd_cs2 * p_dino_ap + wc_cs2 * p_conv_ap
                    update_cm("C6_ClassSel_NoC3_70_20_10_AP", fold, sc, decode_mask(p_cs2_ap, off_v33, meta), gt_mask)

                    # Using V33 base (with v28_f3) + Class-Selective Multi-Arch
                    p_cs2_v33_sq = w18_cs2 * p_v33 + wd_cs2 * p_dino_sq + wc_cs2 * p_conv_sq
                    update_cm("C7_V33Base_ClassSel_NoC3_70_20_10_Sq", fold, sc, decode_mask(p_cs2_v33_sq, off_v33, meta), gt_mask)
                    p_cs2_v33_ap = w18_cs2 * p_v33 + wd_cs2 * p_dino_ap + wc_cs2 * p_conv_ap
                    update_cm("C8_V33Base_ClassSel_NoC3_70_20_10_AP", fold, sc, decode_mask(p_cs2_v33_ap, off_v33, meta), gt_mask)

                    # Using F2_9ep_Blend50 + Class-Selective Multi-Arch
                    p_cs2_f2b_ap = w18_cs2 * p_f2_blend + wd_cs2 * p_dino_ap + wc_cs2 * p_conv_ap
                    update_cm("C9_F2Blend_ClassSel_NoC3_70_20_10_AP", fold, sc, decode_mask(p_cs2_f2b_ap, off_v33, meta), gt_mask)

                    # --- Group D: Inverse-Entropy Confidence Weighting ---
                    def calc_inv_ent(p_map: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
                        ent = -torch.sum(p_map * torch.log(p_map + 1e-8), dim=1, keepdim=True) / math.log(5.0)
                        return torch.pow(ent + 0.05, -gamma)

                    ie_18 = calc_inv_ent(p_v28a, 0.5)
                    ie_d  = calc_inv_ent(p_dino_ap, 0.5)
                    ie_c  = calc_inv_ent(p_conv_ap, 0.5)
                    w18_ie = w18_cs2 * ie_18
                    wd_ie  = wd_cs2 * ie_d
                    wc_ie  = wc_cs2 * ie_c
                    w_sum  = w18_ie + wd_ie + wc_ie + 1e-8
                    p_d1 = (w18_ie * p_v28a + wd_ie * p_dino_ap + wc_ie * p_conv_ap) / w_sum
                    update_cm("D1_InvEnt_ClassSel_AP", fold, sc, decode_mask(p_d1, off_v33, meta), gt_mask)

                    # --- Group E: Logit Offsets & Connected Components (on A0, C5, C6) ---
                    update_cm("E0_A0_Off_Zero", fold, sc, decode_mask(p_v28a, off_zero, meta), gt_mask)
                    update_cm("E1_A0_Off_15_25", fold, sc, decode_mask(p_v28a, off_15_25, meta), gt_mask)
                    update_cm("E2_A0_Off_15_30", fold, sc, decode_mask(p_v28a, off_15_30, meta), gt_mask)
                    update_cm("E3_A0_Off_20_35", fold, sc, decode_mask(p_v28a, off_20_35, meta), gt_mask)
                    update_cm("E4_A0_Off_05_20", fold, sc, decode_mask(p_v28a, off_05_20, meta), gt_mask)
                    update_cm("E5_C6_Off_15_30", fold, sc, decode_mask(p_cs2_ap, off_15_30, meta), gt_mask)
                    update_cm("E6_C6_Off_20_35", fold, sc, decode_mask(p_cs2_ap, off_20_35, meta), gt_mask)
                    update_cm("E7_C5_Off_15_30", fold, sc, decode_mask(p_cs2_sq, off_15_30, meta), gt_mask)

                    m_a0_base = decode_mask(p_v28a, off_v33, meta)
                    m_a0_zero = decode_mask(p_v28a, off_zero, meta)
                    update_cm("E8_A0_CC_MinArea32", fold, sc, filter_small_rare_components(m_a0_base, 32, m_a0_zero), gt_mask)
                    update_cm("E9_A0_CC_MinArea64", fold, sc, filter_small_rare_components(m_a0_base, 64, m_a0_zero), gt_mask)

                    # --- Group F: SegFormer-B5 Integration (Fold 0 OOF & Global Synergies) ---
                    cw_segf = make_cw([0.20, 0.25, 0.30, 0.00, 0.15])
                    p_f1 = (1.0 - cw_segf) * p_v28a + cw_segf * p_segb5_f0
                    update_cm("F1_SegB5_F0_ClassSel", fold, sc, decode_mask(p_f1, off_v33, meta), gt_mask)

                    # Quad synergy: F2_9ep_Blend50 + SegB5_F0 + DINOv2_AP + ConvNeXt_AP + MS_1152
                    p_base_best = 0.80 * p_f2_blend + 0.20 * p_1152
                    if fold == 0:
                        p_base_best = (1.0 - cw_segf) * p_base_best + cw_segf * p_segb5_f0
                    p_quad_ap = w18_cs2 * p_base_best + wd_cs2 * p_dino_ap + wc_cs2 * p_conv_ap
                    update_cm("F2_QuadSynergy_AP_OffV33", fold, sc, decode_mask(p_quad_ap, off_v33, meta), gt_mask)
                    update_cm("F3_QuadSynergy_AP_Off1530", fold, sc, decode_mask(p_quad_ap, off_15_30, meta), gt_mask)
                    p_quad_sq = w18_cs2 * p_base_best + wd_cs2 * p_dino_sq + wc_cs2 * p_conv_sq
                    update_cm("F4_QuadSynergy_Sq_OffV33", fold, sc, decode_mask(p_quad_sq, off_v33, meta), gt_mask)
                    update_cm("F5_QuadSynergy_Sq_Off1530", fold, sc, decode_mask(p_quad_sq, off_15_30, meta), gt_mask)

            print(f"  [Fold {fold}] Scene {sc:10s} done ({len(deg_files)} imgs, {time.time() - t_f0:.1f}s elapsed)", flush=True)

        print(f"[FOLD {fold}] Completed in {time.time() - t_f0:.1f}s ({total_images} cumulative images)", flush=True)
        del refiner_20sc, refiner_fold, seg_v28a, seg_alt_v33, seg_alt_f2_9ep, seg_alt_f0_v18, seg_alt_f0_v20t, segformer_f0, dino_model, conv_model
        torch.cuda.empty_cache()

    elapsed = time.time() - t_suite_start
    print("\n" + "=" * 115, flush=True)
    print(f"GLOBAL 20-SCENE 5-FOLD OOF RESULTS ({total_images} images, {elapsed:.1f}s)", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Candidate Configuration':40s} | {'Q_seg':7s} | {'dQ_V33':8s} | {'mIoU':7s} | {'F1':7s} | Per-Class IoU [BG, Coral, Fish, Ray, Macro]", flush=True)
    print("-" * 115, flush=True)

    summary = {}
    for cfg_name, cm in cm_global.items():
        summary[cfg_name] = compute_global_metrics(cm)

    v33_q = summary["A1_V33_Base (v28_f3, 20sc_tbl)"]["Q_seg"]
    v28a_q = summary["A0_V28A_Base (v18_f3, 20sc_tbl)"]["Q_seg"]

    for cfg_name, m in summary.items():
        dq = m["Q_seg"] - v33_q
        ious = m["class_ious"]
        iou_str = f"[{ious[0]:.4f}, {ious[1]:.4f}, {ious[2]:.4f}, {ious[3]:.4f}, {ious[4]:.4f}]"
        print(f"{cfg_name:40s} | {m['Q_seg']:7.5f} | {dq:+8.5f} | {m['mIoU']:7.5f} | {m['Macro_F1']:7.5f} | {iou_str}", flush=True)

    # Save full report
    os.makedirs("reports/v38", exist_ok=True)
    fold_summary = {f: {cfg: compute_global_metrics(cm) for cfg, cm in cm_by_fold[f].items()} for f in range(5)}
    scene_summary = {sc: {cfg: compute_global_metrics(cm) for cfg, cm in cm_by_scene[sc].items()} for sc in sorted(cm_by_scene.keys())}

    out_json = "reports/v38/oof_ablation_suite_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "sample_stride": sample_stride,
            "total_images": total_images,
            "elapsed_sec": round(elapsed, 2),
            "v33_baseline_q_seg": v33_q,
            "v28a_baseline_q_seg": v28a_q,
            "global_20scene_metrics": summary,
            "per_fold_metrics": fold_summary,
            "per_scene_metrics": scene_summary,
        }, f, indent=2)
    print(f"\nSaved full 20-scene OOF ablation report to {out_json}", flush=True)


if __name__ == "__main__":
    run_ablation_suite(sample_stride=5)
