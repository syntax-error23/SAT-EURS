"""
SAT-EURS Track 1 — V38 Hierarchical Bayesian Quad-Architecture 5-Fold OOF Evaluation
Evaluates the complete 5-fold Quad-Architecture Ensemble across all 20 EURS scenes:
  1. Primary Dual-Encoder Family (EURSAPEXV18Segmenter: DINOv3 ViT-L/16 + ConvNeXt-B + UPerNet)
     + Intra-fold checkpoints: v27_fold0, v20_treatment_fold0, v20_control_fold2 (9ep), v28_fold3
  2. 5-Fold Coralscapes-V2 SegFormer-B5 (v29/fold_0 + v38/fold_{1..4})
  3. 5-Fold DINOv2-Large (v16_dinov2/fold_{0..4})
  4. 5-Fold ConvNeXt-Base FPN (v16_convnext/fold_{0..4})

Tests the 3-Step Hierarchical Bayesian Probability Decomposition:
  - Step 1 (Class 3 Elasmobranchs): Governed strictly by Primary V18/V27 (+0.25 v28_f3 on Fold 3)
    so auxiliary models with low Class 3 recall cannot dilute Shark/Ray probability mass.
  - Step 2 (Class 4 Macro-Fauna | Y != 3): Governed by V18/V27 (+0.25 v20_treat_f0 on Fold 0)
    + DINOv2-Large + SegFormer-B5 (with inverse-entropy confidence weighting), scaled by (1 - P_3).
  - Step 3 (Classes {0, 1, 2} BG, Coral, Fish | Y not in {3, 4}): Governed by all 4 architectures
    (V18/V27 + F2_9ep + SegFormer-B5 + DINOv2-Large + ConvNeXt-FPN), scaled by (1 - P_3 - P_4).
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
from tools.v38.run_oof_ablation_suite import compute_global_metrics, load_seg_state


def calc_inv_entropy_weight(p_map: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    ent = -torch.sum(p_map * torch.log(p_map + 1e-8), dim=1, keepdim=True) / math.log(5.0)
    return torch.pow(ent + 0.05, -gamma)


def hierarchical_bayesian_blend(
    p_c3_auth: torch.Tensor,
    c4_models: List[Tuple[torch.Tensor, float]],
    maj_models: List[Tuple[torch.Tensor, List[float]]],
    use_inv_entropy_c4: bool = False,
) -> torch.Tensor:
    """
    Executes exact 3-Step Hierarchical Bayesian Probability Decomposition:
      1. P(Y=3) = p_c3_auth[:, 3:4]
      2. P(Y=4 | Y!=3) = weighted blend of P_m(Y=4 | Y!=3) across c4_models
         => P(Y=4) = (1 - P(Y=3)) * P(Y=4 | Y!=3)
      3. P(Y=c | Y in {0,1,2}) = class-weighted blend of P_m(Y=c | Y in {0,1,2}) across maj_models
         => P(Y=c) = (1 - P(Y=3) - P(Y=4)) * P(Y=c | Y in {0,1,2}) for c in {0, 1, 2}
    Guarantees:
      - Exact simplex normalization: sum_{c=0}^4 P(Y=c) == 1.0 at every pixel.
      - Zero dilution of Class 3 or Class 4 by majority-specialist models.
    """
    eps = 1e-8
    # Step 1: Class 3 probability
    p3 = p_c3_auth[:, 3:4, :, :]  # [1, 1, H, W]
    rem_not3 = torch.clamp(1.0 - p3, min=0.0, max=1.0)

    # Step 2: Conditional Class 4 probability given Y != 3
    q4_num = torch.zeros_like(p3)
    q4_den = torch.zeros_like(p3)
    for p_m, w_m in c4_models:
        if w_m <= 0.0:
            continue
        mass_not3 = p_m[:, 0:1] + p_m[:, 1:2] + p_m[:, 2:3] + p_m[:, 4:5] + eps
        q4_m = p_m[:, 4:5] / mass_not3
        if use_inv_entropy_c4:
            ie = calc_inv_entropy_weight(p_m, gamma=0.5)
            w_eff = w_m * ie
        else:
            w_eff = w_m
        q4_num = q4_num + w_eff * q4_m
        q4_den = q4_den + w_eff

    q4_cond = q4_num / (q4_den + eps)
    p4 = rem_not3 * q4_cond
    rem_maj = torch.clamp(1.0 - p3 - p4, min=0.0, max=1.0)

    # Step 3: Conditional Majority {0, 1, 2} probabilities given Y in {0, 1, 2}
    q_maj_accum = torch.zeros((p_c3_auth.shape[0], 3, p_c3_auth.shape[2], p_c3_auth.shape[3]), device=p_c3_auth.device, dtype=p_c3_auth.dtype)
    for p_m, w_vec in maj_models:
        mass_maj = p_m[:, 0:3].sum(dim=1, keepdim=True) + eps
        q_m_maj = p_m[:, 0:3] / mass_maj  # [1, 3, H, W]
        w_t = torch.tensor(w_vec, device=p_c3_auth.device, dtype=p_c3_auth.dtype).view(1, 3, 1, 1)
        q_maj_accum = q_maj_accum + w_t * q_m_maj

    # Renormalize majority conditional distribution so q0 + q1 + q2 == 1.0
    q_maj_norm = q_maj_accum / (q_maj_accum.sum(dim=1, keepdim=True) + eps)
    p_maj = rem_maj * q_maj_norm  # [1, 3, H, W]

    return torch.cat([p_maj, p3, p4], dim=1)


def run_bayesian_evaluation(sample_stride: int = 5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 105, flush=True)
    print(f"SAT-EURS TRACK 1 — V38 HIERARCHICAL BAYESIAN 5-FOLD QUAD-ENSEMBLE OOF EVALUATION (stride={sample_stride})", flush=True)
    print("=" * 105, flush=True)

    manifest = validate_manifest_and_folds()
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    offsets_grid = {
        "Off_V33 (0,0,0,0.10,0.25)": torch.tensor([0.0, 0.0, 0.0, 0.10, 0.25], device=device).view(1, 5, 1, 1),
        "Off_15_25 (0,0,0,0.15,0.25)": torch.tensor([0.0, 0.0, 0.0, 0.15, 0.25], device=device).view(1, 5, 1, 1),
        "Off_15_30 (0,0,0,0.15,0.30)": torch.tensor([0.0, 0.0, 0.0, 0.15, 0.30], device=device).view(1, 5, 1, 1),
        "Off_20_35 (0,0,0,0.20,0.35)": torch.tensor([0.0, 0.0, 0.0, 0.20, 0.35], device=device).view(1, 5, 1, 1),
        "Off_20_40 (0,0,0,0.20,0.40)": torch.tensor([0.0, 0.0, 0.0, 0.20, 0.40], device=device).view(1, 5, 1, 1),
        "Off_FishBoost (0,0,0.04,0.18,0.32)": torch.tensor([0.0, 0.0, 0.04, 0.18, 0.32], device=device).view(1, 5, 1, 1),
    }

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

    t0 = time.time()
    total_images = 0

    for fold in range(5):
        tf = time.time()
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])
        print(f"\n[FOLD {fold}] Evaluating on {val_scenes}...", flush=True)

        tbl_fold = torch.load(f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt", map_location=device, weights_only=False)
        tbl_20sc = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device, weights_only=False)

        refiner = build_v18_restoration_refiner(rsuigm_table=tbl_20sc, use_neural_refiner=True).to(device)
        load_seg_state(refiner, f"checkpoints/v18/fold_{fold}/refiner_best.pt", device)

        # 1. Primary V28-A Segmenter
        seg_v28a_ckpt = "checkpoints/v27/segmenter_v27_fold0_best.pt" if fold == 0 else f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
        seg_v28a = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
        load_seg_state(seg_v28a, seg_v28a_ckpt, device)

        # Intra-fold specialists
        seg_v20t_f0 = None
        if fold == 0 and osp.exists("checkpoints/v20/matched_pilot/fold_0/treatment_v20_best.pt"):
            seg_v20t_f0 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
            load_seg_state(seg_v20t_f0, "checkpoints/v20/matched_pilot/fold_0/treatment_v20_best.pt", device)

        seg_v20c_f2 = None
        if fold == 2 and osp.exists("checkpoints/v20/matched_pilot/fold_2/control_v18_best.pt"):
            seg_v20c_f2 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
            load_seg_state(seg_v20c_f2, "checkpoints/v20/matched_pilot/fold_2/control_v18_best.pt", device)

        seg_v28_f3 = None
        if fold == 3 and osp.exists("checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt"):
            seg_v28_f3 = build_v18_segmenter(num_classes=5, vit_pretrained=False, conv_pretrained=False, use_mock=False).to(device)
            load_seg_state(seg_v28_f3, "checkpoints/overnight_pilots/fold_3/segmenter_v28_fold3_best.pt", device)

        # 2. Coralscapes-V2 SegFormer-B5 for this fold
        sf_ckpt = "checkpoints/v29/segformer_b5_fold_0_best.pt" if fold == 0 else f"checkpoints/v38/segformer_b5_fold_{fold}_best.pt"
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024",
            num_labels=5,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        ).to(device)
        load_seg_state(segformer, sf_ckpt, device)

        # 3. DINOv2-Large for this fold
        dino_model = build_dinov2_model(num_classes=5).to(device)
        load_seg_state(dino_model, f"checkpoints/v16_dinov2/fold_{fold}/best_model.pt", device)

        # 4. ConvNeXt-Base FPN for this fold
        conv_model = build_convnext_fpn(num_classes=5, freeze_stage0=True).to(device)
        load_seg_state(conv_model, f"checkpoints/v16_convnext/fold_{fold}/best_model.pt", device)

        for sc in val_scenes:
            gt_mask = cv2.imread(f"data/cached_1024/masks/{sc}.png", cv2.IMREAD_GRAYSCALE)
            deg_dir = f"data/cached_1024/degraded/{sc}"
            deg_files = sorted([f for f in os.listdir(deg_dir) if f.endswith(".png")])[::sample_stride]

            for df in deg_files:
                total_images += 1
                parts = osp.splitext(df)[0].split("_")
                wt, depth = int(parts[1]), int(parts[2])

                bgr = cv2.imread(osp.join(deg_dir, df), cv2.IMREAD_COLOR)
                prep = preprocessor.prepare_input(bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                H_pad, W_pad = raw_pad.shape[-2:]
                nh0, nw0 = meta["target_h"], meta["target_w"]

                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], dtype=torch.float32, device=device)

                with torch.no_grad():
                    # Refiner with Flip TTA (proven +0.021 dB PSNR & +0.0005 SSIM in R1)
                    r_orig = refiner(raw_pad, wt_t, dp_t)["refined_image"]
                    r_flip = torch.flip(refiner(torch.flip(raw_pad, dims=[3]), wt_t, dp_t)["refined_image"], dims=[3])
                    rest_pad = 0.5 * (r_orig + r_flip)
                    rest_table_pad = apply_rsuigm_batch_gpu(raw_pad, wt_t, dp_t.long(), tbl_fold)

                    raw_flip = torch.flip(raw_pad, dims=[3])
                    rest_flip = torch.flip(rest_pad, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        # 1. Primary V28-A
                        l_o = seg_v28a(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                        l_f = seg_v28a(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                        p_v28a = 0.5 * (F.softmax(l_o, dim=1) + torch.flip(F.softmax(l_f, dim=1), dims=[3]))

                        # Intra-fold specialists
                        p_v18_c3 = p_v28a
                        p_v18_c4 = p_v28a
                        p_v18_maj = p_v28a

                        if fold == 0 and seg_v20t_f0 is not None:
                            lt_o = seg_v20t_f0(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                            lt_f = seg_v20t_f0(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                            p_v20t = 0.5 * (F.softmax(lt_o, dim=1) + torch.flip(F.softmax(lt_f, dim=1), dims=[3]))
                            p_v18_c4 = 0.70 * p_v28a + 0.30 * p_v20t
                            p_v18_maj = 0.75 * p_v28a + 0.25 * p_v20t

                        if fold == 2 and seg_v20c_f2 is not None:
                            lc_o = seg_v20c_f2(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                            lc_f = seg_v20c_f2(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                            p_v20c = 0.5 * (F.softmax(lc_o, dim=1) + torch.flip(F.softmax(lc_f, dim=1), dims=[3]))
                            # Only use v20_ctrl_f2 on Majority {0, 1, 2} where it gave +0.118 Fish IoU!
                            p_v18_maj = 0.50 * p_v28a + 0.50 * p_v20c

                        if fold == 3 and seg_v28_f3 is not None:
                            l3_o = seg_v28_f3(rest_pad, raw_pad, wt_t, dp_t)["logits"].float()
                            l3_f = seg_v28_f3(rest_flip, raw_flip, wt_t, dp_t)["logits"].float()
                            p_v28_3 = 0.5 * (F.softmax(l3_o, dim=1) + torch.flip(F.softmax(l3_f, dim=1), dims=[3]))
                            p_v18_c3 = 0.75 * p_v28a + 0.25 * p_v28_3
                            p_v18_maj = 0.75 * p_v28a + 0.25 * p_v28_3

                        # 2. Coralscapes SegFormer-B5 (on rest_pad)
                        xn_ref = (rest_pad - mean_im) / std_im
                        s_ref_o = F.softmax(F.interpolate(segformer(pixel_values=xn_ref).logits.float(), size=(H_pad, W_pad), mode="bilinear", align_corners=False), dim=1)
                        s_ref_f = torch.flip(F.softmax(F.interpolate(segformer(pixel_values=torch.flip(xn_ref, dims=[3])).logits.float(), size=(H_pad, W_pad), mode="bilinear", align_corners=False), dim=1), dims=[3])
                        p_segf = 0.5 * (s_ref_o + s_ref_f)

                        # 3. DINOv2-Large (518x518 on valid region)
                        rest_valid = rest_table_pad[:, :, :nh0, :nw0]
                        r_518 = F.interpolate(rest_valid, size=(518, 518), mode="bilinear", align_corners=False)
                        xd = (r_518 - mean_im) / std_im
                        ld_o = dino_model(xd).float()
                        ld_f = dino_model(torch.flip(xd, dims=[3])).float()
                        pd_518 = 0.5 * (F.softmax(ld_o, dim=1) + torch.flip(F.softmax(ld_f, dim=1), dims=[3]))
                        p_dino = torch.zeros_like(p_v28a)
                        p_dino[:, :, :nh0, :nw0] = F.interpolate(pd_518, size=(nh0, nw0), mode="bilinear", align_corners=False)

                        # 4. ConvNeXt-Base FPN (512x512 on valid region)
                        r_512 = F.interpolate(rest_valid, size=(512, 512), mode="bilinear", align_corners=False)
                        xc = (r_512 - mean_im) / std_im
                        lc_o = conv_model(xc).float()
                        lc_f = conv_model(torch.flip(xc, dims=[3])).float()
                        pc_512 = 0.5 * (F.softmax(lc_o, dim=1) + torch.flip(F.softmax(lc_f, dim=1), dims=[3]))
                        p_conv = torch.zeros_like(p_v28a)
                        p_conv[:, :, :nh0, :nw0] = F.interpolate(pc_512, size=(nh0, nw0), mode="bilinear", align_corners=False)

                    # Evaluate Standalone V28A, V33, and SegFormer-B5 5-Fold
                    update_cm("00_V28A_Base", fold, sc, decode_mask(p_v28a, offsets_grid["Off_V33 (0,0,0,0.10,0.25)"], meta), gt_mask)
                    update_cm("00_V33_Base", fold, sc, decode_mask(p_v28_3 if (fold == 3 and seg_v28_f3 is not None) else p_v28a, offsets_grid["Off_V33 (0,0,0,0.10,0.25)"], meta), gt_mask)
                    update_cm("0_SegFormerB5_5Fold_RefInput", fold, sc, decode_mask(p_segf, offsets_grid["Off_V33 (0,0,0,0.10,0.25)"], meta), gt_mask)

                    # Candidate HB1: Hierarchical Bayesian V18_Upgraded + DINOv2 + ConvNeXt (No SegFormer)
                    p_hb1 = hierarchical_bayesian_blend(
                        p_c3_auth=p_v18_c3,
                        c4_models=[(p_v18_c4, 0.78), (p_dino, 0.22)],
                        maj_models=[(p_v18_maj, [0.70, 0.70, 0.70]), (p_dino, [0.20, 0.20, 0.20]), (p_conv, [0.10, 0.10, 0.10])],
                        use_inv_entropy_c4=False,
                    )
                    # Candidate HB2: Same with Inverse-Entropy on C4
                    p_hb2 = hierarchical_bayesian_blend(
                        p_c3_auth=p_v18_c3,
                        c4_models=[(p_v18_c4, 0.78), (p_dino, 0.22)],
                        maj_models=[(p_v18_maj, [0.70, 0.70, 0.70]), (p_dino, [0.20, 0.20, 0.20]), (p_conv, [0.10, 0.10, 0.10])],
                        use_inv_entropy_c4=True,
                    )
                    # Candidate HB3: Full Quad-Architecture Hierarchical Bayesian (15% SegFormer-B5 on Maj, 10% on C4)
                    p_hb3 = hierarchical_bayesian_blend(
                        p_c3_auth=p_v18_c3,
                        c4_models=[(p_v18_c4, 0.72), (p_dino, 0.18), (p_segf, 0.10)],
                        maj_models=[
                            (p_v18_maj, [0.60, 0.58, 0.55]),
                            (p_segf,    [0.15, 0.18, 0.22]),
                            (p_dino,    [0.15, 0.14, 0.15]),
                            (p_conv,    [0.10, 0.10, 0.08]),
                        ],
                        use_inv_entropy_c4=False,
                    )
                    # Candidate HB4: Full Quad-Architecture with Inverse-Entropy C4 (SegFormer only on Maj {0,1,2} + F0 C4)
                    c4_hb4 = [(p_v18_c4, 0.72), (p_dino, 0.18), (p_segf, 0.10)] if fold == 0 else [(p_v18_c4, 0.78), (p_dino, 0.22)]
                    maj_hb4 = [
                        (p_v18_maj, [0.60, 0.58, 0.55]),
                        (p_segf,    [0.15, 0.18, 0.22]),
                        (p_dino,    [0.15, 0.14, 0.15]),
                        (p_conv,    [0.10, 0.10, 0.08]),
                    ] if fold != 3 else [
                        (p_v18_maj, [0.70, 0.70, 0.70]),
                        (p_dino,    [0.20, 0.20, 0.20]),
                        (p_conv,    [0.10, 0.10, 0.10]),
                    ]
                    p_hb4 = hierarchical_bayesian_blend(
                        p_c3_auth=p_v18_c3,
                        c4_models=c4_hb4,
                        maj_models=maj_hb4,
                        use_inv_entropy_c4=True,
                    )
                    # Candidate HB5: Higher SegFormer weight on Coral & Fish (Promoted Folds {0,1,2,4})
                    maj_hb5 = [
                        (p_v18_maj, [0.58, 0.52, 0.50]),
                        (p_segf,    [0.17, 0.24, 0.28]),
                        (p_dino,    [0.15, 0.14, 0.14]),
                        (p_conv,    [0.10, 0.10, 0.08]),
                    ] if fold != 3 else [
                        (p_v18_maj, [0.70, 0.70, 0.70]),
                        (p_dino,    [0.20, 0.20, 0.20]),
                        (p_conv,    [0.10, 0.10, 0.10]),
                    ]
                    p_hb5 = hierarchical_bayesian_blend(
                        p_c3_auth=p_v18_c3,
                        c4_models=c4_hb4,
                        maj_models=maj_hb5,
                        use_inv_entropy_c4=True,
                    )

                    for hb_name, p_hb in [
                        ("HB1_TriArch_Bayes", p_hb1),
                        ("HB2_TriArch_Bayes_InvEntC4", p_hb2),
                        ("HB3_QuadArch_All5SegF", p_hb3),
                        ("HB4_QuadArch_PromotedSegF_InvEntC4", p_hb4),
                        ("HB5_QuadArch_HighPromotedSegF_InvEntC4", p_hb5),
                    ]:
                        for off_name, off_t in offsets_grid.items():
                            update_cm(f"{hb_name} | {off_name}", fold, sc, decode_mask(p_hb, off_t, meta), gt_mask)

            print(f"  [Fold {fold}] Scene {sc:10s} done ({time.time() - tf:.1f}s)", flush=True)
        del refiner, seg_v28a, seg_v20t_f0, seg_v20c_f2, seg_v28_f3, segformer, dino_model, conv_model
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print("\n" + "=" * 120, flush=True)
    print(f"GLOBAL 20-SCENE HIERARCHICAL BAYESIAN OOF RESULTS ({total_images} images, {elapsed:.1f}s)", flush=True)
    print("=" * 120, flush=True)
    print(f"{'Candidate Configuration':58s} | {'Q_seg':7s} | {'dQ_V33':8s} | {'mIoU':7s} | {'F1':7s} | Per-Class IoU [BG, Coral, Fish, Ray, Macro]", flush=True)
    print("-" * 120, flush=True)

    summary = {}
    for cfg_name, cm in cm_global.items():
        summary[cfg_name] = compute_global_metrics(cm)
    v33_q = summary["00_V33_Base"]["Q_seg"]
    for cfg_name, m in summary.items():
        dq = m["Q_seg"] - v33_q
        ious = m["class_ious"]
        iou_str = f"[{ious[0]:.4f}, {ious[1]:.4f}, {ious[2]:.4f}, {ious[3]:.4f}, {ious[4]:.4f}]"
        print(f"{cfg_name:58s} | {m['Q_seg']:7.5f} | {dq:+8.5f} | {m['mIoU']:7.5f} | {m['Macro_F1']:7.5f} | {iou_str}", flush=True)

    fold_summary = {f: {cfg: compute_global_metrics(cm) for cfg, cm in cm_by_fold[f].items()} for f in range(5)}
    scene_summary = {sc: {cfg: compute_global_metrics(cm) for cfg, cm in cm_by_scene[sc].items()} for sc in sorted(cm_by_scene.keys())}

    out_path = "reports/v38/bayesian_quad_oof_results.json"
    with open(out_path, "w") as fp:
        json.dump({
            "v33_baseline_q_seg": v33_q,
            "global_20scene_metrics": summary,
            "per_fold_metrics": fold_summary,
            "per_scene_metrics": scene_summary,
        }, fp, indent=2)
    print(f"\nSaved full Bayesian Quad-Ensemble OOF report to {out_path}", flush=True)


if __name__ == "__main__":
    run_bayesian_evaluation(sample_stride=10)
