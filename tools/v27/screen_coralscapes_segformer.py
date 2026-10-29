"""
SAT-EURS Track 1 — V27 Experiment 1: CoralscapesV2 SegFormer-B5 Transfer & Fusion Screening
Evaluates pretrained SegFormer-B5 finetuned on CoralscapesV2 (1024x1024) across EURS held-out validation data.

Compares:
1. v18_baseline: V18 Dual-Encoder segmenter alone (Flip TTA)
2. segformer_raw_alone: SegFormer-B5 alone on raw degraded RGB (mapped to 5 EURS classes)
3. segformer_rest_alone: SegFormer-B5 alone on V18-restored RGB (mapped to 5 EURS classes)
4. fusion_raw_w15: Uniform fusion with SegFormer on raw RGB (w=0.15)
5. fusion_rest_w15: Uniform fusion with SegFormer on restored RGB (w=0.15)
6. fusion_class_selective: Class-selective support (protecting V18 Class 3 shark/ray, boosting fish & macro)

Measures exact official challenge metrics:
- Q_seg = mIoU * Macro_F1 * P_cov
- Per-class IoU and Recall for all 5 classes
- Zero cross-scene leakage
"""

import os
import os.path as osp
import sys
import time
import json
import gc
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    load_checkpoint_with_provenance_check,
)
from tools.v18.evaluator import V18OOFEvaluator

# Exact Coralscapes 40-to-5 mapping verified from HuggingFace config.json
CORALSCAPES_TO_EURS_MAP = {
    # Class 0: Background / Non-biological substrate / Trash
    1: 0,   # seagrass
    2: 0,   # trash
    5: 0,   # sand
    7: 0,   # human
    8: 0,   # transect tools
    10: 0,  # algae covered substrate
    12: 0,  # unknown hard substrate
    13: 0,  # background
    14: 0,  # dark
    15: 0,  # transect line
    18: 0,  # rubble
    39: 0,  # dead clam

    # Class 1: Coral
    3: 1,   # other coral dead
    4: 1,   # other coral bleached
    6: 1,   # other coral alive
    16: 1,  # massive/meandering bleached
    17: 1,  # massive/meandering alive
    19: 1,  # branching bleached
    20: 1,  # branching dead
    21: 1,  # millepora
    22: 1,  # branching alive
    23: 1,  # massive/meandering dead
    25: 1,  # acropora alive
    27: 1,  # turbinaria
    28: 1,  # table acropora alive
    31: 1,  # pocillopora alive
    32: 1,  # table acropora dead
    33: 1,  # meandering bleached
    34: 1,  # stylophora alive
    36: 1,  # meandering alive
    37: 1,  # meandering dead

    # Class 2: Fish
    9: 2,   # fish

    # Class 3: Elasmobranchs (sharks/rays) -> NONE in Coralscapes!
    # Strictly protected from false overwriting

    # Class 4: Other Marine Fauna (Invertebrates)
    11: 4,  # other animal
    24: 4,  # clam
    26: 4,  # sea cucumber
    29: 4,  # sponge
    30: 4,  # anemone
    35: 4,  # sea urchin
    38: 4,  # crown of thorn
}


def build_taxonomy_mapping_matrix(device: torch.device) -> torch.Tensor:
    """
    Constructs a fixed [40, 5] linear projection matrix mapping Coralscapes 40 classes
    to EURS 5 classes. Channel 0 (void/unlabeled) has 0 weight in all columns.
    """
    M = torch.zeros((40, 5), dtype=torch.float32, device=device)
    for c_src, c_tgt in CORALSCAPES_TO_EURS_MAP.items():
        M[c_src, c_tgt] = 1.0
    return M


def run_coralscapes_screening(
    test_scenes: List[Tuple[int, str]] = [(0, "seahorse1"), (1, "sharkFish")],
    data_root: str = "data/cached_1024",
    checkpoint_dir: str = "checkpoints/v18",
    alpha_restoration: float = 0.10,
    sigma_restoration: float = 2.0,
    cal_vector: List[float] = [0.0, 0.0, 0.0, 0.10, 0.25],
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 90, flush=True)
    print("SAT-EURS V27 — EXPERIMENT 1: CORALSCAPES-V2 SEGFORMER-B5 TRANSFER SCREENING", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"Test Scenes: {test_scenes}", flush=True)
    print(f"Restoration Alpha: {alpha_restoration} | Logit Calibration: {cal_vector}", flush=True)
    print("=" * 90, flush=True)

    # 1. Load SegFormer-B5
    model_id = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024"
    print(f"Loading Coralscapes SegFormer-B5 from {model_id}...", flush=True)
    segformer = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
    segformer.eval()

    # ImageNet normalization constants for SegFormer
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    mapping_mat = build_taxonomy_mapping_matrix(device)
    cal_t = torch.tensor(cal_vector, device=device).view(1, 5, 1, 1)

    CONFIG_NAMES = [
        "v18_baseline",
        "segformer_raw_alone",
        "segformer_rest_alone",
        "fusion_raw_w15",
        "fusion_rest_w15",
        "fusion_class_selective",
    ]

    evaluators = {cfg: V18OOFEvaluator() for cfg in CONFIG_NAMES}
    manifest = validate_manifest_and_folds()

    t0_all = time.time()
    total_images = 0

    for fold, sc in test_scenes:
        f_data = get_fold_data(manifest, fold)
        val_scenes = f_data["val_scenes"]
        train_scenes = f_data["train_scenes"]

        print(f"\n[Fold {fold}] Loading Refiner and EURSAPEXV18Segmenter for Scene '{sc}'...", flush=True)

        table_path = f"checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{fold}.pt"
        if not osp.exists(table_path):
            table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
        rsuigm_table = torch.load(table_path, map_location=device)

        refiner = build_v18_restoration_refiner(rsuigm_table=rsuigm_table, use_neural_refiner=True).to(device)
        r_path = osp.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        r_weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, train_scenes, val_scenes)
        refiner.load_state_dict(r_weights)
        refiner.eval()

        segmenter = build_v18_segmenter(num_classes=5).to(device)
        s_path = osp.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
        s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, train_scenes, val_scenes)
        segmenter.load_state_dict(s_weights)
        segmenter.eval()

        m_path = osp.join(data_root, "masks", f"{sc}.png")
        gt_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

        print(f"Evaluating 200 degradation variants for scene '{sc}' across {len(CONFIG_NAMES)} modes...", flush=True)
        t_sc = time.time()

        for wt in range(1, 11):
            for depth in range(5, 105, 5):
                deg_path = osp.join(data_root, "degraded", sc, f"{sc}_{wt}_{depth}.png")
                deg_bgr = cv2.imread(deg_path, cv2.IMREAD_COLOR)
                deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                deg_t = torch.from_numpy(deg_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)

                with torch.no_grad():
                    # 1. Restoration
                    out_r = refiner(deg_t, wt_t, dp_t)
                    refined_t = out_r["refined_image"]

                    if alpha_restoration > 0:
                        refined_np = refined_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        blur = cv2.GaussianBlur(refined_np, (0, 0), sigma_restoration)
                        rest_final_np = np.clip(refined_np + alpha_restoration * (refined_np - blur), 0.0, 1.0)
                        rest_input_t = torch.from_numpy(rest_final_np).permute(2, 0, 1).unsqueeze(0).to(device)
                    else:
                        rest_input_t = refined_t

                    # 2. V18 Segmenter Forward with Flip TTA
                    deg_flip = torch.flip(deg_t, dims=[3])
                    rest_flip = torch.flip(rest_input_t, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        v18_orig = segmenter(rest_input_t, deg_t, wt_t, dp_t)
                        p_v18_orig = F.softmax(v18_orig["logits"].float(), dim=1)
                        v18_flip = segmenter(rest_flip, deg_flip, wt_t, dp_t)
                        p_v18_flip = torch.flip(F.softmax(v18_flip["logits"].float(), dim=1), dims=[3])
                        prob_v18 = 0.5 * (p_v18_orig + p_v18_flip)  # [1, 5, 1024, 1024]

                    # 3. SegFormer-B5 on Raw Input
                    norm_raw = (deg_t - mean) / std
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_sf_raw = segformer(norm_raw)
                        logits_sf_raw = F.interpolate(out_sf_raw.logits.float(), size=(1024, 1024), mode="bilinear", align_corners=False)
                        p_sf_40_raw = F.softmax(logits_sf_raw, dim=1)  # [1, 40, 1024, 1024]
                        # Map to 5 EURS classes: [1, 5, 1024, 1024]
                        # Transpose for batch matrix multiplication: (B, H, W, 40) @ (40, 5) -> (B, H, W, 5)
                        prob_sf_raw = torch.einsum("bchw,cd->bdhw", p_sf_40_raw, mapping_mat)
                        # Re-normalize to sum to 1.0 (excluding void)
                        prob_sf_raw = prob_sf_raw / (prob_sf_raw.sum(dim=1, keepdim=True) + 1e-10)

                    # 4. SegFormer-B5 on Restored Input
                    norm_rest = (rest_input_t - mean) / std
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_sf_rest = segformer(norm_rest)
                        logits_sf_rest = F.interpolate(out_sf_rest.logits.float(), size=(1024, 1024), mode="bilinear", align_corners=False)
                        p_sf_40_rest = F.softmax(logits_sf_rest, dim=1)
                        prob_sf_rest = torch.einsum("bchw,cd->bdhw", p_sf_40_rest, mapping_mat)
                        prob_sf_rest = prob_sf_rest / (prob_sf_rest.sum(dim=1, keepdim=True) + 1e-10)

                    # 5. Evaluate all configurations
                    # Config 1: V18 Baseline
                    log_p1 = torch.log(prob_v18 + 1e-10) + cal_t
                    m1 = torch.argmax(log_p1, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    evaluators["v18_baseline"].update_segmentation(m1, gt_mask, scene=sc)

                    # Config 2: SegFormer Raw Alone
                    m2 = torch.argmax(prob_sf_raw, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    evaluators["segformer_raw_alone"].update_segmentation(m2, gt_mask, scene=sc)

                    # Config 3: SegFormer Restored Alone
                    m3 = torch.argmax(prob_sf_rest, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    evaluators["segformer_rest_alone"].update_segmentation(m3, gt_mask, scene=sc)

                    # Config 4: Fusion Raw w=0.15
                    w = 0.15
                    prob_f_raw = (1.0 - w) * prob_v18 + w * prob_sf_raw
                    log_p4 = torch.log(prob_f_raw + 1e-10) + cal_t
                    m4 = torch.argmax(log_p4, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    evaluators["fusion_raw_w15"].update_segmentation(m4, gt_mask, scene=sc)

                    # Config 5: Fusion Restored w=0.15
                    prob_f_rest = (1.0 - w) * prob_v18 + w * prob_sf_rest
                    log_p5 = torch.log(prob_f_rest + 1e-10) + cal_t
                    m5 = torch.argmax(log_p5, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    evaluators["fusion_rest_w15"].update_segmentation(m5, gt_mask, scene=sc)

                    # Config 6: Class-Selective Fusion (Strictly Protect V18 Class 3, support Fish w=0.15, Macro w=0.15, Coral w=0.10)
                    prob_cs = prob_v18.clone()
                    prob_cs[:, 1, :, :] = (1.0 - 0.10) * prob_v18[:, 1, :, :] + 0.10 * prob_sf_rest[:, 1, :, :]
                    prob_cs[:, 2, :, :] = (1.0 - 0.15) * prob_v18[:, 2, :, :] + 0.15 * prob_sf_rest[:, 2, :, :]
                    # Class 3 strictly preserved from V18 (zero external interference)
                    prob_cs[:, 3, :, :] = prob_v18[:, 3, :, :]
                    prob_cs[:, 4, :, :] = (1.0 - 0.15) * prob_v18[:, 4, :, :] + 0.15 * prob_sf_rest[:, 4, :, :]
                    # Normalize
                    prob_cs = prob_cs / (prob_cs.sum(dim=1, keepdim=True) + 1e-10)
                    log_p6 = torch.log(prob_cs + 1e-10) + cal_t
                    m6 = torch.argmax(log_p6, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    evaluators["fusion_class_selective"].update_segmentation(m6, gt_mask, scene=sc)

                del deg_t, wt_t, dp_t, out_r, refined_t, rest_input_t, deg_flip, rest_flip, prob_v18, prob_sf_raw, prob_sf_rest
                total_images += 1

        print(f"  Scene '{sc}' completed in {time.time() - t_sc:.1f}s", flush=True)
        del refiner, segmenter
        gc.collect()
        torch.cuda.empty_cache()

    # Results Compilation
    print("\n" + "=" * 115, flush=True)
    print("           EXPERIMENT 1: CORALSCAPES SEGFORMER-B5 TRANSFER RESULTS SUMMARY", flush=True)
    print("=" * 115, flush=True)
    print(f"{'Configuration':<26} | {'mIoU':<8} | {'Macro F1':<9} | {'P_cov':<6} | {'Q_seg':<8} | {'Delta Q_seg':<12}", flush=True)
    print("-" * 115, flush=True)

    base_qseg = evaluators["v18_baseline"].compute()["segmentation"]["Q_seg"]
    all_results = {}

    for cfg in CONFIG_NAMES:
        seg = evaluators[cfg].compute()["segmentation"]
        qseg = seg["Q_seg"]
        dq = qseg - base_qseg

        print(
            f"{cfg:<26} | {seg['mIoU']:<8.4f} | {seg['Macro_F1']:<9.4f} | "
            f"{seg['coverage_penalty']:<6.1f} | {qseg:<8.4f} | {dq:+10.4f}",
            flush=True
        )
        class_names = ["BG", "Coral", "Fish", "Ray", "Macro"]
        c_ious = [f"{class_names[c]}:{seg['class_ious'][c]:.3f}" for c in range(5)]
        c_recs = [f"{class_names[c]}:{seg['class_recalls'][c]:.3f}" for c in range(5)]
        print(f"       Per-class IoU:    {' | '.join(c_ious)}", flush=True)
        print(f"       Per-class Recall: {' | '.join(c_recs)}", flush=True)

        all_results[cfg] = {
            "metrics": seg,
            "delta_qseg_vs_v18": dq,
        }

    print("=" * 115, flush=True)

    out_json = "reports/v27/exp1_coralscapes_segformer_results.json"
    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved screening results to {out_json}", flush=True)

    # Check Fusion Promotion Gate
    best_fusion_cfg = max(["fusion_raw_w15", "fusion_rest_w15", "fusion_class_selective"], key=lambda c: all_results[c]["delta_qseg_vs_v18"])
    best_dq = all_results[best_fusion_cfg]["delta_qseg_vs_v18"]

    print("\n--- FUSION PROMOTION GATE EVALUATION ---")
    print(f"Promotion Criterion: Delta Q_seg >= +0.010 with no Class 3 regression")
    print(f"Best Fusion Config : {best_fusion_cfg} (Delta Q_seg: {best_dq:+.4f})")
    ray_iou_base = all_results["v18_baseline"]["metrics"]["class_ious"][3]
    ray_iou_best = all_results[best_fusion_cfg]["metrics"]["class_ious"][3]
    print(f"Class 3 (Shark/Ray) IoU: Base={ray_iou_base:.3f} -> Fusion={ray_iou_best:.3f}")

    if best_dq >= 0.010 and ray_iou_best >= ray_iou_base - 0.005:
        print(">>> PROMOTION GATE STATUS: PASSED! Proceeding with full 5-fold evaluation.")
    else:
        print(f">>> PROMOTION GATE STATUS: REJECTED (Delta Q_seg {best_dq:+.4f} < +0.010 threshold).")
        print(">>> Action: Reject external fusion and proceed directly to Priority Two (small-object-aware segmenter training).")

    return all_results


if __name__ == "__main__":
    run_coralscapes_screening()
