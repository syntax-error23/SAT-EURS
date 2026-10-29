"""
SAT-EURS Track 1 — V26 Experiment 3: Multi-Scale & Flip TTA Evaluation
Protocol: Held-out validation scenes covering all key classes:
- seahorse1 (Fold 0: Class 4 Macro-fauna)
- sharkFish (Fold 1: Class 2 Fish & Class 3 Shark)
- fish1 (Fold 0: Class 2 Fish)

Compares:
1. single_1024_noflip: 1.0x single pass (no flip)
2. single_1024_flip:   1.0x with horizontal flip (V18 baseline)
3. two_scale_flip:     [1.0x, 1.125x] (1024, 1152) with horizontal flip
4. three_scale_flip:   [0.875x, 1.0x, 1.125x] (896, 1024, 1152) with horizontal flip

Measures:
- Q_seg, mIoU, Macro F1, Coverage Penalty
- Per-class IoU and Recall (ensuring no Class 3/4 regression)
- Inference latency per image (ms)
- Peak GPU memory allocation (MB)
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

sys.path.insert(0, ".")

from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import (
    validate_manifest_and_folds,
    get_fold_data,
    load_checkpoint_with_provenance_check,
)
from tools.v18.evaluator import V18OOFEvaluator

CONFIGS = [
    "single_1024_noflip",
    "single_1024_flip",
    "two_scale_flip",
    "three_scale_flip",
]


def predict_tta(
    segmenter: torch.nn.Module,
    rest_img: torch.Tensor,     # [1, 3, 1024, 1024]
    deg_img: torch.Tensor,      # [1, 3, 1024, 1024]
    wt_t: torch.Tensor,
    dp_t: torch.Tensor,
    mode: str,
    device: torch.device
) -> torch.Tensor:
    """
    Executes TTA forward passes and returns fused probability map [1, 5, 1024, 1024].
    """
    orig_h, orig_w = 1024, 1024

    if mode == "single_1024_noflip":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = segmenter(rest_img, deg_img, wt_t, dp_t)
            prob = F.softmax(out["logits"].float(), dim=1)
        return prob

    elif mode == "single_1024_flip":
        deg_flip = torch.flip(deg_img, dims=[3])
        rest_flip = torch.flip(rest_img, dims=[3])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p1 = F.softmax(segmenter(rest_img, deg_img, wt_t, dp_t)["logits"].float(), dim=1)
            p2 = torch.flip(F.softmax(segmenter(rest_flip, deg_flip, wt_t, dp_t)["logits"].float(), dim=1), dims=[3])
        return 0.5 * (p1 + p2)

    elif mode == "two_scale_flip":
        scales = [(1024, 1024), (1152, 1152)]
        prob_accum = torch.zeros((1, 5, orig_h, orig_w), device=device)
        for h, w in scales:
            if h == orig_h and w == orig_w:
                r_in = rest_img
                d_in = deg_img
            else:
                r_in = F.interpolate(rest_img, size=(h, w), mode="bilinear", align_corners=False)
                d_in = F.interpolate(deg_img, size=(h, w), mode="bilinear", align_corners=False)

            d_flip = torch.flip(d_in, dims=[3])
            r_flip = torch.flip(r_in, dims=[3])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p1 = F.softmax(segmenter(r_in, d_in, wt_t, dp_t, unpadded_size=(h, w))["logits"].float(), dim=1)
                p2 = torch.flip(F.softmax(segmenter(r_flip, d_flip, wt_t, dp_t, unpadded_size=(h, w))["logits"].float(), dim=1), dims=[3])
                p_scale = 0.5 * (p1 + p2)

            if h != orig_h or w != orig_w:
                p_scale = F.interpolate(p_scale, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

            prob_accum += p_scale
        return prob_accum / len(scales)

    elif mode == "three_scale_flip":
        scales = [(896, 896), (1024, 1024), (1152, 1152)]
        prob_accum = torch.zeros((1, 5, orig_h, orig_w), device=device)
        for h, w in scales:
            if h == orig_h and w == orig_w:
                r_in = rest_img
                d_in = deg_img
            else:
                r_in = F.interpolate(rest_img, size=(h, w), mode="bilinear", align_corners=False)
                d_in = F.interpolate(deg_img, size=(h, w), mode="bilinear", align_corners=False)

            d_flip = torch.flip(d_in, dims=[3])
            r_flip = torch.flip(r_in, dims=[3])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p1 = F.softmax(segmenter(r_in, d_in, wt_t, dp_t, unpadded_size=(h, w))["logits"].float(), dim=1)
                p2 = torch.flip(F.softmax(segmenter(r_flip, d_flip, wt_t, dp_t, unpadded_size=(h, w))["logits"].float(), dim=1), dims=[3])
                p_scale = 0.5 * (p1 + p2)

            if h != orig_h or w != orig_w:
                p_scale = F.interpolate(p_scale, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

            prob_accum += p_scale
        return prob_accum / len(scales)

    else:
        raise ValueError(f"Unknown TTA mode: {mode}")


def run_exp3_tta(
    data_root: str = "data/cached_1024",
    checkpoint_dir: str = "checkpoints/v18",
    alpha_restoration: float = 0.10,
    sigma_restoration: float = 2.0,
    offset_vec: List[float] = [0.0, 0.0, 0.0, 0.15, 0.15],
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print("SAT-EURS V26 — EXPERIMENT 3: MULTI-SCALE AND FLIP TTA BENCHMARK", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"TTA Configurations: {CONFIGS}", flush=True)
    print("=" * 85, flush=True)

    manifest = validate_manifest_and_folds()

    # Test scenes: seahorse1 (Fold 0), fish1 (Fold 0), sharkFish (Fold 1)
    test_eval_scenes = [
        (0, "seahorse1"),
        (0, "fish1"),
        (1, "sharkFish"),
    ]

    evaluators = {cfg: V18OOFEvaluator() for cfg in CONFIGS}
    timings = {cfg: [] for cfg in CONFIGS}
    peak_mem = {cfg: 0 for cfg in CONFIGS}

    cal_t = torch.tensor(offset_vec, device=device).view(1, 5, 1, 1)

    for fold, sc in test_eval_scenes:
        f_data = get_fold_data(manifest, fold)
        val_scenes = f_data["val_scenes"]
        train_scenes = f_data["train_scenes"]

        print(f"\n[Fold {fold}] Loading Refiner and Dual-Encoder Segmenter for Scene '{sc}'...", flush=True)

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

        print(f"Evaluating 200 degradation variants for scene '{sc}' across {len(CONFIGS)} TTA modes...", flush=True)
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
                    out_r = refiner(deg_t, wt_t, dp_t)
                    refined_t = out_r["refined_image"]

                    if alpha_restoration > 0:
                        refined_np = refined_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        blur = cv2.GaussianBlur(refined_np, (0, 0), sigma_restoration)
                        rest_final_np = np.clip(refined_np + alpha_restoration * (refined_np - blur), 0.0, 1.0)
                        rest_input_t = torch.from_numpy(rest_final_np).permute(2, 0, 1).unsqueeze(0).to(device)
                    else:
                        rest_input_t = refined_t

                    for cfg in CONFIGS:
                        torch.cuda.reset_peak_memory_stats(device)
                        t0 = time.time()

                        prob_fused = predict_tta(segmenter, rest_input_t, deg_t, wt_t, dp_t, mode=cfg, device=device)
                        log_prob = torch.log(prob_fused + 1e-10)
                        pred_mask = torch.argmax(log_prob + cal_t, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                        dt = (time.time() - t0) * 1000.0  # ms
                        timings[cfg].append(dt)
                        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                        if mem_mb > peak_mem[cfg]:
                            peak_mem[cfg] = mem_mb

                        evaluators[cfg].update_segmentation(pred_mask, gt_mask, scene=sc)

                del deg_t, wt_t, dp_t, out_r, refined_t, rest_input_t, prob_fused, log_prob, pred_mask

        print(f"  Scene '{sc}' completed in {time.time() - t_sc:.1f}s", flush=True)

        del refiner, segmenter
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 125, flush=True)
    print("                      EXPERIMENT 3: MULTI-SCALE & FLIP TTA BENCHMARK RESULTS", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Configuration':<25} | {'mIoU':<8} | {'Macro F1':<9} | {'P_cov':<6} | {'Q_seg':<8} | {'Latency/img':<12} | {'Peak Mem':<10}", flush=True)
    print("-" * 125, flush=True)

    results = {}
    base_qseg = None

    for cfg in CONFIGS:
        seg_res = evaluators[cfg].compute()["segmentation"]
        qseg = seg_res["Q_seg"]
        if base_qseg is None:
            base_qseg = qseg

        mean_latency = float(np.mean(timings[cfg]))
        mem = peak_mem[cfg]

        print(
            f"{cfg:<25} | {seg_res['mIoU']:<8.4f} | {seg_res['Macro_F1']:<9.4f} | "
            f"{seg_res['coverage_penalty']:<6.1f} | {qseg:<8.4f} | {mean_latency:<8.1f} ms | {mem:<8.1f} MB",
            flush=True
        )

        class_names = ["BG", "Coral", "Fish", "Ray", "Macro"]
        c_ious = [f"{class_names[c]}:{seg_res['class_ious'][c]:.3f}" for c in range(5)]
        c_recs = [f"{class_names[c]}:{seg_res['class_recalls'][c]:.3f}" for c in range(5)]
        print(f"       Per-class IoU:    {' | '.join(c_ious)}", flush=True)
        print(f"       Per-class Recall: {' | '.join(c_recs)}", flush=True)

        results[cfg] = {
            "metrics": seg_res,
            "mean_latency_ms": mean_latency,
            "peak_gpu_mem_mb": mem,
            "delta_qseg_vs_noflip": qseg - base_qseg
        }

    print("=" * 125, flush=True)

    out_json = "reports/v26/exp3_multiscale_tta_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Exp3 benchmark results to {out_json}", flush=True)

    return results


if __name__ == "__main__":
    run_exp3_tta()
