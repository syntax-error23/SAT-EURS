"""
SAT-EURS Track 1 — V37 External Foundation Model Pilot
Architecture: CoralscapesV2 SegFormer-B5 (EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024)
Head: Decoupled 5-Class Head for EURS:
  0: background
  1: coral
  2: fish
  3: elasmobranch (Shark / Ray)
  4: other (Macro-fauna / Invertebrates)

Experimental Protocol:
  1. Complete Smoke Test: load local weights, load real EURS batch, execute forward/backward, verify finite loss.
  2. Train Pilot on Fold 0 (16 train scenes, strictly 4 held-out val scenes).
  3. Train Pilot on Fold 2 (16 train scenes, strictly 4 held-out val scenes).
  4. Out-of-fold Evaluation & Probability Blending:
     - Compare Standalone SegFormer-B5
     - Compare V36 Baseline
     - Compare Probability-Space Blends (weight 0.15 and 0.25)
     - Evaluate Q_seg, mIoU, Macro F1, Coverage, Absent-Class False Positives, and per-scene breakdown.
  5. Promotion Gating:
     - If blend yields positive delta Q_seg without rare-class regression, promote to V37.
     - Else, report measured results and maintain V36 fallback.
"""

import os
import os.path as osp
import sys
import time
import json
import random
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

sys.path.insert(0, ".")

from data.v28.dataset import V28DegradationAwareDataset, V28SceneBalancedSampler
from models.v18.restoration_refiner import build_v18_restoration_refiner
from models.v18.segmenter import build_v18_segmenter
from tools.v18.run_v18_pipeline import V18CanonicalPreprocessor, get_fold_data, validate_manifest_and_folds
from losses.v4_losses import PresentClassLovaszSoftmaxLoss
from metrics.segmentation_metrics import SegmentationEvaluator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)

IM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def build_coralscapes_model(model_id: str = "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024") -> nn.Module:
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=5,
        ignore_mismatched_sizes=True,
        local_files_only=True
    ).to(device)
    return model


def run_smoke_test(model: nn.Module, refiner: nn.Module) -> bool:
    print("\n" + "=" * 70, flush=True)
    print("[SMOKE TEST] Verifying Model, Data, Loss, and Backprop Pipeline...", flush=True)
    print("=" * 70, flush=True)

    dummy_ds = V28DegradationAwareDataset(
        scenes=["fish1", "fish11"],
        target_size=512,
        is_train=True,
        object_crop_prob=0.70,
        min_crop_ratio=0.75,
        base_seed=999
    )
    dummy_loader = DataLoader(dummy_ds, batch_size=2, shuffle=False)
    batch = next(iter(dummy_loader))

    deg = batch["deg_rgb"].to(device)
    mask = batch["mask"].to(device)
    wt = batch["wt"].to(device)
    dp = batch["depth"].to(device)

    print(f"  Input batch shapes: deg={deg.shape}, mask={mask.shape}, wt={wt.shape}, dp={dp.shape}", flush=True)
    assert set(torch.unique(mask).cpu().numpy()).issubset({0, 1, 2, 3, 4, 255}), "Invalid mask class labels!"

    with torch.no_grad():
        restored = refiner(deg, wt, dp)["refined_image"]
    print(f"  Restored shape: {restored.shape}", flush=True)

    x_norm = (restored - IM_MEAN) / IM_STD
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x_norm)
        logits = F.interpolate(out.logits.float(), size=(512, 512), mode="bilinear", align_corners=False)
        print(f"  Output logits shape: {logits.shape}", flush=True)

        ce_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
        l_ce = F.cross_entropy(logits, mask.long(), weight=ce_weights, ignore_index=255)
        loss_lov = PresentClassLovaszSoftmaxLoss().to(device)
        l_lov = loss_lov(logits, mask.long())
        total_loss = 0.60 * l_ce + 0.40 * l_lov

    print(f"  Calculated losses: CE={l_ce.item():.4f}, Lovasz={l_lov.item():.4f}, Total={total_loss.item():.4f}", flush=True)
    assert torch.isfinite(total_loss), "Loss is not finite!"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("  [OK] Forward pass, loss calculation, backward pass, and optimizer step SUCCEEDED!", flush=True)
    print("=" * 70, flush=True)
    return True


def train_pilot_fold(
    fold: int,
    epochs: int = 4,
    batch_size: int = 8,
    lr: float = 3e-5,
    crop_size: int = 512,
    out_dir: str = "checkpoints/v37"
) -> str:
    print("\n" + "=" * 80, flush=True)
    print(f"TRAINING V37 SEGFOMER-B5 PILOT — FOLD {fold}", flush=True)
    print("=" * 80, flush=True)

    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    train_scenes = sorted(f_data["train_scenes"])
    val_scenes = sorted(f_data["val_scenes"])

    print(f"Train scenes ({len(train_scenes)}): {train_scenes}", flush=True)
    print(f"Val scenes   ({len(val_scenes)}): {val_scenes}", flush=True)

    # 1. Load frozen refiner
    refiner_ckpt = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=table, use_neural_refiner=True).to(device)
    r_state = torch.load(refiner_ckpt, map_location=device)
    refiner.load_state_dict(r_state["state_dict"] if "state_dict" in r_state else r_state)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad = False

    # 2. Build model
    model = build_coralscapes_model()
    model.train()

    # 3. Setup Dataset
    train_ds = V28DegradationAwareDataset(
        scenes=train_scenes,
        target_size=crop_size,
        is_train=True,
        object_crop_prob=0.70,
        min_crop_ratio=0.75,
        base_seed=100 + fold
    )
    sampler = V28SceneBalancedSampler(train_ds, batch_size=batch_size, seed=100 + fold)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    # 4. Losses & Optimizer
    ce_weights = torch.tensor([0.80, 1.00, 1.20, 2.00, 2.00], device=device)
    loss_ce = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=255)
    loss_lov = PresentClassLovaszSoftmaxLoss().to(device)

    # Backbone vs Head LR
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "decode_head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.20},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    ckpt_path = osp.join(out_dir, f"segformer_b5_fold{fold}_best.pt")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_ds.set_epoch(epoch)
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            deg = batch["deg_rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            wt = batch["wt"].to(device, non_blocking=True)
            dp = batch["depth"].to(device, non_blocking=True)

            with torch.no_grad():
                restored = refiner(deg, wt, dp)["refined_image"]

            x_norm = (restored - IM_MEAN) / IM_STD
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x_norm)
                logits = F.interpolate(out.logits.float(), size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                l1 = loss_ce(logits, mask.long())
                l2 = loss_lov(logits, mask.long())
                loss = 0.60 * l1 + 0.40 * l2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        scheduler.step()
        ep_loss = running_loss / max(1, steps)
        print(f"  Epoch {epoch:02d}/{epochs:02d} complete in {time.time() - ep_start:.1f}s | Mean Loss: {ep_loss:.4f}", flush=True)

    torch.save({
        "epoch": epochs,
        "fold": fold,
        "state_dict": model.state_dict(),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
    }, ckpt_path)
    print(f"  Saved Fold {fold} pilot checkpoint to {ckpt_path} (Training time: {time.time() - t0:.1f}s)", flush=True)
    return ckpt_path


def evaluate_fold_blends(
    fold: int,
    pilot_ckpt: str,
    eval_stride: int = 5
) -> Dict[str, Any]:
    print("\n" + "=" * 80, flush=True)
    print(f"EVALUATING FOLD {fold} HELD-OUT SCENES & BLEND ABLATION", flush=True)
    print("=" * 80, flush=True)

    manifest = validate_manifest_and_folds()
    f_data = get_fold_data(manifest, fold)
    val_scenes = sorted(f_data["val_scenes"])

    # 1. Load Refiner
    refiner_ckpt = f"checkpoints/v18/fold_{fold}/refiner_best.pt"
    table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    refiner = build_v18_restoration_refiner(rsuigm_table=table, use_neural_refiner=True).to(device)
    r_state = torch.load(refiner_ckpt, map_location=device)
    refiner.load_state_dict(r_state["state_dict"] if "state_dict" in r_state else r_state)
    refiner.eval()

    # 2. Load V36 Segmenter
    if fold == 0:
        v36_ckpt = "checkpoints/v27/segmenter_v27_fold0_best.pt"
    else:
        v36_ckpt = f"checkpoints/v18/fold_{fold}/segmenter_best.pt"
    v36_model = build_v18_segmenter(num_classes=5).to(device)
    v36_state = torch.load(v36_ckpt, map_location=device)
    v36_model.load_state_dict(v36_state["state_dict"] if "state_dict" in v36_state else v36_state)
    v36_model.eval()

    # 3. Load Pilot Model
    pilot_model = build_coralscapes_model()
    p_state = torch.load(pilot_ckpt, map_location=device)
    pilot_model.load_state_dict(p_state["state_dict"] if "state_dict" in p_state else p_state)
    pilot_model.eval()

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    evaluators = {
        "V36_Baseline": SegmentationEvaluator(num_classes=5),
        "V37_Pilot_Standalone": SegmentationEvaluator(num_classes=5),
        "V37_Blend_0.15": SegmentationEvaluator(num_classes=5),
        "V37_Blend_0.25": SegmentationEvaluator(num_classes=5),
    }

    per_scene_metrics = {sc: {} for sc in val_scenes}

    for sc in val_scenes:
        gt_mask = cv2.imread(f"data/cached_1024/masks/{sc}.png", cv2.IMREAD_UNCHANGED)
        gt_t = torch.from_numpy(gt_mask).long().to(device)

        deg_files = sorted(os.listdir(f"data/cached_1024/degraded/{sc}"))[::eval_stride]
        print(f"  Evaluating Scene {sc} ({len(deg_files)} variants)...", flush=True)

        sc_evals = {k: SegmentationEvaluator(num_classes=5) for k in evaluators.keys()}

        for fn in deg_files:
            parts = fn.replace(".png", "").split("_")
            wt, dp = int(parts[1]), int(parts[2])
            bgr = cv2.imread(f"data/cached_1024/degraded/{sc}/{fn}")
            prep = preprocessor.prepare_input(bgr)
            raw_t = prep["rgb_tensor"].to(device)
            meta = prep["meta"]
            wt_t = torch.tensor([wt], device=device)
            dp_t = torch.tensor([dp], dtype=torch.float32, device=device)

            with torch.no_grad():
                rest = refiner(raw_t, wt_t, dp_t)["refined_image"]

                # V36 prediction
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    v36_logits = v36_model(rest, raw_t, wt_t, dp_t)["logits"].float()
                    prob_v36 = F.softmax(v36_logits, dim=1)

                    # Pilot prediction
                    x_norm = (rest - IM_MEAN) / IM_STD
                    pilot_out = pilot_model(x_norm)
                    pilot_logits = F.interpolate(pilot_out.logits.float(), size=v36_logits.shape[-2:], mode="bilinear", align_corners=False)
                    prob_pilot = F.softmax(pilot_logits, dim=1)

            # Blends
            prob_b15 = 0.85 * prob_v36 + 0.15 * prob_pilot
            prob_b25 = 0.75 * prob_v36 + 0.25 * prob_pilot

            # Unpad and resize to GT mask size
            probs = {
                "V36_Baseline": prob_v36,
                "V37_Pilot_Standalone": prob_pilot,
                "V37_Blend_0.15": prob_b15,
                "V37_Blend_0.25": prob_b25,
            }

            for k, pr in probs.items():
                p_np = pr.squeeze(0).permute(1, 2, 0).cpu().numpy()
                p_unpad = preprocessor.unpad_output(p_np, meta)
                p_res = cv2.resize(p_unpad, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                pred_m = np.argmax(p_res, axis=-1).astype(np.uint8)
                pred_t = torch.from_numpy(pred_m).to(device)

                evaluators[k].update(pred_t.unsqueeze(0), gt_t.unsqueeze(0))
                sc_evals[k].update(pred_t.unsqueeze(0), gt_t.unsqueeze(0))

        for k in evaluators.keys():
            per_scene_metrics[sc][k] = sc_evals[k].compute()

    results = {}
    print("\n" + "-" * 75, flush=True)
    print(f"FOLD {fold} HELD-OUT RESULTS ACROSS VARIANTS", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Configuration':<24} | {'mIoU':<7} | {'Macro F1':<8} | {'Q_seg':<7} | {'Coverage':<8}")
    print("-" * 75, flush=True)
    for k, ev in evaluators.items():
        comp = ev.compute()
        results[k] = comp
        print(f"{k:<24} | {comp['mIoU']:<7.4f} | {comp['macro_F1']:<8.4f} | {comp['Q_seg']:<7.4f} | {comp['coverage_penalty']:<8.1f}", flush=True)
        print(f"   Class IoUs: {comp['per_class_IoU']}", flush=True)

    print("-" * 75, flush=True)
    print("Per-Scene Q_seg Breakdown:", flush=True)
    for sc in val_scenes:
        v36_q = per_scene_metrics[sc]["V36_Baseline"]["Q_seg"]
        b15_q = per_scene_metrics[sc]["V37_Blend_0.15"]["Q_seg"]
        b25_q = per_scene_metrics[sc]["V37_Blend_0.25"]["Q_seg"]
        print(f"  {sc:12s}: V36={v36_q:.4f} | Blend_0.15={b15_q:.4f} (Delta: {b15_q - v36_q:+.4f}) | Blend_0.25={b25_q:.4f} (Delta: {b25_q - v36_q:+.4f})", flush=True)

    return {
        "fold": fold,
        "global_results": results,
        "per_scene_results": per_scene_metrics
    }


def main():
    print("=" * 85, flush=True)
    print("SAT-EURS TRACK 1 — V37 EXTERNAL FOUNDATION MODEL PILOT EXECUTION", flush=True)
    print("Architecture: CoralscapesV2 SegFormer-B5 Transfer", flush=True)
    print("=" * 85, flush=True)

    # 1. Smoke test
    refiner_table = torch.load("checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt", map_location=device)
    smoke_refiner = build_v18_restoration_refiner(rsuigm_table=refiner_table, use_neural_refiner=True).to(device)
    smoke_model = build_coralscapes_model()
    run_smoke_test(smoke_model, smoke_refiner)
    del smoke_model, smoke_refiner
    torch.cuda.empty_cache()

    # 2. Train Fold 0 Pilot
    f0_ckpt = train_pilot_fold(fold=0, epochs=4, batch_size=8, lr=3e-5)

    # 3. Evaluate Fold 0 Pilot & Blends
    f0_results = evaluate_fold_blends(fold=0, pilot_ckpt=f0_ckpt)

    # 4. Train Fold 2 Pilot
    f2_ckpt = train_pilot_fold(fold=2, epochs=4, batch_size=8, lr=3e-5)

    # 5. Evaluate Fold 2 Pilot & Blends
    f2_results = evaluate_fold_blends(fold=2, pilot_ckpt=f2_ckpt)

    # 6. Global Analysis & Promotion Gate
    print("\n" + "=" * 85, flush=True)
    print("GLOBAL V37 PILOT COMPARISON & PROMOTION VERDICT", flush=True)
    print("=" * 85, flush=True)

    f0_base_q = f0_results["global_results"]["V36_Baseline"]["Q_seg"]
    f0_b15_q = f0_results["global_results"]["V37_Blend_0.15"]["Q_seg"]
    f0_b25_q = f0_results["global_results"]["V37_Blend_0.25"]["Q_seg"]
    f0_delta15 = f0_b15_q - f0_base_q
    f0_delta25 = f0_b25_q - f0_base_q

    f2_base_q = f2_results["global_results"]["V36_Baseline"]["Q_seg"]
    f2_b15_q = f2_results["global_results"]["V37_Blend_0.15"]["Q_seg"]
    f2_b25_q = f2_results["global_results"]["V37_Blend_0.25"]["Q_seg"]
    f2_delta15 = f2_b15_q - f2_base_q
    f2_delta25 = f2_b25_q - f2_base_q

    mean_delta15 = 0.5 * (f0_delta15 + f2_delta15)
    mean_delta25 = 0.5 * (f0_delta25 + f2_delta25)

    print(f"Fold 0: V36 Q_seg = {f0_base_q:.4f} | Blend_0.15 Delta: {f0_delta15:+.4f} | Blend_0.25 Delta: {f0_delta25:+.4f}", flush=True)
    print(f"Fold 2: V36 Q_seg = {f2_base_q:.4f} | Blend_0.15 Delta: {f2_delta15:+.4f} | Blend_0.25 Delta: {f2_delta25:+.4f}", flush=True)
    print(f"Mean Delta across Evaluated Folds: Blend_0.15 = {mean_delta15:+.4f} | Blend_0.25 = {mean_delta25:+.4f}", flush=True)

    # Check rare-class regression
    rare_regression = False
    for f_idx, res in [(0, f0_results), (2, f2_results)]:
        b_ious = res["global_results"]["V36_Baseline"]["per_class_IoU"]
        m_ious = res["global_results"]["V37_Blend_0.15"]["per_class_IoU"]
        for c in range(1, 5):
            if b_ious[c] > 0.20 and (m_ious[c] - b_ious[c]) < -0.05:
                rare_regression = True
                print(f"  [ALERT] Rare class {c} regression on Fold {f_idx}: {b_ious[c]:.4f} -> {m_ious[c]:.4f}", flush=True)

    promoted = (mean_delta15 > 0.005 or mean_delta25 > 0.005) and (not rare_regression)
    verdict = "PROMOTED TO V37" if promoted else "REJECTED (V36 Retained as Primary Champion)"
    best_weight = 0.25 if mean_delta25 > mean_delta15 else 0.15
    print(f"\nFinal Verdict: {verdict}", flush=True)
    print("=" * 85, flush=True)

    # Save comprehensive report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_architecture": "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024",
        "fold_0_results": f0_results,
        "fold_2_results": f2_results,
        "mean_delta_blend_0.15": round(mean_delta15, 5),
        "mean_delta_blend_0.25": round(mean_delta25, 5),
        "promoted": promoted,
        "verdict": verdict,
        "best_blend_weight": best_weight if promoted else 0.0
    }
    with open("reports/v37/v37_pilot_evaluation.json", "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data

if __name__ == "__main__":
    main()
