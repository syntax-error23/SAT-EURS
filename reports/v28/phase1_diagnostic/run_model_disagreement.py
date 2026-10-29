"""
SAT-EURS Track 1 — Phase 1 Model Disagreement & Dilution Diagnostic
Compares:
1. Full-20 SWA alone
2. Fold 0 (V27 Targeted) alone
3. 5-Fold Ensemble (Folds 0-4) alone
4. Production V27 Hybrid (50% 5-Fold + 50% Full-20)
5. Targeted Pair (50% Fold 0 + 50% Full-20, excluding naive folds 1-4)
6. 75% Full-20 + 25% 5-Fold
Across the 30 stratified test images.
"""

import os
import sys
import json
import time
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
    parse_scene_wt_depth,
    V18CanonicalPreprocessor,
)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[PREFLIGHT] Running on device: {device} ({torch.cuda.get_device_name(device)})")

    out_dir = "reports/v28/phase1_diagnostic"
    table_path = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt"
    checkpoint_dir = "checkpoints/v18"
    v27_checkpoint_dir = "checkpoints/v27"

    # 1. 30 Stratified test images
    test_dir = "data/test/degraded"
    conditions = [
        ("mild", 1, 10),
        ("mid", 3, 30),
        ("deep", 5, 60),
        ("turbid", 8, 40),
        ("extreme", 10, 90),
    ]
    scenes = ["test04", "test09", "test10", "test11", "test19", "test20"]
    sample_stems = []
    for sc in scenes:
        for lbl, wt, dp in conditions:
            sample_stems.append(f"{sc}_{wt}_{dp}")

    print(f"[Step 1/5] Selected {len(sample_stems)} representative diagnostic images.")

    # 2. Load restoration models
    print(f"[Step 2/5] Loading restoration refiner ensemble...")
    base_table = torch.load(table_path, map_location=device)
    manifest = validate_manifest_and_folds()
    refiner_models = []
    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        r_model = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_path = os.path.join(checkpoint_dir, f"fold_{fold}", "refiner_best.pt")
        r_weights = load_checkpoint_with_provenance_check(r_path, "ConditionedRestorationRefinerV18", fold, f_data["train_scenes"], f_data["val_scenes"])
        r_model.load_state_dict(r_weights)
        r_model.eval()
        refiner_models.append(r_model)

    # 3. Load segmenter models
    print(f"[Step 3/5] Loading segmenter models...")
    # Fold 0 (V27 targeted)
    fold0_model = build_v18_segmenter(num_classes=5).to(device)
    f0_ckpt = torch.load(os.path.join(v27_checkpoint_dir, "segmenter_v27_fold0_best.pt"), map_location=device)
    fold0_model.load_state_dict(f0_ckpt["state_dict"])
    fold0_model.eval()

    # Folds 1-4 (V18 baseline)
    fold_models = [fold0_model]
    for fold in range(1, 5):
        f_data = get_fold_data(manifest, fold)
        s_model = build_v18_segmenter(num_classes=5).to(device)
        s_path = os.path.join(checkpoint_dir, f"fold_{fold}", "segmenter_best.pt")
        s_weights = load_checkpoint_with_provenance_check(s_path, "EURSAPEXV18Segmenter", fold, f_data["train_scenes"], f_data["val_scenes"])
        s_model.load_state_dict(s_weights)
        s_model.eval()
        fold_models.append(s_model)

    # Full-20 SWA model
    full20_model = build_v18_segmenter(num_classes=5).to(device)
    f20_ckpt = torch.load(os.path.join(v27_checkpoint_dir, "segmenter_full20_swa.pt"), map_location=device)
    f20_sd = f20_ckpt["state_dict"] if "state_dict" in f20_ckpt else f20_ckpt
    full20_model.load_state_dict(f20_sd)
    full20_model.eval()

    print(f"  All segmenter models loaded into GPU memory successfully.")

    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)
    offset_vector = [0.0, 0.0, 0.0, 0.10, 0.25]
    cal_t = torch.tensor(offset_vector, device=device).view(1, 5, 1, 1)

    # 4. Run comparative inference on the 30 images
    print(f"[Step 4/5] Running multi-model diagnostic inference on 30 images...")
    
    CONFIGS = [
        "Full20_Alone",
        "Fold0_Targeted_Alone",
        "FiveFold_Ensemble",
        "V27_Hybrid_50_50",
        "Targeted_Pair_Fold0_Full20",
        "Full20_Heavy_75_25"
    ]

    results_by_config = {cfg: {
        "total_pixels": 0,
        "class_pixels": np.zeros(5, dtype=np.int64),
        "scene_class_pixels": {sc: np.zeros(5, dtype=np.int64) for sc in scenes}
    } for cfg in CONFIGS}

    pairwise_disagreement = {f"{c1}_vs_{c2}": [] for c1 in CONFIGS for c2 in CONFIGS if c1 < c2}

    t0 = time.time()
    for idx, stem in enumerate(sample_stems):
        sc, wt, dp = parse_scene_wt_depth(stem)
        img_path = os.path.join(test_dir, sc, f"{stem}.png")
        bgr = cv2.imread(img_path)
        prep = preprocessor.prepare_input(bgr)
        raw_pad = prep["rgb_tensor"].to(device)
        meta = prep["meta"]
        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth := dp], device=device)

        with torch.inference_mode():
            # Restoration
            rest_acc = torch.zeros_like(raw_pad)
            for r_m in refiner_models:
                rest_acc += r_m(raw_pad.float(), wt_t, dp_t)["refined_image"]
            rest_fused = rest_acc / len(refiner_models)

            raw_flip = torch.flip(raw_pad, dims=[3])
            rest_flip = torch.flip(rest_fused, dims=[3])

            # Predictions with amp
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Full20 alone
                p_f20_orig = F.softmax(full20_model(rest_fused, raw_pad, wt_t, dp_t)["logits"].float(), dim=1)
                p_f20_flip = torch.flip(F.softmax(full20_model(rest_flip, raw_flip, wt_t, dp_t)["logits"].float(), dim=1), dims=[3])
                p_full20 = 0.5 * (p_f20_orig + p_f20_flip)

                # Fold 0 alone
                p_f0_orig = F.softmax(fold0_model(rest_fused, raw_pad, wt_t, dp_t)["logits"].float(), dim=1)
                p_f0_flip = torch.flip(F.softmax(fold0_model(rest_flip, raw_flip, wt_t, dp_t)["logits"].float(), dim=1), dims=[3])
                p_fold0 = 0.5 * (p_f0_orig + p_f0_flip)

                # Folds 1-4
                p_folds_acc = p_fold0.clone()
                for s_m in fold_models[1:]:
                    p_orig = F.softmax(s_m(rest_fused, raw_pad, wt_t, dp_t)["logits"].float(), dim=1)
                    p_flip = torch.flip(F.softmax(s_m(rest_flip, raw_flip, wt_t, dp_t)["logits"].float(), dim=1), dims=[3])
                    p_folds_acc += 0.5 * (p_orig + p_flip)
                p_fivefold = p_folds_acc / len(fold_models)

                # Hybrids
                p_v27_hybrid = 0.50 * p_fivefold + 0.50 * p_full20
                p_targeted_pair = 0.50 * p_fold0 + 0.50 * p_full20
                p_full20_heavy = 0.75 * p_full20 + 0.25 * p_fivefold

            prob_dict = {
                "Full20_Alone": p_full20,
                "Fold0_Targeted_Alone": p_fold0,
                "FiveFold_Ensemble": p_fivefold,
                "V27_Hybrid_50_50": p_v27_hybrid,
                "Targeted_Pair_Fold0_Full20": p_targeted_pair,
                "Full20_Heavy_75_25": p_full20_heavy,
            }

            mask_dict = {}
            for cfg, p_tensor in prob_dict.items():
                log_prob = torch.log(p_tensor + 1e-10) + cal_t
                mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                mask_unpad = preprocessor.unpad_output(mask_pad, meta)
                mask_dict[cfg] = mask_unpad

                counts = np.bincount(mask_unpad.ravel(), minlength=5)
                results_by_config[cfg]["total_pixels"] += mask_unpad.size
                results_by_config[cfg]["class_pixels"] += counts[:5]
                results_by_config[cfg]["scene_class_pixels"][sc] += counts[:5]

            # Compute pairwise disagreement
            for c1 in CONFIGS:
                for c2 in CONFIGS:
                    if c1 < c2:
                        diff = np.mean(mask_dict[c1] != mask_dict[c2])
                        pairwise_disagreement[f"{c1}_vs_{c2}"].append(diff)

        if (idx + 1) % 5 == 0:
            print(f"  Processed [{idx+1:02d}/30] images in {time.time() - t0:.1f}s...")

    # [Step 5/5] Synthesize findings
    summary = {
        "configurations": {},
        "pairwise_disagreement_mean_percent": {},
    }

    CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]

    print("\n" + "=" * 90)
    print(f"{'CONFIGURATION':<28} | {'BG %':<6} | {'CORAL %':<7} | {'FISH %':<6} | {'RAY %':<6} | {'FAUNA %':<7} | {'FOREGROUND %':<10}")
    print("=" * 90)

    for cfg in CONFIGS:
        tot = results_by_config[cfg]["total_pixels"]
        cp = results_by_config[cfg]["class_pixels"]
        pcts = [round((cp[c] / tot) * 100, 3) for c in range(5)]
        fg_pct = round(sum(pcts[1:]), 3)
        print(f"{cfg:<28} | {pcts[0]:5.2f}% | {pcts[1]:6.2f}% | {pcts[2]:5.2f}% | {pcts[3]:5.2f}% | {pcts[4]:6.2f}% | {fg_pct:6.2f}%")

        summary["configurations"][cfg] = {
            "total_pixels": int(tot),
            "class_pixels": [int(x) for x in cp],
            "class_percentages": pcts,
            "foreground_percentage": fg_pct,
            "per_scene_breakdown": {
                sc: [int(x) for x in results_by_config[cfg]["scene_class_pixels"][sc]]
                for sc in scenes
            }
        }

    print("=" * 90)
    print("\nPAIRWISE DISAGREEMENT MATRIX (Pixel % Mismatch across 30 Diagnostic Images):")
    print("-" * 70)
    for pair, diffs in sorted(pairwise_disagreement.items()):
        mean_diff = round(float(np.mean(diffs)) * 100, 3)
        summary["pairwise_disagreement_mean_percent"][pair] = mean_diff
        print(f"  {pair:<45}: {mean_diff:6.2f}% disagreement")
    print("-" * 70)

    out_json = os.path.join(out_dir, "model_disagreement_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved disagreement summary to {out_json}")

if __name__ == "__main__":
    main()
