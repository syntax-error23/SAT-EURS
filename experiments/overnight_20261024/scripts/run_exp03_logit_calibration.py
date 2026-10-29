import os
import os.path as osp
import sys
import time
import json
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
    V18CanonicalPreprocessor,
)
from experiments.overnight_20261024.scripts.benchmark_engine import compute_metrics_from_conf_mat

CLASS_NAMES = ["Background", "Coral", "Fish", "Ray", "MacroFauna"]

CALIBRATION_CONFIGS = {
    "cfg0_baseline": {"offset": [0.0, 0.0, 0.0, 0.10, 0.25], "min_comp": 0},
    "cfg1_macro20": {"offset": [0.0, 0.0, 0.0, 0.10, 0.20], "min_comp": 0},
    "cfg2_macro30": {"offset": [0.0, 0.0, 0.0, 0.10, 0.30], "min_comp": 0},
    "cfg3_macro35": {"offset": [0.0, 0.0, 0.0, 0.10, 0.35], "min_comp": 0},
    "cfg4_macro40": {"offset": [0.0, 0.0, 0.0, 0.10, 0.40], "min_comp": 0},
    "cfg5_ray05": {"offset": [0.0, 0.0, 0.0, 0.05, 0.25], "min_comp": 0},
    "cfg6_ray15": {"offset": [0.0, 0.0, 0.0, 0.15, 0.25], "min_comp": 0},
    "cfg7_fish05": {"offset": [0.0, 0.0, 0.05, 0.10, 0.25], "min_comp": 0},
    "cfg8_fish10": {"offset": [0.0, 0.0, 0.10, 0.10, 0.25], "min_comp": 0},
    "cfg9_fish05_macro30": {"offset": [0.0, 0.0, 0.05, 0.10, 0.30], "min_comp": 0},
    "cfg10_fish05_macro35": {"offset": [0.0, 0.0, 0.05, 0.10, 0.35], "min_comp": 0},
    "cfg11_balanced_opt": {"offset": [0.0, 0.0, 0.05, 0.08, 0.30], "min_comp": 0},
    "cfg12_filter_min5": {"offset": [0.0, 0.0, 0.0, 0.10, 0.25], "min_comp": 5},
    "cfg13_filter_min10": {"offset": [0.0, 0.0, 0.0, 0.10, 0.25], "min_comp": 10},
    "cfg14_fish05_macro30_min5": {"offset": [0.0, 0.0, 0.05, 0.10, 0.30], "min_comp": 5},
}


def run_logit_calibration_sweep(
    segmenter_ckpts: Dict[int, str],
    refiner_ckpts: Dict[int, str],
    table_path: str = "checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt",
    data_root: str = "data/cached_1024",
    sample_stride: int = 5,
    device_id: int = 0
) -> Dict[str, Any]:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    manifest = validate_manifest_and_folds()
    base_table = torch.load(table_path, map_location=device)
    preprocessor = V18CanonicalPreprocessor(max_side=1024, pad_multiple=32)

    # Pre-build offset tensors on GPU
    cal_tensors = {
        name: torch.tensor(cfg["offset"], device=device).view(1, 5, 1, 1)
        for name, cfg in CALIBRATION_CONFIGS.items()
    }

    # Confusion matrix trackers
    conf_mats = {name: np.zeros((5, 5), dtype=np.int64) for name in CALIBRATION_CONFIGS}

    print("=" * 80)
    print("SAT-EURS TRACK 1 — EXP-03 / EXP-05: LOGIT CALIBRATION & FILTER SWEEP")
    print(f"Testing {len(CALIBRATION_CONFIGS)} candidate configurations simultaneously across 20 scenes...")
    print("=" * 80)

    t0 = time.time()
    total_images = 0

    for fold in range(5):
        f_data = get_fold_data(manifest, fold)
        val_scenes = sorted(f_data["val_scenes"])

        # Load fold refiner
        refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(device)
        r_obj = torch.load(refiner_ckpts[fold], map_location=device)
        refiner.load_state_dict(r_obj["state_dict"] if "state_dict" in r_obj else r_obj)
        refiner.eval()

        # Load fold segmenter
        segmenter = build_v18_segmenter(num_classes=5).to(device)
        s_obj = torch.load(segmenter_ckpts[fold], map_location=device)
        s_sd = s_obj["state_dict"] if "state_dict" in s_obj else s_obj
        segmenter.load_state_dict(s_sd)
        segmenter.eval()

        print(f"\nFold {fold} evaluating scenes: {val_scenes}...", flush=True)

        for sc in val_scenes:
            mask_path = osp.join(data_root, "masks", f"{sc}.png")
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gh, gw = gt_mask.shape[:2]
            y_t = gt_mask.reshape(-1)

            deg_scene_dir = osp.join(data_root, "degraded", sc)
            deg_files = sorted([f for f in os.listdir(deg_scene_dir) if f.endswith(".png")])
            sampled_files = deg_files[::sample_stride]

            for fname in sampled_files:
                stem = osp.splitext(fname)[0]
                parts = stem.split("_")
                wt = int(parts[1])
                depth = int(parts[2])

                img_path = osp.join(deg_scene_dir, fname)
                bgr = cv2.imread(img_path)
                if bgr is None:
                    continue

                prep = preprocessor.prepare_input(bgr)
                raw_pad = prep["rgb_tensor"].to(device)
                meta = prep["meta"]
                wt_t = torch.tensor([wt], device=device)
                dp_t = torch.tensor([depth], device=device)

                with torch.inference_mode():
                    out_r = refiner(raw_pad.float(), wt_t, dp_t)
                    restored_fused = out_r["refined_image"]

                    # Horizontal flip TTA
                    raw_pad_flip = torch.flip(raw_pad, dims=[3])
                    restored_flip = torch.flip(restored_fused, dims=[3])

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out_orig = segmenter(restored_fused, raw_pad, wt_t, dp_t)
                        p_orig = F.softmax(out_orig["logits"].float(), dim=1)

                        out_flip = segmenter(restored_flip, raw_pad_flip, wt_t, dp_t)
                        p_flip = torch.flip(F.softmax(out_flip["logits"].float(), dim=1), dims=[3])

                        prob_fused = 0.5 * (p_orig + p_flip)
                        log_prob_base = torch.log(prob_fused + 1e-10)

                total_images += 1

                # Evaluate all candidate configs
                for name, cfg in CALIBRATION_CONFIGS.items():
                    cal_t = cal_tensors[name]
                    min_comp = cfg["min_comp"]

                    log_prob = log_prob_base + cal_t
                    mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    pred_mask = preprocessor.unpad_output(mask_pad, meta)

                    if pred_mask.shape != (gh, gw):
                        pred_mask = cv2.resize(pred_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)

                    if min_comp > 0:
                        for c in range(1, 5):
                            binary = (pred_mask == c).astype(np.uint8)
                            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                            for lbl in range(1, num_labels):
                                if stats[lbl, cv2.CC_STAT_AREA] < min_comp:
                                    pred_mask[labels == lbl] = 0

                    y_p = pred_mask.reshape(-1)
                    indices = 5 * y_t + y_p
                    counts = np.bincount(indices, minlength=25).reshape(5, 5)
                    conf_mats[name] += counts

        print(f"Fold {fold} complete. Total processed images: {total_images}", flush=True)

    elapsed = time.time() - t0
    print(f"\nAll 20 scenes evaluated ({total_images} images) in {elapsed:.1f}s.")

    # Compute and rank results
    results = {}
    base_q_seg = None
    for name, cmat in conf_mats.items():
        m = compute_metrics_from_conf_mat(cmat)
        if name == "cfg0_baseline":
            base_q_seg = m["Q_seg"]
        results[name] = {
            "config": CALIBRATION_CONFIGS[name],
            "metrics": m,
            "delta_vs_baseline": round(m["Q_seg"] - (base_q_seg if base_q_seg else m["Q_seg"]), 5)
        }

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Config Name':<28} | {'Q_seg':<8} | {'mIoU':<8} | {'F1':<8} | {'Bg IoU':<8} | {'Coral':<8} | {'Fish':<8} | {'Ray':<8} | {'Macro':<8} | {'Delta Q_seg'}")
    print("=" * 110)
    sorted_items = sorted(results.items(), key=lambda x: x[1]["metrics"]["Q_seg"], reverse=True)
    for name, res in sorted_items:
        m = res["metrics"]
        c_ious = m["class_ious"]
        delta_str = f"{res['delta_vs_baseline']:+.4f}"
        print(f"{name:<28} | {m['Q_seg']:<8.4f} | {m['mIoU']:<8.4f} | {m['Macro_F1']:<8.4f} | {c_ious[0]:<8.4f} | {c_ious[1]:<8.4f} | {c_ious[2]:<8.4f} | {c_ious[3]:<8.4f} | {c_ious[4]:<8.4f} | {delta_str}")
    print("=" * 110)

    return {"results": results, "elapsed_s": round(elapsed, 2), "total_images": total_images}


def main():
    segmenter_ckpts = {
        0: 'checkpoints/v28/segmenter_v28_fold0_best.pt',
        1: 'checkpoints/v18/fold_1/segmenter_best.pt',
        2: 'checkpoints/v18/fold_2/segmenter_best.pt',
        3: 'checkpoints/v18/fold_3/segmenter_best.pt',
        4: 'checkpoints/v18/fold_4/segmenter_best.pt',
    }

    refiner_ckpts = {
        0: 'checkpoints/v18/fold_0/refiner_best.pt',
        1: 'checkpoints/v18/fold_1/refiner_best.pt',
        2: 'checkpoints/v18/fold_2/refiner_best.pt',
        3: 'checkpoints/v18/fold_3/refiner_best.pt',
        4: 'checkpoints/v18/fold_4/refiner_best.pt',
    }

    out = run_logit_calibration_sweep(
        segmenter_ckpts=segmenter_ckpts,
        refiner_ckpts=refiner_ckpts,
        sample_stride=5,
        device_id=0
    )

    out_file = "experiments/overnight_20261024/predictions/exp03_calibration_results.json"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
