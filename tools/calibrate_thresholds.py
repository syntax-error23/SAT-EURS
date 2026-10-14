"""
Threshold Calibration & Precision-Recall Optimization Script (Exp-02)
Evaluates class-wise logit offsets on the local validation split.
Measures per-class IoU, Precision, Recall, Macro F1, and Q_seg across candidate offsets.
"""

import os
import os.path as osp
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.eurs_net_v5 import EURSNetV5
from metrics.segmentation_metrics import SegmentationEvaluator


def run_threshold_grid_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_scenes = ["fish1", "fish12", "seahorse1", "sharkFish"]
    data_masks_dir = "data/cached_1024/masks"
    data_deg_dir = "data/cached_1024/degraded"

    print("=" * 72)
    print("   RUNNING THRESHOLD CALIBRATION GRID SEARCH (EXP-02)")
    print("=" * 72)

    student = EURSNetV5(num_classes=5, bounded_restoration=True, bound_scale=0.20, pretrained_imagenet=False).to(device)
    ckpt = torch.load("checkpoints/v5_clinical_distilled_best.pt", map_location=device, weights_only=False)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    student.eval()

    # Pre-compute all student logits on validation images to make grid search instantaneous
    print("Pre-computing validation logits...", flush=True)
    val_data = []
    for sc in val_scenes:
        mask_p = osp.join(data_masks_dir, f"{sc}.png")
        deg_folder = osp.join(data_deg_dir, sc)

        with Image.open(mask_p) as m_im:
            mask_np = np.asarray(m_im, dtype=np.int64)

        deg_files = sorted([osp.join(deg_folder, f) for f in os.listdir(deg_folder) if f.endswith(".png")])
        sampled = deg_files[::10] # 20 images per scene = 80 images total

        for df in sampled:
            with Image.open(df) as d_im:
                deg_np = np.asarray(d_im.convert("RGB"), dtype=np.float32) / 255.0
            deg_tensor = torch.from_numpy(deg_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                _, logits = student(deg_tensor)
                val_data.append((logits.cpu(), torch.from_numpy(mask_np).unsqueeze(0)))

    print(f"Pre-computed {len(val_data)} validation logit tensors into memory!\n")

    # Candidate offsets for Class 3 (Sharks) and Class 4 (Starfish)
    delta3_cands = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5]
    delta4_cands = [0.0, 0.5, 1.0, 1.5]

    best_q_seg = 0.0
    best_config = None
    best_results = None

    print(f"{'Offset C3':<10} | {'Offset C4':<10} | {'mIoU':<8} | {'Macro F1':<10} | {'Q_seg':<8} | {'C3 F1':<7} | {'C4 F1':<7}")
    print("-" * 72)

    evaluator = SegmentationEvaluator(num_classes=5)

    for d3 in delta3_cands:
        for d4 in delta4_cands:
            evaluator.reset()
            offset = torch.tensor([0.0, 0.0, 0.0, d3, d4]).view(1, 5, 1, 1)

            for logits, target in val_data:
                adjusted_logits = logits + offset
                evaluator.update(adjusted_logits, target)

            res = evaluator.compute()
            q_seg = res["Q_seg"]
            c3_f1 = res["per_class_F1"][3]
            c4_f1 = res["per_class_F1"][4]

            if q_seg > best_q_seg:
                best_q_seg = q_seg
                best_config = (d3, d4)
                best_results = res
                marker = " *"
            else:
                marker = ""

            print(f"{d3:<10.1f} | {d4:<10.1f} | {res['mIoU']:<8.4f} | {res['macro_F1']:<10.4f} | {q_seg:<8.4f} | {c3_f1:<7.4f} | {c4_f1:<7.4f}{marker}")

    print("=" * 72)
    print(f"Optimal Calibration Found: Delta_3 = {best_config[0]:.1f}, Delta_4 = {best_config[1]:.1f}")
    print(f"Baseline Q_seg: 0.4424  --->  Optimized Q_seg: {best_q_seg:.4f} (+{best_q_seg - 0.4424:.4f} gain!)")
    print(f"Per-Class F1 with Calibration: {best_results['per_class_F1']}")
    print(f"Per-Class Precision:          {best_results['per_class_precision']}")
    print(f"Per-Class Recall:             {best_results['per_class_recall']}")
    print("=" * 72)

    return best_config, best_results


if __name__ == "__main__":
    run_threshold_grid_search()
