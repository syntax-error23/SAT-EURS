"""
Synthetic Unseen-Host Diagnostic Probe (Section 3 Protocol 2 Supplementary)
Tests whether models trained on known rare taxa can recognize them when pasted
into completely unseen fish host scenes at matching (wt, depth) conditions.

Constraints:
- Donors strictly from that fold's train_scenes only.
- Hosts strictly from that fold's held-out val_fish_hosts only.
- Exact (wt, depth) suffix matching.
- Labeled strictly synthetic_unseen_host_probe (never substituted for natural OOF).
"""

import os
import os.path as osp
import sys
import time
import json
import random
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..")))

from metrics.segmentation_metrics import SegmentationEvaluator
from deploy.export_v5_and_track1 import build_rsuigm_poly_features_gpu
from tools.train_track1_screening import build_dinov2_model, build_mit_b3_model, apply_rsuigm_batch_gpu
from tools.pilot_restoration_evaluation import fit_leak_free_rsuigm_table
from data.controlled_copy_paste import ControlledCopyPasteAugmenter


def evaluate_synthetic_unseen_host_probe(
    checkpoint_path: str,
    fold_id: str = "fold_0",
    model_type: str = "dinov2_l",
    use_rsuigm: bool = True,
    n_probe_samples: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"RUNNING SYNTHETIC UNSEEN-HOST PROBE FOR: {checkpoint_path}")
    print(f"  Fold: {fold_id} | Model: {model_type} | Samples: {n_probe_samples}")
    print("=" * 72)

    with open("splits/manifest_5fold.json", "r") as f:
        manifest = json.load(f)

    f_meta = manifest["folds"][fold_id]
    train_scenes = f_meta["train_scenes"]
    val_fish_hosts = f_meta["val_fish_hosts"]

    print(f"  Train Donors ({len(train_scenes)} scenes): {train_scenes}")
    print(f"  Unseen Fish Hosts ({len(val_fish_hosts)} scenes): {val_fish_hosts}")

    # Build fold copy-paste augmenter (extracts donors strictly from train_scenes)
    cp_augmenter = ControlledCopyPasteAugmenter(
        train_scenes=train_scenes,
        data_root="data/cached_1024",
        paste_prob=1.0, # Always paste on probe samples
        max_pastes=1,
        base_seed=seed
    )

    # Fit fold RSUIGM table if required
    rsuigm_table = None
    if use_rsuigm:
        rsuigm_table = fit_leak_free_rsuigm_table(train_scenes, device=device)

    # Load model
    if model_type == "dinov2_l":
        model = build_dinov2_model(num_classes=5).to(device)
    else:
        model = build_mit_b3_model(num_classes=5).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean_im = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_im = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    evaluator = SegmentationEvaluator(num_classes=5)
    rng = random.Random(seed)

    # Collect host images
    deg_dir = "data/cached_1024/degraded"
    masks_dir = "data/cached_1024/masks"
    host_items = []
    for h_sc in val_fish_hosts:
        h_deg_dir = osp.join(deg_dir, h_sc)
        m_path = osp.join(masks_dir, f"{h_sc}.png")
        h_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        deg_files = sorted(os.listdir(h_deg_dir))
        for df in deg_files:
            parts = osp.splitext(df)[0].split("_")
            wt = int(parts[1])
            depth = int(parts[2])
            host_items.append((h_sc, osp.join(h_deg_dir, df), wt, depth, h_mask))

    sampled_hosts = rng.sample(host_items, min(n_probe_samples, len(host_items)))
    print(f"  Constructing {len(sampled_hosts)} synthetic unseen-host probe images...")

    pastes_recorded = 0
    t0 = time.time()

    for idx, (h_sc, deg_path, wt, depth, h_mask) in enumerate(sampled_hosts):
        bgr = cv2.imread(deg_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Paste known training rare donor onto unseen fish host
        p_rgb, _, p_mask = cp_augmenter(
            img_rgb=rgb,
            clean_rgb=rgb,
            mask=h_mask,
            wt=wt,
            depth=depth,
            seed=idx,
            epoch=0
        )

        # Check if paste actually occurred
        if (p_mask == 3).any() or (p_mask == 4).any():
            pastes_recorded += 1

        # Dynamic resolution based on model_type
        cs = 518 if model_type == "dinov2_l" else 512
        p_rgb_sub = cv2.resize(p_rgb, (cs, cs), interpolation=cv2.INTER_LINEAR)
        p_mask_sub = cv2.resize(p_mask, (cs, cs), interpolation=cv2.INTER_NEAREST)

        x_raw = torch.from_numpy(p_rgb_sub.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        y_mask = torch.from_numpy(p_mask_sub.astype(np.int64)).long().unsqueeze(0).to(device)
        wt_t = torch.tensor([wt], device=device)
        dp_t = torch.tensor([depth], device=device)

        if use_rsuigm and rsuigm_table is not None:
            x_in = apply_rsuigm_batch_gpu(x_raw, wt_t, dp_t, rsuigm_table)
        else:
            x_in = x_raw

        x_norm = (x_in - mean_im) / std_im

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_norm)

        evaluator.update(logits, y_mask)

    res = evaluator.compute()
    elapsed = time.time() - t0

    probe_report = {
        "diagnostic_protocol": "synthetic_unseen_host_probe",
        "checkpoint": checkpoint_path,
        "fold_id": fold_id,
        "model_type": model_type,
        "use_rsuigm": use_rsuigm,
        "total_probes": len(sampled_hosts),
        "pastes_recorded": pastes_recorded,
        "elapsed_sec": round(elapsed, 2),
        "metrics": res,
        "note": "Evaluates transfer of training-seen rare taxa to unseen fish background scenes. Diagnostic only; not official OOF."
    }

    print("-" * 72)
    print(f"SYNTHETIC PROBE RESULTS ({pastes_recorded} active rare object pastes onto unseen hosts):")
    print(f"  mIoU: {res['mIoU']:.4f} | Macro F1: {res['macro_F1']:.4f} | Q_seg: {res['Q_seg']:.4f}")
    print(f"  Class 3 (Shark/Ray) IoU: {res['per_class_IoU'][3]:.4f} | F1: {res['per_class_F1'][3]:.4f}")
    print(f"  Class 4 (Other)     IoU: {res['per_class_IoU'][4]:.4f} | F1: {res['per_class_F1'][4]:.4f}")
    print("=" * 72)

    return probe_report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold_id", type=str, default="fold_0")
    parser.add_argument("--model_type", type=str, default="dinov2_l")
    parser.add_argument("--raw_input", dest="use_rsuigm", action="store_false", default=True)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()

    evaluate_synthetic_unseen_host_probe(
        checkpoint_path=args.checkpoint,
        fold_id=args.fold_id,
        model_type=args.model_type,
        use_rsuigm=args.use_rsuigm,
        n_probe_samples=args.samples
    )
