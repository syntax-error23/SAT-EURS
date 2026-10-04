"""
Class Feasibility Audit Tool
Computes per-scene pixel mass vectors [m0, m1, m2, m3, m4].
Audits whether foreground classes can mathematically appear in both Train and Val
without violating scene-disjointness:
  - If scenes(class_c) >= 2: Feasible to stratify.
  - If scenes(class_c) < 2:  NOT feasible to stratify. Constraint documented.
Outputs:
  audits/class_feasibility.json
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import argparse
from typing import Dict, Any
import cv2
import numpy as np


CLASS_NAMES = {
    0: "Background",
    1: "Coral",
    2: "Fish",
    3: "Elasmobranchii",
    4: "Other"
}


def audit_class_feasibility(data_dir: str, audits_dir: str = "audits") -> Dict[str, Any]:
    if not os.path.exists(os.path.join(data_dir, 'masks')) and os.path.exists(os.path.join(data_dir, 'mini_dev', 'masks')):
        data_dir = os.path.join(data_dir, 'mini_dev')

    masks_dir = os.path.join(data_dir, 'masks')
    assert os.path.exists(masks_dir), f"Masks directory not found: {masks_dir}"

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    scene_mass: Dict[str, Dict[str, Any]] = {}
    class_scene_counts = {c: 0 for c in range(5)}
    total_class_pixels = {c: 0 for c in range(5)}

    print("=" * 60)
    print(f"[FEASIBILITY] Auditing class distribution across scenes in: {masks_dir}")
    print("=" * 60)

    for fname in mask_files:
        scene_id = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim > 2 and mask.shape[2] > 1:
            mask = mask[:, :, 0]

        total_pixels = mask.size
        counts = {c: int(np.sum(mask == c)) for c in range(5)}

        for c in range(5):
            total_class_pixels[c] += counts[c]
            if counts[c] > 0:
                class_scene_counts[c] += 1

        scene_mass[scene_id] = {
            "total_pixels": total_pixels,
            "pixel_counts": counts,
            "foreground_pixels": sum(counts[c] for c in range(1, 5)),
            "present_classes": [c for c in range(5) if counts[c] > 0]
        }

    # Feasibility analysis per class
    feasibility = {}
    for c in range(5):
        num_scenes = class_scene_counts[c]
        is_feasible = (num_scenes >= 2) if c > 0 else True
        feasibility[c] = {
            "class_name": CLASS_NAMES.get(c, f"Class {c}"),
            "scene_count": num_scenes,
            "total_pixels": total_class_pixels[c],
            "feasible_stratification": is_feasible,
            "constraint_note": "OK" if is_feasible else "Only appears in 1 scene; must be assigned to Train exclusively to prevent scene leakage."
        }

    report = {
        "total_scenes_audited": len(scene_mass),
        "class_feasibility": feasibility,
        "per_scene_mass": scene_mass
    }

    os.makedirs(audits_dir, exist_ok=True)
    out_path = os.path.join(audits_dir, "class_feasibility.json")
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[FEASIBILITY] Audit complete. Saved to: {out_path}")
    for c, info in feasibility.items():
        status = "[FEASIBLE]" if info["feasible_stratification"] else "[CONSTRAINT]"
        print(f"  {status} Class {c} ({info['class_name']}): present in {info['scene_count']} scenes, {info['total_pixels']:,} pixels")

    print("=" * 60)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audit Class Feasibility")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--audits_dir", type=str, default="audits", help="Audits directory")
    args = parser.parse_args()

    audit_class_feasibility(args.data_dir, args.audits_dir)
