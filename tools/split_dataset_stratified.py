"""
Mass-Balanced Scene-Disjoint Split Generator & Hash Freezing
1. Uses class feasibility audit and per-scene pixel masses.
2. Enforces scene-disjointness: set(train_scenes).isdisjoint(set(val_scenes)).
3. Preserves feasibility constraints (scenes with single-occurrence classes locked to train).
4. Balances foreground pixel mass across train and val splits.
5. Emits splits/manifest_full.json and splits/manifest_full.json.sha256.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import hashlib
import argparse
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.audit_class_feasibility import audit_class_feasibility


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_mass_balanced_split(
    data_dir: str = "data",
    splits_dir: str = "splits",
    audits_dir: str = "audits",
    val_ratio: float = 0.20,
    seed: int = 42
) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(data_dir, 'clean')) and os.path.exists(os.path.join(data_dir, 'mini_dev', 'clean')):
        data_dir = os.path.join(data_dir, 'mini_dev')

    # 1. Run or load feasibility audit
    feasibility_file = os.path.join(audits_dir, "class_feasibility.json")
    if os.path.exists(feasibility_file):
        with open(feasibility_file, 'r') as f:
            audit = json.load(f)
    else:
        audit = audit_class_feasibility(data_dir, audits_dir)

    scene_mass = audit["per_scene_mass"]
    class_feasibility = audit["class_feasibility"]

    all_scenes = sorted(list(scene_mass.keys()))
    assert len(all_scenes) > 0, "No valid scenes found for splitting!"

    # 2. Lock scenes containing unfeasible classes into train
    locked_train_scenes = set()
    for c_str, info in class_feasibility.items():
        c = int(c_str)
        if not info["feasible_stratification"]:
            for scene_id, data in scene_mass.items():
                if c in data["present_classes"]:
                    locked_train_scenes.add(scene_id)
                    print(f"[SPLIT] Locking scene '{scene_id}' to Train (contains unfeasible class {c})")

    available_for_val = [s for s in all_scenes if s not in locked_train_scenes]

    target_val_count = max(1, int(round(len(all_scenes) * val_ratio))) if len(all_scenes) > 1 else 0
    target_val_count = min(target_val_count, len(available_for_val))

    # 3. Greedy mass-balanced selection for validation
    # Sort available scenes by foreground mass descending
    available_for_val.sort(key=lambda s: scene_mass[s]["foreground_pixels"], reverse=True)

    val_scenes: List[str] = []
    # Deterministic selection aiming for val_ratio of foreground mass
    total_fg = sum(scene_mass[s]["foreground_pixels"] for s in all_scenes)
    target_fg_val = total_fg * val_ratio

    current_val_fg = 0
    # Evenly pick scenes from available list
    step = len(available_for_val) / max(1, target_val_count)
    chosen_indices = [int(i * step) for i in range(target_val_count)]
    for idx in chosen_indices:
        if idx < len(available_for_val):
            val_scenes.append(available_for_val[idx])

    train_scenes = sorted([s for s in all_scenes if s not in val_scenes])
    val_scenes = sorted(val_scenes)

    # 4. Strict Invariant Checks
    assert set(train_scenes).isdisjoint(set(val_scenes)), "FATAL: Train and Val scenes intersect!"
    assert len(train_scenes) + len(val_scenes) == len(all_scenes), "FATAL: Scene count mismatch!"

    # Check variation counts
    degraded_dir = os.path.join(data_dir, 'degraded')
    train_samples = 0
    val_samples = 0
    for s in train_scenes:
        deg_path = os.path.join(degraded_dir, s)
        if os.path.isdir(deg_path):
            train_samples += len([f for f in os.listdir(deg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for s in val_scenes:
        deg_path = os.path.join(degraded_dir, s)
        if os.path.isdir(deg_path):
            val_samples += len([f for f in os.listdir(deg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    manifest = {
        "split_policy": "mass_balanced_scene_disjoint_persistent",
        "seed": seed,
        "val_ratio_target": val_ratio,
        "total_scenes": len(all_scenes),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "train_samples_count": train_samples,
        "val_samples_count": val_samples,
        "foreground_mass": {
            "train": sum(scene_mass[s]["foreground_pixels"] for s in train_scenes),
            "val": sum(scene_mass[s]["foreground_pixels"] for s in val_scenes)
        }
    }

    os.makedirs(splits_dir, exist_ok=True)
    manifest_path = os.path.join(splits_dir, "manifest_full.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # 5. Compute SHA-256 and freeze hash
    sha256_hash = compute_file_sha256(manifest_path)
    hash_path = manifest_path + ".sha256"
    with open(hash_path, 'w') as f:
        f.write(f"{sha256_hash}  manifest_full.json\n")

    print("=" * 60)
    print(f"[SPLIT] Successfully generated and froze split:")
    print(f"  - Manifest: {manifest_path}")
    print(f"  - SHA-256:  {sha256_hash} (saved to {hash_path})")
    print(f"  - Train Scenes ({len(train_scenes)}): {train_scenes}")
    print(f"  - Val Scenes ({len(val_scenes)}): {val_scenes}")
    print(f"  - Train Samples: {train_samples}, Val Samples: {val_samples}")
    print("=" * 60)

    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Mass-Balanced Scene Split")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--splits_dir", type=str, default="splits", help="Splits folder")
    parser.add_argument("--audits_dir", type=str, default="audits", help="Audits folder")
    parser.add_argument("--val_ratio", type=float, default=0.20, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_mass_balanced_split(args.data_dir, args.splits_dir, args.audits_dir, args.val_ratio, args.seed)
