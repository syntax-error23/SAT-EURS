"""
Audit Dataset Distribution & Generate Scene-Disjoint Split Manifest
Inspects the dataset directory, measures scene variation distribution,
validates clean/mask/degraded matching, and outputs:
1. audits/dataset_stats.json
2. splits/manifest.json (frozen persistent train/val split by scene_id)
"""

import os
import re
import json
import argparse
import statistics


def audit_and_split(data_dir: str, splits_dir: str, audits_dir: str, val_ratio: float = 0.25):
    clean_dir = os.path.join(data_dir, 'clean')
    masks_dir = os.path.join(data_dir, 'masks')
    degraded_dir = os.path.join(data_dir, 'degraded')

    assert os.path.exists(clean_dir), f"Missing clean directory: {clean_dir}"
    assert os.path.exists(masks_dir), f"Missing masks directory: {masks_dir}"
    assert os.path.exists(degraded_dir), f"Missing degraded directory: {degraded_dir}"

    clean_files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Extract scene IDs
    clean_scenes = {os.path.splitext(f)[0]: f for f in clean_files}
    mask_scenes = {os.path.splitext(f)[0]: f for f in mask_files}

    # Verify 1-to-1 clean to mask match
    assert set(clean_scenes.keys()) == set(mask_scenes.keys()), (
        f"Mismatch between clean and mask scenes! Clean only: {set(clean_scenes.keys()) - set(mask_scenes.keys())}, "
        f"Mask only: {set(mask_scenes.keys()) - set(clean_scenes.keys())}"
    )

    all_scenes = sorted(list(clean_scenes.keys()))

    # Degraded variants per scene
    scene_variants = {}
    total_degraded_count = 0

    for scene in all_scenes:
        scene_deg_path = os.path.join(degraded_dir, scene)
        if os.path.isdir(scene_deg_path):
            variants = sorted([
                f for f in os.listdir(scene_deg_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            variants = []
        scene_variants[scene] = len(variants)
        total_degraded_count += len(variants)

    counts = list(scene_variants.values())
    stats = {
        "dataset_root": os.path.abspath(data_dir),
        "total_unique_scenes": len(all_scenes),
        "scenes": all_scenes,
        "total_degraded_images": total_degraded_count,
        "variants_per_scene": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(statistics.mean(counts), 2) if counts else 0,
            "median": statistics.median(counts) if counts else 0,
            "distribution": scene_variants
        }
    }

    os.makedirs(audits_dir, exist_ok=True)
    stats_path = os.path.join(audits_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[AUDIT] Saved dataset statistics to {stats_path}")
    print(f"  - Total scenes: {len(all_scenes)}")
    print(f"  - Total degraded variations: {total_degraded_count}")
    print(f"  - Variants per scene: min={stats['variants_per_scene']['min']}, max={stats['variants_per_scene']['max']}, mean={stats['variants_per_scene']['mean']}")

    # Scene-disjoint split creation (deterministic)
    # Reserve at least 1 scene for val if total_scenes >= 2
    os.makedirs(splits_dir, exist_ok=True)
    manifest_path = os.path.join(splits_dir, 'manifest.json')

    # If manifest already exists, verify and do not overwrite to keep it strictly persistent
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"[SPLIT] Existing manifest found at {manifest_path}. Preserving frozen split:")
        print(f"  - Train scenes ({len(manifest['train_scenes'])}): {manifest['train_scenes']}")
        print(f"  - Val scenes ({len(manifest['val_scenes'])}): {manifest['val_scenes']}")
    else:
        num_val = max(1, int(round(len(all_scenes) * val_ratio))) if len(all_scenes) > 1 else 0
        # Deterministic choice: pick every k-th scene or last scenes
        val_scenes = all_scenes[1::int(1/val_ratio)] if val_ratio > 0 and len(all_scenes) > 2 else all_scenes[-num_val:] if num_val > 0 else []
        train_scenes = [s for s in all_scenes if s not in val_scenes]

        manifest = {
            "split_policy": "scene_disjoint_persistent",
            "val_ratio_target": val_ratio,
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "total_scenes": len(all_scenes),
            "train_samples_count": sum(scene_variants[s] for s in train_scenes),
            "val_samples_count": sum(scene_variants[s] for s in val_scenes)
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"[SPLIT] Created new frozen manifest at {manifest_path}:")
        print(f"  - Train scenes ({len(train_scenes)}): {train_scenes}")
        print(f"  - Val scenes ({len(val_scenes)}): {val_scenes}")

    # Enforce disjoint assertion
    assert set(manifest['train_scenes']).isdisjoint(set(manifest['val_scenes'])), "FATAL: Train and Val scenes overlap!"
    print("  [PASS] Scene-disjoint invariant verified: 0% data leakage.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audit dataset and generate scene-disjoint manifest.")
    parser.add_argument('--data_dir', type=str, default='d:/MACVI/data/mini_dev', help='Path to dataset directory')
    parser.add_argument('--splits_dir', type=str, default='d:/MACVI/splits', help='Path to splits directory')
    parser.add_argument('--audits_dir', type=str, default='d:/MACVI/audits', help='Path to audits directory')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='Validation fraction (scene-level)')
    args = parser.parse_args()

    audit_and_split(args.data_dir, args.splits_dir, args.audits_dir, args.val_ratio)
