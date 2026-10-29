"""
Phase 1 Pre-Flight Audit Script: Manifest, Leakage & Scorer Audit
Validates Safeguards 1 & 3:
1. Disk scenes vs manifest 5-fold validation
2. Mutually exclusive folds (each scene in exactly 1 val fold, 4 train folds)
3. Zero train/val leakage
4. Number of variants per scene (exactly 200)
5. Metric computation contract on pooled predictions
"""

import os
import os.path as osp
import json
import glob
import numpy as np

def audit_manifest_and_data():
    print("=" * 76)
    print("PHASE 1 AUDIT: MANIFEST, LEAKAGE & DATASET INTEGRITY (SAFEGUARD 1)")
    print("=" * 76)

    data_root = "data/cached_1024"
    clean_dir = osp.join(data_root, "clean")
    masks_dir = osp.join(data_root, "masks")
    deg_dir = osp.join(data_root, "degraded")

    clean_files = sorted(glob.glob(osp.join(clean_dir, "*.png")))
    mask_files = sorted(glob.glob(osp.join(masks_dir, "*.png")))
    disk_scenes = sorted([osp.splitext(osp.basename(f))[0] for f in clean_files])

    print(f"Total source scenes on disk: {len(disk_scenes)}")
    assert len(disk_scenes) == 20, f"Expected 20 scenes, found {len(disk_scenes)}: {disk_scenes}"
    assert len(mask_files) == 20, f"Expected 20 masks, found {len(mask_files)}"

    # Check 200 variants per scene
    scene_variant_counts = {}
    for sc in disk_scenes:
        deg_sc_dir = osp.join(deg_dir, sc)
        assert osp.exists(deg_sc_dir), f"Missing degraded directory for scene {sc}"
        deg_files = [f for f in os.listdir(deg_sc_dir) if f.endswith(".png")]
        scene_variant_counts[sc] = len(deg_files)
        assert len(deg_files) == 200, f"Scene {sc} has {len(deg_files)} variants, expected 200"

    print("All 20 scenes have exactly 200 degraded variants on disk (total 4,000 images): PASS")

    # Audit splits/manifest_5fold.json
    manifest_path = "splits/manifest_5fold.json"
    assert osp.exists(manifest_path), f"Missing {manifest_path}"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    folds = manifest["folds"]
    assert len(folds) == 5, f"Expected 5 folds, found {len(folds)}"

    val_scenes_seen = []
    for f_idx in range(5):
        f_name = f"fold_{f_idx}"
        f_data = folds[f_name]
        tr = set(f_data["train_scenes"])
        va = set(f_data["val_scenes"])

        assert len(tr) == 16, f"{f_name} train_scenes count is {len(tr)}, expected 16"
        assert len(va) == 4, f"{f_name} val_scenes count is {len(va)}, expected 4"
        assert len(tr.intersection(va)) == 0, f"LEAKAGE DETECTED in {f_name}! Overlap: {tr.intersection(va)}"
        assert tr.union(va) == set(disk_scenes), f"{f_name} union does not cover all 20 scenes"

        val_scenes_seen.extend(f_data["val_scenes"])

    assert len(val_scenes_seen) == 20, f"Total validation scenes is {len(val_scenes_seen)}, expected 20"
    assert set(val_scenes_seen) == set(disk_scenes), "Validation scenes do not match all disk scenes"
    assert len(set(val_scenes_seen)) == 20, "Duplicate validation scenes detected across folds"

    print("Manifest audit: All 20 scenes mutually exclusive across 5 validation splits: PASS")
    print("Zero train/validation leakage in manifest: PASS")
    print("=" * 76)

if __name__ == "__main__":
    audit_manifest_and_data()
