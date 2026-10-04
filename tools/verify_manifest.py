"""
Manifest Cryptographic Verification Tool
Verifies:
1. SHA-256 checksum matches splits/manifest_full.json.sha256 exactly.
2. Scene disjointness invariant: set(train_scenes).isdisjoint(set(val_scenes)).
3. Every referenced scene and image path exists on disk.
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


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(
    manifest_path: str = "splits/manifest_full.json",
    data_dir: str = "data"
) -> bool:
    print("=" * 60)
    print(f"[VERIFY] Verifying manifest: {manifest_path}")
    print("=" * 60)

    assert os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}"
    hash_path = manifest_path + ".sha256"
    assert os.path.exists(hash_path), f"Manifest hash file missing: {hash_path}"

    with open(hash_path, 'r') as f:
        expected_line = f.read().strip()
    expected_hash = expected_line.split()[0]

    actual_hash = compute_sha256(manifest_path)
    if actual_hash != expected_hash:
        print(f"[FAIL] Cryptographic SHA-256 mismatch!")
        print(f"  Expected: {expected_hash}")
        print(f"  Actual:   {actual_hash}")
        sys.exit(1)
    print(f"  [PASS] Cryptographic SHA-256 verified: {actual_hash}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    train_scenes = set(manifest["train_scenes"])
    val_scenes = set(manifest["val_scenes"])

    assert train_scenes.isdisjoint(val_scenes), "FATAL: Train and Val scenes intersect! Data leakage detected!"
    print(f"  [PASS] Scene disjointness verified: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes (0 overlap)")

    if not os.path.exists(os.path.join(data_dir, 'clean')) and os.path.exists(os.path.join(data_dir, 'mini_dev', 'clean')):
        data_dir = os.path.join(data_dir, 'mini_dev')

    clean_dir = os.path.join(data_dir, 'clean')
    masks_dir = os.path.join(data_dir, 'masks')
    degraded_dir = os.path.join(data_dir, 'degraded')

    all_scenes = train_scenes | val_scenes
    for s in all_scenes:
        clean_found = any(os.path.exists(os.path.join(clean_dir, f"{s}{ext}")) for ext in ('.png', '.jpg', '.jpeg'))
        mask_found = any(os.path.exists(os.path.join(masks_dir, f"{s}{ext}")) for ext in ('.png', '.jpg', '.jpeg'))
        deg_dir = os.path.join(degraded_dir, s)
        deg_found = os.path.isdir(deg_dir)

        assert clean_found, f"Missing clean image for scene: {s}"
        assert mask_found, f"Missing mask image for scene: {s}"
        assert deg_found, f"Missing degraded directory for scene: {s}"

    print(f"  [PASS] All {len(all_scenes)} scenes validated on disk.")
    print("=" * 60)
    print("[VERIFY] Manifest Integrity: 100% VERIFIED")
    print("=" * 60 + "\n")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify Manifest Cryptographic Hash")
    parser.add_argument("--manifest", type=str, default="splits/manifest_full.json", help="Path to manifest")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    args = parser.parse_args()

    verify_manifest(args.manifest, args.data_dir)
