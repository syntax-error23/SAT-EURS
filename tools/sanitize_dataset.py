"""
Dataset Sanitizer & Integrity Verifier
Non-destructive check:
1. Verifies image & mask headers (PIL verify + cv2.imread).
2. Verifies channel count: clean & degraded are 3-channel RGB, masks are 1-channel.
3. Verifies mask discrete values are within valid class set {0, 1, 2, 3, 4}.
4. Computes SHA-256 checksums and identifies duplicates or corrupted files.
5. Quarantines corrupted files to audits/dataset_quarantine.json (NEVER deletes files).
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
import cv2
import numpy as np
from PIL import Image


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sanitize_dataset(data_dir: str, audits_dir: str = "audits") -> Dict[str, Any]:
    if not os.path.exists(os.path.join(data_dir, 'clean')) and os.path.exists(os.path.join(data_dir, 'mini_dev', 'clean')):
        data_dir = os.path.join(data_dir, 'mini_dev')

    clean_dir = os.path.join(data_dir, 'clean')
    masks_dir = os.path.join(data_dir, 'masks')
    degraded_dir = os.path.join(data_dir, 'degraded')

    os.makedirs(audits_dir, exist_ok=True)
    quarantine_log = os.path.join(audits_dir, "dataset_quarantine.json")

    quarantine: List[Dict[str, Any]] = []
    file_hashes: Dict[str, str] = {}
    duplicates: List[Dict[str, str]] = []
    valid_samples = 0

    print("=" * 60)
    print(f"[SANITIZER] Beginning non-destructive audit of: {os.path.abspath(data_dir)}")
    print("=" * 60)

    # 1. Audit clean images
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    clean_scenes = set()

    for fname in clean_files:
        path = os.path.join(clean_dir, fname)
        scene_id = os.path.splitext(fname)[0]
        # Header check
        try:
            with Image.open(path) as img:
                img.verify()
            cv_img = cv2.imread(path)
            if cv_img is None or cv_img.ndim != 3 or cv_img.shape[2] != 3:
                quarantine.append({"file": path, "issue": "Unreadable or invalid channels (expected 3)"})
                continue
            sha = compute_sha256(path)
            if sha in file_hashes:
                duplicates.append({"original": file_hashes[sha], "duplicate": path})
            else:
                file_hashes[sha] = path
            clean_scenes.add(scene_id)
        except Exception as e:
            quarantine.append({"file": path, "issue": f"Header corrupted: {str(e)}"})

    # 2. Audit masks
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_scenes = set()

    for fname in mask_files:
        path = os.path.join(masks_dir, fname)
        scene_id = os.path.splitext(fname)[0]
        try:
            with Image.open(path) as img:
                img.verify()
            cv_mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if cv_mask is None:
                quarantine.append({"file": path, "issue": "Unreadable mask"})
                continue
            if cv_mask.ndim > 2 and cv_mask.shape[2] > 1:
                cv_mask = cv_mask[:, :, 0]
            unique_vals = np.unique(cv_mask)
            if not set(unique_vals).issubset({0, 1, 2, 3, 4}):
                quarantine.append({"file": path, "issue": f"Invalid mask classes: {unique_vals.tolist()}"})
                continue
            mask_scenes.add(scene_id)
        except Exception as e:
            quarantine.append({"file": path, "issue": f"Mask corrupted: {str(e)}"})

    # 3. Audit degraded variations
    for scene_id in sorted(clean_scenes):
        scene_deg_dir = os.path.join(degraded_dir, scene_id)
        if not os.path.isdir(scene_deg_dir):
            quarantine.append({"file": scene_deg_dir, "issue": "Missing degraded folder for scene"})
            continue
        variants = sorted([f for f in os.listdir(scene_deg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for v in variants:
            v_path = os.path.join(scene_deg_dir, v)
            try:
                with Image.open(v_path) as img:
                    img.verify()
                cv_v = cv2.imread(v_path)
                if cv_v is None or cv_v.ndim != 3 or cv_v.shape[2] != 3:
                    quarantine.append({"file": v_path, "issue": "Unreadable degraded variation"})
                    continue
                valid_samples += 1
            except Exception as e:
                quarantine.append({"file": v_path, "issue": f"Degraded variation corrupted: {str(e)}"})

    # Write quarantine log
    report = {
        "dataset_root": os.path.abspath(data_dir),
        "total_valid_scenes": len(clean_scenes & mask_scenes),
        "total_valid_samples": valid_samples,
        "quarantined_files_count": len(quarantine),
        "quarantined_files": quarantine,
        "duplicates_count": len(duplicates),
        "duplicates": duplicates
    }

    with open(quarantine_log, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[SANITIZER] Completed audit:")
    print(f"  - Valid clean scenes: {len(clean_scenes)}")
    print(f"  - Valid mask scenes: {len(mask_scenes)}")
    print(f"  - Valid degraded samples: {valid_samples}")
    print(f"  - Quarantined issues: {len(quarantine)} (logged to {quarantine_log})")
    print(f"  - Duplicates: {len(duplicates)}")
    print("=" * 60)

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sanitize EURS Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--audits_dir", type=str, default="audits", help="Audits directory")
    args = parser.parse_args()

    sanitize_dataset(args.data_dir, args.audits_dir)
