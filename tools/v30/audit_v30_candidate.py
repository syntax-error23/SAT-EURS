"""
SAT-EURS Track 1 — Candidate V30 Forensic Integrity Audit & Statistical Comparison
Compares Candidate V30 (Scene Consensus SOTA) with V28-A (Baseline 0.634) and V29 (Regressed 0.597).
"""

import os
import os.path as osp
import sys
import zipfile
import hashlib
import numpy as np
import cv2
from collections import Counter

sys.path.insert(0, ".")

V30_ZIP = "deploy/v30/submission_v30_consensus_sota.zip"
V29_ZIP = "deploy/v29/submission_v29_dual_backbone_sota.zip"
V28A_ZIP = "deploy/v28/submission_v28a_fivefold_ablation.zip"

EXPECTED_V28A_SHA = "471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39"
EXPECTED_V29_SHA = "b73155c757376c721b7cfff9ee7f160e050a4237cf8c01f5cb2943405599d6c3"


def audit_archive(zip_path: str, expected_sha: str = None, name: str = ""):
    print("=" * 85)
    print(f"AUDITING {name} ({zip_path})")
    print("=" * 85)

    if not osp.exists(zip_path):
        print(f"  [ERROR] File does not exist: {zip_path}")
        return None

    # 1. SHA-256 Check
    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest().lower()
    if expected_sha:
        assert actual_sha == expected_sha, f"SHA MISMATCH! Expected {expected_sha}, got {actual_sha}"
        print(f"  [OK] SHA-256 Verified: {actual_sha}")
    else:
        print(f"  [OK] SHA-256 Generated: {actual_sha}")

    # 2. Inspect ZIP Structure
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        r_files = sorted([n for n in namelist if n.startswith("restored/") and n.endswith(".jpg")])
        m_files = sorted([n for n in namelist if n.startswith("seg_masks/") and n.endswith(".png")])

        print(f"  [OK] Restored images count: {len(r_files)}")
        print(f"  [OK] Segmentation masks count: {len(m_files)}")
        assert len(r_files) == 1200, f"Expected 1,200 restored images, got {len(r_files)}"
        assert len(m_files) == 1200, f"Expected 1,200 segmentation masks, got {len(m_files)}"

        r_stems = [osp.splitext(osp.basename(f))[0] for f in r_files]
        m_stems = [osp.splitext(osp.basename(f))[0] for f in m_files]
        assert r_stems == m_stems, "Stem mismatch between restored and masks!"

        # 3. Analyze Masks & Class Distribution
        total_pixels = Counter()
        illegal_pixels = 0
        min_dim = (99999, 99999)
        max_dim = (0, 0)

        for i, mf in enumerate(m_files):
            data = zf.read(mf)
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            assert arr.ndim == 2, f"Mask {mf} is not 2D grayscale!"
            assert arr.dtype == np.uint8, f"Mask {mf} is not uint8!"

            min_dim = (min(min_dim[0], arr.shape[0]), min(min_dim[1], arr.shape[1]))
            max_dim = (max(max_dim[0], arr.shape[0]), max(max_dim[1], arr.shape[1]))

            unq, cnts = np.unique(arr, return_counts=True)
            for u, c in zip(unq, cnts):
                if u > 4:
                    illegal_pixels += c
                total_pixels[int(u)] += int(c)

        assert illegal_pixels == 0, f"Found {illegal_pixels} illegal pixel values (> 4)!"

        tot = sum(total_pixels.values())
        print("  Pixel Class Distribution:")
        class_names = ["Background", "Coral", "Fish", "Elasmobranch", "Macrofauna"]
        for c in range(5):
            cnt = total_pixels[c]
            pct = 100.0 * cnt / tot
            print(f"    Class {c} ({class_names[c]:12s}): {cnt:14,d} pixels ({pct:6.2f}%)")
        print(f"  Dimension Range: min={min_dim}, max={max_dim}")
        print("=" * 85)
        return {"sha256": actual_sha, "pixels": total_pixels, "total": tot}


if __name__ == "__main__":
    p28 = audit_archive(V28A_ZIP, EXPECTED_V28A_SHA, "V28-A (Production Baseline 0.634)")
    p29 = audit_archive(V29_ZIP, EXPECTED_V29_SHA, "V29 (Dual-Backbone SOTA Candidate)")
    p30 = audit_archive(V30_ZIP, None, "V30 (Cross-Degradation Consensus SOTA)")
