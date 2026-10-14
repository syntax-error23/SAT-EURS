# EURS-Net Submission Artifact Audit (Phase 1.1)

**Date:** October 14, 2026  
**Auditor:** Antigravity CV Research Team  
**Focus:** Forensic Inventory, Hash Verification, and Integrity Validation of Track 1 Submission Archives  

---

## 1. Archive Inventory & SHA256 Integrity Verification

All submission archives in `deploy/` have been audited, decompressed in memory, and verified against the official MaCVi specification:

| Archive Path | File Size | SHA256 (First 16) | Restored Images | Seg Masks | Pairing & Naming Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [`deploy/eurs_track1_ultimate_recovery_submission.zip`](file:///c:/Projects/SAT-EURS/deploy/eurs_track1_ultimate_recovery_submission.zip) | 572.29 MB | `2b9b71e9e88dc7af` | 1,200 | 1,200 | **100% Exact 1:1 Match** |
| [`deploy/eurs_track1_paired_A_D_submission.zip`](file:///c:/Projects/SAT-EURS/deploy/eurs_track1_paired_A_D_submission.zip) | 574.97 MB | `3daf756f635c366a` | 1,200 | 1,200 | **100% Exact 1:1 Match** |
| [`deploy/eurs_track1_verified_submission.zip`](file:///c:/Projects/SAT-EURS/deploy/eurs_track1_verified_submission.zip) | 902.71 MB | `c3ef083f119c7bbc` | 1,200 | 1,200 | **100% Exact 1:1 Match** |
| [`deploy/eurs_track1_dev_compact.zip`](file:///c:/Projects/SAT-EURS/deploy/eurs_track1_dev_compact.zip) | 735.84 MB | `0ef1a7f120852824` | 1,200 | 1,200 | **100% Exact 1:1 Match** |
| [`deploy/eurs_track1_dev_submission.zip`](file:///c:/Projects/SAT-EURS/deploy/eurs_track1_dev_submission.zip) | 1,240.59 MB | `65d8cae3f31587ea` | 1,200 | 1,200 | **Over 1024 MB Limit** |

---

## 2. Directory Structure & Content Validation

For both `deploy/eurs_track1_ultimate_recovery_submission.zip` and `deploy/eurs_track1_paired_A_D_submission.zip`:
1. **Archive Directory Structure:**
   - Canonical root:
     ```
     submission.zip
     ├── restored/
     │   ├── fish10_0001.jpg (or .png)
     │   └── ... (exactly 1,200 files)
     └── seg_masks/
         ├── fish10_0001.png
         └── ... (exactly 1,200 files)
     ```
   - Zero local machine absolute paths, drive letters (`C:\`), or hidden OS files (`.DS_Store`, `Thumbs.db`).
2. **Dimension & Channel Verification:**
   - Evaluated all 1,200 pairs: Dimension mismatches = **0**.
   - Masks: Single-channel `uint8` (`mode="L"`), pixel values strictly $\in \{0, 1, 2, 3, 4\}$.
   - Restored Images: 3-channel RGB float32 $\to$ uint8 $\in [0, 255]$ saved at JPEG quality 98.
3. **Target Sizing:**
   - Resized according to the official contract `scale_to_max_side(w, h, 1024)`.
   - Native dimensions up to $6000 \times 8000$ properly scaled to $768 \times 1024$ or $1024 \times 768$.

---

## 3. Provenance & Artifact Sourcing

1. **`deploy/eurs_track1_paired_A_D_submission.zip`:**
   - `restored/`: Sourced from `deploy/track1_dev_v5/restored` (the exact files responsible for Result A / #18672: PSNR = 25.49 dB, SSIM = 0.810, Sharpness = 0.893).
   - `seg_masks/`: Sourced from `deploy/track1_dev_submission_v6/seg_masks` (the exact files responsible for Result D: $Q_{\text{seg}} = 0.424$, $\text{mIoU} = 0.583$, $\text{Macro } F_1 = 0.727$).
2. **`deploy/eurs_track1_ultimate_recovery_submission.zip`:**
   - `restored/`: Sourced from `deploy/track1_dev_v5/restored` (Result A).
   - `seg_masks/`: Sourced from `deploy/eurs_track1_verified_submission/seg_masks` (4-way ensemble).
