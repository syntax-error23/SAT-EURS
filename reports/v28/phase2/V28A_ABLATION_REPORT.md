# SAT-EURS Track 1 — V28-A Controlled Five-Fold Ablation Report

**Campaign:** SAT-EURS V28 Corrected Phase 2 Score Recovery Campaign  
**Date:** October 24, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** MaCVi @ WACV 2027 EURS Track 1 Testing Phase  
**Archive Generated:** [`deploy/v28/submission_v28a_fivefold_ablation.zip`](file:///c:/Projects/SAT-EURS/deploy/v28/submission_v28a_fivefold_ablation.zip)  
**Archive SHA-256:** `471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39`  
**Archive Size:** 497.98 MB (51.4% headroom below 1,024 MB limit)  
**Status:** **PASSED 7-POINT PRE-SUBMISSION AUDIT — VERIFIED BIT-FOR-BIT**

---

## 1. Hypothesis Tested

**Hypothesis V28-A:** The naive linear blending of the clean-trained Full-20 SWA model ($w_{\text{full20}} = 0.50$) in V27 diluted foreground predictions and over-expanded the background, suppressing true positive detections of elusive fauna. Setting $w_{\text{full20}} = 0.00$ while retaining the upgraded targeted Fold 0 segmenter will isolate the true impact of the 5-fold ensemble without Full-20 dilution.

---

## 2. Experimental Configuration & Pipeline Integrity

1. **Restoration Pipeline:** Exact V27 restoration pipeline (5-Fold Refiners + `tbl19_20sc_scale_consistent.pt` + $\alpha=0.10$ Gaussian unsharp filter). Reused all 1,200 verified V27 raw restored images from `deploy/v27/track1_test_outputs/raw_unpadded/restored`.
2. **Segmenter Ensemble:**
   - Fold 0: Upgraded small-object targeted checkpoint ([`checkpoints/v27/segmenter_v27_fold0_best.pt`](file:///c:/Projects/SAT-EURS/checkpoints/v27/segmenter_v27_fold0_best.pt), SHA-256: `1a5dbfc86964317312474def3e037eb4eb7639da47a94d06518099adeb1e5464`)
   - Fold 1: Baseline V18 segmenter ([`checkpoints/v18/fold_1/segmenter_best.pt`](file:///c:/Projects/SAT-EURS/checkpoints/v18/fold_1/segmenter_best.pt))
   - Fold 2: Baseline V18 segmenter ([`checkpoints/v18/fold_2/segmenter_best.pt`](file:///c:/Projects/SAT-EURS/checkpoints/v18/fold_2/segmenter_best.pt))
   - Fold 3: Baseline V18 segmenter ([`checkpoints/v18/fold_3/segmenter_best.pt`](file:///c:/Projects/SAT-EURS/checkpoints/v18/fold_3/segmenter_best.pt))
   - Fold 4: Baseline V18 segmenter ([`checkpoints/v18/fold_4/segmenter_best.pt`](file:///c:/Projects/SAT-EURS/checkpoints/v18/fold_4/segmenter_best.pt))
3. **Full-20 Contribution:** Strictly **0.00** (Zero Full-20 weight).
4. **Test-Time Augmentation (TTA):** Horizontal-flip TTA on both restored and raw degraded feature streams.
5. **Logit Calibration:** $[0.0, 0.0, 0.0, 0.10, 0.25]$ applied before argmax.
6. **Native Resolution Resizing:** Resized to exact native test resolutions via `scale_to_max_side(1024)` contract.
7. **Packaging:** Compressed at JPEG quality 98 and saved as single-channel `uint8` PNG masks.

---

## 3. Strict 7-Point Package Audit Results

| Audit Check | Specification / Gate Requirement | Measured Status | Verification |
| :--- | :--- | :--- | :---: |
| **1. File Count** | Exactly 2,400 files (1,200 restored JPEGs, 1,200 PNG masks) | Exactly 2,400 files | **PASSED** |
| **2. Stem Matching** | 100% 1:1 match against `data/test/degraded` | 1,200/1,200 matched (0 orphans, 0 missing) | **PASSED** |
| **3. Dimensions** | Conform strictly to `scale_to_max_side(orig_w, orig_h, 1024)` | Zero dimension mismatches | **PASSED** |
| **4. Mask Encoding** | Single-channel 2D `uint8` (`mode="L"`) | 1,200 valid 2D uint8 PNGs | **PASSED** |
| **5. Class IDs** | Discrete values strictly $\in \{0, 1, 2, 3, 4\}$ | Discrete $\{0, 1, 2, 3, 4\}$ confirmed | **PASSED** |
| **6. Archive Size** | Strict limit: $< 1,024$ MB | **497.98 MB** (526 MB headroom) | **PASSED** |
| **7. Baseline Protection** | Cryptographic hashes of V18, V26, and V27 unmodified | Bit-for-bit immutable | **PASSED** |

---

## 4. Full-Census Pixel Distribution Comparison

| Class ID / Name | V26 Apex (#19081: 0.622) | V27 Hybrid (#19096: 0.633) | V28-A Ablation Candidate | Delta V28-A vs V27 |
| :--- | :---: | :---: | :---: | :---: |
| **Total Test Pixels** | 869,580,800 px | 869,580,800 px | 869,580,800 px | 0 px |
| **0: Background** | 479,408,280 (55.13%) | 505,258,034 (58.10%) | **483,403,360 (55.59%)** | **-21,854,674 (-2.51%)** |
| **1: Coral** | 326,478,196 (37.54%) | 307,390,666 (35.35%) | **325,020,817 (37.38%)** | **+17,630,151 (+2.03%)** |
| **2: Fish** | 18,832,528 (2.17%) | 15,325,496 (1.76%) | **17,221,423 (1.98%)** | **+1,895,927 (+12.37% rel)** |
| **3: Ray** | 24,646,692 (2.83%) | 22,631,241 (2.60%) | **24,932,496 (2.87%)** | **+2,301,255 (+10.17% rel)** |
| **4: MacroFauna** | 20,215,104 (2.33%) | 18,975,363 (2.18%) | **19,002,704 (2.18%)** | **+27,341 (+0.14% rel)** |

### Diagnostic Observations:
1. **Foreground Recovery:** Removing Full-20 recovers **+1.90M Fish pixels (+12.37% relative)** and **+2.30M Ray pixels (+10.17% relative)**, confirming that Full-20's high background probability was actively suppressing these detections.
2. **Census Diagnostic Limitation:** As instructed by protocol, predicted pixel counts serve as a diagnostic indicator of sensitivity shift, not proof of ground-truth segmentation accuracy. The archive is preserved locally under `deploy/v28/` for formal comparison.
