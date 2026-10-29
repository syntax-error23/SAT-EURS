# SAT-EURS Track 1 — V21 Restoration Candidate Evaluation Benchmark

**Date:** 2026-10-18 17:02:22 UTC  
**Baseline SHA-256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52` (100% Intact)  
**Evaluated Folds:** [0, 2] (1600 Held-Out Images, 1774.72s Runtime)  

---

## 1. Pooled Restoration Candidate Benchmark Table

| Candidate ID | Candidate Architecture | Fold Safe | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | UIQM | UCIQE | $\Delta Q_{\text{FR}}$ | $\Delta S_{\text{comp}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cand1_raw_degraded` | Raw Degraded Input | YES | **12.72** | 0.4776 | 0.0693 | **0.2219** | 0.657 | 0.230 | -0.4271 | **-0.12813** |
| `cand2_r0_table_b_raw` | Strict Fold-Safe RSUIGM Candidate B (No Unsharp) | YES | **24.79** | 0.8047 | 0.3970 | **0.6222** | 1.126 | 0.335 | -0.0268 | **-0.00804** |
| `cand3_r0_table_b_unsharp` | Strict Fold-Safe RSUIGM Candidate B + Unsharp 1.34 | YES | **24.43** | 0.7983 | 0.6442 | **0.6357** | 1.287 | 0.373 | -0.0133 | **-0.00399** |
| `cand4_v18_refiner_fold_safe` | Strict Fold-Safe V18 Refiner (Fold Table + Fold Refiner) | YES | **25.41** | 0.8192 | 0.4621 | **0.6490** | 1.292 | 0.398 | +0.0000 | **+0.00000** |
| `cand5_v18_dev_ensemble` | V18 Dev Deployment Configuration (20-Scene Table + 5-Fold Ensemble) | NO (20sc) | **26.01** | 0.8338 | 0.4410 | **0.6669** | 1.232 | 0.339 | +0.0179 | **+0.00537** |

---

## 2. Fold-by-Fold Restoration Breakdown

### FOLD_0 Results:

| Candidate | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | UIQM | UCIQE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw Degraded Input | 13.99 | 0.4979 | 0.0652 | 0.2306 | 0.661 | 0.230 |
| Strict Fold-Safe RSUIGM Candidate B (No Unsharp) | 25.05 | 0.8002 | 0.3798 | 0.6242 | 1.120 | 0.342 |
| Strict Fold-Safe RSUIGM Candidate B + Unsharp 1.34 | 24.70 | 0.7943 | 0.6307 | 0.6388 | 1.275 | 0.391 |
| Strict Fold-Safe V18 Refiner (Fold Table + Fold Refiner) | 25.71 | 0.8184 | 0.4363 | 0.6528 | 1.309 | 0.423 |
| V18 Dev Deployment Configuration (20-Scene Table + 5-Fold Ensemble) | 26.24 | 0.8362 | 0.4282 | 0.6721 | 1.228 | 0.347 |

### FOLD_2 Results:

| Candidate | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | UIQM | UCIQE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Raw Degraded Input | 11.45 | 0.4574 | 0.0735 | 0.2132 | 0.653 | 0.230 |
| Strict Fold-Safe RSUIGM Candidate B (No Unsharp) | 24.54 | 0.8092 | 0.4142 | 0.6202 | 1.132 | 0.327 |
| Strict Fold-Safe RSUIGM Candidate B + Unsharp 1.34 | 24.15 | 0.8022 | 0.6578 | 0.6326 | 1.299 | 0.355 |
| Strict Fold-Safe V18 Refiner (Fold Table + Fold Refiner) | 25.11 | 0.8201 | 0.4878 | 0.6452 | 1.275 | 0.372 |
| V18 Dev Deployment Configuration (20-Scene Table + 5-Fold Ensemble) | 25.77 | 0.8313 | 0.4537 | 0.6617 | 1.236 | 0.330 |

