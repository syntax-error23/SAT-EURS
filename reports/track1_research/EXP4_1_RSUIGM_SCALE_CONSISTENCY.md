# SAT-EURS Track 1 — Experiment R1: Scale-Consistent RSUIGM Fitting

**Date:** 2026-10-16 17:21:02  
**Hardware:** NVIDIA GeForce RTX 5090 (31.8 GB VRAM)  
**Total Duration:** 3712.55s  
**Verdict:** **PROMOTE_CANDIDATE_B**  

---

## 1. Executive Summary & Core Finding

This experiment tested the hypothesis of a **scale mismatch** between the features used to fit the 19-term RSUIGM inversion table and those evaluated at inference:
- **Control A (Existing Fitting Scale):** Degraded and clean images downsampled from 1024x1024 to 128x128 via `avg_pool2d(kernel=8, stride=8)`. Features computed on 128x128 degraded image; regressed against 128x128 clean targets.
- **Candidate B (Inference-Scale Fitting):** Features computed at full 1024x1024 resolution using `build_rsuigm_poly_features_gpu()`, exactly matching inference. Stratified jittered sampling (1 sample per 8x8 tile = 128x128 samples per image) with point-sampled clean targets at identical coordinates.
- **Strict Isolation:** Both models evaluated across all 5 folds (20 scenes, 4,000 out-of-fold images) with identical regularizer (`1e-2`), identical post-processing (Gaussian unsharp 1.34 / -0.34, sigma 2.0), and zero data leakage.

### Pooled 5-Fold Benchmark (4,000 Out-of-Fold Images)

| Metric | Control A (Downsampled 128x128) | Candidate B (Full-Res Stratified) | Delta (B - A) | Superior |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | 24.0545 | 24.1945 | **+0.1400** | Candidate B |
| **PSNR_norm** | 0.4527 | 0.4597 | **+0.0070** | Candidate B |
| **SSIM** | 0.7475 | 0.7530 | **+0.0055** | Candidate B |
| **Sharpness Ratio** | 0.5083 | 0.4882 | **-0.0201** | Control A |
| **Q_FR (Composite)** | **0.5909** | **0.5945** | **+0.0036** | **Candidate B** |
| **Clipping Fraction** | 4.4020% | 3.9900% | -0.4120% | — |

---

## 2. Promotion Criteria Scorecard

1. **Improves Pooled $Q_{\text{FR}}$:** `PASS` (Delta: `+0.00360`)  
2. **Improves $Q_{\text{FR}}$ on $\ge 4$ of 5 Folds:** `PASS` (5/5 folds improved)  
3. **No Material PSNR/SSIM Sacrifice:** `PASS` (dPSNR: `+0.1400` dB, dSSIM: `+0.00550`)  
4. **No Severe Fold Regression / Clipping Anomaly:** `PASS` (Min fold dQ_FR: `+0.00060`)  

**FINAL VERDICT:** **PROMOTE_CANDIDATE_B**

---

## 3. Fold-by-Fold Breakdown

| Fold | Val Scenes | Control A $Q_{\text{FR}}$ | Candidate B $Q_{\text{FR}}$ | $\Delta Q_{\text{FR}}$ | Control A PSNR | Candidate B PSNR | $\Delta$ PSNR | Control A SSIM | Candidate B SSIM | $\Delta$ SSIM | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | fish1, fish11, fish12, seahorse1 | 0.6283 | 0.6341 | **+0.0058** | 24.35 | 24.56 | +0.21 | 0.7866 | 0.7932 | +0.0066 | WIN (B) |
| Fold 1 | fish13, fish14, fish15, sharkFish | 0.5902 | 0.5908 | **+0.0006** | 24.50 | 24.58 | +0.09 | 0.7489 | 0.7509 | +0.0020 | WIN (B) |
| Fold 2 | fish17, fish18, fish2, ray1 | 0.6172 | 0.6237 | **+0.0065** | 23.80 | 24.00 | +0.20 | 0.7908 | 0.7996 | +0.0088 | WIN (B) |
| Fold 3 | fish20, fish21, fish4, starfish2 | 0.5264 | 0.5274 | **+0.0010** | 23.34 | 23.37 | +0.03 | 0.6671 | 0.6725 | +0.0054 | WIN (B) |
| Fold 4 | fish5, fish7, fish9, turtles1 | 0.5925 | 0.5966 | **+0.0041** | 24.28 | 24.46 | +0.17 | 0.7439 | 0.7487 | +0.0048 | WIN (B) |

---

## 4. Per-Scene Granular Results (All 20 Held-Out Scenes)

| Scene | Fold | Control A $Q_{\text{FR}}$ | Candidate B $Q_{\text{FR}}$ | $\Delta Q_{\text{FR}}$ | Control A PSNR | Candidate B PSNR | $\Delta$ PSNR | Control A SSIM | Candidate B SSIM | Clip A | Clip B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fish1` | Fold 0 | 0.6126 | 0.6211 | **+0.0085** | 24.80 | 25.07 | +0.27 | 0.7575 | 0.7688 | 3.43% | 3.29% |
| `fish11` | Fold 0 | 0.5556 | 0.5645 | **+0.0089** | 22.12 | 22.38 | +0.26 | 0.7588 | 0.7721 | 1.68% | 1.77% |
| `fish12` | Fold 0 | 0.7351 | 0.7405 | **+0.0054** | 27.27 | 27.54 | +0.26 | 0.8617 | 0.8626 | 12.66% | 7.82% |
| `seahorse1` | Fold 0 | 0.6099 | 0.6103 | **+0.0004** | 23.22 | 23.26 | +0.04 | 0.7685 | 0.7691 | 1.71% | 2.41% |
| `fish13` | Fold 1 | 0.6923 | 0.6902 | **-0.0021** | 26.55 | 26.66 | +0.11 | 0.8435 | 0.8400 | 14.16% | 10.99% |
| `fish14` | Fold 1 | 0.5485 | 0.5506 | **+0.0021** | 22.70 | 22.84 | +0.15 | 0.7300 | 0.7319 | 10.43% | 10.00% |
| `fish15` | Fold 1 | 0.4306 | 0.4370 | **+0.0064** | 20.60 | 20.82 | +0.22 | 0.6234 | 0.6307 | 0.03% | 0.03% |
| `sharkFish` | Fold 1 | 0.6892 | 0.6855 | **-0.0037** | 28.15 | 28.01 | -0.14 | 0.7987 | 0.8009 | 0.62% | 0.77% |
| `fish17` | Fold 2 | 0.6134 | 0.6131 | **-0.0003** | 23.93 | 24.05 | +0.12 | 0.8066 | 0.8081 | 9.19% | 7.98% |
| `fish18` | Fold 2 | 0.6314 | 0.6393 | **+0.0079** | 23.17 | 23.32 | +0.15 | 0.8051 | 0.8154 | 0.21% | 0.21% |
| `fish2` | Fold 2 | 0.5438 | 0.5524 | **+0.0086** | 22.64 | 22.92 | +0.27 | 0.7212 | 0.7344 | 1.63% | 1.88% |
| `ray1` | Fold 2 | 0.6804 | 0.6901 | **+0.0097** | 25.45 | 25.72 | +0.27 | 0.8303 | 0.8403 | 2.01% | 1.85% |
| `fish20` | Fold 3 | 0.5355 | 0.5296 | **-0.0059** | 23.41 | 23.24 | -0.17 | 0.6848 | 0.6854 | 1.15% | 1.71% |
| `fish21` | Fold 3 | 0.4477 | 0.4451 | **-0.0026** | 24.21 | 24.16 | -0.06 | 0.5064 | 0.5074 | 4.12% | 4.74% |
| `fish4` | Fold 3 | 0.6191 | 0.6273 | **+0.0082** | 24.06 | 24.29 | +0.23 | 0.7724 | 0.7830 | 6.41% | 5.31% |
| `starfish2` | Fold 3 | 0.5032 | 0.5075 | **+0.0043** | 21.68 | 21.78 | +0.11 | 0.7047 | 0.7140 | 0.46% | 0.60% |
| `fish5` | Fold 4 | 0.5148 | 0.5195 | **+0.0047** | 21.53 | 21.67 | +0.14 | 0.7152 | 0.7223 | 3.60% | 3.95% |
| `fish7` | Fold 4 | 0.5452 | 0.5426 | **-0.0026** | 24.92 | 24.99 | +0.07 | 0.6453 | 0.6444 | 7.45% | 9.65% |
| `fish9` | Fold 4 | 0.6542 | 0.6614 | **+0.0072** | 24.87 | 25.18 | +0.31 | 0.8208 | 0.8250 | 7.05% | 4.82% |
| `turtles1` | Fold 4 | 0.6559 | 0.6628 | **+0.0069** | 25.80 | 25.97 | +0.17 | 0.7943 | 0.8030 | 0.05% | 0.01% |

---

## 5. Architectural & Mathematical Analysis

### Why the Scale Mismatch Affected the Parameters
1. **Veiling Light Kernel Semantics:**
   - In Control A, `build_rsuigm_poly_features_gpu` operates on $128 \times 128$ inputs.
   - The $16 \times 16$ pooling filter covers $1/8$ of the image spatial dimension ($128$ pixels at native $1024$ scale).
   - The $64 \times 64$ pooling filter covers $1/2$ of the image spatial dimension ($512$ pixels at native $1024$ scale).
   - At inference on $1024 \times 1024$, the exact same pooling filters cover $1/64$ ($16$ px) and $1/16$ ($64$ px) of the image.
   - Candidate B resolves this disparity by extracting the veiling basis directly at native $1024 \times 1024$ scale during fitting.

2. **Point Extremes vs. Spatial Averages:**
   - In Control A, polynomial terms $(x, x^2, x^3)$ are regressed against $8 \times 8$ average-pooled targets.
   - Average pooling reduces high-frequency variance and attenuates edge contrast.
   - In Candidate B, the stratified point samples match high-variance point targets, providing scale-consistent polynomial coefficients.

3. **Inference Consistency:**
   - Candidate B guarantees mathematical isomorphism between feature distribution at fitting time and inference time.

---

## 6. Safety & Preservation Attestation

- Production tables `checkpoints/track1_screening/tbl19_20sc_clean.pt` and `tbl19_16sc_fold*.pt`: **UNMODIFIED**.
- Production submission archives `deploy/submission_v16_apex_sota.zip` and `deploy/submission_v17_dual_backbone_sota.zip`: **UNMODIFIED**.
- Frozen V17 candidate manifest `deploy/FROZEN_CANDIDATE_V17_MANIFEST.json`: **UNMODIFIED**.
- No neural restorer was attached without empirical validation.
- Local metrics evaluated via exact `RestorationEvaluator` without confounding no-reference approximations.
