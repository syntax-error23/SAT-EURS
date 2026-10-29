# SAT-EURS Track 1 — V21-B Physics-Anchored Restoration Correction Pilot Report

**Date:** 2026-10-18 19:33:01 UTC  
**Baseline SHA-256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52` (100% Verified Intact)  
**Pilot Scope:** Limited Two-Fold Matched Benchmark (Folds 0 & 2, 1,600 Held-Out Images)  
**Pre-Registered Decision Gate Verdict:** **NO-GO (CEILING / DEFICIENCY IDENTIFIED — STOP ARCHITECTURE EXPERIMENTS)**  

## 1. Matched Comparison Table (Control vs. Treatment)

| Evaluation Split | Model / Pipeline | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | UIQM | UCIQE | CCF | $0.30 Q_{\text{FR}} + 0.10 Q_{\text{NR}}$ | $\Delta$ Composite Contribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | Control: V18 Fold Table + Refiner | 25.71 | 0.8184 | 0.4363 | 0.6533 | 1.309 | 0.423 | 0.0122 | 0.27918 | — |
| | **Treatment: V18 + Anchored Corrector** | **25.7** | **0.8221** | **0.4386** | **0.655** | 1.304 | 0.42 | 0.0121 | **0.2792** | **+0.00002** |
| **Fold 2** | Control: V18 Fold Table + Refiner | 25.11 | 0.8201 | 0.4878 | 0.6477 | 1.275 | 0.372 | 0.0093 | 0.27136 | — |
| | **Treatment: V18 + Anchored Corrector** | **25.09** | **0.8225** | **0.4897** | **0.6488** | 1.283 | 0.376 | 0.0095 | **0.27237** | **+0.00101** |
| **Pooled (0 & 2)** | Control: V18 Fold Table + Refiner | 25.41 | 0.8193 | 0.4621 | 0.6505 | 1.292 | 0.397 | 0.0107 | 0.27527 | — |
| | **Treatment: V18 + Anchored Corrector** | **25.39** | **0.8223** | **0.4642** | **0.6519** | 1.293 | 0.398 | 0.0108 | **0.27579** | **+0.00052** |

## 2. Depth Stratification & Error Breakdown

| Regime | Control PSNR (dB) | Treatment PSNR (dB) | $\Delta$ PSNR (dB) |
| :--- | :---: | :---: | :---: |
| **Shallow (5–30m)** | 25.51 | 25.5 | -0.01 dB |
| **Medium (35–60m)** | 26.2 | 26.18 | -0.02 dB |
| **Deep (65–100m)** | 24.73 | 24.73 | +0.00 dB |

### Per-Channel MAE & Clipping:
- **Control Per-Channel MAE:** Red=0.0512, Green=0.0418, Blue=0.0517
- **Treatment Per-Channel MAE:** Red=0.0511, Green=0.042, Blue=0.0517
- **Clipped Pixel Fraction:** Control=1.71% vs Treatment=0.994%

## 3. Decision Gate Criteria Evaluation

1. **Pooled restoration contribution $\ge +0.010$ composite points:** Measured **+0.00052** -> **FAIL**
2. **Positive delta on both folds:** Fold 0 (True), Fold 2 (True) -> **PASS**
3. **No material PSNR/SSIM regression:** PSNR (True), SSIM (True) -> **PASS**
4. **Deep water stability & bounded clipping:** Deep PSNR (True), Clipping (True) -> **PASS**
5. **Strict scene isolation & reproducibility:** 16 train / 4 held-out scenes -> **PASS**

## 4. Computational Overhead

- **Corrector Parameters:** 63,004 (246.1 KB float32)
- **Corrector Latency:** 1.59 ms per 1024x1024 image
- **Peak VRAM:** 2554.0 MB on NVIDIA GeForce RTX 5090
