# SAT-EURS Track 1 — V21 ConditionedBilateralAffineNet Pilot Benchmark Report

**Date:** 2026-10-18 18:02:05 UTC  
**Baseline SHA-256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52` (100% Verified Intact)  
**Pilot Scope:** Limited Two-Fold Matched Benchmark (Folds 0 & 2, 1,600 Held-Out Images)  
**Final Pre-Registered Screening Verdict:** **NO-GO (CEILING / DEFICIENCY IDENTIFIED)**  

## 1. Matched Comparison Table (Control vs. Treatment)

| Evaluation Split | Model / Pipeline | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | UIQM | UCIQE | CCF | $0.30 Q_{\text{FR}} + 0.10 Q_{\text{NR}}$ | $\Delta$ Composite Contribution |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | Control: V18 Fold Table + Refiner | 25.71 | 0.8184 | 0.4363 | 0.6533 | 1.309 | 0.423 | 0.0122 | 0.27918 | — |
| | **Treatment: BilateralAffineNet** | **23.06** | **0.7847** | **0.3287** | **0.569** | 1.138 | 0.379 | 0.0114 | **0.24417** | **-0.03501** |
| **Fold 2** | Control: V18 Fold Table + Refiner | 25.11 | 0.8201 | 0.4878 | 0.6477 | 1.275 | 0.372 | 0.0093 | 0.27136 | — |
| | **Treatment: BilateralAffineNet** | **22.15** | **0.7906** | **0.3366** | **0.5568** | 1.115 | 0.307 | 0.0071 | **0.23254** | **-0.03882** |
| **Pooled (Folds 0 & 2)** | Control: V18 Fold Table + Refiner | 25.41 | 0.8193 | 0.4621 | 0.6505 | 1.292 | 0.397 | 0.0107 | 0.27527 | — |
| | **Treatment: BilateralAffineNet** | **22.6** | **0.7876** | **0.3327** | **0.5629** | 1.127 | 0.343 | 0.0093 | **0.23835** | **-0.03692** |

## 2. Pre-Registered Screening Criteria Evaluation

1. **Pooled restoration contribution $\ge +0.010$ composite points:** Measured **-0.03692** -> **FAIL**
2. **Positive delta on both folds:** Fold 0 (False), Fold 2 (False) -> **FAIL**
3. **No material PSNR/SSIM regression:** Fold 0 PSNR 23.06 vs 25.71, Fold 2 PSNR 22.15 vs 25.11 -> **FAIL**
4. **Strict scene-level isolation:** 16 training scenes, 4 completely held-out scenes per fold -> **PASS**
5. **Submission inference path compatibility:** Direct per-image execution using filename metadata (`wt`, `depth`) -> **PASS**

## 3. Computational Footprint & Latency

- **Parameters:** 272,132 (1.04 MB float32)
- **Steady-State Latency:** 1.01 ms per 1024x1024 image (990.9 FPS)
- **Peak VRAM:** 1334.3 MB on NVIDIA GeForce RTX 5090
- **External Dependencies:** Zero (Native PyTorch, CUDA, OpenCV)

