# SAT-EURS Track 2 — Sprint Phase 4 Report: Stride-2 Architecture Branch Experiment

**Date:** 2026-10-24  
**Author:** Antigravity Senior Computer Vision Research & Optimization Team  
**Sprint Status:** PHASE 4 COMPLETE — STRIDE-2 BRANCH RIGOROUSLY EVALUATED & DISCARDED  
**Hardware Limit Check:** PASSED (Params: 0.4602 M <= 1.0 M, FLOPs: 1.9107 GFLOPs <= 12.0 GFLOPs)  
**Evaluation Protocol:** Tested on identical Fold 0 and Fold 1 held-out validation scenes (320 images).

---

## 1. Experimental Comparison

| Metric | Phase 3 Champion (`SAT-EURS-LiteV3`) | Phase 4 Stride-2 Branch (`SATEURSLiteV3Stride2`) | Delta | Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Parameters** | **0.4795 M** | 0.4602 M | -0.0193 M | Compliant |
| **GFLOPs** | **1.6280** | 1.9107 | +0.2827 | Compliant |
| **mIoU** | **0.4168** | 0.1073 | -0.3095 | Severe Degradation |
| **Macro-F1** | **0.5460** | 0.1691 | -0.3769 | Severe Degradation |
| **Coverage Penalty ($P_{\text{cov}}$)** | **1.0** | 0.0 | -1.0 | **Coverage Failure** |
| **$Q_{\text{seg}}$** | **0.2276** | **0.0000** | **-0.2276** | **Catastrophic Drop** |
| **PSNR (dB)** | **19.80 dB** | 8.32 dB | -11.48 dB | Failure |
| **SSIM** | **0.6867** | 0.0847 | -0.6020 | Failure |
| **Composite Score ($S_{\text{comp}}$)** | **0.3379** | 0.0857 | -0.2522 | Rejected |

---

## 2. Failure Analysis & Architectural Diagnosis

1. **Gradient Dilution at Fine Stride:**
   Injecting uncurated Stride-2 shallow features (Layer 1, $192 \times 384$) directly into the decoder flooded the semantic representations with high-frequency noise and underwater backscatter before high-level receptive fields could establish class context.
2. **Backpropagation Instability:**
   The gradient from the high-resolution loss dominated early layers, destabilizing the optical routing representations and causing complete restoration collapse (PSNR dropped from 19.80 dB to 8.32 dB).
3. **Mandate Compliance:**
   As instructed by Phase 4 (*"Discard this branch if it does not yield genuine held-out gains"*), this branch is **permanently discarded**.

---

## 3. Champion Architecture Locked

The canonical **`SAT-EURS-LiteV3`** architecture (Progressive Strides 4, 8, 16, 32 + Edge-Gated Sharpening + Phase 3 Class Weighting) remains the single locked, verified architecture for production training in Phase 5.
