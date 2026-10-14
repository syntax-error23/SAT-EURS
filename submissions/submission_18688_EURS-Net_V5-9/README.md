# Submission #18688 — EURS-Net V5-9

**Submission ID:** `#18688`  
**Method Name:** `EURS-Net V5-9`  
**Submission Date:** 14-Oct-2026  
**Archive:** `eurs_track1_submission_18688_v5_9.zip`  
**Generating Script:** [`deploy/generate_verified_track1_submission.py`](file:///c:/Projects/SAT-EURS/deploy/generate_verified_track1_submission.py) / [`deploy/generate_compliant_track1_submission.py`](file:///c:/Projects/SAT-EURS/deploy/generate_compliant_track1_submission.py)  

---

## 1. Official Leaderboard Score

| Metric | Measured Score | Official Weight in Composite | Delta vs. Prior (#18672) |
|:---|:---:|:---:|:---:|
| **Composite Score** | **0.449** | 100% | **-0.027 (Series Low)** |
| **$Q_{\text{seg}}$** | **0.300** | 60% | **-0.024 (Series Low)** |
| **mIoU** | **0.499** | — | **-0.011 (Fell below 0.50)** |
| **Macro $F_1$** | **0.602** | — | **-0.033 (Severe Drop)** |
| **Coverage Penalty** | **1.000** | — | 1.000 |
| **PSNR** | **24.01 dB** | 13.5% | **-1.48 dB** |
| **SSIM** | **0.812** | 13.5% | +0.002 |
| **Sharpness Ratio** | **0.852** | 3.0% | -0.041 |
| **UIQM** | **0.810** | 4.0% | -0.011 |
| **UCIQE** | **0.563** | 4.0% | -0.003 |
| **CCF** | **0.042** | 2.0% | -0.008 |

---

## 2. Architecture & Design

### Restoration Pipeline (Independent-Image Compliant)
- **Eliminated Cross-Image Consensus**: Every image processed strictly from its own individual degraded input, resolving the competition integrity violation.
- **Physical Inversion + Edge-Preserving Sharpening**:
  - Closed-form GPU RSUIGM polynomial lookup table $W_{(\text{wt}, \text{depth})}$.
  - Single-image bilateral unsharp sharpening on GPU to preserve sharpness without cross-image leakage.

### Segmentation Ensemble
- **4-Way Normalized Ensemble**:
  - **25% SegFormer MiT-B3** (Vision Transformer, 45.2M params)
  - **25% DeepLabV3-ResNet101** (Fixed ImageNet normalization, 60.9M params)
  - **25% EURS-Net V4 Full20** (0.72M params)
  - **25% EURS-Net V5 Attention** (0.78M params)
- **Calibrated Logit Gating**:
  - Starfish boost $+2.0$ (designed to rescue #18668's recall collapse)
  - Elasmobranch correction $-2.5$

---

## 3. What Was Promised vs. What Was Actually Achieved

- **Promised / Projected Score**:
  - Composite Score: $\ge \mathbf{0.51 - 0.55}$
  - Expected that fixing DeepLabV3 normalization and enforcing compliant independent image restoration would produce a clean, compliant $0.51+$ composite.
- **Actual Leaderboard Score**:
  - Composite Score: **0.449** (Dropped to the lowest in the entire experimental series)
  - $Q_{\text{seg}}$: **0.300** (First time falling to 0.30)
  - mIoU: **0.499** (First time falling below 0.50)
  - PSNR: **24.01 dB** (Dropped from 25.49 dB)

---

## 4. Why the Composite and $Q_{\text{seg}}$ Were Less Than Promised (Root Causes)

1. **Loss of Consensus Smoothing on Restoration**:
   - The depth-weighted scene consensus in #18672 had artificially boosted PSNR by $+1.48\text{ dB}$ (to $25.49\text{ dB}$) by averaging out noise and chromatic aberration across all 200 variations.
   - Processing each image independently exposed the raw degraded inputs at extreme depths ($> 70\text{m}$) where physical polynomial inversion suffers from noise amplification.
   - PSNR fell to **$24.01\text{ dB}$**, dropping the restoration component score.
2. **The 4-Way Ensemble Domain Collapse**:
   - Equal 25% averaging across 4 disparate models diluted high-confidence predictions:
     * SegFormer, DeepLabV3, V4, and V5 had different boundary errors on unseen Dev scenes.
     * When averaged equally, ambiguous boundary logits fell below the decision threshold, causing pixels to default to Background.
   - Macro $F_1$ collapsed from $0.635 \to \mathbf{0.602}$ ($-0.033$), and mIoU dropped below $0.50$ ($0.499$).
   - $Q_{\text{seg}} = 0.499 \times 0.602 = \mathbf{0.300}$.
3. **The Metadata Fragility**:
   - The RSUIGM table lookup relied on parsing `_wt_depth` from filenames. While valid on Dev split, any variance in water parameter mapping across depths caused misaligned physical matrices.

---

## 5. What Plan Was Made to Fix It (The Pivot to Clean Leakage-Free Foundations)

- **The Realization**:
  - Continuing to tune ensembles of models trained on all 20 scenes was futile because in-sample evaluation hid real-world generalization failures.
  - Scene-name heuristics and filename parsing were brittle crutches that collapsed under strict evaluation.
- **The Recovery Plan**:
  1. Halt all unverified submissions and leaderboard uploads.
  2. Establish a **strict 16/4 scene-disjoint split** with zero validation scenes seen during training.
  3. Train a clean, single-stream SegFormer baseline ([`checkpoints/exp_baseline_segformer/best_model.pt`](file:///c:/Projects/SAT-EURS/checkpoints/exp_baseline_segformer/best_model.pt)) to establish true held-out out-of-distribution metrics ($Q_{\text{seg}} = 0.2016$).
  4. Investigate the semantic failure on rare classes (Class 4 / Seahorse) systematically using controlled loss experiments (Focal+Dice and Mild Loss) before making any further leaderboard uploads.
