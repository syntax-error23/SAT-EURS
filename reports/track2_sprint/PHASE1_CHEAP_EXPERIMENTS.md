# SAT-EURS Track 2 — Sprint Phase 1 Report: Cheap Experiments Before Retraining

**Date:** 2026-10-24  
**Author:** Antigravity Senior Computer Vision Research & Optimization Team  
**Sprint Status:** GATE 1 EVALUATION COMPLETE  
**Evaluated Data:** Held-out validation sets across 5 scene folds (800 validation images)

---

## 1. Restoration Sharpening Filter Sweep

We conducted an empirical comparison of four sharpening filter variants across the held-out validation sets of Fold 0 and Fold 1. Because the restored image feeds directly into the downstream segmentation backbone (`MobileNetV3`), both restoration and downstream segmentation were evaluated end-to-end.

### 1.1 Empirical Comparison Table

| Variant | PSNR (dB) | $\text{PSNR}_{\text{norm}}$ | SSIM | Sharpness (vs Clean) | $Q_{\text{FR}}$ | mIoU | Macro-F1 | $P_{\text{cov}}$ | $Q_{\text{seg}}$ | $S_{\text{composite}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Existing Global Unsharp ($\gamma = 1.20$)** | 18.92 | 0.1960 | 0.6841 | 0.0786 | 0.4039 | 0.3955 | 0.5196 | 1.0 | 0.2055 | 0.3188 |
| **2. No Unsharp Filtering ($\gamma = 1.00$)** | 18.86 | 0.1930 | 0.6779 | 0.0611 | 0.3980 | 0.3979 | 0.5230 | 1.0 | 0.2081 | 0.3186 |
| **3. Reduced Global Sharpening ($\gamma = 1.10$)** | 18.89 | 0.1946 | 0.6813 | 0.0695 | 0.4011 | 0.3968 | 0.5215 | 1.0 | 0.2069 | 0.3188 |
| **4. Edge-Gated Sharpening ($\text{base} = 0.45, \tau = 0.08$)** | **18.95** | **0.1977** | **0.6866** | **0.0977** | **0.4077** | **0.4001** | **0.5241** | **1.0** | **0.2097** | **0.3224** |

### 1.2 Analysis of Sharpening Variants

1. **Edge-Gated Sharpening Wins Across All Metrics:**
   - **Restoration Quality:** Achieves the highest PSNR (18.95 dB), highest SSIM (0.6866), and highest Sharpness (0.0977, a **+24.3% relative improvement** over global unsharp).
   - **Downstream Segmentation:** mIoU increases from 0.3955 to 0.4001; Macro-F1 increases from 0.5196 to 0.5241; $Q_{\text{seg}}$ increases from 0.2055 to 0.2097.
   - **Class 3 (Elasmobranch) Boost:** Class 3 IoU increases from 0.2425 to 0.2629 (+0.0204); Class 3 F1 increases from 0.3903 to 0.4163 (+0.0260).
   - **Why It Works:** Global unsharp filtering amplifies noise and turbidity haze in open-water regions, causing false background boundaries. Edge-gating restricts high-frequency sharpening strictly to pixels with high gradient magnitude (coral edges, shark contours, and fish fins), leaving flat open water clean.

---

## 2. Post-Hoc Per-Class Logit Offset Calibration

We conducted rigorous cross-fold calibration across all 5 folds (800 validation images). For each fold $k \in \{0, 1, 2, 3, 4\}$, the offset vector $\beta = [0, \beta_1, \beta_2, \beta_3, \beta_4]$ was optimized on the remaining 4 folds to maximize official $Q_{\text{seg}}$ and then tested on the strictly held-out 5th fold.

### 2.1 Empirical Cross-Fold Results

| Configuration | Offsets $\beta = [\beta_0, \beta_1, \beta_2, \beta_3, \beta_4]$ | mIoU | Macro-F1 | $P_{\text{cov}}$ | $Q_{\text{seg}}$ | $\Delta Q_{\text{seg}}$ vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Uncalibrated)** | $[0.0, 0.0, 0.0, 0.0, 0.0]$ | **0.3574** | **0.4519** | 1.0 | **0.1615** | Control |
| **Argon Proposed Offsets** | $[0.0, 0.0, 0.0, 0.10, 0.25]$ | 0.3580 | 0.4532 | 1.0 | 0.1623 | $+0.0008$ (negligible) |
| **Cross-Fold Out-of-Sample Calibrated** | Rotation-specific $\beta^{(-k)}$ | 0.3482 | 0.4390 | 1.0 | 0.1528 | **$-0.0087$ (DEGRADATION)** |

### 2.2 Per-Class Breakdown: Baseline vs. Cross-Fold Calibrated

| Class ID & Name | Baseline IoU | Calibrated IoU | Baseline F1 | Calibrated F1 | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **0: Background** | 0.7082 | 0.7122 | 0.8292 | 0.8319 | Minor increase |
| **1: Coral** | 0.6882 | 0.6832 | 0.8153 | 0.8118 | Slight drop |
| **2: Fish** | 0.3262 | 0.2970 | 0.4919 | 0.4580 | **Drop (-0.0292 IoU)** |
| **3: Elasmobranch** | 0.0554 | 0.0415 | 0.1049 | 0.0796 | **Drop (-0.0139 IoU)** |
| **4: Other** | 0.0092 | 0.0068 | 0.0183 | 0.0136 | **Drop (-0.0024 IoU)** |

### 2.3 Diagnostic Finding on Logit Calibration

When offsets are optimized in-sample on 4 folds, the search boosts rare-class biases (e.g. $+0.50$ to $+1.75$) to capture true positives in the known training scenes. However, when applied to a genuinely unseen scene with different water turbidity and lighting, these positive offsets generate massive false-positive noise, collapsing precision and causing a net **$-0.0087$ drop in official out-of-fold $Q_{\text{seg}}$**.

---

## 3. Gate 1 Decisions

1. **Edge-Gated Sharpening ($\text{base} = 0.45, \tau = 0.08$):**
   - **DECISION: RETAINED & ADOPTED.**
   - Verified held-out improvement across PSNR (+0.03 dB), SSIM (+0.0025), Sharpness (+24.3%), $Q_{\text{FR}}$ (+0.0038), $Q_{\text{seg}}$ (+0.0042), and Composite ($0.3188 \to 0.3224$). Zero coverage penalty.

2. **Post-Hoc Per-Class Logit Offsets:**
   - **DECISION: REJECTED.**
   - Fails Gate 1: Strict cross-fold evaluation proves that offset tuning causes out-of-sample overconfidence and degrades out-of-fold $Q_{\text{seg}}$ from 0.1615 to 0.1528. Production offsets remain $\beta = \mathbf{0}$.

**GATE 1 STATUS: PASSED.**
