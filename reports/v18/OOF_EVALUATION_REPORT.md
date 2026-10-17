# SAT-EURS Track 1 — V18 5-Fold Out-Of-Fold Evaluation Report

**Date:** 2026-10-17 14:59:12 UTC  
**Evaluated Folds:** [0, 1, 2, 3, 4] (Covering all 20 EURS scenes)  

## 1. Overall Track 1 Challenge Metrics

| Metric | Formula / Source | Measured OOF Value |
| :--- | :--- | :---: |
| **$S_{\text{exact\_90pct}}$** | $0.60 Q_{\text{seg}} + 0.30 Q_{\text{FR}}$ (Verified Local Score) | **0.44469** |
| **$Q_{\text{seg}}$ (Present)** | $\text{Macro F1}_{\text{present}} \times \text{mIoU}_{\text{present}} \times \text{coverage}$ | **0.43645** |
| **$Q_{\text{seg}}$ (5-Class Zero)** | $\text{Macro F1}_{5\text{cls}} \times \text{mIoU}_{5\text{cls}} \times \text{coverage}$ | **0.43645** |
| **mIoU (Present Classes)** | Mean over classes with $N_{\text{gt}} > 0$ (5/5) | **0.59441** |
| **mIoU (5-Class Conservative)** | 5-class mean with absent classes set to 0.0 | **0.59441** |
| **Macro F1 (Present Classes)** | Mean over classes with $N_{\text{gt}} > 0$ (5/5) | 0.73425 |
| **Macro F1 (5-Class Conservative)** | 5-class mean with absent classes set to 0.0 | 0.73425 |
| **Coverage Penalty** | $\prod_{c \in C_{\text{present}}} \mathbb{I}(\text{TP}_c > 0)$ (Provisional contract) | 1.0 (UNVERIFIED_PROVISIONAL) |
| **$Q_{\text{FR}}$** | $0.45 \text{PSNR}_{\text{norm}} + 0.45 \text{SSIM} + 0.10 \text{sharp}$ | **0.60942** |
| **PSNR (dB)** | Dataset Mean Peak Signal-to-Noise Ratio | 25.04 dB |
| **SSIM** | Mean Structural Similarity Index | 0.77614 |
| **Sharpness Ratio** | Ratio of high-frequency energy | 0.34198 |
| **$Q_{\text{NR}}$ Status** | Server-side UIQM, UCIQE, CCF | UNAVAILABLE_LOCALLY |
| **$S_{\text{composite}}$ Range** | Sensitivity bracket for $Q_{\text{NR}} \in [0.40, 0.80]$ | **[0.48469, 0.52469]** |

> [!NOTE] Official $Q_{\text{NR}}$ is evaluated on the competition server using UIQM, UCIQE, and CCF. Local offline parity cannot be computed without server-side normalization constants. $S_{\text{exact\_90pct}}$ isolates verified local metrics without unverified assumptions.

## 2. Per-Class Segmentation Performance & Pixel Counts

| Class ID | Class Name | Status | GT Pixels | Pred Pixels | TP Pixels | FP Pixels | FN Pixels | IoU | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Background | PRESENT | 2,031,879,200 | 2,122,909,836 | 1,781,227,677 | 341,682,159 | 250,651,523 | 0.7504 | 0.8574 |
| 1 | Coral | PRESENT | 1,759,327,400 | 1,639,672,635 | 1,421,073,498 | 218,599,137 | 338,253,902 | 0.7185 | 0.8362 |
| 2 | Fish | PRESENT | 248,840,000 | 316,053,101 | 213,658,230 | 102,394,871 | 35,181,770 | 0.6083 | 0.7565 |
| 3 | Elasmobranchii | PRESENT | 32,287,600 | 39,194,480 | 25,503,101 | 13,691,379 | 6,784,499 | 0.5547 | 0.7136 |
| 4 | Macro-Fauna | PRESENT | 121,969,800 | 76,473,948 | 50,369,585 | 26,104,363 | 71,600,215 | 0.3402 | 0.5077 |
