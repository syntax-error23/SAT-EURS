# SAT-EURS Track 1 — V28 Scoring Contract Reconciliation & Baseline Audit

**Campaign:** SAT-EURS V28 Corrected Phase 2 Score Recovery Campaign  
**Date:** October 24, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** MaCVi @ WACV 2027 EURS Track 1 Testing Phase  
**Production Baseline:** V27 Official Submission #19096, Official Composite Score: **0.633**

---

## 1. Executive Summary & Scoring Contract Reconciliation

The Phase 1 diagnostic report contained an estimated decomposition based on provisional local metrics ($Q_{\text{seg}} \approx 0.438$, $Q_{\text{FR}} \approx 0.665$, $Q_{\text{NR}} \approx 0.600$).

This document formally reconciles the baseline using the **exact official leaderboard numbers** reported by the challenge evaluation server for V27 (Submission #19096):

- **Official Composite ($S$):** **0.633**
- **Official Segmentation Quality ($Q_{\text{seg}}$):** **0.568**
- **Official Mean IoU ($\text{mIoU}$):** **0.700**
- **Official Macro $F_1$ Score:** **0.810**
- **Official PSNR:** **25.48 dB**
- **Official SSIM:** **0.829**
- **Official Sharpness Ratio:** **0.973**

---

## 2. Mathematical Score Breakdown & Implied $Q_{\text{NR}}$

### 2.1 Challenge Scoring Formula
$$S = 0.60 \times Q_{\text{seg}} + 0.30 \times Q_{\text{FR}} + 0.10 \times Q_{\text{NR}}$$

### 2.2 Full-Reference Restoration ($Q_{\text{FR}}$)
Normalized according to the published Track 1 window of $[15.0, 35.0]$ dB:
$$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0}, 0.0, 1.0\right) = \frac{25.48 - 15.0}{20.0} = \mathbf{0.5240}$$

$$Q_{\text{FR}} = 0.45 \times \text{PSNR}_{\text{norm}} + 0.45 \times \text{SSIM} + 0.10 \times \text{Sharpness}$$
$$Q_{\text{FR}} = 0.45(0.5240) + 0.45(0.8290) + 0.10(0.9730) = 0.23580 + 0.37305 + 0.09730 = \mathbf{0.70615}$$

### 2.3 Segmentation Quality ($Q_{\text{seg}}$)
$$\text{Product: } \text{mIoU} \times F_1 = 0.700 \times 0.810 = 0.5670 \approx \mathbf{0.568} \text{ (matching leaderboard)}$$

### 2.4 Reconciliation of Implied Non-Reference Quality ($Q_{\text{NR}}$)
$$S = 0.60(0.5680) + 0.30(0.70615) + 0.10(Q_{\text{NR}}) = 0.633$$
$$0.34080 + 0.211845 + 0.10(Q_{\text{NR}}) = 0.633$$
$$0.552645 + 0.10(Q_{\text{NR}}) = 0.633$$
$$0.10(Q_{\text{NR}}) = 0.633 - 0.552645 = 0.080355$$
$$Q_{\text{NR}} \approx \mathbf{0.80355}$$

The implied server-evaluated non-reference quality metric (UIQM, UCIQE, and CCF combination) is approximately **0.804**, representing a healthy baseline on raw image statistics.

---

## 3. Corrected Target Analysis to Reach 0.720+

| Component | Official Weight | Baseline V27 Value | Points Contributed (V27) | Target for $S = 0.720$ | Points Needed ($S = 0.720$) | Delta Needed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Segmentation ($Q_{\text{seg}}$)** | **60%** | **0.5680** | **0.34080 pts** | **0.7130** | **0.42780 pts** | **+0.1450 pts** |
| **Restoration ($Q_{\text{FR}}$)** | **30%** | **0.7062** | **0.21185 pts** | **0.7062** | **0.21185 pts** | Stable baseline |
| **Non-Reference ($Q_{\text{NR}}$)** | **10%** | **0.8036** | **0.08035 pts** | **0.8036** | **0.08035 pts** | Stable baseline |
| **Official Composite ($S$)** | **100%** | **0.6330** | **0.63300 pts** | **0.7200** | **0.72000 pts** | **+0.0870 pts** |

### Critical Correction of Phase 1 Promotion Gate:
- **INVALID GATE STRICKEN:** In `PHASE1_DECISION_REPORT.md`, Gate 5 mistakenly stated that $Q_{\text{seg}} \ge 0.550$ would imply an official composite $S \ge 0.720$. This was an arithmetic artifact of assuming $Q_{\text{seg}} \approx 0.438$. In reality, V27 already achieved $Q_{\text{seg}} = 0.568$, yet its composite was $0.633$.
- **CORRECTED PROMOTION REQUIREMENT:** To reach $S \ge 0.720$, held-out segmentation quality $Q_{\text{seg}}$ must reach **$\ge 0.7130$** (requiring $\text{mIoU} \ge 0.810$ and $\text{Macro } F_1 \ge 0.880$ under full coverage).
- Any candidate composite estimate must strictly distinguish measured local validation metrics (computed on held-out EURS scenes) from the hidden-test leaderboard scores.
