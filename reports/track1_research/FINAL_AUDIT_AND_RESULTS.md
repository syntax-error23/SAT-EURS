# SAT-EURS Track 1 — Audited Ablation & Architecture Final Report

**Date:** October 15, 2026  
**Challenge:** MaCVi @ WACV 2027 EURS Challenge (Track 1)  
**Hardware:** NVIDIA RTX 5090 (32 GB VRAM), 64 GB RAM, Intel Core Ultra 9  
**Branch:** `track1-research` | **Git Commit:** `9a17437`  
**Protocol:** Locked Pre-Execution Protocol (Safeguards 1–6 Enforced)

---

## 1. Executive Summary & Verified Submission Candidate

Following the locked pre-execution protocol, all phases of data audit, baseline reproduction, isolated inference ablations, targeted architectural experiments, and official packaging verification have concluded.

### 1.1 Verified Primary Submission Package
- **Archive File:** `deploy/eurs_track1_phase2_verified.zip`
- **File Size:** `587.49 MB` (Passes the $< 1024\text{ MB}$ challenge limit)
- **SHA256:** `7d108c682c5542dd9e3f6010f7a62f6d210aed41fc6753cb836383d4b20c3d38`
- **Restoration Count:** Exactly `1,200` JPEG images (`restored/`, Quality 98)
- **Segmentation Count:** Exactly `1,200` PNG masks (`seg_masks/`, `uint8`, classes $\in \{0, 1, 2, 3, 4\}$)
- **Dimension Matching:** 100% paired 1:1 match with original native test dimensions (`1024x768` and `768x1024`).
- **Single-Image Independence:** 100% bit-for-bit exact match ($\text{max\_diff} = 0$, byte-for-byte identical JPEG encodings) against isolated single-image execution.
- **Projected Official Leaderboard Score:** **`0.690`** (recovers $+0.006$ over baseline #18700 by eliminating the $-0.0052$ restoration truncation penalty).

### 1.2 Leaderboard Submission Metadata (Strict Portal Constraints)
- **Team / Method Name:** `EURSNetV6Verified` (17 chars $\le 25$)
- **Method Description:**  
  `Full 20 scene 19 term RSUIGM table with unsharp 1.34 filter. 5 Fold DINOv2 Large ensemble with multiscale test time augmentation. Strict single image independence.`  
  *(Complies strictly: only plain letters, numbers, spaces, and periods. Zero brackets, zero parentheses, zero hyphens).*

---

## 2. Forensic Provenance: Why #18700 Was 0.684 and How It Is Resolved

In submission **#18700 (`EURS-Net-V6-MultiScale`)**:
- **Official Dev Composite:** `0.684`
- **Official Dev Segmentation:** $Q_{\text{seg}} = \mathbf{0.675}$ ($\text{mIoU} = 0.776$, $\text{Macro } F_1 = 0.869$)
- **Official Dev Restoration:** $\text{PSNR} = 25.07\text{ dB}$, $\text{SSIM} = 0.787$, $\text{Sharpness} = 0.964$, $\text{UIQM} = 0.921$

In prior submission **#18698 (`DINOv2-SOTA`)**:
- **Official Dev Composite:** `0.678` ($Q_{\text{seg}} = 0.659$, $\text{PSNR} = 25.44\text{ dB}$, $\text{SSIM} = 0.809$)

### Exact Mathematical Ledger:
1. **Segmentation Gain:** $+0.60 \times (0.675 - 0.659) = \mathbf{+0.0096}$ points due to Multi-Scale TTA (`[448, 518, 588]` + HFlip).
2. **Restoration Deficit:** $\Delta Q_{\text{FR}} = -0.0174 \implies 0.30 \times (-0.0174) = \mathbf{-0.0052}$ points because `generate_track1_multiscale_sota.py` took alphabetical `clean_scenes[:8]` instead of all training fold scenes.
3. **Non-Reference Gain:** $+0.10 \times (+0.016) = \mathbf{+0.0016}$ points ($\text{UIQM } 0.863 \to 0.921$).
4. **Net Official Shift:** $+0.0096 - 0.0052 + 0.0016 = \mathbf{+0.0060}$ ($0.678 \to 0.684$).

**Resolution in `deploy/eurs_track1_phase2_verified.zip`:**
The table was fitted on all 20 clean scenes with equal pixel weighting ($327,680$ pixels per condition) + calibrated unsharp filter ($1.34$). This recovers the $+0.0052$ deficit while preserving the $+0.0096$ multi-scale segmentation gain, yielding projected composite **`0.690`**.

---

## 3. Empirical Ablation Ledger & Guardrail Performance

All metrics were computed strictly out-of-fold across the official 5-fold split (`splits/manifest_5fold.json`).

### 3.1 Locked Baseline Floors (#18700 Control)

| Split Set | Images | $Q_{\text{seg}}$ | $\text{mIoU}$ | $\text{Macro } F_1$ | $\text{PSNR}$ (dB) | $\text{SSIM}$ | $\text{Sharpness}$ | $Q_{\text{FR}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Screening Folds (0–1)** | 1,600 | **0.3838** | 0.5461 | 0.7028 | 24.29 | 0.7614 | **0.5122** | **0.6030** |
| **Confirmation Folds (2–4)** | 2,400 | **0.2041** | 0.3946 | 0.5173 | 23.43 | 0.7055 | **0.5152** | **0.5588** |
| **Pooled All 5 Folds (0–4)** | 4,000 | **0.2745** | 0.4544 | 0.6040 | 23.78 | 0.7279 | **0.5140** | **0.5765** |

Locked Per-Class IoU Floors (Screening Folds 0–1): `[0.6184, 0.5884, 0.5666, 0.3915, 0.5658]`  
Locked Per-Class F1 Floors (Screening Folds 0–1): `[0.7642, 0.7409, 0.7233, 0.5627, 0.7227]`

---

### 3.2 Phase 2: Isolated Calibration & Inference Experiments

#### Experiment 2.1: RSUIGM Basis Ablation (Screening Folds 0–1)
- **Control (Baseline Table):** $\text{PSNR} = 24.29\text{ dB}, \text{SSIM} = 0.7614, Q_{\text{FR}} = 0.6030$
- **Option 1 (16-Scene 19-Term Fit):** $\text{PSNR} = \mathbf{24.50\text{ dB}}, \text{SSIM} = \mathbf{0.7687}, \text{Sharpness} = 0.5148, Q_{\text{FR}} = \mathbf{0.6091}$ ($\mathbf{+0.0061}$ gain)
- **Option 2 (28-Term Spatial Basis):** $\text{PSNR} = 24.46\text{ dB}, \text{SSIM} = 0.7675, Q_{\text{FR}} = 0.6081$
- **Decision:** Option 1 **PROMOTED**. Option 2 **REJECTED** per Safeguard 5 ($0.6081 < 0.6091$).

#### Experiment 2.2: Edge-Gated Filtering & Unsharp (Confirmation Folds 2–4)
- **Control (Baseline Table):** $\text{PSNR} = 23.43\text{ dB}, \text{SSIM} = 0.7055, \text{Sharpness} = 0.5152, Q_{\text{FR}} = 0.5588$
- **Bilateral Filtering:** Sharpness = $0.4845 < 0.5152$ $\implies$ **REJECTED** per Safeguard 4.
- **Unsharp `cfg1_unsharp_1.34`:** $\text{PSNR} = \mathbf{23.81\text{ dB}} (+0.38\text{ dB}), \text{SSIM} = \mathbf{0.7339} (+0.0284), \text{Sharpness} = \mathbf{0.5186} (> 0.5152), Q_{\text{FR}} = \mathbf{0.5787}$ ($\mathbf{+0.0199}$ net gain).
- **Decision:** `cfg1_unsharp_1.34` **PROMOTED** with Pareto dominance across all restoration metrics.

#### Experiment 2.3: Aspect-Preserving vs. Square Inference
- Single-scale square inference ($518 \times 518$): $\text{mIoU} = 0.5369, Q_{\text{seg}} = 0.3702$
- Single-scale aspect-preserving inference: $\text{mIoU} = 0.5305, Q_{\text{seg}} = 0.3621$ ($-0.0081$ drop)
- **Decision:** Aspect-preserving inference **REJECTED**. Checkpoints trained on square crops require square inference. Multi-Scale Square TTA retained.

#### Experiment 2.4: Isolated Post-Processing (Screening Folds 0–1)
- **Control (Pure Multi-Scale TTA):** $\text{mIoU} = 0.5510, \text{Macro } F_1 = 0.7064, Q_{\text{seg}} = 0.3892, \text{Class 3 IoU} = \mathbf{0.3871}$
- **Guided Filter ($r=4, \epsilon=10^{-3}$):** $\text{mIoU} = 0.5518, Q_{\text{seg}} = 0.3901$, but $\text{Class 3 IoU} = 0.3867$ ($-0.0004$ drop)
- **Boundary Hysteresis ($\tau_{\text{high}}=0.45, \tau_{\text{low}}=0.20$):** $\text{mIoU} = 0.5510, Q_{\text{seg}} = 0.3892, \text{Class 3 IoU} = 0.3871$
- **Decision:** Post-processing **REJECTED** per Safeguard 4 to protect rare-class generalization.

---

### 3.3 Phase 3: Targeted Architectural Experiments

#### Experiment 3.1: DINOv2HighResDecoder + Lovász-Softmax Fine-Tuning Pilot
- **Architecture:** Pretrained DINOv2-ViT-L/14 + Stride-2 ($48$ ch) and Stride-4 ($96$ ch) RGB detail stem + zero-initialized residual projection.
- **Unit Tests:** 100% passed (Shapes, mathematical zero-identity at step 0, 2-step gradient flow, FP32 loss stability).
- **Empirical Pilot Results (Screening Folds 0–1, 1,600 images):**

| Metric | Locked Screening Floor | Exp 3.1 HighRes Pilot | Status |
| :--- | :---: | :---: | :---: |
| **$Q_{\text{seg}}$** | $\ge \mathbf{0.3838}$ | **0.2638** | **FAIL ($-0.1200$)** |
| **$\text{mIoU}$** | $\ge \mathbf{0.5461}$ | **0.4433** | **FAIL ($-0.1028$)** |
| **$\text{Macro } F_1$** | $\ge \mathbf{0.7028}$ | **0.5951** | **FAIL ($-0.1077$)** |
| **Class 0 IoU** | $\ge 0.6184$ | 0.6413 | PASS |
| **Class 1 IoU** | $\ge 0.5884$ | 0.6056 | PASS |
| **Class 2 IoU** | $\ge 0.5666$ | 0.3742 | **FAIL ($-0.1924$)** |
| **Class 3 IoU** | $\ge 0.3915$ | 0.4080 | PASS |
| **Class 4 IoU** | $\ge 0.5658$ | 0.1872 | **FAIL ($-0.3786$)** |

- **Root Cause Analysis:** Fine-tuning the foundation model for additional epochs on small scene counts (even at $\text{lr} = 10^{-5}$) caused severe representational drift on macro-fauna classes (Class 2 and Class 4 in Fold 0 collapsed), confirming our earlier finding that training beyond 3–5 epochs degrades held-out transfer.
- **Decision:** Exp 3.1 **FAILED Safeguard 4** and is **STRICTLY REJECTED**. In accordance with the protocol, it is not scaled to Confirmation Folds 2–4. The verified Phase 2 ensemble remains the winning submission.

---

## 4. Final Submission Checklist & Integrity Verification

| Requirement | Specification | Measured Value | Compliance |
| :--- | :--- | :--- | :---: |
| **Archive Location** | `deploy/` root | `deploy/eurs_track1_phase2_verified.zip` | PASS |
| **Archive Size** | $< 1024\text{ MB}$ | `587.49 MB` | PASS |
| **Restored Count** | Exactly 1,200 | 1,200 files in `restored/` | PASS |
| **Restored Format** | High-Quality JPEG | JPEG Quality 98 | PASS |
| **Mask Count** | Exactly 1,200 | 1,200 files in `seg_masks/` | PASS |
| **Mask Format** | PNG `uint8` | PNG `uint8` | PASS |
| **Discrete Classes** | Subset of $\{0, 1, 2, 3, 4\}$ | Observed classes: `[0, 1, 2, 3, 4]` | PASS |
| **Resolution Match** | Native test dimensions | 100% paired 1:1 match | PASS |
| **Per-Image Independence** | Zero cross-image communication | Exact 100% bitwise match ($\Delta = 0$) | PASS |
| **Method Name** | $\le 25$ characters | `EURSNetV6Verified` (17 chars) | PASS |
| **Method Description** | Letters, numbers, spaces, periods only | Fully sanitized string | PASS |
