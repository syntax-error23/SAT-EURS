# SAT-EURS Track 1 — V7 Architectural Breakthrough Experiment Ledger

**Date:** October 15, 2026  
**Auditor:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)  
**Hardware:** NVIDIA RTX 5090 (32 GB VRAM)  
**Current Best Leaderboard Result:** Submission #18713 ($S = 0.689$, $Q_{\text{seg}} = 0.676$, $\text{PSNR} = 25.41\text{ dB}$, $\text{SSIM} = 0.816$, $\text{Sharpness} = 0.960$, $\text{UIQM} = 0.844$)

---

## 1. V7 Master Experiment Ledger (Official Metric Scale)

The official Track 1 Composite Formula is:
$$S = 0.60 \, Q_{\text{seg}} + 0.135 \, \text{PSNR}_{\text{norm}} + 0.135 \, \text{SSIM} + 0.030 \, \text{Sharpness} + 0.10 \, Q_{\text{NR}}$$
where $Q_{\text{FR}} = 0.45 \, \text{PSNR}_{\text{norm}} + 0.45 \, \text{SSIM} + 0.10 \, \text{Sharpness}$ and $\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{20.0}, 0.0, 1.0\right)$.

| Stage / Model Candidate | Folds | $Q_{\text{seg}}$ | $\text{mIoU}$ | $\text{Macro } F_1$ | $\text{PSNR}$ (dB) | $\text{SSIM}$ | $\text{Sharpness}$ | $\text{UIQM}$ | $\text{UCIQE}$ | $Q_{\text{FR}}$ | Composite Score | Promotion Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#18700 Baseline (`EURS-Net-V6-MultiScale`)** | Official | 0.675 | 0.776 | 0.869 | 25.07 | 0.787 | 0.964 | 0.921 | 0.567 | 0.677 | **0.684** | Prior SOTA Baseline |
| **#18713 Baseline (`EURSNetV6Verified`)** | Official | 0.676 | 0.777 | 0.870 | 25.41 | 0.816 | 0.960 | 0.844 | 0.559 | 0.697 | **0.689** | Current SOTA Control |
| **Stage 1: Forensic Diagnostic Audit** | 5-Fold OOF | 0.2745* | 0.4544* | 0.6040* | 23.78* | 0.7279* | 0.5140* | 1.074 | 0.380 | 0.5765* | — | Bottleneck Identified: 15px boundary strip causes 68.4% of IoU loss; UIQM dropped due to 20-scene MSE linear shrinkage. |
| **Stage 2: PhysicsGuidedNAFNet + Chroma** | Folds 0–1 | *Training* | — | — | *Training* | *Training* | *Training* | *Training* | *Training* | *Training* | *Pending* | Running (`task-992`) on RTX 5090 |
| **Stage 3: DINOv2HighResDPT (700x700)** | Folds 0–1 | *Queued* | — | — | — | — | — | — | — | — | *Pending* | Script ready (`train_v7_stage3_dpt.py`) |
| **Stage 4: ConvNeXt-Base FPN (640x640)** | Folds 0–1 | *Queued* | — | — | — | — | — | — | — | — | *Pending* | Script ready (`train_v7_stage4_convnext.py`) |
| **Stage 5: 10-Model Ensemble + Calibrated b** | All 5 Folds | *Queued* | — | — | — | — | — | — | — | — | **0.78–0.80+ (Target)** | Final Submission Packaging Gate |

*\*Note: Unnormalized raw local OOF metrics on 4,000 un-augmented full-scene validation images. Dev leaderboard metrics on the server correspond to 0.676 segmentation and 25.41 dB restoration.*

---

## 2. Stage-by-Stage Implementation & Verification Matrix

### Stage 1: Bottleneck Diagnosis
- **Artifact:** [`reports/track1_research/V6_BOTTLENECK_DIAGNOSIS.md`](file:///c:/Projects/SAT-EURS/reports/track1_research/V6_BOTTLENECK_DIAGNOSIS.md)
- **Key Findings:**
  1. Class 3 (Shark/Ray) recall is 0.2907 (IoU: 0.2011).
  2. Class 4 (Macro-fauna) IoU is 0.2982 due to patch-14 tokenization blurring thin appendages.
  3. 15px boundary band accounts for 8.1% of pixels but drops mIoU by **0.1708** relative to interiors (0.5471 vs. 0.3764).
  4. 20-scene linear ridge regression muted chromatic variance and local contrast, causing UIQM to drop from 0.921 to 0.844.

### Stage 2: Single-Image Neural Restoration (`PhysicsGuidedNAFNet`)
- **Module:** [`models/physics_guided_nafnet.py`](file:///c:/Projects/SAT-EURS/models/physics_guided_nafnet.py) (8-channel single-image input with bounded residual $\pm 0.25$, anchored at RSUIGM prior).
- **Loss:** [`losses/restoration_v7_loss.py`](file:///c:/Projects/SAT-EURS/losses/restoration_v7_loss.py) ($\mathcal{L}_{\text{Charbonnier}} + 0.40 \mathcal{L}_{\text{SSIM}} + 0.20 \mathcal{L}_{\text{Grad}} + 0.15 \mathcal{L}_{\text{ChromaContrast}}$).
- **Unit Tests:** [`tests/test_physics_guided_nafnet.py`](file:///c:/Projects/SAT-EURS/tests/test_physics_guided_nafnet.py) — **100% Passed**.
- **Execution Script:** [`tools/train_v7_stage2_restorer.py`](file:///c:/Projects/SAT-EURS/tools/train_v7_stage2_restorer.py). Currently training under task `task-992`.

### Stage 3: High-Resolution DPT Segmentation (`DINOv2HighResDPT`)
- **Module:** [`models/dinov2_highres_dpt.py`](file:///c:/Projects/SAT-EURS/models/dinov2_highres_dpt.py) (Early blocks 0..15 frozen; blocks 16..23 fine-tuned; Stride-2 & Stride-4 detail stems + zero-initialized residual detail head).
- **Loss:** [`losses/segmentation_v7_loss.py`](file:///c:/Projects/SAT-EURS/losses/segmentation_v7_loss.py) ($0.30 \mathcal{L}_{\text{WCE}} + 0.35 \mathcal{L}_{\text{Lovász}} + 0.25 \mathcal{L}_{\text{SoftMacroF1}} + 0.10 \mathcal{L}_{\text{Boundary}}$).
- **Unit Tests:** [`tests/test_dinov2_highres_dpt.py`](file:///c:/Projects/SAT-EURS/tests/test_dinov2_highres_dpt.py) and [`tests/test_segmentation_v7_loss.py`](file:///c:/Projects/SAT-EURS/tests/test_segmentation_v7_loss.py) — **100% Passed**.
- **Execution Script:** [`tools/train_v7_stage3_dpt.py`](file:///c:/Projects/SAT-EURS/tools/train_v7_stage3_dpt.py).

### Stage 4: Complementary Stride-4 Hierarchical Backbone (`ConvNeXt-Base FPN`)
- **Module:** [`models/convnext_fpn.py`](file:///c:/Projects/SAT-EURS/models/convnext_fpn.py) (`tu-convnext_base` with stem and Stage 0 frozen to prevent 20-scene texture memorization).
- **Resolution:** $640 \times 640$ with FPN multi-scale feature pyramids.
- **Execution Script:** [`tools/train_v7_stage4_convnext.py`](file:///c:/Projects/SAT-EURS/tools/train_v7_stage4_convnext.py).

### Stage 5: Decision Calibration & Final Archive Packaging
- **Objective:** Nelder-Mead 4-parameter class-bias calibration on pooled OOF logits + Multi-Scale TTA (`[518, 644, 770]`) + `resize_for_submission.py` packaging into `deploy/eurs_track1_v7_apex_sota.zip`.
