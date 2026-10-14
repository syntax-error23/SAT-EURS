# Submission #18666 — EURS-Net V5-5

**Submission ID:** `#18666`  
**Method Name:** `EURS-Net V5-5`  
**Submission Date:** 13-Oct-2026  
**Archive:** `eurs_track1_submission_18666_v5_5.zip`  
**Generating Script:** [`deploy/generate_track1_submission_sota.py`](file:///c:/Projects/SAT-EURS/deploy/generate_track1_submission_sota.py)  

---

## 1. Official Leaderboard Score

| Metric | Measured Score | Official Weight in Composite |
|:---|:---:|:---:|
| **Composite Score** | **0.473** | 100% |
| **$Q_{\text{seg}}$** | **0.347** | 60% |
| **mIoU** | **0.521** | — |
| **Macro $F_1$** | **0.666** | — |
| **Coverage Penalty** | **1.000** | — |
| **PSNR** | **24.50 dB** | 13.5% |
| **SSIM** | **0.821** | 13.5% |
| **Sharpness Ratio** | **0.611** | 3.0% |
| **UIQM** | **0.711** | 4.0% |
| **UCIQE** | **0.565** | 4.0% |
| **CCF** | **0.041** | 2.0% |

---

## 2. Architecture & Design

### Restoration Pipeline
- **Physics Inversion via RSUIGM Polynomial Pseudo-Inverse**: Evaluates a 19-dimensional polynomial basis over local multi-scale veiling light on GPU, multiplied by pre-fitted weight matrices $W_{(\text{wt}, \text{depth})}$.
- **Bilateral Edge Preservation**: Bilateral filtering to smooth chrominance haze while holding luminance edge gradients.
- **Single-Image Processing**: Evaluated per image with closed-form polynomial lookup.

### Segmentation Ensemble
- **Tri-Model Ensemble**:
  - **40% SegFormer MiT-B3 DeepLabV3Plus** (45.2M parameters)
  - **30% EURS-Net V4** (0.72M parameters, full20 checkpoint)
  - **30% EURS-Net V5 Attention-Guided** (0.78M parameters, gan_attn checkpoint)
- **Heuristic Decision Boundary Routing**:
  - Starfish (Class 4) logit boost: `logits[4] += 1.8`
  - Rare-class suppression: `logits[3] -= 4.0; logits[4] -= 4.0` on non-rare scenes
- **Inference Augmentation**: Horizontal-flip Test-Time Augmentation (TTA).

---

## 3. What Was Promised vs. What Was Actually Achieved

- **Promised / Projected Score**:
  - Composite Score: $\ge \mathbf{0.70 - 0.78}$
  - $Q_{\text{seg}}$: $\ge \mathbf{0.68 - 0.74}$
  - PSNR: $\ge 26.0\text{ dB}$
- **Actual Leaderboard Score**:
  - Composite Score: **0.473** ($-0.227$ below promise)
  - $Q_{\text{seg}}$: **0.347** ($-0.350$ below promise)
  - PSNR: **24.50 dB**

---

## 4. Why the Composite and $Q_{\text{seg}}$ Were Less Than Promised (Root Causes)

1. **The In-Sample Memorization Trap**:
   - EURS-Net V4 (`v4_full20`) and V5 (`v5_gan_attn`) were trained on all 20 scenes in `data/train`.
   - Local validation was conducted on 4 of those 20 scenes (`fish1`, `fish12`, `seahorse1`, `sharkFish`), achieving local $Q_{\text{seg}} \approx 0.71$.
   - On the official server's 6 completely unseen Dev scenes (`fish10`, `fish19`, `fish3`, `fish6`, `sharRayFish`, `starfish1`), the ensemble had zero memorized priors. True generalization $Q_{\text{seg}}$ collapsed to **$0.347$**.
2. **Heuristic Logit Manipulation**:
   - The manual $+1.8$ Starfish boost was a rigid constant. While it elevated Starfish recall on `starfish1` ($F_1 = 0.666$), it was not adaptive to image turbidity or depth.
3. **Restoration Sub-Optimization**:
   - Standard RSUIGM polynomial inversion achieved 24.50 dB, but edge sharpness ($0.611$) was unsharpened, missing potential high-frequency detail.

---

## 5. What Plan Was Made to Fix It

- **Plan for Next Submission (#18668)**:
  1. Add a massive, pre-trained clean-domain context model: **DeepLabV3-ResNet101** (60.9M parameters).
  2. Rebalance ensemble weights: 40% SegFormer + 35% DeepLabV3 + 15% V4 + 10% V5.
  3. Tighten false-positive suppression: Increase rare penalty from $-4.0$ to $-5.0$ to eliminate noisy specks in fish scenes.
  4. Moderate the Starfish boost from $+1.8$ down to $+0.8$ to improve background precision.

---

## 6. Why We Flawed Again in the Next Submission (#18668)

- **The Fix Backfired**:
  - Weakening the Starfish boost from $+1.8 \to +0.8$ and over-suppressing rare classes with $-5.0$ destroyed Starfish recall.
  - DeepLabV3 was accidentally passed unnormalized inputs `[0, 1]` instead of ImageNet standardized `(x - mean) / std`, collapsing its standalone $Q_{\text{seg}}$ to $0.0682$ and poisoning the ensemble.
  - As a result, $Q_{\text{seg}}$ dropped from $0.347 \to 0.336$, and total composite dropped from $0.473 \to 0.467$.
