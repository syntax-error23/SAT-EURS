# SAT-EURS Track 1 — Phase 2 Packaging Gate & Phase 3 Progress Report

**Date:** October 15, 2026  
**Auditor:** Antigravity Research Team  
**Challenge:** MaCVi @ WACV 2027 EURS Track 1  
**Hardware:** NVIDIA RTX 5090 (32 GB VRAM)  
**Current Best Leaderboard Result:** Submission #18700 ($S = 0.684$, $Q_{\text{seg}} = 0.675$, $\text{PSNR} = 25.07\text{ dB}$, $\text{SSIM} = 0.787$)

---

## 1. Phase 2 Packaging Gate (COMPLETED & VERIFIED)

In accordance with Section 4 of the Pre-Execution Protocol, the surviving Phase 2 improvements were compiled and packaged into the official candidate archive before initiating Phase 3 training.

### 1.1 Verified Candidate Specifications
- **Archive Path:** `deploy/eurs_track1_phase2_verified.zip`
- **File Size:** `587.49 MB` (Strictly complies with the $< 1024\text{ MB}$ limit)
- **SHA256:** `7d108c682c5542dd9e3f6010f7a62f6d210aed41fc6753cb836383d4b20c3d38`
- **Restoration Count:** Exactly `1,200` JPEG images (`restored/`, Quality 98)
- **Segmentation Count:** Exactly `1,200` PNG masks (`seg_masks/`, `uint8`, classes $\in \{0, 1, 2, 3, 4\}$)
- **Dimension Match:** 100% paired 1:1 match across all test scenes (`1024x768` and `768x1024` native aspect-scaled resolutions).

### 1.2 Algorithmic Innovations in Phase 2 Candidate
1. **Full 20-Scene 19-Term RSUIGM Calibration:**
   - Overcomes the accidental `clean_scenes[:8]` alphabetical truncation in #18700.
   - Solved with equal pixel representation ($20 \times 128 \times 128 = 327,680$ pixels per $(wt, \text{depth})$ condition).
   - Saved in `checkpoints/track1_screening/tbl19_20sc_clean.pt`.
2. **Calibrated Unsharp Filtering (`cfg1_unsharp_1.34`):**
   - High-fidelity edge enhancement with Gaussian blur $\sigma = 2.0$ and weighting $1.34 \cdot I_{\text{rest}} - 0.34 \cdot I_{\text{blur}}$.
   - Confirmed on held-out validation folds with $+0.0199$ $Q_{\text{FR}}$ gain while strictly passing the local sharpness floor ($0.5186 > 0.5152$).
3. **Multi-Scale TTA Ensemble Segmentation:**
   - 5-Fold DINOv2-Large ensemble with scales $[448, 518, 588]$ (weights $[0.20, 0.60, 0.20]$) and horizontal flip TTA.
4. **Deterministic Single-Image Independence Verification:**
   - Evaluated isolated single images through [`tests/test_phase2_single_image_independence.py`](file:///c:/Projects/SAT-EURS/tests/test_phase2_single_image_independence.py).
   - Bit-for-bit identical segmentation masks ($\text{max\_diff} = 0$) and 100% identical restored JPEG bytes.

---

## 2. Phase 3 — Experiment 3.1: High-Resolution RGB Detail Stem + Lovász-Softmax

### 2.1 Architectural Design (`models/dinov2_highres_decoder.py`)
- **Backbone:** DINOv2-ViT-L/14 with 4 multi-level feature extractions (Blocks 5, 11, 17, 23).
- **Detail Stem:** Parallel convolutional stem operating directly on full-resolution RGB inputs:
  - Stride 2 stage ($48$ channels at $H/2 \times W/2$).
  - Stride 4 stage ($96$ channels at $H/4 \times W/4$).
- **Top-Down Feature Fusion:**
  - ViT coarse representations ($H/14 \times W/14$) progressively fused with Stride-4 and Stride-2 detail maps.
- **Zero-Initialized Residual Projection:**
  - $\text{Logits} = \text{Base\_Logits} + \text{ZeroConv}(\text{Detail\_Features})$.
  - $\text{max\_abs\_diff} = 0.00000000$ at Step 0 against baseline checkpoints.
  - Zero initial degradation guarantee.

### 2.2 Unit Test Suite (`tests/test_dinov2_highres_decoder.py`)
- **Shapes:** Verified on $448 \times 448$, $518 \times 518$, and $588 \times 588$ tensors.
- **Identity Check:** $100\%$ identity to `DINOV2_CP_F0/best_model.pt`.
- **Gradient Flow:** 2-step verification confirmed active gradients across all 7 functional modules (`encoder`, `proj`, `fusion`, `stem`, `fuse_stage4`, `fuse_stage2`, `zero_conv`).
- **Loss Stability:** Weighted CE + Present-Class Lovász-Softmax runs stable with zero NaNs or Infs.

### 2.3 Screening Execution (`tools/train_exp3_1_highres_pilot.py`)
- Currently running under task `task-825` on Screening Folds 0 & 1 (4 epochs each).
- Evaluates on 1,600 held-out images with Multi-Scale TTA against locked baseline floors:
  - $Q_{\text{seg}} \ge 0.3838$
  - $\text{mIoU} \ge 0.5461$
  - $\text{Macro } F_1 \ge 0.7028$
  - Class-wise IoU floors: $[0.6184, 0.5884, 0.5666, 0.3915, 0.5658]$
