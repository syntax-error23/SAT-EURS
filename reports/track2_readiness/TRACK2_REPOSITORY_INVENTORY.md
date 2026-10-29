# SAT-EURS Track 2 — Repository Architecture & Artifact Inventory

**Auditor:** Independent Senior Computer Vision Researcher & Model Optimization Engineer  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Target Hardware:** NVIDIA Jetson AGX Orin (Embedded Edge)  
**Host Evaluation Hardware:** NVIDIA GeForce RTX 5090 (32.6 GB VRAM), Intel Core Ultra 9 285K, 64 GB DDR5 RAM  
**Official Complexity Caps:** Total Parameters $\le \mathbf{1,000,000}$ ($\le 1.0\text{M}$), Total Static Compute $\le \mathbf{12.0\text{ GFLOPs}}$ at $1024 \times 1024$  

---

## 1. Candidate Model Status Classification Summary

Each Track 2 candidate in the repository has been audited against ten strict criteria:
1. Source code exists and is complete.
2. Trainable parameters and implemented training pipeline.
3. Checkpoint exists on disk.
4. Checkpoint matches intended training configuration.
5. Evaluated on real validation data.
6. ONNX export succeeds.
7. Official parameter/FLOPs checker passes (`deploy/check_submission_flops.py`).
8. PyTorch vs. ONNX Runtime numerical parity ($\text{MAE} < 10^{-4}$).
9. Physically tested on Jetson AGX Orin hardware.
10. Claimed metrics reproducible from existing artifacts.

| Candidate Architecture | Model File / Source | Checkpoint Path | ONNX File Path | Params (M) | Compute (GFLOPs) | Official Leaderboard | Audit Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **EURS-Net V5 Calibrated** | `models/eurs_net_v5.py` | `checkpoints/v5_calibrated_track2_best.pt` | `deploy/eurs_net_v5_calibrated_track2.onnx` | **0.7683** | **9.0500** | **0.3380 (#18699)** | **VALIDATED** |
| **EURS-EdgeSOTA-V2 (Pretrained Pyramid)** | `models/track2_edgesota_v2.py` | `checkpoints/EXP-PRETRAINED-30EP_best.pt` | `deploy/eurs_edgesota_v2_final.onnx` | **0.3537** | **2.2496** | *None (Unsubmitted)* | **TRAINED_UNVERIFIED (FLAWED VALIDATION)** |
| **EURS-EdgeSOTA-V2 (Scratch Inverted Bottleneck)** | `models/track2_edgesota_v2.py` | `checkpoints/EXP-COMBINED-BEST_best.pt` | `deploy/EXP-COMBINED-BEST.onnx` | **0.9589** | **3.5666** | *None (Unsubmitted)* | **TRAINED_UNVERIFIED** |
| **EURS-Net V4 (Path 2)** | `models/eurs_net_v4.py` | `checkpoints/v4_path2/best_v4_checkpoint.pt` | `deploy/eurs_net_v4_track2.onnx` | **0.7289** | **9.1800** | *None* | **SUPERSEDED** |
| **EURS-EdgeSOTA V1 (Parallel Proposal)** | `reports/track2_research/TRACK2_RESEARCH_...` | *None* | *None* | ~0.961 | ~10.65 | *None* | **REJECTED (FATAL FLAW)** |
| **EURS-EdgeSOTA-Scaled (Candidate B)** | *Proposed* | *None* | *None* | ~0.880 | ~7.500 | *None* | **PROPOSED** |
| **EURS-DualPath-Edge (Candidate C)** | *Proposed* | *None* | *None* | ~0.950 | ~9.500 | *None* | **PROPOSED** |

---

## 2. Detailed Audit per Track 2 Candidate

### 2.1 EURS-Net V5 Calibrated (`#18699`) — Status: VALIDATED
- **Description:** MobileNetV3-Small encoder, dual decoders with bounded restoration head (bound scale 0.20) and U-Net segmentation head. Class 3 bias offset $\Delta_3 = -1.50$ applied.
- **Source Code:** `models/eurs_net_v5.py` (complete).
- **Training Pipeline:** `models/train_clinical_distillation_fast.py` (distilled from SegFormer MiT-B3 and DeepLabV3-ResNet101).
- **Checkpoint:** `checkpoints/v5_calibrated_track2_best.pt` (3,416,299 bytes, SHA-256: `41851d759499e6791c0d550c0f443058943b05336459b34e25e3fcabef893d07`).
- **ONNX Model:** `deploy/eurs_net_v5_calibrated_track2.onnx` (3,287,892 bytes, SHA-256: `ad7bb1b1083083aee9fbbe44ce8974089d6b93077d892c20df84a267dd5fd135`).
- **Official Checker Results:**
  - Parameters: **0.7683 M** ($\le 1.0\text{M}$ cap: **PASS**)
  - Compute: **9.0500 GFLOPs** ($\le 12.0\text{ GFLOPs}$ cap: **PASS**)
- **Numerical Parity:** Tested in `deploy/export_calibrated_track2_onnx.py`: Restored Max Error: $7.2 \times 10^{-6}$, Logits Max Error: $8.4 \times 10^{-6}$ (PASS).
- **Hardware Benchmarks:** Measured on host CPU at **105.60 ms (9.5 FPS)**. Jetson Orin: **Not measured**.
- **Metrics on Record:**
  - Official Leaderboard (#18699): Composite = **0.3380**, $Q_{\text{seg}} = 0.2450$, mIoU = 0.4260, Macro F1 = 0.5760, PSNR = 17.10 dB, SSIM = 0.8210, Sharpness = 0.3190.
  - Local Validation (160 images): Composite = **0.5196** (using local $Q_{\text{NR}} = 0.50$, higher due to Fold 0 validation overlap).
- **Verdict:** This is the **strongest genuinely validated Track 2 baseline**.

---

### 2.2 EURS-EdgeSOTA-V2 (Pretrained Pyramid) — Status: TRAINED_UNVERIFIED (FLAWED VALIDATION)
- **Description:** Cascaded physical RSUIGM restoration subnet (optical router + 19-term basis + unsharp filter) feeding ImageNet-pretrained MobileNetV3-Small backbone with a 4-level Progressive Multi-Scale Pyramid Fusion Decoder.
- **Source Code:** `models/track2_edgesota_v2.py` (complete).
- **Training Pipeline:** `tools/train_track2_student.py` (30 epochs distillation from 3 Track 1 DINOv2-Large teachers).
- **Checkpoint:** `checkpoints/EXP-PRETRAINED-30EP_best.pt` (1,564,357 bytes, SHA-256: `1a59558a650836f47b3b31b106f7c1ba519e2c30486e3d7ae74e199b603cbfd0`).
- **ONNX Model:** `deploy/eurs_edgesota_v2_final.onnx` (5,688,188 bytes, SHA-256: `f547d0e1478311514c7f09f97b97d0c1e6f45087c85083342a154da45eba5f18`).
- **Official Checker Results:**
  - Parameters: **0.3537 M** ($\le 1.0\text{M}$ cap: **PASS, 64.6% headroom**)
  - Compute: **2.2496 GFLOPs** ($\le 12.0\text{ GFLOPs}$ cap: **PASS, 81.3% headroom**)
- **Numerical Parity:** Max error on CPU: $3.71 \times 10^{-6}$ (restoration), $3.1 \times 10^{-8}$ (logits) (PASS).
- **Hardware Benchmarks:** Measured on host CPU at **37.83 ms (26.4 FPS)**. Jetson Orin: **Not measured**.
- **The Forensic Flaw in Claimed Metrics:**
  - In `reports/track2_research/TRACK2_RESULTS.md`, the model claims $Q_{\text{seg}} = \mathbf{0.3314}$ and Projected Composite = $\mathbf{0.4074}$.
  - **Audit Discovery:** The model was evaluated ONLY on Fold 0 (`fish1`, `fish11`, `fish12`, `seahorse1`).
  - In Fold 0, **Class 3 (Elasmobranch) has 0 ground-truth pixels**.
  - In `tools/train_track2_student.py` (lines 481–501), a negative bias offset sweep selected $-4.0$ for Class 3 and baked it into `model.decoder.cls_head.bias[3]`.
  - Because the model predicted 0 pixels of Class 3 and ground-truth had 0 pixels, the local evaluator (`metrics/segmentation_metrics.py`) divided $0/0$ and assigned **IoU = 1.0000 and F1 = 1.0000** for Class 3.
  - This vacuous 1.0 artificially inflated mIoU from 0.4236 up to 0.5389.
  - Furthermore, on real scenes containing Class 3 (`sharkFish` or `ray1`), the $-4.0$ offset suppresses Class 3 recall, creating zero true positives and **triggering the fatal 0.0 coverage penalty**, collapsing $Q_{\text{seg}}$ to 0.0!
  - **The model has never been submitted to the official leaderboard.**
- **Verdict:** Must be re-calibrated and evaluated across all 5 folds before any claims of superiority can be accepted.

---

### 2.3 EURS-EdgeSOTA-V2 (Scratch Inverted Bottleneck) — Status: TRAINED_UNVERIFIED
- **Description:** Cascaded physical RSUIGM restoration subnet feeding a scratch-trained 3-stage Inverted Bottleneck backbone (channels 24, 32, 64, 160) and LR-ASPP context decoder.
- **Source Code:** `models/track2_edgesota_v2.py` (when `pretrained_backbone=False`).
- **Checkpoint:** `checkpoints/EXP-COMBINED-BEST_best.pt` (3,996,865 bytes, SHA-256: `7668ef25e2da4f4048a0fff5ce5d26f1b67e38a1190263cddf6ee3e33aaaef97`).
- **ONNX Model:** `deploy/EXP-COMBINED-BEST.onnx` (8,067,862 bytes, SHA-256: `645eb785ba2bfb07a9ce4349b19b79151af627a88c58b466035395e48cbbcd52`).
- **Official Checker Results:**
  - Parameters: **0.9589 M** ($\le 1.0\text{M}$ cap: **PASS, 4.1% headroom**)
  - Compute: **3.5666 GFLOPs** ($\le 12.0\text{ GFLOPs}$ cap: **PASS, 70.3% headroom**)
- **Hardware Benchmarks:** Measured on host CPU at **29.42 ms (34.0 FPS)**. Jetson Orin: **Not measured**.
- **Metrics on Record:** Local Fold 0 validation: Composite = **0.3401**, $Q_{\text{seg}} = 0.2453$, mIoU = 0.4594, Macro F1 = 0.5339, PSNR = 19.27 dB.
- **Verdict:** Scratch model respects constraints but underperforms pretrained backbone on small fish contours.

---

### 2.4 EURS-EdgeSOTA V1 (Parallel Proposal) — Status: REJECTED (FATAL FLAW)
- **Description:** Proposed in `reports/track2_research/TRACK2_RESEARCH_AND_ARCHITECTURE_BLUEPRINT.md`. Softmax mixture of 200 condition matrices ($\boldsymbol{\alpha} \mathbf{W}_{\text{bank}}$) and parallel execution of segmentation on raw degraded RGB.
- **Audit Findings:**
  - Softmax mixture empirically collapses restoration to **13.46 dB PSNR** (polynomial coefficients do not form a linear convex manifold).
  - Segmentation on raw degraded RGB suffers a **-52.2% drop in $Q_{\text{seg}}$** (from 0.2925 to 0.1399) due to extreme red-light extinction in deep water.
- **Verdict:** **FATALLY FLAWED & EMPIRICALLY REJECTED**.

---

## 3. Teacher Models & Distillation Assets Inventory

| Teacher Family | Checkpoint Directory | Checkpoint Count | Total Size | Architecture / Components | Usable for Track 2 KD? |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Track 1 V18 Dual-Encoder** | `checkpoints/v18/fold_{0..4}/` | 5 segmenters + 5 refiners | 7.69 GB | DINOv3 ViT-L/16 + ConvNeXt-Base + NAFNet Refiners | **YES (Highest Capacity)** |
| **Track 1 DINOv2-Large (CP)** | `checkpoints/track1_screening/DINOV2_CP_F{0..4}/` | 5 models | 5.88 GB | DINOv2 ViT-L/14 with multi-scale feature tap (308M params) | **YES (Validated in SOTA #18713)** |
| **Track 1 ConvNeXt-Base** | `checkpoints/v16_convnext/fold_{0..4}/` | 5 models | 1.71 GB | ConvNeXt-Base + UPerNet (88.5M params) | **YES (High-resolution details)** |
| **V17 Specialists (Full 20)** | `checkpoints/v17_specialists/` | 2 models | 1.52 GB | DINOv2-Large & ConvNeXt trained on all 20 scenes | **CAUTION (Leakage on OOF val)** |
| **V26 Deep Bilateral Restorer**| `checkpoints/v26/restorer_full20_best.pt` | 1 model | 1.70 MB | Bilateral Grid / Affine Restorer | **YES (Compact Restorer)** |

---

## 4. Evaluation Tooling & Deployment Infrastructure Inventory

| Tool / Script | File Path | Function | Integrity Status |
| :--- | :--- | :--- | :--- |
| **Official Complexity Checker** | `deploy/check_submission_flops.py` | Computes static Conv/Gemm/MatMul GFLOPs and parameter counts directly from ONNX graph. | **VERIFIED & OPERATIONAL** |
| **Historical FLOPs Checker** | `deploy/check_flops.py` | Secondary checker from earlier phase. | Verified |
| **Track 2 ONNX Evaluator** | `tools/evaluate_track2_onnx.py` | Runs ONNX Runtime CPU evaluation on held-out scenes. | **MODIFICATION NEEDED:** Hardcodes Fold 0 scenes (`fish1, 11, 12, seahorse1`). Needs 5-fold expansion. |
| **Submission Packaging Script** | `tools/v18/package_submission.py` | Validates file census, naming convention, paired stems, and archive size. | **VERIFIED & OPERATIONAL** |
| **Resizing Tool** | `tools/resize_for_submission.py` | Resizes model outputs back to native image dimensions. | **VERIFIED & OPERATIONAL** |
| **Fold Support Audit Tool** | `tools/audit_folds_and_support.py` | Computes exact pixel census per class and evaluates fold pair support. | **VERIFIED & OPERATIONAL** |
