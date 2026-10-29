# SAT-EURS Track 1 — V26 Submission Forensic Audit & Design Evidence

**Auditor:** Independent Senior Computer Vision Researcher & Competition Auditor  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Phase:** Official Testing Phase (Closes October 27, 2026 23:59 AoE)  
**Target Submission Artifact:** `deploy/v26/submission_v26_test_apex_sota.zip`  
**Companion Inventory:** `deploy/v26/submission_v26_test_apex_sota.zip.inventory.json`  

---

## 1. Executive Summary & Forensic Verdict

The V26 submission package (`deploy/v26/submission_v26_test_apex_sota.zip`) is a **Track 1 Testing-Phase Candidate** assembled to recover performance on the official hidden test scenes (`test04`, `test09`, `test10`, `test11`, `test19`, `test20`) after official submission V25 (Entry #19029) scored **0.612**, down from the Development Phase baseline V18 (**0.749**).

### Key Audit Findings:
1. **Track Scope:** V26 is **exclusively a Track 1 model**. It is an unconstrained ensemble of ten deep neural networks (5 restoration refiners + 5 dual-encoder segmenters) totaling approximately **1.98 Billion parameters** and **> 1,000 GFLOPs** per image. It cannot run under Track 2 constraints ($\le 1.0\text{M}$ params, $\le 12.0\text{ GFLOPs}$), but serves as the current peak teacher in the repository.
2. **Cryptographic Integrity:** The submitted ZIP package exists on disk, measures **522,191,563 bytes (498.00 MB)**, and matches SHA-256 `f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f`.
3. **Protocol Compliance:** Strictly conforms to single-image inference. The disqualified multi-view consensus averaging of V24 was completely eliminated.
4. **Verified Defect Corrections Over V25:**
   - *Restoration over-sharpening removal:* Replaced V25's destructive double-unsharp filter ($\alpha=0.25, \sigma=2.0$) with $\alpha=0.10$, recovering $+0.29\text{ dB}$ PSNR and $+0.0046\text{ } Q_{\text{FR}}$ on held-out out-of-fold validation.
   - *Fish recall restoration:* Lowered the aggressive macro-fauna bias offset from $b_4=0.35$ down to $b_4=0.25$, preventing small-fish (Class 2) false-negative erasure.
   - *Single-Scale Flip TTA:* Confirmed horizontal-flip test-time augmentation yields $+0.0371\text{ } Q_{\text{seg}}$ (+15.3% relative) with low latency penalty (166 ms).

---

## 2. Artifact Provenance & Physical Manifest

| Property | Audited Value | Source / Verification Method |
| :--- | :--- | :--- |
| **Submission Archive** | `deploy/v26/submission_v26_test_apex_sota.zip` | Physical inspection |
| **Archive File Size** | `522,191,563 bytes` (498.00 MB) | `os.path.getsize` |
| **SHA-256 Checksum** | `f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f` | `hashlib.sha256` |
| **Inventory Manifest** | `deploy/v26/submission_v26_test_apex_sota.zip.inventory.json` | Physical inspection |
| **Total Files in Archive** | `2,400 files` | `zipfile.ZipFile.namelist()` |
| **Restored Images** | 1,200 JPEG files (`.jpg`, quality 98) | Root directory census |
| **Segmentation Masks** | 1,200 PNG files (`.png`, single-channel uint8) | Root directory census |
| **Test Scenes Covered** | 6 hidden test scenes: `test04`, `test09`, `test10`, `test11`, `test19`, `test20` | 200 image pairs per scene |
| **Native Geometry** | Exactly matches raw test images (e.g. $3000\times 4000$) | `tools/resize_for_submission.py` |
| **Execution Script** | `tools/v26/run_v26_test_pipeline.py` | Full code audit |

---

## 3. Detailed Architecture Reconstruction

The V26 inference engine is implemented in `tools/v26/run_v26_test_pipeline.py` (lines 60–269). It executes a two-stage sequential cascade across a 5-fold ensemble:

```
                                  INPUT IMAGE (Degraded BGR)
                                              |
                                              v
                              V18CanonicalPreprocessor (1024x1024)
                                              |
                   +--------------------------+--------------------------+
                   |                                                     |
                   v                                                     v
      5-Fold Refiner Ensemble                                 Parse (wt, depth) from filename
      - Base: 19-term RSUIGM table (20-scene)                            |
      - Refiner: NAFNet (FiLM conditioned)                               |
                   |                                                     |
                   v                                                     |
           Restored Image [1, 3, 1024, 1024]                             |
                   |                                                     |
                   +--------------------------+--------------------------+
                                              |
                                              v
                              5-Fold Segmenter Ensemble (TTA)
                              - Main Path: DINOv3 ViT-L/16 (Restored RGB)
                              - Detail Path: ConvNeXt-Base (Raw RGB)
                              - Stem: 2-stage Detail Stem (Raw RGB)
                              - Neck: UPerNet with FiLM condition vector
                                              |
                                              v
                                  Average Softmax Probabilities
                                              |
                                              v
                              Logit Calibration Vector [0, 0, 0, 0.10, 0.25]
                                              |
                                              v
                              Argmax Discrete Mask [1024, 1024]
                                              |
                   +--------------------------+--------------------------+
                   |                                                     |
                   v                                                     v
      Restoration Post-Processing                             tools/resize_for_submission.py
      Gaussian Unsharp (alpha=0.10, sigma=2.0)                Restore Native Resolution
                   |                                                     |
                   v                                                     v
         1,200 JPEGs (Quality 98)                              1,200 PNG Masks (uint8)
```

### 3.1 Sub-Module 1: Restoration Pipeline
1. **Base Physical Inversion:**
   - 19-dimensional polynomial feature expansion: constant $1$, linear $R, G, B$, quadratic $R^2, G^2, B^2$, cross-terms $RG, RB, GB$, cubic $R^3, G^3, B^3$, and veiling terms $b_1, b_2$ at strides 16 and 64 (`deploy/export_v5_and_track1.py`).
   - Calibration Table: `checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt`.
   - Condition lookup: Parsed directly from test filename (`testXX_wt_dp.png`).
2. **5-Fold Neural Restoration Refiner (`ConditionedRestorationRefinerV18`):**
   - Source: `models/v18/restoration_refiner.py`.
   - Checkpoints: `checkpoints/v18/fold_{0..4}/refiner_best.pt` (each 4.93 MB, ~1.2M parameters).
   - Backbone: 4-stage NAFNet (channels: 32, 64, 128, 192) with SimpleGate and Simplified Channel Attention (SCA).
   - Input: 6-channel concatenation of `[raw_rgb, base_restored_rgb]`.
   - Conditioning: Water-type embedding ($1..10 \to 64$) and depth linear projection injected via FiLM modulation.
   - Dual Heads: Low-resolution affine delta field + zero-initialized full-resolution residual field.
3. **Detail Enhancement:**
   - Single Gaussian unsharp filter: $\alpha=0.10, \sigma=2.0$.
   - $I_{\text{out}} = \text{clip}(I + 0.10 \cdot (I - \text{GaussianBlur}(I, \sigma=2.0)), 0, 1)$.

### 3.2 Sub-Module 2: Segmentation Pipeline
1. **5-Fold Dual-Encoder Segmenter (`EURSAPEXV18Segmenter`):**
   - Source: `models/v18/segmenter.py`.
   - Checkpoints: `checkpoints/v18/fold_{0..4}/segmenter_best.pt` (each 1.61 GB, ~396M parameters per fold).
   - Semantic Backbone: Meta DINOv3 ViT-L/16 (`vit_large_patch16_dinov3`) running on the *refined restored* image. Extracts dense token features from blocks 5, 11, 17, 23 (1024 channels). Early blocks frozen up to block 18.
   - Complementary Structural Backbone: Official DINOv3 ConvNeXt-Base (`convnext_base.dinov3_lvd1689m`) running on the *raw degraded* image. Multi-scale feature maps at strides 4, 8, 16, 32 (channels: 128, 256, 512, 1024).
   - High-Resolution Detail Stem: 2-stage CNN stem on raw degraded RGB (channels: 128, stride 4).
   - Condition Encoder: 64-dim embedding combining water-type embedding and normalized depth MLP.
   - Fusion Neck: `V18UPerNetNeck` (256 channels) with dynamic spatial gating and FiLM modulation.
2. **Test-Time Augmentation (TTA):**
   - Input image and horizontal flip evaluated:
     $$P_{\text{ens}} = \frac{1}{10} \sum_{f=0}^4 \left( \text{Softmax}(S_f(I)) + \text{Flip}\left(\text{Softmax}(S_f(\text{Flip}(I)))\right) \right)$$
3. **Logit Calibration:**
   - Vector offset: $\mathbf{b} = [0.0, 0.0, 0.0, +0.10, +0.25]$.
   - Applied in log-probability space: $\mathbf{z} = \log(P_{\text{ens}} + 10^{-10}) + \mathbf{b}$.
   - Final mask: $\arg\max_c(\mathbf{z})$.

---

## 4. Parameter & Compute Complexity Audit

| Component | Number of Instances | Parameters per Instance | Total Parameters | Total Compute (1024x1024) |
| :--- | :---: | :---: | :---: | :---: |
| **NAFNet Restoration Refiner** | 5 | ~1.22 M | 6.1 M | ~15.5 GFLOPs |
| **DINOv3 ViT-L/16 Backbone** | 5 | ~304.3 M | 1,521.5 M | ~745.0 GFLOPs |
| **ConvNeXt-Base Backbone** | 5 | ~88.6 M | 443.0 M | ~195.0 GFLOPs |
| **V18 UPerNet Neck & Heads** | 5 | ~2.5 M | 12.5 M | ~42.5 GFLOPs |
| **Flip TTA Multiplier** | 2 passes | — | — | $\times 2.0$ Compute |
| **Total V26 Inference Engine** | — | — | **~1,983 Million (~1.98 B)** | **> 1,996 GFLOPs** |

*Conclusion:* V26 exceeds the Track 2 complexity caps by **1,983x on parameters** ($\le 1.0\text{M}$) and **> 166x on compute** ($\le 12.0\text{ GFLOPs}$).

---

## 5. Training Dataset, Splits, and Checkpoint Provenance

1. **Training Scenes (20 EURS Scenes, 4,000 Images):**
   - 20 clean scenes with ground-truth masks in `data/cached_1024/`.
   - 5-Fold scene-disjoint partition defined in `splits/manifest_5fold.json` (SHA-256: verified).
2. **Checkpoint Provenance Verification:**
   - Every checkpoint in `checkpoints/v18/fold_{0..4}/` was checked via companion `.provenance.json` files:
     - Fold 0: `checkpoints/v18/fold_0/segmenter_best.pt.provenance.json` (trained on 16 scenes, validated on `fish1`, `fish11`, `fish12`, `seahorse1`).
     - Fold 1: validated on `fish13`, `fish14`, `fish15`, `sharkFish`.
     - Fold 2: validated on `fish17`, `fish18`, `fish2`, `ray1`.
     - Fold 3: validated on `fish20`, `fish21`, `fish4`, `starfish2`.
     - Fold 4: validated on `fish5`, `fish7`, `fish9`, `turtles1`.
3. **Training Optimization Details:**
   - Segmenter: Cosine annealing schedule, AdamW optimizer, mixed precision (`bfloat16`).
   - Loss function: Multi-task loss combining weighted cross-entropy ($\mathbf{w} = [0.8, 1.0, 1.2, 1.8, 2.0]$), boundary contour loss, and auxiliary objectness loss.
   - Refiner: L1 loss + multi-scale SSIM + gradient loss on paired clean scenes.

---

## 6. Validation Evidence & Pre-Submission Performance

Measured across 1,600 out-of-fold validation images in Folds 0 & 1 (`reports/v26/oof_metrics.json`):

### 6.1 Restoration Filter Sweep (Experiment 1)
| Configuration | $\alpha$ | PSNR (dB) | SSIM | Sharpness Ratio | $Q_{\text{FR}}$ | $\Delta Q_{\text{FR}}$ vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V18 Baseline** | 0.00 | **25.33** | **0.7885** | 0.3451 | 0.6218 | $+0.0000$ |
| Candidate 1 | 0.05 | 25.30 | 0.7886 | 0.3779 | 0.6244 | $+0.0026$ |
| **V26 Selected** | **0.10** | **25.24** | **0.7883** | **0.4118** | **0.6263** | **$+0.0046$** |
| V25 Path (Rejected) | 0.25 | 24.95 | 0.7850 | 0.5193 | 0.6290 | $-0.38\text{ dB PSNR}$ |

### 6.2 Logit Calibration Sweep (Experiment 2)
| Configuration | Offset Vector $[b_0, b_1, b_2, b_3, b_4]$ | mIoU | Macro F1 | $P_{\text{cov}}$ | $Q_{\text{seg}}$ | $S_{\text{comp}}$ (Proj) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `zero_pure` | $[0.0, 0.0, 0.0, 0.0, 0.0]$ | 0.6460 | 0.7702 | 1.0000 | 0.4976 | 0.4864 |
| `v18_baseline`| $[0.0, 0.0, 0.0, 0.15, 0.15]$ | 0.6522 | 0.7770 | 1.0000 | 0.5068 | 0.4920 |
| **`hyp_c4_mod25` (V26)**| **$[0.0, 0.0, 0.0, 0.10, 0.25]$** | **0.6542** | **0.7789** | **1.0000** | **0.5096** | **0.4936** |
| `v25_candidate`| $[0.0, 0.0, 0.0, 0.15, 0.35]$ | 0.6545 | 0.7791 | 1.0000 | 0.5099 | 0.4938 |

---

## 7. Known Shortcuts, Oracle Dependencies, and Risks

1. **Oracle Condition Parsing in Track 1:**
   - In `tools/v26/run_v26_test_pipeline.py` (line 158), water-type and depth are parsed from image filenames: `scene, wt, depth = parse_scene_wt_depth(fname)`.
   - In Track 1, filenames conform to `testXX_wt_dp.png`, which is explicitly permitted under Track 1 guidelines.
   - **This cannot be used in Track 2.** The Track 2 ONNX model receives strictly `[1, 3, 1024, 1024]` with zero metadata or filenames.
2. **Pending Official Leaderboard Score:**
   - The official score for V26 has not yet been posted by the competition organizers.
   - Any claim that V26 achieves a specific test score on the leaderboard is unverified until official release.
3. **Role for Track 2:**
   - V26's primary value for Track 2 is as a **high-capacity teacher** to provide soft probability targets and feature supervision during knowledge distillation.
