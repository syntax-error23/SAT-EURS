# SAT-EURS Track 1 — V19 P2: Heterogeneous Ensemble Feasibility Audit

**Auditor:** Senior Computer Vision Research Engineer & Empirical ML Specialist  
**Challenge:** MaCVi @ WACV 2027 Challenge — Track 1 (Underwater Restoration & Semantic Segmentation)  
**Date:** October 18, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 5090 (31.84 GB VRAM), 64 GB Host RAM, Intel Core Ultra 9  
**Target Baseline:** Official V18 Submission (`deploy/v18/submission_v18_dual_encoder_sota.zip`)  
**Official Evaluation Baseline:** Composite: **0.749**, $Q_{\text{seg}}$: **0.768**, mIoU: **0.842**, Macro F1: **0.912**, PSNR: **25.89 dB**, SSIM: **0.833**  
**Local 5-Fold OOF Baseline:** $Q_{\text{seg}}$: **0.43645**, mIoU: **0.59441**, Macro F1: **0.73425**, PSNR: **25.04 dB**, SSIM: **0.7761**  

---

## Executive Verdict: PASS (With Mandatory Scope Demarcation)

The feasibility audit concludes with an explicit **PASS** for a strictly defined, leak-free, 5-fold heterogeneous probability-level ensemble combining:
1. **V18 Dual-Encoder** (`EURSAPEXV18Segmenter`: DINOv3 ViT-Large/16 + ConvNeXt-Base + UPerNet neck, 401.8M params)
2. **V16 DINOv2-Large** (`DINOv2Segmenter`: DINOv2 ViT-Large/14 + Linear Projection + Fusion neck, 304.3M params)
3. **V17 ConvNeXt-Base FPN** (`tu-convnext_base` + FPN Decoder, 88.5M params)

All **15 required model checkpoints** (5 folds $\times$ 3 models) exist on disk, have verified SHA256 hashes, strictly adhere to the identical 16-train / 4-val scene-disjoint splits of `splits/manifest_5fold.json`, and share identical 5-class target ontologies.

```
========================================================================================================
                                 FEASIBILITY AUDIT DECISION MATRIX
========================================================================================================
Configuration Proposed                       | 5-Fold Provenance | Leak-Free OOF? | Verdict
---------------------------------------------|-------------------|----------------|---------------------
Tri-Ensemble (V18 + V16-DINOv2 + V17-ConvNeXt)| 100% (15/15 Folds)| YES (0% Leak)  | PASS (Recommended)
V17 Full-20 Specialists Ensembling           | Missing Folds     | NO (100% Leak) | BLOCKED (Disqualified)
Phase 2 SegFormer MiT-B3 Ensembling          | Missing Folds     | NO (80% Leak)  | BLOCKED (Disqualified)
========================================================================================================
```

> [!CAUTION]
> **Fatal Data Leakage Risk Disclosed & Blocked:**
> - The nominal "V17 Specialists" (`checkpoints/v17_specialists/dinov2_full20_best.pt` and `convnext_full20_best.pt`) were trained simultaneously across **all 20 clean training scenes**. Evaluating them on held-out validation scenes introduces **100% in-sample data leakage**. They are **BLOCKED** from OOF evaluation.
> - The Phase 2 teacher `checkpoints/segformer_mit_b3_best.pt` is a single checkpoint trained without cross-validation. It is **BLOCKED** from OOF evaluation.
> - Consequently, the only mathematically valid, leak-free ensemble across the three generations is the **Tri-Ensemble** formed by V18's 5 folds, V16's 5 DINOv2 folds, and V17's 5 ConvNeXt folds.

---

## 1. Immutable Baseline & Safety Verification

The official V18 submission archive has been audited and cryptographically verified:
- **Archive Path:** `deploy/v18/submission_v18_dual_encoder_sota.zip`
- **File Size:** 568,690,404 bytes (542.34 MB)
- **Expected SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Verified SHA256 (Before Audit):** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Integrity Status:** **100% INTACT & UNMODIFIED**

No model weights, restoration tables, split manifests, or production deployment files were modified during this audit. Zero GPU training or inference was executed.

---

## 2. Complete Inventory of All Three Model Generations

A forensic audit of the repository identified exactly **122 checkpoint (`.pt`) files**, from which all candidate models were inspected.

### Checkpoint Inventory Table

| Generation / Component | Checkpoint Path | Size (MB) | SHA256 Checksum | 5-Fold Status | Training Split Conformance |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **V18 Fold 0 Segmenter** | `checkpoints/v18/fold_0/segmenter_best.pt` | 1532.33 | `2c94bc9c295c6a98b9d07ce5a8eecb8b17cb903526606ac9da95f124ad9e850b` | Present | Strict `manifest_5fold.json` (Val: fish1, fish11, fish12, seahorse1) |
| **V18 Fold 1 Segmenter** | `checkpoints/v18/fold_1/segmenter_best.pt` | 1532.33 | `a89ccb69220a4cbec050a11dd70104ea59e658c946546754a0c4a17d9b765c82` | Present | Strict `manifest_5fold.json` (Val: fish13, fish14, fish15, sharkFish) |
| **V18 Fold 2 Segmenter** | `checkpoints/v18/fold_2/segmenter_best.pt` | 1532.33 | `ec1c96e087b5479566cd03ace43e4a59e2b97f17180c96105691f1e700513a18` | Present | Strict `manifest_5fold.json` (Val: fish17, fish18, fish2, ray1) |
| **V18 Fold 3 Segmenter** | `checkpoints/v18/fold_3/segmenter_best.pt` | 1532.33 | `e21a26951b084fa1616a5a80678b3c379a86b12d5bce42c0cdcf1e1efc4c5449` | Present | Strict `manifest_5fold.json` (Val: fish20, fish21, fish4, starfish2) |
| **V18 Fold 4 Segmenter** | `checkpoints/v18/fold_4/segmenter_best.pt` | 1532.33 | `72dd0a5bf56fb2a5a7dc704aac43fb541ca6ebd85c762b3ed20fc1758378f5e7` | Present | Strict `manifest_5fold.json` (Val: fish5, fish7, fish9, turtles1) |
| **V18 Fold 0 Refiner** | `checkpoints/v18/fold_0/refiner_best.pt` | 4.70 | `c728728c8737aca546404dd0343415e3d6f8917a626ded32cd0c6a6703003130` | Present | Strict `manifest_5fold.json` |
| **V18 Fold 1 Refiner** | `checkpoints/v18/fold_1/refiner_best.pt` | 4.70 | `ebbab3801f8724b2426f69c0549ecf4c678e67c99b83981247b10ada29938e7c` | Present | Strict `manifest_5fold.json` |
| **V18 Fold 2 Refiner** | `checkpoints/v18/fold_2/refiner_best.pt` | 4.70 | `bac9a3a38071ed312b592b805864383fd70aee8c60c1dcc060f22ed2e3c58b01` | Present | Strict `manifest_5fold.json` |
| **V18 Fold 3 Refiner** | `checkpoints/v18/fold_3/refiner_best.pt` | 4.70 | `6ebfe5f292251edc4ec18e0cf8bef378c7a2ba34de3e0938eddd9d6f8bc55f14` | Present | Strict `manifest_5fold.json` |
| **V18 Fold 4 Refiner** | `checkpoints/v18/fold_4/refiner_best.pt` | 4.70 | `34efebbe00224e4f7481f9bd41de37260f06d83e7c7b70d13905d93a4edd8d34` | Present | Strict `manifest_5fold.json` |
| **V16 DINOv2 Fold 0** | `checkpoints/v16_dinov2/fold_0/best_model.pt` | 1175.37 | `3bbe037c421f14ed8e961d2535b00e404ea45f4f8344484c09767e020196acf8` | Present | Strict `manifest_5fold.json` |
| **V16 DINOv2 Fold 1** | `checkpoints/v16_dinov2/fold_1/best_model.pt` | 1175.37 | `2cfa306d39e4b55d51900da073fea9bd504ed27469a002c707c71b4d6b7dc59f` | Present | Strict `manifest_5fold.json` |
| **V16 DINOv2 Fold 2** | `checkpoints/v16_dinov2/fold_2/best_model.pt` | 1175.37 | `e68b1317fdf2a5892e153208d7369fa5b2c8c69669f8a26b0cffe8db8b8514be` | Present | Strict `manifest_5fold.json` |
| **V16 DINOv2 Fold 3** | `checkpoints/v16_dinov2/fold_3/best_model.pt` | 1175.37 | `1e4bc8053e04bf80ab28fa928174a80470b55886f58c19a67603e7d4b5c98b2b` | Present | Strict `manifest_5fold.json` |
| **V16 DINOv2 Fold 4** | `checkpoints/v16_dinov2/fold_4/best_model.pt` | 1175.37 | `8398d538d0fe092bc23d852cda4027d2bb784d29ba7d8fd36d365574e53a162f` | Present | Strict `manifest_5fold.json` |
| **V17 ConvNeXt Fold 0** | `checkpoints/v16_convnext/fold_0/best_model.pt` | 342.26 | `ed36e5d39fcddfbf2d64e0e9aa6b959f4972bcdee9f493dcf90f770a8b8accbd` | Present | Strict `manifest_5fold.json` |
| **V17 ConvNeXt Fold 1** | `checkpoints/v16_convnext/fold_1/best_model.pt` | 342.26 | `4ec49c010213f3a6ffdd03a9a237f1428e7c3bf3fb48173b5a6b2492cbbb7c4e` | Present | Strict `manifest_5fold.json` |
| **V17 ConvNeXt Fold 2** | `checkpoints/v16_convnext/fold_2/best_model.pt` | 342.26 | `c5b2aabcf614cc42e2aad627f0fb8d01beb711ac6d211cb851f535f3b8a90f97` | Present | Strict `manifest_5fold.json` |
| **V17 ConvNeXt Fold 3** | `checkpoints/v16_convnext/fold_3/best_model.pt` | 342.26 | `bcdf6ddca9599667ebc71f810ff4a3eefa07c1b9772313188ccb0c0528cb05d0` | Present | Strict `manifest_5fold.json` |
| **V17 ConvNeXt Fold 4** | `checkpoints/v16_convnext/fold_4/best_model.pt` | 342.26 | `e6bf76698d4857f49c31c44710ce61f8a5aa49849b864f7f85e6c7c289266519` | Present | Strict `manifest_5fold.json` |
| *V17 Full-20 DINOv2* | `checkpoints/v17_specialists/dinov2_full20_best.pt` | 1175.37 | `db767eab069ca900ba9cc5c9a34388716072e910663f710b757cd9cc8705cc71` | Single | **LEAKED (Trained on all 20 scenes)** |
| *V17 Full-20 ConvNeXt* | `checkpoints/v17_specialists/convnext_full20_best.pt` | 342.26 | `066a5dc723b8595693226eff14263da8bacd5a4a13276d9c7cc2bfda13bc4461` | Single | **LEAKED (Trained on all 20 scenes)** |
| *Phase 2 SegFormer* | `checkpoints/segformer_mit_b3_best.pt` | 172.77 | `03df63f3b3f1694c19cc92068a3283fe1e330eeb5301f951516390a570ff4bcc` | Single | **LEAKED (Single-split checkpoint)** |

---

## 3. Comparative Architecture & Preprocessing Analysis

```
========================================================================================================
                                 ARCHITECTURAL & PIPELINE COMPARISON
========================================================================================================
Dimension                  | V18 Dual-Encoder              | V16 DINOv2-Large             | V17 ConvNeXt-Base FPN
---------------------------|-------------------------------|------------------------------|---------------------
Architecture Class         | EURSAPEXV18Segmenter          | DINOv2Segmenter              | smp.FPN
Parameter Count            | 401.8M params                 | 304.3M params                | 88.5M params
Vision Backbones           | DINOv3 ViT-L/16 + ConvNeXt-B  | DINOv2 ViT-L/14              | tu-convnext_base
Neck / Fusion              | UPerNet Neck (256-d) + FiLM   | 4-level Linear + 3-Conv Head | Multi-scale FPN
Input Geometry             | 1024 x 1024 (native)          | 518 x 518 (patch-14 aligned) | 512 x 512 (stride-32)
Auxiliary Inputs           | Water Type (1..10) + Depth (m)| None                         | None
Image Normalization        | ImageNet mean & std           | ImageNet mean & std          | ImageNet mean & std
Output Resolution          | 1024 x 1024                   | 518 x 518 (upsampled 1024)   | 512 x 512 (upsampled 1024)
Output Representation      | Raw Logits (Unbounded)        | Raw Logits (Unbounded)       | Raw Logits (Unbounded)
Target Class Count         | 5 Classes                     | 5 Classes                    | 5 Classes
========================================================================================================
```

### Class Mapping Verification
All three model architectures share the identical 5-class target ontology:
- **Class 0:** Background (sand substrate, open ocean, water column, rocks)
- **Class 1:** Coral Reef (hard coral formations, soft corals, branching colonies)
- **Class 2:** Fish (teleost species, schooling fishes)
- **Class 3:** Elasmobranchii (sharks, rays, batoids) — Rare class
- **Class 4:** Macro-Fauna (sea turtles, seahorses, starfish) — Rare class

No label remapping or index permutation is required.

---

## 4. Why an Ensemble is Technically Feasible and Complementary

### The Complementarity Hypothesis
In our forensic post-run audit of V18 and V16:
1. **V18 Dominates Semantics and Large Marine Animals:**
   V18 achieved an out-of-fold $Q_{\text{seg}} = 0.43645$ and $\text{mIoU} = 0.59441$, vastly outperforming V16 ($Q_{\text{seg}} = 0.2951$, $\text{mIoU} = 0.4815$). On Class 3 (Elasmobranchii), V18 reached **0.55467 IoU** compared to V16's **0.23410 IoU** (+0.32057 IoU gain), largely due to the condition-guided FiLM modulation and large receptive field of DINOv3 ViT-L.
2. **V16/V17 Preserves Higher Recall on Class 4 Macro-Fauna:**
   Despite lower overall scores, V16's DINOv2 + ConvNeXt combination achieved **0.5318 recall on Class 4** (Macro-Fauna), whereas V18 achieved only **0.41297 recall**. Specifically on thin, delicate structures (the snout of `seahorse1` and the arms of `starfish2`), ConvNeXt-Base FPN operates with stride-4 convolutions that maintain high-frequency boundary fidelity without the patch tokenization boundary blurring of pure ViTs.
3. **Probability Simplex Convex Blending:**
   Combining the global semantic certainty of V18 with the fine-grained high-recall contours of V16/V17 can resolve V18's false-negative bias on Class 4 without compromising Class 3 or Background precision.

---

## 5. Compatibility Adapter Design

Because the models originate from different codebases and feature disparate spatial resolutions and temperature scales, direct logit summation is strictly invalid. The compatibility adapter executes the following mathematical sequence:

```
[Raw Degraded (1024x1024) + Water Type + Depth]
                     │
                     ▼
       ┌───────────────────────────────┐
       │ V18 Restoration Refiner       │ (Conditioned Neural Refiner + Scale-Consistent Table B)
       └───────────────────────────────┘
                     │
                     ▼
       [Restored RGB (1024x1024)]
         │           │           │
         ▼           ▼           ▼
   ┌───────────┐┌──────────┐┌──────────┐
   │    V18    ││  DINOv2  ││ ConvNeXt │
   │ (1024x1024││ (518x518)││ (512x512)│
   └───────────┘└──────────┘└──────────┘
         │           │           │
   (Logits 1024) (Logits 518) (Logits 512)
         │           │           │
      Softmax     Softmax     Softmax
         │           │           │
       P_V18       P_dino      P_conv
         │           │           │
         │       Bilinear    Bilinear
         │       (to 1024)   (to 1024)
         │           │           │
         ▼           ▼           ▼
   ┌───────────────────────────────────┐
   │   Convex Combination (Simplex)    │
   │ P_ens = w18*P_V18 + wd*Pd + wc*Pc │
   └───────────────────────────────────┘
                     │
                     ▼
   ┌───────────────────────────────────┐
   │ Log-Space Calibration & Offset    │
   │ LogP = log(P_ens + 1e-10) + b     │
   └───────────────────────────────────┘
                     │
                     ▼
   [Final Argmax Segmentation Mask (1024x1024)]
```

### Detailed Adapter Specifications:
1. **Restoration Normalization:**
   All models are evaluated on the identical restored image generated by the V18 Refiner (`ConditionedRestorationRefinerV18` + `table_candidate_B_fold_{k}.pt`). This guarantees 100% optical consistency across backbones.
2. **Spatial Resampling:**
   - DINOv2: Downsamples restored image to $518 \times 518$, evaluates logits, applies softmax, and upsamples probability map to $1024 \times 1024$ via bilinear interpolation (`align_corners=False`).
   - ConvNeXt: Downsamples restored image to $512 \times 512$, evaluates logits, applies softmax, and upsamples probability map to $1024 \times 1024$ via bilinear interpolation.
3. **Probability-Level Fusion (Not Logits):**
   Uncalibrated logit ranges differ substantially across models. Softmax transforms each model's output into a probability distribution over the 5-simplex before convex combination:
   $$P_{\text{ens}} = w_{18} P_{\text{V18}} + w_{\text{dino}} P_{\text{dino}} + w_{\text{conv}} P_{\text{conv}}$$
   where $w_{18} + w_{\text{dino}} + w_{\text{conv}} = 1.0$ and $w_i \ge 0$.
4. **Horizontal-Flip TTA:**
   Horizontal flip TTA is applied independently within each model branch before spatial upsampling:
   $$P_m = \frac{1}{2} \left( \text{Softmax}(L_m(x)) + \text{Flip}\left(\text{Softmax}(L_m(\text{Flip}(x)))\right) \right)$$
5. **Post-Fusion Calibration:**
   Logit calibration offsets ($b_3 = +0.15, b_4 = +0.33$, validated in EXP-V19-P0) are applied strictly **post-fusion** in log-probability space:
   $$L_{\text{cal}} = \log(P_{\text{ens}} + 10^{-10}) + [0, 0, 0, 0.15, 0.33]$$
   $$\hat{Y} = \arg\max_{c \in \{0..4\}} L_{\text{cal}, c}$$

---

## 6. Smallest Useful Experiment: Nested Cross-Fitting Protocol

To prevent meta-overfitting, the ensemble weights must NOT be selected on the same held-out fold used for final evaluation. A **nested 5-fold cross-fitting protocol** is pre-registered:

### Nested Cross-Fitting Scheme
For each outer fold $k \in \{0, 1, 2, 3, 4\}$ (800 validation images):
1. **Inner Tuning:** Compute predictions across the remaining 4 training folds $\{j \ne k\}$ (3,200 images).
2. **Weight Selection:** Find the weight tuple $w^*(k) = (w_{18}, w_{\text{dino}}, w_{\text{conv}})$ that maximizes $Q_{\text{seg}}$ on the 3,200 inner images across the candidate grid.
3. **Outer Evaluation:** Apply $w^*(k)$ to predict outer fold $k$.
4. **Pooled Metric:** Aggregate outer fold predictions across all 20 scenes (4,000 images) to compute the completely unbiased cross-fitted score.

### Pre-Registered Candidate Weight Grid

| Configuration ID | $w_{18}$ (V18) | $w_{\text{dino}}$ (DINOv2) | $w_{\text{conv}}$ (ConvNeXt) | Design Rationale |
| :---: | :---: | :---: | :---: | :--- |
| **C0 (Control)** | **1.00** | **0.00** | **0.00** | **V18 Baseline Control (Reference)** |
| **C1** | 0.80 | 0.20 | 0.00 | Mild DINOv2 feature smoothing |
| **C2** | 0.70 | 0.30 | 0.00 | Moderate DINOv2 feature smoothing |
| **C3** | 0.85 | 0.00 | 0.15 | ConvNeXt boundary injection |
| **C4** | 0.70 | 0.20 | 0.10 | Balanced Tri-Ensemble |
| **C5** | 0.60 | 0.25 | 0.15 | High-Diversity Tri-Ensemble |
| **C6** | 0.50 | 0.35 | 0.15 | Equal Weighting between V18 and V16/V17 |

---

## 7. Explicit GO / NO-GO Criteria

The candidate experiment will be evaluated against strict pre-registered statistical gates:

```
========================================================================================================
                                    PRE-REGISTERED SUCCESS CRITERIA
========================================================================================================
Gate Name               | Metric                          | Baseline | Required Gate | Status Rule
------------------------|---------------------------------|----------|---------------|------------------
Primary Success Gate    | Cross-Fitted Delta Q_seg        | 0.43645  | >= +0.01000   | PROCEED if Met
Secondary Support Gate  | Cross-Fitted Delta mIoU         | 0.59441  | >= +0.00800   | Confirmatory
Class 4 Safety Floor    | Macro-Fauna IoU (Class 4)       | 0.34016  | >= 0.34016    | NO REGRESSION
Class IoU Safety Floors | Maximum allowable drop per class| N/A      | <= -0.00500   | REJECT if Dropped
========================================================================================================
```

### Empirical Grounding & Target Reality Check
- **Can Ensembling Reach 0.800+ Alone?**
  **NO.** Current V18 official score is 0.749. A gap of $+0.051$ Composite points requires an increase of $+0.085$ in $Q_{\text{seg}}$ on the official test set. In historical cross-validation, ensembling DINOv2 and ConvNeXt yielded $+0.0185$ $Q_{\text{seg}}$ locally and $+0.009$ on Dev.
  Therefore, an ensemble is realistically expected to deliver **$+0.010$ to $+0.015$ $Q_{\text{seg}}$** (moving Dev Composite from **0.749 to ~0.755–0.760**). It represents a valuable, rigorous incremental improvement, but does NOT by itself achieve the 0.800 threshold. False promises of instant 0.800+ via ensembling alone are empirically rejected.

---

## 8. Resource & Compute Feasibility

- **GPU Hardware:** NVIDIA GeForce RTX 5090 (31.84 GB VRAM)
- **Peak VRAM Consumption:**
  - V18 Dual-Encoder forward: ~7.2 GB
  - DINOv2-Large forward: ~2.5 GB
  - ConvNeXt-Base FPN forward: ~1.2 GB
  - **Total Concurrent VRAM:** **~10.9 GB (34.2% of 31.84 GB)**
- **Runtime Budget:**
  - 6 forward passes per image (2 passes per model family $\times$ 3 models)
  - Estimated throughput: ~45 ms per image
  - 800 images per fold: ~260 seconds (~4.3 minutes)
  - Full 5-fold evaluation (4,000 images): **~22 minutes total wall-clock time**

---

## 9. Baseline Archive Hash Re-Verification

Following completion of all analysis and artifact generation, the baseline submission archive was re-verified:
- **Target File:** `deploy/v18/submission_v18_dual_encoder_sota.zip`
- **Pre-Audit SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Post-Audit SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Integrity:** **IDENTICAL & PRESERVED**

---

## 10. Summary & Recommendation

1. **Conclusion:** **PASS** for the fold-safe Tri-Ensemble (V18 + V16-DINOv2 + V17-ConvNeXt).
2. **Missing Components:** 5-fold checkpoints for V17 Specialists and Phase 2 SegFormer do not exist. Attempting to use them would constitute fatal validation data leakage and is strictly prohibited.
3. **Next Smallest Experiment:** If approved by the user, execute `tools/v19/run_ensemble_benchmark.py` evaluating the nested cross-fitted candidate grid across the 4,000 out-of-fold images to empirically verify whether $\Delta Q_{\text{seg}} \ge +0.01000$ is achieved.
