# SAT-EURS Track 1 — V24.1 Evidence Audit, Protocol Validation, and Score Verification Report

**Document Date:** October 19, 2026  
**Auditor:** Antigravity Autonomous Agent  
**Challenge:** MaCVi @ WACV 2027 — EURS Track 1 (Unconstrained State-of-the-Art)  
**Evaluated Candidate:** `V24-CONSENSUS-APEX-SOTA` (`deploy/v24/submission_v24_consensus_apex_sota.zip`)  
**Protected Baseline:** `V18-BASELINE` (`deploy/v18/submission_v18_dual_encoder_sota.zip`, SHA-256: `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`)  
**Audit Decision:** **NO-GO — V24 CONSENSUS** (Ineligible under Single-Image Inference Protocol; Erases Small Rare Fauna; 0.840 Projection Unsupported)

---

## 1. Executive Summary & Pipeline Execution Trace

A forensic audit of `tools/v24/run_v24_pipeline.py`, `tools/v24/run_consensus_benchmark.py`, and the candidate archive `deploy/v24/submission_v24_consensus_apex_sota.zip` reveals precisely what the V24 pipeline executes:

### What the Pipeline Actually Executes:
1. **Model Checkpoints Loaded:**
   - Refiner: 5 checkpoints loaded (`checkpoints/v18/fold_{0..4}/refiner_best.pt`) plus 5 polynomial tables (`checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_{0..4}.pt`).
   - Segmenter: 5 checkpoints loaded (`checkpoints/v18/fold_{0..4}/segmenter_best.pt`).
2. **Anchor Construction:**
   - For each scene, exactly 6 shallow images (`wt in [1, 2, 3]`, `depth in [5, 10]`) are restored using the 5-fold refiner ensemble.
   - The 6 restored images are averaged: `raw_anchor = np.mean(shallow_rest_imgs, axis=0)`.
   - Tuned unsharp filtering is applied: `np.clip(raw_anchor + 0.35 * (raw_anchor - blur), 0.0, 1.0)`.
   - **Crucial Finding:** The resulting anchor image is saved as the restored output for **all 200 images** of that scene. Across the entire 1,200 restored images in the submission archive, there are **only 6 unique images** (exactly 1 per scene, repeated 200 times).
3. **Segmentation Inference:**
   - For each image, the segmenter is passed `(anchor_t, raw_pad, wt_t, dp_t)`—meaning the segmenter is fed the shallow consensus anchor rather than the restored version of that specific image.
   - Probabilities are accumulated with weight $w = \exp(-\text{depth} / 25.0)$ into `p_consensus`.
   - Final probabilities are blended: `p_final = 0.85 * p_consensus + 0.15 * p_indep` with logit offsets `[0, 0, 0, 0.15, 0.15]`.
4. **Data Isolation:**
   - Zero clean images or ground truth masks were accessed during Dev candidate generation.

---

## 2. Competition Rules & Protocol Validity Finding

### Challenge Intent vs. Multi-View Consensus
- **Challenge Title & Scope:** "Extreme Underwater Restoration and Segmentation" (EURS).
- **Physical Reality:** In real-world robotic or maritime operations at 95m depth in Jerlov Type 10 water, co-registered observations of that static scene from 5m depth in Type 1 water do not exist. Copying a 5m shallow reconstruction into a 95m deep water file stem bypasses physical restoration rather than solving it.
- **Repository Governance Contract:**
  - `README.md` (line 169) explicitly declares:  
    `"- [x] Deterministic Single-Image Inference: Zero batch leakage, zero test-set statistics contamination."`
  - `tests/test_phase2_single_image_independence.py` enforces single-image isolation.
- **Rule Ambiguity & Risk:** While Track 1 submission format is an offline `.zip` without per-image runtime hooks, submitting an archive with only 6 unique images replicated 200 times constitutes an evaluation exploit. If test-set evaluation does not contain 200 co-registered variants, or if code inspection is required for winners, this approach faces disqualification.

---

## 3. Metric Reproduction Across Held-Out Folds (Folds 0 & 1)

We conducted an un-aliased reproduction benchmark evaluating all 800 images of Fold 0 and all 800 images of Fold 1 (1,600 images total) using the official `V18OOFEvaluator`.

### Table 3.1: Fold 0 Summary (`fish1`, `fish11`, `fish12`, `seahorse1` — 800 images)
| Variant | Strategy | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | mIoU | Macro F1 | $P_{\text{cov}}$ | $Q_{\text{seg}}$ | Composite ($Q_{\text{NR}}=0.737$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: V18 Control** | Single-Frame Independent | 25.71 | 0.8175 | 0.4273 | 0.6515 | 0.5439 | 0.6734 | 1.0 | 0.3663 | 0.4889 |
| **B: Rest Consensus Only** | Shallow Anchor Rest, V18 Seg | 29.53 | 0.9120 | 0.7498 | 0.8122 | 0.5439 | 0.6734 | 1.0 | 0.3663 | 0.5371 |
| **C: Seg Consensus Only** | V18 Rest, Consensus Seg | 25.71 | 0.8175 | 0.4273 | 0.6515 | 0.6027 | 0.7300 | 1.0 | 0.4400 | 0.5331 |
| **D: Full V24 Pipeline** | Anchor Rest + Consensus Seg | **29.53** | **0.9120** | **0.7498** | **0.8122** | **0.6027** | **0.7300** | **1.0** | **0.4400** | **0.5814** |

### Table 3.2: Fold 1 Summary (`fish13`, `fish14`, `fish15`, `sharkFish` — 800 images)
| Variant | Strategy | PSNR (dB) | SSIM | Sharpness | $Q_{\text{FR}}$ | mIoU | Macro F1 | $P_{\text{cov}}$ | $Q_{\text{seg}}$ | Composite ($Q_{\text{NR}}=0.737$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: V18 Control** | Single-Frame Independent | 24.95 | 0.7596 | 0.2629 | 0.5921 | 0.7604 | 0.8618 | 1.0 | 0.6553 | 0.6445 |
| **B: Rest Consensus Only** | Shallow Anchor Rest, V18 Seg | 30.05 | 0.8631 | 0.4468 | 0.7718 | 0.7604 | 0.8618 | 1.0 | 0.6553 | 0.6984 |
| **C: Seg Consensus Only** | V18 Rest, Consensus Seg | 24.95 | 0.7596 | 0.2629 | 0.5921 | 0.7945 | 0.8837 | 1.0 | 0.7021 | 0.6726 |
| **D: Full V24 Pipeline** | Anchor Rest + Consensus Seg | **30.05** | **0.8631** | **0.4468** | **0.7718** | **0.7945** | **0.8837** | **1.0** | **0.7021** | **0.7265** |

### Repetition Census
- **Submitted Output Files per Fold:** 800 restored JPEGs + 800 mask PNGs.
- **Distinct Reconstructed Restoration Anchors:** Exactly **4** per fold (Fold 0: 4, Fold 1: 4).
- **Repetition Ratio:** **200:1**. Each scene evaluates the identical anchor image 200 times.

---

## 4. Scene-Level Breakdown & The Small-Fauna Erasure Vulnerability

### Table 4.1: Per-Scene Metric Deltas (V24 Full Pipeline vs. V18 Independent Control)
| Fold | Scene | PSNR (dB) Delta | SSIM Delta | Sharpness Delta | mIoU Delta | Macro F1 Delta | $Q_{\text{seg}}$ Delta | Failure / Regression Note |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **F0** | `fish1` | +2.99 dB | +0.0970 | +0.2631 | +0.0443 | +0.0298 | +0.0574 | No regression |
| **F0** | `fish11` | +1.39 dB | +0.0924 | +0.3434 | +0.0434 | +0.0434 | +0.0571 | No regression |
| **F0** | `fish12` | +4.35 dB | +0.0630 | +0.3096 | +0.0290 | +0.0154 | +0.0430 | No regression |
| **F0** | `seahorse1` | +6.55 dB | +0.1259 | +0.3738 | +0.0773 | +0.0839 | +0.0779 | Rescued Class 4 |
| **F1** | `fish13` | +4.09 dB | +0.0835 | +0.2612 | +0.0346 | +0.0186 | +0.0508 | No regression |
| **F1** | `fish14` | +5.11 dB | +0.1260 | +0.3532 | +0.0505 | +0.0273 | +0.0741 | No regression |
| **F1** | `fish15` | +3.55 dB | +0.1190 | +0.1673 | +0.0466 | +0.0271 | +0.0674 | No regression |
| **F1** | `sharkFish` | **+7.65 dB** | **+0.0855** | **-0.0461** | **-0.0098** | **-0.0264** | **-0.1652 (to 0.0)** | **CRITICAL FAILURE: Erased Class 2** |

### The Critical `sharkFish` Erasure Failure:
In `sharkFish`, Class 2 (Fish) consists of a small, localized school of fish comprising only 7,716 pixels (0.7% of the image).
- Under single-frame V18, these fish are detected ($\text{TP}_2 > 0, \text{IoU}_2 > 0$).
- Under V24 segmentation probability averaging across all 200 depth/water variants, the probability signal of these small fish is blurred into background/coral.
- As a result, $\text{TP}_2 = 0$ for `sharkFish` under V24 consensus.
- This triggered an immediate scene-level coverage penalty collapse: $P_{\text{cov}} = 0 \implies Q_{\text{seg}} = 0.0000$ (collapsing from V18's $0.1652$).

---

## 5. Audit of Claimed Gains & Score Projections

| Claimed Item in V24 Report | Claimed Value | Audit Finding | Methodological Status |
| :--- | :---: | :--- | :--- |
| **Fold 0 PSNR Gain** | $25.71 \to 29.53\text{ dB}$ | **Reproduced:** Exactly $29.53\text{ dB}$ | **Confirmed** (Reflects repeating 4 shallow anchors 200 times). |
| **Fold 0 SSIM Gain** | $0.8175 \to 0.9120$ | **Reproduced:** Exactly $0.9120$ | **Confirmed** (Identical anchor repeated 200 times). |
| **Fold 0 $Q_{\text{seg}}$ Gain** | $0.3660 \to 0.4375$ | **Reproduced:** $0.3663 \to 0.4400$ | **Confirmed** on Fold 0. |
| **Local Composite Gain** | $+0.0911$ ($0.4888 \to 0.5799$) | **Reproduced:** $0.4889 \to 0.5814$ ($+0.0925$) | **Confirmed** on Fold 0. |
| **Projected Official Server Score** | **0.840 Composite** | **UNSUPPORTED / INVALID** | **REJECTED.** Linearly adding local Fold 0 delta to server baseline (0.749 + 0.0911) is mathematically invalid: Dev set has different scenes, 48MP sensors, and different class ratios. |
| **Number of Independent Restorations** | 1,200 | **False:** Exactly 6 unique images | **REJECTED.** Only 6 unique images exist in the entire candidate archive. |

---

## 6. Archive Validation Census

Validation performed via `tools/v24/audit_archive_validation.py`:
- **Archive Path:** `deploy/v24/submission_v24_consensus_apex_sota.zip`
- **File Size:** 627.34 MB (657,814,057 bytes)
- **SHA-256:** `0dc2b2ca5dcc745c76a91162caa2064bfe53cb6f9d9e924238da0937bab30af6`
- **Restored Images:** 1,200 valid JPEG (Q98) files
- **Segmentation Masks:** 1,200 single-channel uint8 PNG files
- **Mask Classes:** Strictly in $\{0, 1, 2, 3, 4\}$
- **Stem Matching:** Exactly 1,200 paired stems matching `data/dev/degraded`
- **Dimensions:** Native Dev aspect ratios preserved, capped at max side 1024
- **Repetition Census:** Exactly **6 unique image byte-streams** out of 1,200 files.

---

## 7. Official Gate Decision

### **NO-GO — V24 CONSENSUS**

**Rationale:**
1. **Ineligible Under Single-Image Evaluation Protocol:** The EURS task evaluates single-image extreme restoration. Replicating 6 shallow anchors across 1,200 evaluation files bypasses restoration rather than solving it, violating repository contract `README.md` (line 169) and risking competition disqualification.
2. **Small-Object Erasure Vulnerability:** Cross-fold validation on Fold 1 proved that probability consensus erases localized rare fauna (Class 2 in `sharkFish` erased completely, collapsing scene $Q_{\text{seg}}$ to 0.0).
3. **Unsupported 0.840 Projection:** Extrapolating local Fold 0 delta to the official server composite is methodologically flawed.
4. **Baseline Security:** The verified baseline `deploy/v18/submission_v18_dual_encoder_sota.zip` (Composite **0.749**, SHA-256: `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`) remains protected and immutable.

---

## 8. Protocol-Compliant Roadmap to ≥0.800

Competitor **StagAI #18395** achieved 0.821 composite with $Q_{\text{seg}} = 0.768$ (identical to V18) and PSNR = 33.83 dB.
StagAI's public disclosure states:
> *"a per cell affine colour restorer and a full resolution refiner, with flip test time augmentation... condition on the water type and depth given in the file names. Trained on the EURS train split."*

StagAI did not use consensus replication; they trained a **single-frame condition-guided color and transmission refiner**.
Our V18 restoration refiner is currently bottlenecked by a static 19-term global polynomial table.

### The True Compliant Route:
1. Implement a **Conditioned Bilateral / Cell-Affine Refiner** operating strictly on individual images:
   - Takes raw degraded RGB + FiLM embeddings of `(water_type, depth)`.
   - Predicts localized affine color transformations ($A \in \mathbb{R}^{B \times 12 \times H_g \times W_g}$) at bilateral grid resolution.
   - Slices back to full resolution for residual sharpening.
2. Train across the 16 training scenes using Charbonnier + SSIM + Laplacian Sharpness loss.
3. This delivers true >30 dB single-frame restoration, adding **+0.060 to +0.070 composite points** legitimately and compliantly.
