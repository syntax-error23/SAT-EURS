# SAT-EURS Track 1 — Forensic Investigation Report: V29 Regression Root Cause & Final Submission Strategy

**Author:** Senior Computer Vision Research & Evaluation Auditor  
**Date:** October 26, 2026  
**Challenge:** MaCVi @ WACV 2027 EURS Track 1  
**Lead Artifacts Audited:**
- **Baseline V28-A (Official #19096):** `deploy/v28/submission_v28a_fivefold_ablation.zip` (SHA-256 `471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39`, Official Composite: **0.634**)
- **Candidate V29 (Official #19214):** `deploy/v29/submission_v29_dual_backbone_sota.zip` (SHA-256 `b73155c757376c721b7cfff9ee7f160e050a4237cf8c01f5cb2943405599d6c3`, Official Composite: **0.597**)

---

## 1. Executive Summary & Final Recommendation

### 1.1 Executive Summary
Official submission **#19214 (V29-Dual-Backbone-SOTA)** scored **0.597**, suffering a severe **-0.037 composite drop** from production champion **V28-A (0.634)**. The degradation was concentrated entirely in semantic segmentation:
- **Composite Score:** $0.634 \to 0.597$ (**-0.037**)
- **Segmentation Quality ($Q_{\text{seg}}$):** $0.570 \to 0.507$ (**-0.063**)
- **mIoU:** $0.704 \to 0.652$ (**-0.052**)
- **Macro F1:** $0.810 \to 0.779$ (**-0.031**)
- **Restoration Quality:** 100% byte-for-byte identical (PSNR 25.48 dB, SSIM 0.829, Sharpness 0.973, MAE: 0.000000).

Through end-to-end code tracing, pixel-level archive diffing, and controlled out-of-fold (OOF) re-evaluation, the exact root cause has been conclusively established:

> **Primary Root Cause (CONFIRMED):**  
> V29 introduced a dual-backbone probability ensemble that assigned **60% voting weight** to a fine-tuned `SegFormer-B5` model (`checkpoints/v29/segformer_b5_full20_best.pt`), marginalizing the established 5-fold DINOv3 ensemble down to 40%.  
> While SegFormer-B5 appeared superior during local training ($Q_{\text{seg}} = 0.9320$), this score was a **catastrophic artifact of in-sample training memorization and partial-split metric truncation** ($max\_eval = 160$ on `fish1`, where absent classes vacuously received $1.0000$ IoU).  
> When evaluated on genuine held-out validation scenes, SegFormer-B5's true out-of-fold $Q_{\text{seg}}$ collapsed to **0.2721** (vs. **0.5694** for DINOv3), suffering from severe rare-class hallucinations and condition blindness in turbid/deep water (water type 10, depth 80–119m, where 36.21% of all mask pixels flipped).

### 1.2 Final Submission Recommendation
1. **RETAIN BASELINE V28-A AS PRODUCTION SUBMISSION:**  
   Candidate `deploy/v28/submission_v28a_fivefold_ablation.zip` (SHA-256 `471bf9b9...`) remains our highest performing, rigorously verified submission (0.634).
2. **DO NOT DEPLOY V29 OR ANY SEGFORMER-B5 ENSEMBLE:**  
   SegFormer-B5 lacks the condition-aware FiLM neck required for optical depth invariant underwater segmentation and suffers severe generalization collapse on held-out scenes.
3. **PRESERVE ALL CHECKPOINTS AND RUNS:**  
   The baseline archive and inference lineage are locked and verified.

---

## 2. Verified V28-A and V29 Architecture & Checkpoint Lineage

A critical correction to prior reporting: **The foundational V18/V27/V28 segmentation architecture is DINOv3, not DINOv2.**

```
                               MODEL LINEAGE HIERARCHY
                               
      Meta DINOv3 ViT-L/16           Official DINOv3 ConvNeXt-Base
   ('vit_large_patch16_dinov3')       ('convnext_base.dinov3_lvd1689m')
               │                                   │
               └─────────────────┬─────────────────┘
                                 ▼
                     EURSAPEXV18Segmenter
              (ConditionEncoder + V18UPerNetNeck)
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
     [V28-A Pipeline]                             [V29 Pipeline]
 5-Fold DINOv3 Ensemble                    40% 5-Fold DINOv3 Ensemble
 (Fold 0: V27 Targeted                     + 60% SegFormer-B5 Full-20
  Folds 1-4: Baseline V18)                 + 0.05 Anti-Trap Gate
         │                                               │
         ▼                                               ▼
 Official Test Score: 0.634                      Official Test Score: 0.597
```

### 2.1 Model & Checkpoint Lineage Table

| Submission | Component | Exact Architecture | Underlying Backbone(s) | Pretrained Source | Checkpoint Path | Training Scene Scope |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V28-A** | Refiner | `ConditionedRestorationRefinerV18` | RSUIGM Physical Prior + Zero-Residual ConvNet | Calibrated 20-scene table (`tbl19_20sc_scale_consistent.pt`) | `checkpoints/v18/fold_{0..4}/refiner_best.pt` | 16 train scenes / fold (Fold-disjoint) |
| **V28-A** | Segmenter (Fold 0) | `EURSAPEXV18Segmenter` | Dual-Encoder: DINOv3 ViT-L/16 + ConvNeXt-Base | timm `dinov3_lvd1689m` | `checkpoints/v27/segmenter_v27_fold0_best.pt` | Fold 0 train scenes (16 scenes; val `fish1,11,12,seahorse1` excluded) |
| **V28-A** | Segmenter (Folds 1–4)| `EURSAPEXV18Segmenter` | Dual-Encoder: DINOv3 ViT-L/16 + ConvNeXt-Base | timm `dinov3_lvd1689m` | `checkpoints/v18/fold_{1..4}/segmenter_best.pt` | Folds 1–4 train scenes (Fold-disjoint) |
| **V29** | Refiner | Reused V27/V28-A Restored Images | Byte-for-byte identical | — | `deploy/v27/track1_test_outputs/raw_unpadded/restored` | Identical to V28-A |
| **V29** | Segmenter Path A | `EURSAPEXV18Segmenter` (5 folds) | Dual-Encoder: DINOv3 ViT-L/16 + ConvNeXt-Base | timm `dinov3_lvd1689m` | Same 5 checkpoints as V28-A | Same as V28-A |
| **V29** | Segmenter Path B | `SegformerForSemanticSegmentation` | MiT-B5 + MLP Decode Head | HuggingFace `EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024` | `checkpoints/v29/segformer_b5_full20_best.pt` | **All 20 Scenes (4,000 images)** (No held-out split!) |

---

## 3. Exact Code & Configuration Differences

A line-by-line inspection of [`tools/v28/run_v28a_ablation.py`](file:///c:/Projects/SAT-EURS/tools/v28/run_v28a_ablation.py) vs. [`tools/v29/run_v29_candidate.py`](file:///c:/Projects/SAT-EURS/tools/v29/run_v29_candidate.py) reveals the following operational divergences:

| Pipeline Stage | V28-A Baseline (`run_v28a_ablation.py`) | V29 Candidate (`run_v29_candidate.py`) | Status | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Restoration Inputs** | 5-Fold Refiners / V27 cache | Reused V27 cache directly | **VERIFIED IDENTICAL** | 100% byte-for-byte identical; 0.000000 MAE. |
| **Segmentation Ensemble**| 5-Fold DINOv3 models, uniform $1/5$ weight | 5-Fold DINOv3 ($40\%$) + SegFormer-B5 ($60\%$) | **VERIFIED DIVERGENT** | SegFormer given majority control. |
| **Condition Encoding** | Water type & depth encoded via FiLM modulation | SegFormer-B5 receives **NO condition vector** | **VERIFIED DIVERGENT** | Severe collapse under high turbidity/depth. |
| **Rare-Class Gating** | None (pure continuous logit offsets) | $\text{threshold} = 0.05$ clamp to $-\infty$ if $P_{\max} < 0.05$ | **VERIFIED DIVERGENT** | Erased valid small macrofauna detections. |
| **Logit Offsets** | $[0.0, 0.0, 0.0, 0.10, 0.25]$ | $[0.0, 0.0, 0.0, 0.10, 0.25]$ | **VERIFIED IDENTICAL** | Identical post-fusion offsets. |
| **TTA Strategy** | Horizontal flip TTA across all 5 models | Horizontal flip TTA on DINOv3 and SegFormer | **VERIFIED IDENTICAL** | Both use $0.5 \times (P_{\text{orig}} + P_{\text{flip}})$. |
| **Resizing Engine** | `tools/resize_for_submission.py` | `tools/resize_for_submission.py` | **VERIFIED IDENTICAL** | Exact same native aspect-preserved output. |

---

## 4. Archive-Level Image and Mask Comparison Results

### 4.1 Restored Images Verification
- **Total Test Images:** 1,200
- **Byte-for-byte identical files:** 1,200 / 1,200 (**100.0%**)
- **Mean Absolute Error (MAE):** 0.000000
- **Maximum Pixel Difference:** 0.0000
- **Leaderboard Parity:** PSNR 25.48, SSIM 0.829, Sharpness 0.973 on both submissions.

### 4.2 Segmentation Masks Global Comparison
- **Total Pixels Evaluated:** 869,580,800
- **Identical Pixels:** 810,327,086 (**93.19%**)
- **Changed Pixels:** 59,253,714 (**6.81%**)

### 4.3 5x5 Class Transition Matrix (V28-A vs. V29)

```
                       PIXEL CLASS TRANSITION MATRIX
Row = V28-A Predicted Class | Column = V29 Predicted Class
-----------------------------------------------------------------------------------------------------------
V28-A \ V29       | Background    | Coral         | Fish          | Elasmobranch  | Macrofauna    | Total V28-A
-----------------------------------------------------------------------------------------------------------
0: Background     | 465,451,072   |  17,163,093   |     528,684   |     228,441   |      32,070   | 483,403,360
1: Coral          |  27,562,302   | 296,654,449   |     736,118   |       8,959   |      58,989   | 325,020,817
2: Fish           |   4,731,587   |   1,094,058   |  11,355,910   |      33,714   |       6,154   |  17,221,423
3: Elasmobranch   |   1,403,488   |       7,462   |     603,163   |  22,692,045   |     226,338   |  24,932,496
4: Macrofauna     |     317,465   |   1,837,822   |   2,668,822   |       4,985   |  14,173,610   |  19,002,704
-----------------------------------------------------------------------------------------------------------
Total V29         | 499,465,914   | 316,756,884   |  15,892,697   |  22,968,144   |  14,497,161   | 869,580,800
Net Change        | +16,062,554   |  -8,263,933   |  -1,328,726   |  -1,964,352   |  -4,505,543   |
Net Change %      | +3.32%        | -2.54%        | -7.72%        | -7.88%        | -23.71%       |
```

### 4.4 Key Statistical Observations
1. **Severe Macrofauna Deficit:** V29 lost **4,505,543 Macrofauna pixels (-23.71%)**, with 2.67M converted to Fish and 1.84M converted to Coral.
2. **Fish Pixel Erosion:** Fish lost **4.73M pixels to Background** (-7.72% net).
3. **Coral-to-Background Bleed:** 27.56M Coral pixels in V28-A were flipped to Background in V29, while 17.16M Background pixels flipped to Coral, indicating significant boundary distortion along benthic structures.
4. **Scene Concentration:** Two scenes (`test04` and `test11`) account for **46,333,098 of the 59,253,714 changed pixels (78.20% of all changes in the entire test set)**.

### 4.5 Degradation-Stratified Impact
- **Water Type 10:** 11.91% pixels changed (vs. 5.3%–6.8% for water types 1–9).
- **Depth 80–99m:** 28.91% pixels changed in WT 10.
- **Depth 100–119m:** **36.21% pixels changed in WT 10**.
- SegFormer-B5 has no optical depth conditioning; under severe attenuation and green/blue scatter, its predictions degraded severely.

---

## 5. Root Causes: Forensic Audit & Classification

| Suspected Cause | Classification | Supporting Artifact / Measurement | Test / Evidence |
| :--- | :--- | :--- | :--- |
| **1. SegFormer-B5 Out-of-Fold Generalization Failure** | **CONFIRMED** | Held-out OOF benchmark (`reports/v29_regression_visuals/oof_comparative_experiment.json`) | SegFormer-B5 scored **0.2721 $Q_{\text{seg}}$** on held-out scenes vs. **0.5694** for DINOv3. |
| **2. In-Sample Evaluation Metric Leakage** | **CONFIRMED** | `tools/train_v29_segformer_full20.py` and log `task-6187.log` | Training on all 20 scenes achieved $Q_{\text{seg}} = 0.9320$ via memorization, misleading the deployment decision. |
| **3. Metric Calculation Truncation ($max\_eval = 160$)** | **CONFIRMED** | `train_v29_segformer_b5.py:196` | Evaluator truncated at 160 images (`fish1` only); absent classes 3 and 4 vacuously received $1.0000$ IoU, masking rare-class failure. |
| **4. Lack of Physical Optical Condition Encoding in SegFormer** | **CONFIRMED** | `scratch/analyze_degradation_breakdown.py` | WT 10 / depth >80m showed 36.21% pixel divergence. SegFormer lacks the FiLM condition encoder present in DINOv3. |
| **5. Anti-Trap Presence Gating Suppressing Valid Detections** | **CONFIRMED** | `scratch/evaluate_oof_comparison.py` | On `seahorse1`, gating at $\tau = 0.05$ dropped score from $0.9351 \to 0.5877$ (-37.1% relative drop). |
| **6. Checkpoint Loading or State Dict Mismatch** | **REJECTED** | `scratch/inspect_checkpoints.py` | State dicts loaded cleanly with zero missing/unexpected keys across all backbones. |
| **7. Resizing or Geometry Packaging Regression** | **REJECTED** | `scratch/audit_v29_candidate.py` | Native aspect-preserved geometry, uint8 $\{0..4\}$ values, and 2,400 file counts verified 100% compliant. |
| **8. Restoration Pipeline Regression** | **REJECTED** | `scratch/compare_v28a_v29_detailed.py` | Restored images are 100% byte-for-byte identical (MAE 0.000000). |

---

## 6. Validation Methodology and Leakage Audit

### 6.1 The Disjoint Scene Protocol
The SAT-EURS Track 1 dataset consists of 20 distinct underwater scenes, each rendered under 200 synthetic degradation conditions (water types 1–10, depths 5–100m) for a total of 4,000 images.
- **Critical Leakage Rule:** An algorithm must NEVER be evaluated on a degradation variant of a scene that was present in its training set. Random image splitting causes severe leakage, as the network memorizes the static background reef topography.
- **Manifest:** `splits/manifest_5fold.json` strictly enforces 16 train scenes and 4 validation scenes per fold, with zero scene overlap.

### 6.2 The In-Sample Fallacy of Full-20 Training
In EXP-02-Full20, SegFormer-B5 was trained on all 20 scenes (4,000 images). Its reported validation score of **0.9320** was calculated on `fish1` — a scene it was actively training on. This was pure training memorization, completely unrepresentative of unseen test scenes (`test04`, `test09`, `test10`, `test11`, `test19`, `test20`).

---

## 7. Experiment Table: Controlled Held-Out Validations

All experiments below were executed using the **identical evaluation harness**:
- Exact held-out scenes (`fish1`, `fish11`, `fish12`, `seahorse1`)
- Native 1024 geometry with 32-pad and official unpadding
- Horizontal-flip TTA
- Calibrated offset vector $[0.0, 0.0, 0.0, 0.10, 0.25]$

| Experiment ID | Model / Configuration | $Q_{\text{seg}}$ | $\text{mIoU}$ | Macro F1 | Class 0 (Back) | Class 1 (Coral) | Class 2 (Fish) | Class 3 (Elasmo) | Class 4 (Macro) | `seahorse1` $Q_{\text{seg}}$ | Official Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-00-V18** | DINOv3 Fold 0 Baseline | 0.5434 | 0.6864 | 0.7918 | 0.777 | 0.806 | 0.434 | 1.000 | 0.414 | 0.3120 | 0.612 |
| **EXP-00-V28A**| **DINOv3 Fold 0 Targeted (Champion)** | **0.5694** | **0.7047** | **0.8080** | **0.774** | **0.814** | **0.529** | **1.000** | **0.406** | **0.3036** | **0.634** |
| **EXP-01-V28B**| DINOv3 Fold 0 (#19185 ckpt) | 0.5898 | 0.7184 | 0.8209 | 0.777 | 0.809 | 0.555 | 1.000 | 0.451 | 0.3180 | 0.622 (offset 0.35) |
| **EXP-02-SF0** | SegFormer-B5 Fold 0 OOF | 0.2721 | 0.4827 | 0.5637 | 0.741 | 0.829 | 0.716 | **0.000** | **0.127** | **0.1147** | — |
| **EXP-02-SFF** | SegFormer-B5 Full-20 (In-Sample) | 0.9376* | 0.9583* | 0.9785* | 0.969* | 0.973* | 0.929* | 1.000* | 0.921* | 0.9351* | — |
| **EXP-03-V29** | **V29 Candidate ($0.60$ SF + $0.40$ DINOv3 + gate)** | **0.9395***| **0.9595***| **0.9791***| **0.970***| **0.973***| **0.928***| **1.000***| **0.927***| **0.5877***| **0.597 (OFFICIAL)** |
| **EXP-04-B35** | Pure Blend (0.35 SegF0 + 0.65 DINOv3) | 0.3132 | 0.5109 | 0.6130 | 0.780 | 0.815 | 0.550 | **0.000** | 0.409 | 0.2933 | — |
| **EXP-04-B20** | Pure Blend (0.20 SegF0 + 0.80 DINOv3) | 0.5765 | 0.7099 | 0.8121 | 0.783 | 0.816 | 0.537 | 1.000 | 0.413 | 0.3029 | Expected ~0.634 |

*\*Note: Scores marked with asterisk (\*) reflect in-sample training scene memorization. When evaluated on unseen test data, the model collapsed to 0.597.*

---

## 8. Best Implementable Candidate & Safety Assessment

### 8.1 Can Any Variant Safely Beat V28-A?
- Blending SegFormer-B5 into DINOv3 at $\ge 35\%$ causes immediate collapse on Class 3 ($IoU \to 0.000$), because SegFormer hallucinates elasmobranchs on absent scenes.
- While a very small blend (e.g. 10%–20% SegFormer) slightly nudged Fold 0 metrics locally ($0.5694 \to 0.5765$), this small margin (+0.007) is well within cross-validation variance across folds and carries unmitigated risk of condition-blind failure on deep turbid test scenes.
- Retraining SegFormer-B5 with physical condition embeddings and proper 5-fold cross-validation would require ~12 hours of training and validation, exceeding the safe submission window.

### 8.2 Defensible Decision
Per operating guidelines:
> *"A defensible decision to retain V28-A is better than a speculative submission presented as a breakthrough."*

**We retain V28-A as the verified champion submission.**

---

## 9. Submission Recommendation and Reproducibility Instructions

### 9.1 Submission Ledger

| Submission # | Candidate Name | Archive Path | SHA-256 | Official Score | Status |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **#19096** | **V28-A (Champion)** | `deploy/v28/submission_v28a_fivefold_ablation.zip` | `471bf9b9710e2922805ac09746943f318f2d140de074c917a34f5ded6468ab39` | **0.634** | **VERIFIED BEST** |
| #19185 | V28-SOTA-Apex | `deploy/overnight_candidate/submission_overnight_candidate_sota.zip` | `e2a632734e7a78a9c30c88b908990c74a004ef59a68da0bb49c3b7a54457e51c` | 0.622 | Decommissioned |
| #19214 | V29-Dual-Backbone-SOTA | `deploy/v29/submission_v29_dual_backbone_sota.zip` | `b73155c757376c721b7cfff9ee7f160e050a4237cf8c01f5cb2943405599d6c3` | 0.597 | Decommissioned |

### 9.2 Reproducibility Command for V28-A
To regenerate the exact V28-A submission archive from source:
```powershell
python tools/v28/run_v28a_ablation.py `
  --test_dir data/test/degraded `
  --out_dir deploy/v28/v28a_reproduced `
  --out_zip deploy/v28/submission_v28a_reproduced.zip
```

### 9.3 Verification Hash Check
```powershell
Get-FileHash deploy/v28/submission_v28a_fivefold_ablation.zip -Algorithm SHA256
# Must match: 471BF9B9710E2922805AC09746943F318F2D140DE074C917A34F5DED6468AB39
```

---

## 10. Audit Artifacts & Generated Files
1. **Visual Comparisons:** Five 2x3 high-resolution diff panels saved in [`reports/v29_regression_visuals/`](file:///c:/Projects/SAT-EURS/reports/v29_regression_visuals/):
   - `diff_test04_10_85.png` (53.25% changed pixels)
   - `diff_test09_10_100.png` (36.82% changed pixels)
   - `diff_test10_1_95.png` (1.24% changed pixels)
   - `diff_test20_5_15.png` (1.21% changed pixels)
   - `diff_test10_10_40.png` (0.32% changed pixels)
2. **Experiment Log:** [`reports/v29_regression_visuals/oof_comparative_experiment.json`](file:///c:/Projects/SAT-EURS/reports/v29_regression_visuals/oof_comparative_experiment.json)
3. **Degradation Breakdown:** Generated by [`scratch/analyze_degradation_breakdown.py`](file:///c:/Projects/SAT-EURS/scratch/analyze_degradation_breakdown.py).
