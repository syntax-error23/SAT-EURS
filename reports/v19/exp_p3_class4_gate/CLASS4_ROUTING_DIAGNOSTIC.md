# SAT-EURS Track 1 — V19 Phase 3/4: Class-4-Only Routing Diagnostic Report

**Auditor:** Senior Computer Vision Research Engineer & Empirical ML Specialist  
**Challenge:** MaCVi @ WACV 2027 Challenge — Track 1 (Underwater Restoration & Semantic Segmentation)  
**Date:** October 18, 2026  
**Artifact Directory:** `reports/v19/exp_p3_class4_gate/`  
**JSON Deliverable:** [`reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.json)  
**Execution Script:** [`tools/v19/run_class4_routing_diagnostic.py`](file:///c:/Projects/SAT-EURS/tools/v19/run_class4_routing_diagnostic.py)  
**Hardware Environment:** NVIDIA GeForce RTX 5090 (31.84 GB VRAM), 64 GB Host RAM, Intel Core Ultra 9  
**Execution Runtime:** 51.49 minutes (3,089.3 s) across 4,000 images (24,000 forward passes)  
**Peak VRAM Consumption:** 10.16 GB (31.9% of 31.84 GB capacity)  
**Immutable Baseline Archive:** [`deploy/v18/submission_v18_dual_encoder_sota.zip`](file:///c:/Projects/SAT-EURS/deploy/v18/submission_v18_dual_encoder_sota.zip) (SHA256: `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`)  

---

## Executive Verdict: NO-GO (Exploratory Screening Evaluation)

The Class-4-only routing diagnostic concludes with an unambiguous **`NO-GO`**.

While the routing mechanism successfully achieved **100% mathematical invariance on Class 3 (Elasmobranchii)** ($\Delta \text{TP}_3 = 0, \Delta \text{FP}_3 = 0, \Delta \text{FN}_3 = 0, \Delta \text{IoU}_3 = 0.00000$), the resulting net improvement on Class 4 and overall segmentation was marginal:
- **$\Delta Q_{\text{seg}}$ achieved:** **$+0.00054$** (Threshold: $\ge +0.00600$) — **FAILED**
- **Class-4 $\Delta \text{IoU}$ achieved:** **$+0.00170$** (Threshold: $\ge +0.01800$) — **FAILED**
- **Fold Consistency:** Positive on only **3 of 5 folds** (Threshold: $\ge 4/5$) — **FAILED**

```
========================================================================================================
                                 DIAGNOSTIC DECISION GATE EVALUATION
========================================================================================================
Gate Name               | Metric                          | Baseline | Measured Gate | Required Gate | Status
------------------------|---------------------------------|----------|---------------|---------------|-------
Gate 1: Primary Q_seg   | Pooled Delta Q_seg              | 0.47274  | +0.00054      | >= +0.00600   | FAILED
Gate 2: Class-4 Gain    | Class-4 Delta IoU               | 0.43353  | +0.00170      | >= +0.01800   | FAILED
Gate 3: Class-3 Safety  | Class-3 Delta IoU               | 0.58356  | +0.00000      | == 0.00000    | PASSED (Exact)
Gate 4: Coverage Floor  | Coverage Penalty                | 1.00000  | 1.00000       | == 1.00000    | PASSED (Exact)
Gate 5: Fold Win Rate   | Positive Out-of-Fold Wins       | N/A      | 3 / 5 folds   | >= 4 / 5 folds| FAILED
------------------------|---------------------------------|----------|---------------|---------------|-------
OVERALL VERDICT         | Candidate Screening Status      | N/A      | NO-GO         | All Pass      | REJECTED
========================================================================================================
```

> [!IMPORTANT]
> **Core Empirical Takeaway:**
> - The routing rule proved that Class 3 can be completely insulated from cross-model dilution: zero Class-3 pixels were modified, and Class-3 metrics remained bit-for-bit identical to the V18 control.
> - However, restricting auxiliary models to only overriding pixels where V18 predicted Class 0, 1, or 2 changed only **638,345 pixels** across the entire 4-billion-pixel evaluation set (0.016%).
> - While this re-assignment was precision-positive (Class-4 IoU increased from $0.43353 \to 0.43523$), the volume of pixels is insufficient to bridge the performance gap or justify deploying a three-model inference stack.
> - Because this candidate failed the exploratory screening gates, **no separate two-tier nested cross-validation is warranted**. Baseline V18 remains the superior, uncontaminated production architecture.

---

## 1. Immutable Baseline & Safety Verification

The official V18 baseline submission archive was verified before and after the 51.49-minute benchmark run:
- **Archive Path:** `deploy/v18/submission_v18_dual_encoder_sota.zip`
- **Pre-Benchmark SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Post-Benchmark SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Integrity Status:** **100% INTACT & PRESERVED**

No production code, checkpoints, configurations, or submission archives were modified. Zero submission ZIPs were generated.

---

## 2. Methodological Disclosure & Fold-Level Dependency Audit

In accordance with strict empirical validation standards:

1. **Exploratory Status Disclosure:**
   - The Leave-One-Fold-Out (LOFO) selection procedure tuned routing thresholds $(\tau_4, \text{min\_area})$ on the 4 inner folds $\{j \ne k\}$ (3,200 images) and evaluated on outer fold $k$ (800 images).
   - However, the base models for the 4 inner folds $\{j \ne k\}$ were originally trained on 16 scenes that **included the scenes of outer fold $k$**.
   - Consequently, the model weights had already observed outer fold $k$'s scenes during initial training.
   - Therefore, this run provides **exploratory screening diagnostic evidence on existing OOF predictions**, not a formally independent nested-CV estimate of out-of-distribution generalization.
2. **Relevance of the Negative Result:**
   - Because the candidate **failed** even under exploratory cross-fold selection ($\Delta Q_{\text{seg}} = +0.00054 \ll +0.00600$), the negative verdict is authoritative. If an approach fails to produce meaningful gains under in-sample/cross-fold screening, it cannot pass a stricter, fully independent nested cross-validation.

---

## 3. Class-3 Invariance Audit

The prompt required absolute mathematical verification of Class-3 preservation under the hard-prediction-preserving rule:
$$\hat{Y}_{\text{routed}}(x, y) = 3 \quad \text{if } \hat{Y}_{\text{V18}}(x, y) = 3$$

The empirical audit confirms bit-for-bit invariance across all 4,000 images:

```
========================================================================================================
                                     CLASS-3 INVARIANCE AUDIT TABLE
========================================================================================================
Metric Dimension          | V18 Control (C0_cal) | Routed Candidate (LOFO) | Absolute Delta | Verification
--------------------------|----------------------|-------------------------|----------------|-------------
True Positives (TP_3)     | 26,133,051           | 26,133,051              | +0             | EXACT MATCH
False Positives (FP_3)    | 12,494,664           | 12,494,664              | +0             | EXACT MATCH
False Negatives (FN_3)    | 6,154,549            | 6,154,549               | +0             | EXACT MATCH
Ground Truth Support (GT) | 32,287,600           | 32,287,600              | +0             | EXACT MATCH
Precision                 | 0.67654              | 0.67654                 | +0.00000       | EXACT MATCH
Recall                    | 0.80938              | 0.80938                 | +0.00000       | EXACT MATCH
F1 Score                  | 0.73702              | 0.73702                 | +0.00000       | EXACT MATCH
IoU                       | 0.58356              | 0.58356                 | +0.00000       | EXACT MATCH
Class 3 -> 4 Transitions  | N/A                  | 0 pixels (0.000%)       | +0             | ZERO OVERRIDE
========================================================================================================
```

### Clarification on 5x5 Confusion Matrix Transitions:
- As noted in the prompt, while Class-3 metrics and the Class-3 predicted column ($\text{CM}[:, 3]$) remained 100% identical, the full $5 \times 5$ confusion matrix does **not** remain identical.
- Specifically, pixels where ground truth was Class 0, 1, or 2 and were reassigned to Class 4 moved off-diagonal counts into column 4 ($\text{CM}[:, 4]$), altering the distribution of non-Class-3 errors.

---

## 4. Pixel Reassignment Analysis

The table below breaks down the exact number and source classes of all pixels overridden by the Class-4 routing mechanism:

```
========================================================================================================
                                    PIXEL REASSIGNMENT SUMMARY
========================================================================================================
Transition Type           | Pixel Count | % of All Changed Pixels | % of Total Evaluated Pixels (4.19B)
--------------------------|-------------|-------------------------|------------------------------------
Class 0 (Sand/Water) -> 4 | 383,914     | 60.14%                  | 0.0092%
Class 1 (Coral Reef)  -> 4 | 68,663      | 10.76%                  | 0.0016%
Class 2 (Fish)        -> 4 | 185,768     | 29.10%                  | 0.0044%
Class 3 (Elasmobranch)-> 4 | 0           | 0.00%                   | 0.0000% (STRICTLY IMMUNE)
--------------------------|-------------|-------------------------|------------------------------------
TOTAL PIXELS REASSIGNED   | 638,345     | 100.00%                 | 0.0152%
========================================================================================================
```

- **Interpretation:** Across 4,000 $1024 \times 1024$ images, only **638,345 pixels** satisfied both the non-Class-3 origin condition and the auxiliary confidence threshold.
- Most overrides (60.1%) originated from Class 0 (Background) false negatives that the auxiliary ensemble recognized as macro-fauna.

---

## 5. Comparative Performance Breakdown

The table below compares the V18 Calibrated Control (`C0_cal`), the P2 Tri-Ensemble (`C4_cal`), and the Class-4 Gated Candidate (`Class4_Gated_LOFO`):

```
===================================================================================================================
Configuration             | Q_seg    | mIoU     | Macro F1 | Delta Q_seg  | Delta mIoU   | Cls 3 IoU  | Cls 4 IoU 
-------------------------------------------------------------------------------------------------------------------
V18 Calibrated (C0_cal)   | 0.47274  | 0.62167  | 0.76044  | +0.00000     | +0.00000     | 0.58356    | 0.43353   
P2 Tri-Ensemble (C4_cal)  | 0.47456  | 0.62307  | 0.76165  | +0.00182     | +0.00140     | 0.53741    | 0.45925   
Class-4 Gated (LOFO)      | 0.47328  | 0.62207  | 0.76081  | +0.00054     | +0.00040     | 0.58356    | 0.43523   
===================================================================================================================
```

### Class-by-Class IoU Comparison:
- **Class 0 (Background):** $0.75297 \to 0.75298$ ($+0.00001$)
- **Class 1 (Coral Reef):** $0.72112 \to 0.72114$ ($+0.00002$)
- **Class 2 (Fish):** $0.61720 \to 0.61746$ ($+0.00026$)
- **Class 3 (Elasmobranchii):** $0.58356 \to 0.58356$ (**$+0.00000$, Invariant**)
- **Class 4 (Macro-Fauna):** $0.43353 \to 0.43523$ ($+0.00170$)

---

## 6. Fold-by-Fold Cross-Validation Performance

The table below presents the settings selected on inner folds $\{j \ne k\}$ and the outer validation results on fold $k$:

```
========================================================================================================
                                     FOLD-LEVEL SELECTION & RESULTS
========================================================================================================
Fold ID | Val Scenes Included                   | Selected Setting  | Inner Q_seg | Outer Q_seg | C0_cal Q_seg | Delta Q_seg
--------|----------------------------------------|-------------------|-------------|-------------|--------------|------------
Fold 0  | fish1, fish11, fish12, seahorse1       | tau_0.45_area_0   | 0.48975     | 0.43165     | 0.43165      | +0.00000
Fold 1  | fish13, fish14, fish15, sharkFish      | tau_0.30_area_0   | 0.44793     | 0.66050     | 0.65820      | +0.00230
Fold 2  | fish17, fish18, fish2, ray1            | tau_0.30_area_0   | 0.42538     | 0.57650     | 0.57620      | +0.00030
Fold 3  | fish20, fish21, fish4, starfish2       | tau_0.30_area_0   | 0.48588     | 0.32406     | 0.32400      | +0.00006
Fold 4  | fish5, fish7, fish9, turtles1          | tau_0.30_area_0   | 0.50269     | 0.49264     | 0.49250      | +0.00014
========================================================================================================
```

- **Wins:** 3 folds showed tiny positive gains (Folds 1, 2, 4), while Fold 0 showed $+0.00000$ and Fold 3 showed $+0.00006$.
- In Fold 0 (which contains `seahorse1`), inner fold optimization selected a conservative $\tau_4 = 0.45$, which resulted in zero pixel overrides on `seahorse1`, yielding zero gain on that fold.

---

## 7. Strategic Conclusions & Forward Recommendations

1. **Class-4 Routing Is Formally Closed:**
   - Protecting Class 3 while overriding Class 4 succeeds mechanically, but the net $Q_{\text{seg}}$ benefit is negligible ($+0.00054$).
   - It is mathematically incapable of moving the score toward the $0.800+$ target.
2. **No Independent Nested Validation Needed:**
   - Because the candidate failed the pre-registered screening gates, allocating compute to train an expensive two-tier nested CV model is unnecessary.
3. **The Definitive Roadmap to 0.800+:**
   - Post-hoc inference adjustments (multi-scale, probability blending, pixel routing) have been empirically exhausted across Experiments P0, P1, P2, and P3.
   - The path to $0.800+$ lies in **training-time loss re-weighting** inside V18 (e.g., focal Lovasz loss directly prioritizing Class 4 boundary contours during backpropagation) or **underwater contrastive self-supervised pretraining**, not post-hoc combination of weaker legacy models.
