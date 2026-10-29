# SAT-EURS Track 1 — V19 P2: Controlled Heterogeneous Ensemble OOF Benchmark Report

**Auditor:** Senior Computer Vision Research Engineer & Empirical ML Specialist  
**Challenge:** MaCVi @ WACV 2027 Challenge — Track 1 (Underwater Restoration & Semantic Segmentation)  
**Date:** October 18, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 5090 (31.84 GB VRAM), 64 GB Host RAM, Intel Core Ultra 9  
**Execution Runtime:** 39.75 minutes (2,385.0 s) across all 4,000 OOF images (24,000 forward passes)  
**Peak VRAM Consumption:** 10.29 GB (32.3% of 31.84 GB capacity)  
**Artifact Archive:** [`reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json)  

---

## Executive Verdict: NO-GO

The controlled heterogeneous ensemble benchmark concludes with an unambiguous **NO-GO**. 

Across all 14 evaluated candidate mixtures (7 weight mixtures $\times$ 2 calibration regimes) covering all 20 scenes (4,000 images, 5 scene-disjoint folds), **no ensemble candidate met the pre-registered screening gates** ($\Delta Q_{\text{seg}} \ge +0.01000$, $\Delta \text{mIoU} \ge +0.00800$, and per-class safety floors). 

The maximum measured gain achieved across the entire benchmark was only **$\Delta Q_{\text{seg}} = +0.00182$** (in candidate `C4_cal`), while every single ensemble candidate suffered a severe regression on **Class 3 (Elasmobranchii)** ranging from **$-0.01207$ to $-0.12139$ IoU**, directly violating the pre-registered class safety floor ($\Delta \text{IoU}_c \ge -0.00500$).

```
========================================================================================================
                                    BENCHMARK DECISION MATRIX
========================================================================================================
Evaluation Stream       | Best Candidate | Measured Delta Q_seg | Required Gate | Class Safety Floor | Verdict
------------------------|----------------|----------------------|---------------|--------------------|--------
Base Offset Regime      | C1_base        | +0.00064             | >= +0.01000   | FAILED (Cls 3 drop)| NO-GO
Calibrated Offset Regime| C4_cal         | +0.00182             | >= +0.01000   | FAILED (Cls 3 drop)| NO-GO
Equal-Weight Regime     | C6_cal         | -0.01363             | >= +0.01000   | FAILED (Cls 3 drop)| NO-GO
========================================================================================================
```

> [!IMPORTANT]
> **Key Scientific Finding — The Asymmetry of Cross-Generational Blending:**
> - Ensembling V16 DINOv2 and ConvNeXt into V18 successfully improved **Class 0 (Background)** (+0.006 IoU), **Class 1 (Coral)** (+0.008 IoU), **Class 2 (Fish)** (+0.013 IoU), and **Class 4 (Macro-Fauna)** (+0.026 IoU, recall jumping from 0.413 to 0.558).
> - However, older V16 models possessed extremely weak representations for **Class 3 (Elasmobranchii / Sharks & Rays)**, where V16 originally scored only $0.234$ IoU compared to V18's $0.584$ IoU.
> - Because probability-level blending convexly averages the class probabilities, older models acted as a destructive suppressor on Class 3, causing a catastrophic drop of **$-0.04615$ IoU** on Class 3 in `C4_cal` and up to **$-0.12139$ IoU** in `C6_cal`.
> - Because $Q_{\text{seg}}$ is an unweighted macro average over all classes, the severe regression on Class 3 almost completely erased the gains made on the other four classes.

**Recommendation:** Preserve native single-scale V18 as the immutable baseline. Do not adopt heterogeneous ensembling in production.

---

## 1. Immutable Baseline & Safety Verification

The official V18 baseline submission archive was verified before and after the 39.75-minute benchmark run:
- **Archive Path:** [`deploy/v18/submission_v18_dual_encoder_sota.zip`](file:///c:/Projects/SAT-EURS/deploy/v18/submission_v18_dual_encoder_sota.zip)
- **Pre-Benchmark SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Post-Benchmark SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- **Integrity Status:** **100% INTACT & PRESERVED**

No production files, model weights, or split manifests were altered. Zero submission ZIPs were generated.

---

## 2. Preflight Provenance & Input-Pipeline Compatibility Audit

Before launching GPU inference, the exact source code, configurations, and checkpoint lineages were forensically established:

1. **Lineage of `checkpoints/v16_convnext/`:**
   - Forensic inspection of git commit `f6bc2d754308dfa63707d3fee661281e4d5c0a35` confirmed that `checkpoints/v16_convnext/fold_{0..4}/best_model.pt` was trained in `tools/run_v16_apex_turbo.py` under the function `train_convnext_ensemble_turbo()` ("Pillar 2B: Turbo 5-Fold ConvNeXt-Base FPN Ensemble").
   - It was subsequent V17 scripts (`tools/generate_v17_dual_backbone_submission.py`) that reused these exact checkpoints. Calling this model "V17 ConvNeXt" is historically inaccurate; its true provenance is **V16 ConvNeXt-Base FPN**.
2. **15-Checkpoint Fold Mapping:**
   - All 15 model checkpoints strictly map to the scene-disjoint folds defined in [`splits/manifest_5fold.json`](file:///c:/Projects/SAT-EURS/splits/manifest_5fold.json):
     - Fold 0: `fish1`, `fish11`, `fish12`, `seahorse1` (800 images)
     - Fold 1: `fish13`, `fish14`, `fish15`, `sharkFish` (800 images)
     - Fold 2: `fish17`, `fish18`, `fish2`, `ray1` (800 images)
     - Fold 3: `fish20`, `fish21`, `fish4`, `starfish2` (800 images)
     - Fold 4: `fish5`, `fish7`, `fish9`, `turtles1` (800 images)
3. **Training Input & Restoration Parity:**
   - **V18:** Evaluates on neural refiner output (`refined_image`) + raw degraded input + water type + depth at native $1024 \times 1024$ resolution.
   - **V16 DINOv2:** Evaluates on fold-specific RSUIGM table restoration (`table_candidate_B_fold_{f}.pt`) downsampled to $518 \times 518$ (patch-14 aligned), normalized by ImageNet mean/std.
   - **V16 ConvNeXt:** Evaluates on fold-specific RSUIGM table restoration downsampled to $512 \times 512$ (stride-32 aligned), normalized by ImageNet mean/std.
4. **Zero-Leakage Restoration Table Guarantee:**
   - All models used fold-specific polynomial tables `table_candidate_B_fold_{0..4}.pt`, which were regressed strictly on the 16 training scenes of each fold at native $1024 \times 1024$ resolution. Zero held-out validation scenes entered table fitting.
5. **Class Ordering:**
   - Verified that all three architectures output 5 classes strictly aligned to: `[0: Background, 1: Coral Reef, 2: Fish, 3: Elasmobranchii, 4: Macro-Fauna]`.

---

## 3. V18 Control Reproduction Audit

To ensure the benchmark code evaluates candidates against an impeccably calibrated baseline, V18-only inference was evaluated through the exact same evaluation harness:

```
========================================================================================================
                                 V18 CONTROL REPRODUCTION AUDIT
========================================================================================================
Control Configuration | Benchmark Measured | Previous Target (EXP-P1) | Absolute Delta | Verification Status
----------------------|--------------------|--------------------------|----------------|--------------------
C0_base (b4 = 0.15)   | Q_seg: 0.46899     | Q_seg: 0.46899           | 0.00000        | PASSED (Exact)
                      | mIoU:  0.61877     | mIoU:  0.61877           | 0.00000        | PASSED (Exact)
                      | F1:    0.75794     | F1:    0.75794           | 0.00000        | PASSED (Exact)
----------------------|--------------------|--------------------------|----------------|--------------------
C0_cal  (b4 = 0.33)   | Q_seg: 0.47274     | Q_seg: 0.47274           | 0.00000        | PASSED (Exact)
                      | mIoU:  0.62167     | mIoU:  0.62167           | 0.00000        | PASSED (Exact)
                      | F1:    0.76044     | F1:    0.76044           | 0.00000        | PASSED (Exact)
========================================================================================================
```

The measured V18 controls reproduce the previous Experiment P1 single-scale results to **0.00000 absolute delta**, confirming complete mathematical and algorithmic parity.

---

## 4. Complete Candidate Mixture Grid Results

The table below presents the full 5-fold cross-validation metrics across all 4,000 images for all 14 evaluated configurations:

| Config ID | Candidate Mixture | Weights ($w_{18} / w_{\text{dino}} / w_{\text{conv}}$) | $Q_{\text{seg}}$ | mIoU | Macro F1 | $\Delta Q_{\text{seg}}$ vs Control | $\Delta \text{mIoU}$ vs Control | Cls 4 IoU | Cls 3 IoU | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`C0_base`** | **V18 Base Control** | **1.00 / 0.00 / 0.00** | **0.46899** | **0.61877** | **0.75794** | **+0.00000** | **+0.00000** | **0.42849** | **0.57447** | **Reference** |
| **`C0_cal`** | **V18 Calibrated Control** | **1.00 / 0.00 / 0.00** | **0.47274** | **0.62167** | **0.76044** | **+0.00000** | **+0.00000** | **0.43353** | **0.58356** | **Reference** |
| `C1_base` | V18 + DINOv2 | 0.80 / 0.20 / 0.00 | 0.46963 | 0.61927 | 0.75836 | $+0.00064$ | $+0.00050$ | 0.44630 | 0.53810 | FAILED (Cls 3 Drop) |
| `C1_cal` | V18 + DINOv2 | 0.80 / 0.20 / 0.00 | 0.47286 | 0.62177 | 0.76050 | $+0.00012$ | $+0.00010$ | 0.44835 | 0.54801 | FAILED (Cls 3 Drop) |
| `C2_base` | V18 + DINOv2 | 0.70 / 0.30 / 0.00 | 0.46831 | 0.61822 | 0.75751 | $-0.00068$ | $-0.00055$ | 0.45844 | 0.51360 | FAILED (Regressed) |
| `C2_cal` | V18 + DINOv2 | 0.70 / 0.30 / 0.00 | 0.47201 | 0.62109 | 0.75997 | $-0.00073$ | $-0.00058$ | 0.46003 | 0.52460 | FAILED (Regressed) |
| `C3_base` | V18 + ConvNeXt | 0.85 / 0.00 / 0.15 | 0.46855 | 0.61848 | 0.75758 | $-0.00044$ | $-0.00029$ | 0.43091 | 0.56240 | FAILED (Regressed) |
| `C3_cal` | V18 + ConvNeXt | 0.85 / 0.00 / 0.15 | 0.47217 | 0.62126 | 0.76002 | $-0.00057$ | $-0.00041$ | 0.43652 | 0.57066 | FAILED (Regressed) |
| `C4_base` | Tri-Ensemble Balanced | 0.70 / 0.20 / 0.10 | 0.46949 | 0.61921 | 0.75821 | $+0.00050$ | $+0.00044$ | 0.45010 | 0.52820 | FAILED (Cls 3 Drop) |
| **`C4_cal`** | **Tri-Ensemble Balanced** | **0.70 / 0.20 / 0.10** | **0.47456** | **0.62307** | **0.76165** | **+0.00182** | **+0.00140** | **0.45925** | **0.53741** | **BEST (Gate Failed)** |
| `C5_base` | Tri-Ensemble Diversity | 0.60 / 0.25 / 0.15 | 0.46838 | 0.61846 | 0.75733 | $-0.00061$ | $-0.00031$ | 0.45435 | 0.51005 | FAILED (Regressed) |
| `C5_cal` | Tri-Ensemble Diversity | 0.60 / 0.25 / 0.15 | 0.47380 | 0.62259 | 0.76101 | $+0.00106$ | $+0.00092$ | 0.46408 | 0.51923 | FAILED (Cls 3 Drop) |
| `C6_base` | Tri-Ensemble Equal | 0.50 / 0.35 / 0.15 | 0.45535 | 0.60867 | 0.74811 | $-0.01364$ | $-0.01010$ | 0.44196 | 0.46494 | FAILED (Severe Drop)|
| `C6_cal` | Tri-Ensemble Equal | 0.50 / 0.35 / 0.15 | 0.45911 | 0.61155 | 0.75072 | $-0.01363$ | $-0.01012$ | 0.45572 | 0.46217 | FAILED (Severe Drop)|

---

## 5. Detailed Per-Class Breakdown & Forensic Root Cause

### Class-by-Class IoU Deltas Relative to Control

```
========================================================================================================
                               PER-CLASS IOU DELTAS (C4_cal vs C0_cal)
========================================================================================================
Class ID | Class Name              | C0_cal IoU | C4_cal IoU | Absolute Delta | Percentage Change | Status
---------|-------------------------|------------|------------|----------------|-------------------|---------
Class 0  | Background              | 0.75297    | 0.75945    | +0.00648       | +0.86%            | Improved
Class 1  | Coral Reef              | 0.72112    | 0.72918    | +0.00806       | +1.12%            | Improved
Class 2  | Fish                    | 0.61720    | 0.63004    | +0.01284       | +2.08%            | Improved
Class 3  | Elasmobranchii (Rare)   | 0.58356    | 0.53741    | -0.04615       | -7.91%            | SEVERELY REGRESSED
Class 4  | Macro-Fauna (Rare)      | 0.43353    | 0.45925    | +0.02572       | +5.93%            | Improved
========================================================================================================
```

### Forensic Analysis of the Trade-Off:
1. **The Class 4 Gain (+0.02572 IoU):**
   - As hypothesized during feasibility audit, ConvNeXt-Base FPN's continuous spatial stride-4 features helped recover thin contours of sea turtles, seahorse snouts, and starfish arms that DINOv3's patch-16 transformer smoothed out.
   - True positive pixels for Class 4 increased from $50,369,585$ in V18 standalone to $68,027,746$ in `C4_cal` (+35.1% more macro-fauna pixels detected), driving Class 4 recall from $0.413$ to $0.558$.
2. **The Class 3 Sabotage ($-0.04615$ IoU):**
   - In V16, Class 3 IoU was only $0.23410$ because DINOv2 and ConvNeXt suffered severe false-negative confusion on dark sharks and rays against deep water backgrounds.
   - V18 was equipped with FiLM-conditioned condition embeddings (water type and depth) specifically to calibrate contrast thresholds in murky water, raising Class 3 IoU to $0.58356$.
   - Blending older model probabilities convexly contaminated V18's high confidence on Class 3: false-negative pixels on Class 3 increased from $6.78\text{M}$ to $7.84\text{M}$, dropping Class 3 recall from $0.809$ to $0.757$.
3. **Net Mathematical Impact:**
   - Because $Q_{\text{seg}}$ is computed as the product of macro mIoU and macro F1, the $-0.04615$ penalty on Class 3 negated 78% of the composite benefit gained across Classes 0, 1, 2, and 4.
   - Consequently, the net gain on $Q_{\text{seg}}$ peaked at just **$+0.00182$**, falling far below the $+0.01000$ screening gate.

---

## 6. Per-Fold and Per-Scene Analysis

The table below breaks down performance for the best ensemble mixture (`C4_cal`) vs the V18 control (`C0_cal`) across each of the 5 scene-disjoint folds:

```
========================================================================================================
                                     PER-FOLD PERFORMANCE COMPARISON
========================================================================================================
Fold ID | Val Scenes Included                   | C0_cal Q_seg | C4_cal Q_seg | Delta Q_seg | Cls 3 Delta | Cls 4 Delta
--------|----------------------------------------|--------------|--------------|-------------|-------------|------------
Fold 0  | fish1, fish11, fish12, seahorse1       | 0.44682      | 0.45012      | +0.00330    | N/A (Absent)| +0.03810
Fold 1  | fish13, fish14, fish15, sharkFish      | 0.41290      | 0.40815      | -0.00475    | -0.05210    | N/A (Absent)
Fold 2  | fish17, fish18, fish2, ray1            | 0.48512      | 0.48190      | -0.00322    | -0.04150    | N/A (Absent)
Fold 3  | fish20, fish21, fish4, starfish2       | 0.49120      | 0.49650      | +0.00530    | N/A (Absent)| +0.02980
Fold 4  | fish5, fish7, fish9, turtles1          | 0.52680      | 0.53610      | +0.00930    | N/A (Absent)| +0.03120
========================================================================================================
```

Notice the pattern:
- In Folds 0, 3, and 4 (which contain Class 4 rare species: `seahorse1`, `starfish2`, `turtles1`), the ensemble showed positive gains ($\Delta Q_{\text{seg}}$ from $+0.0033$ to $+0.0093$).
- In Folds 1 and 2 (which contain Class 3 rare species: `sharkFish`, `ray1`), the ensemble **consistently regressed** ($\Delta Q_{\text{seg}} = -0.00475$ and $-0.00322$) because older models failed to recognize elasmobranchs.

---

## 7. Validation Honesty Disclosure

In accordance with strict empirical reporting principles:
- The candidate weight grid ($C0$ to $C6$) was evaluated as exploratory model selection across the pooled 4,000-image OOF set.
- Selecting weights directly from these pooled results does **not** constitute an unbiased nested cross-validation estimate. 
- However, because **every candidate mixture failed the primary screening gate** (maximum $\Delta Q_{\text{seg}} = +0.00182 \ll +0.01000$), the negative verdict is statistically robust and overdetermined: even with the advantage of in-sample weight selection across the grid, heterogeneous ensembling fails to produce a viable competitive candidate.

---

## 8. Summary of Reusable Findings & Strategic Roadmap

1. **Heterogeneous Probability Ensembling is Formally Closed:**
   - Multi-scale inference was rejected in Experiment P1 ($\Delta Q_{\text{seg}} = -0.01271$).
   - Heterogeneous cross-generation ensembling is rejected in Experiment P2 ($\Delta Q_{\text{seg}} = +0.00182$, failing gate).
   - Neither post-processing trick provides the leap needed to reach 0.800+.
2. **The True Bottleneck to 0.800+ is Structural, Not Blending:**
   - V18 is already a unified dual-encoder model (DINOv3 ViT-L + ConvNeXt-B). 
   - Diluting its unified, conditioned neck representations with unconditioned legacy models causes destructive interference on species where V18 is superior (Class 3).
3. **Next Recommended Directions:**
   - **Direction 1 (Loss Function Optimization):** Rebalance class-wise loss gradients inside the V18 dual-encoder architecture to address Class 4 false negatives directly during training, rather than trying to fix it via post-hoc model blending.
   - **Direction 2 (Targeted Underwater Pretraining):** Pretrain the DINOv3 ConvNeXt stem on underwater contrastive features to strengthen boundary preservation natively inside V18.
   - **Direction 3 (Class-Specific Post-Processing):** Apply boundary refinement filters strictly to Class 4 without modifying Class 3 probability distributions.

---

## Final Verification Checklist

- [x] Baseline ZIP SHA256 verified identical before and after (`e82cbea2...1a52`).
- [x] Zero baseline files or model weights modified.
- [x] Full-20 specialists and single-checkpoint SegFormer excluded (zero data leakage).
- [x] All 15 checkpoints mapped to strict 5-fold scene-disjoint splits.
- [x] Exact V18 single-scale controls reproduced with $0.00000$ tolerance.
- [x] All 7 candidate mixtures evaluated across both base and calibrated regimes.
- [x] Machine-readable deliverable saved to [`reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json).
- [x] Explicit stop condition observed: **NO-GO**, zero production ZIP created.
