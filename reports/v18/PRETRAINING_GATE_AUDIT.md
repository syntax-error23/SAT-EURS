# SAT-EURS Track 1 — V18 Pre-Training Gate Audit Report

**Date:** 2026-10-17  
**Gate Status:** **`PASS`**  
**Audited Target:** V18 Architecture, Pilot Execution Artifacts, Metric Engine & Fold Isolation  
**Hardware Verified:** NVIDIA GeForce RTX 5090 (31.84 GB VRAM)  
**Strict Policy:** Zero GPU training, zero external downloads, zero mutation of V16/V17 baselines during audit.

---

## Executive Summary & Gate Verdict

Following the successful execution of the single-epoch Fold 0 GPU pilot benchmark, a rigorous pre-training source-code and artifact audit was conducted across four mandatory verification pillars:
1. **Absent-Class Metric Handling:** Investigated and resolved the distortion where an absent validation class received $1.0$, inflating Fold 0 validation mIoU from $0.4731$ to $0.5785$. Corrected `tools/v18/evaluator.py` to transparently report per-class pixel metrics ($N_{\text{gt}}, N_{\text{pred}}, \text{TP}, \text{FP}, \text{FN}$), dual macro averages (present-class mean vs conservative 5-class zero mean), and confirmed with regression unit tests covering all edge cases.
2. **Restoration Table Fold-Safety:** Proved that `table_candidate_B_fold_0.pt` was fitted strictly on the 16 training scenes defined in `splits/manifest_5fold.json`. Clean targets and degraded inputs of held-out validation scenes (`fish1`, `fish11`, `fish12`, `seahorse1`) were strictly excluded. Verified zero contamination across all fold tables and calibration stages.
3. **Out-Of-Fold (OOF) Scoring Protocol:** Verified that `run_five_fold_oof_pipeline()` accumulates a single global $5 \times 5$ confusion matrix across all 20 scenes (4,000 images) prior to score evaluation, eliminating fold-averaging distortion. Server-only metrics ($Q_{\text{NR}}$ and coverage penalty IoU-floor) remain strictly isolated as provisional.
4. **Pilot Execution Forensic Audit:** Confirmed checkpoint integrity, optimizer step counts (800 steps per stage), finite loss convergence, real DINOv3 ViT-Large/16 + ConvNeXt-Base backbones (401.6M parameters), 16.70 GB peak VRAM, and 15.14 GB headroom on the RTX 5090.

**Verdict:** **`PASS`** — The V18 pipeline is formally verified and ready for full 5-fold training.

---

## 1. Absent-Class Metric Distortion & Transparent Policy

### 1.1 Pilot Metric Discrepancy & Reproduction
In `reports/v18/pilot/PILOT_BENCHMARK_REPORT.json`, the validation metrics for Fold 0 after Epoch 1 reported:
- **Reported mIoU:** `0.57852`
- **Reported Macro F1:** `0.66243`
- **Class 3 IoU:** `1.0`, **Class 3 F1:** `1.0`

#### Forensic Reproduction from Raw Confusion Matrix
The pilot recorded the following raw $5 \times 5$ confusion matrix across all 800 validation images ($800 \times 512 \times 512 = 209,715,200$ pixels):
```
[[59925495, 25913259,   428828, 0,  25218],   <- Class 0 (Background)
 [ 1531967, 96630115,   378318, 0,      0],   <- Class 1 (Coral)
 [  831957,  1353831,  6756612, 0,      0],   <- Class 2 (Fish)
 [       0,        0,        0, 0,      0],   <- Class 3 (Elasmobranchii / Rare)
 [  950793, 11782749,  3206058, 0,      0]]   <- Class 4 (Macro-Fauna)
```

In Fold 0 validation scenes (`fish1`, `fish11`, `fish12`, `seahorse1`), Class 3 has zero ground-truth pixels ($N_{\text{gt}, 3} = 0$), and the model predicted zero pixels ($N_{\text{pred}, 3} = 0$).

Under the legacy code in `tools/v18/evaluator.py`:
```python
if support == 0 and pred_support == 0:
    ious.append(1.0)
    f1s.append(1.0)
```
This vacuous truth convention assigned Class 3 $\text{IoU} = 1.0$ and $\text{F1} = 1.0$.

When taking the 5-class arithmetic mean:
$$\text{mIoU}_{\text{legacy}} = \frac{0.66876 + 0.70230 + 0.52152 + \mathbf{1.00000} + 0.00000}{5} = \frac{2.89258}{5} = \mathbf{0.57852}$$
$$\text{Macro F1}_{\text{legacy}} = \frac{0.80150 + 0.82512 + 0.68553 + \mathbf{1.00000} + 0.00000}{5} = \frac{3.31215}{5} = \mathbf{0.66243}$$

Assigning $1.0$ directly inflated single-fold mIoU by **$+0.1054$** over the present-class mean, and by **$+0.2000$** over a conservative 5-class mean.

### 1.2 Official Server Parity Assessment: `UNVERIFIED`
Whether the official MaCVi / EURS evaluation server includes absent classes in the macro denominator, treats them as $0.0$, or excludes them on partial splits cannot be verified locally because official challenge evaluation server code is proprietary and closed-source. On the complete 20-scene dataset, all 5 classes are present, so the issue only manifests on partial scene-disjoint folds.

Therefore, official server parity for absent classes on partial splits is formally marked: **`UNVERIFIED`**.

### 1.3 Transparent Local Policy Implemented
`tools/v18/evaluator.py` has been updated to provide full transparency:
1. **Present-Class Mean ($\text{mIoU}_{\text{present}}$, $\text{Macro F1}_{\text{present}}$):** Averages strictly over classes present in ground truth ($N_{\text{gt}, c} > 0$). Matches standard semantic segmentation benchmark practices (e.g. Cityscapes, ADE20K).
2. **Conservative 5-Class Zero Mean ($\text{mIoU}_{5\text{cls}}$, $\text{Macro F1}_{5\text{cls}}$):** Treats absent classes as $0.0$, providing the absolute lower bound.
3. **Per-Class Pixel Accounting:** Every evaluation reports $N_{\text{gt}}, N_{\text{pred}}, \text{TP}, \text{FP}, \text{FN}$ and explicit status flags (`PRESENT`, `ABSENT_GT_AND_PRED`, `ABSENT_GT_BUT_PREDICTED`).

#### Recalculated Fold 0 Pilot Benchmark Results
| Class ID | Class Name | Status | Ground-Truth Pixels ($N_{\text{gt}}$) | Predicted Pixels ($N_{\text{pred}}$) | True Positives ($\text{TP}$) | False Positives ($\text{FP}$) | False Negatives ($\text{FN}$) | IoU | F1 | Precision | Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Background | `PRESENT` | 86,292,800 | 63,240,212 | 59,925,495 | 3,314,717 | 26,367,305 | 0.6688 | 0.8015 | 0.9476 | 0.6944 |
| **1** | Coral | `PRESENT` | 98,540,400 | 135,679,954 | 96,630,115 | 39,049,839 | 1,910,285 | 0.7023 | 0.8251 | 0.7122 | 0.9806 |
| **2** | Fish | `PRESENT` | 8,942,400 | 10,769,816 | 6,756,612 | 4,013,204 | 2,185,788 | 0.5215 | 0.6855 | 0.6274 | 0.7556 |
| **3** | Elasmobranchii | `ABSENT_GT_AND_PRED` | 0 | 0 | 0 | 0 | 0 | **0.0000** | **0.0000** | 0.0000 | 0.0000 |
| **4** | Macro-Fauna | `PRESENT` | 15,939,600 | 25,218 | 0 | 25,218 | 15,939,600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- **Present Classes ($N_{\text{gt}} > 0$):** `[0, 1, 2, 4]` (4 of 5 classes).
- **Corrected Present-Class Mean:**
  - $\text{mIoU}_{\text{present}} = \frac{0.66876 + 0.70230 + 0.52152 + 0.00000}{4} = \mathbf{0.47314}$
  - $\text{Macro F1}_{\text{present}} = \frac{0.80150 + 0.82512 + 0.68553 + 0.00000}{4} = \mathbf{0.57804}$
- **Conservative 5-Class Zero Mean:**
  - $\text{mIoU}_{5\text{cls}} = \frac{0.66876 + 0.70230 + 0.52152 + 0.00000 + 0.00000}{5} = \mathbf{0.37852}$
  - $\text{Macro F1}_{5\text{cls}} = \frac{0.80150 + 0.82512 + 0.68553 + 0.00000 + 0.00000}{5} = \mathbf{0.46243}$
- **Coverage Penalty:** $\mathbf{0.0}$ (Class 4 has $N_{\text{gt}} = 15.9\text{M} > 0$ but $\text{TP}_4 = 0$ in Epoch 1).
- **$Q_{\text{seg}}$:** $\mathbf{0.0}$ (correctly penalized by coverage penalty).

### 1.4 Regression Unit Tests Verified
A dedicated unit test `test_absent_class_metric_handling()` was added to `tests/v18/test_v18_synthetic.py` covering:
1. Class absent from both GT and prediction ($N_{\text{gt}} = 0, N_{\text{pred}} = 0$).
2. Class absent from GT but predicted ($N_{\text{gt}} = 0, N_{\text{pred}} > 0$ — hallucinated).
3. Class present in GT but never predicted ($N_{\text{gt}} > 0, N_{\text{pred}} = 0$ — missed / drops coverage).
4. Normal present class ($N_{\text{gt}} > 0, N_{\text{pred}} > 0$).
5. Dual present-class and 5-class zero macro averages.

**Result:** `13/13` unit tests passed cleanly on CPU.

---

## 2. Restoration Table Fold-Safety & Provenance Audit

### 2.1 Fold 0 Table Provenance & Contamination Status
The table used during Fold 0 training is:
- **File:** `checkpoints/exp4_1_scale_consistent/table_candidate_B_fold_0.pt`
- **SHA256:** `BF74A414A5CA5F3FF969E7FD5AC71F0A59D34AA681CB879676832C0EC6F7C269`
- **Fitting Source Code:** `tools/exp4_1_rsuigm_scale_consistent.py` (`fit_candidate_b_fold()`).

#### Scene Separation Verification
From `splits/manifest_5fold.json`:
- **Fold 0 Train Scenes (16):** `fish13`, `fish14`, `fish15`, `fish17`, `fish18`, `fish2`, `fish20`, `fish21`, `fish4`, `fish5`, `fish7`, `fish9`, `ray1`, `sharkFish`, `starfish2`, `turtles1`.
- **Fold 0 Held-Out Val Scenes (4):** `fish1`, `fish11`, `fish12`, `seahorse1`.

In `tools/exp4_1_rsuigm_scale_consistent.py`, lines 121–156:
```python
clean_samples_dict: Dict[str, torch.Tensor] = {}
for s in train_scenes:
    c_rgb = cache_dict["clean"][s]
    clean_samples_dict[s] = torch.from_numpy(sampled_c).to(device=device)

# Loop over degraded images:
for b_start in range(0, len(train_scenes), batch_scenes):
    b_scenes = train_scenes[b_start : b_start + batch_scenes]
    b_deg = [cache_dict["degraded"][(s, wt, depth)] for s in b_scenes]
```
The clean targets and degraded inputs of `fish1`, `fish11`, `fish12`, and `seahorse1` were **never loaded or sampled** during the fitting of `table_candidate_B_fold_0.pt`.

**Contamination Status:** **`CLEAN (ZERO CONTAMINATION)`**.

### 2.2 Complete Table Inventory & SHA256 Hashes
All fold-specific and full-dataset restoration tables are distinct and cryptographically verified:
| Checkpoint File | Scope | Scenes Used | SHA256 Hash |
| :--- | :---: | :---: | :--- |
| `table_candidate_B_fold_0.pt` | Fold 0 | 16 train scenes | `BF74A414A5CA5F3FF969E7FD5AC71F0A59D34AA681CB879676832C0EC6F7C269` |
| `table_candidate_B_fold_1.pt` | Fold 1 | 16 train scenes | `AEA0DA723D6EE588B78D5333E5BE3F29F66EB27FD9F4D4AD742B46A1DC4A8DCB` |
| `table_candidate_B_fold_2.pt` | Fold 2 | 16 train scenes | `5765C89CD1817F9FB4643216A030D003C55A5F184ACFC3928544066A103A146E` |
| `table_candidate_B_fold_3.pt` | Fold 3 | 16 train scenes | `FD8A4759CBA0ACBC2506BF26BD77E4E9E4FD4F63C5D93960F05418718F6F2403` |
| `table_candidate_B_fold_4.pt` | Fold 4 | 16 train scenes | `22D2FFD88F6F01DD123EE912BC288CD30BD941148B8537655451C5BC64E06037` |
| `tbl19_20sc_scale_consistent.pt` | Dev Deploy | All 20 scenes | `097CC9380FE11CA672F712481EC8187BBCE2E3EE84F1B17D2DC2288793B0E9A4` |

### 2.3 Stage Usage Contract in `tools/v18/run_v18_pipeline.py`
- **Stage 1 (Refiner Training):** Loads `table_candidate_B_fold_{fold}.pt` (16 training scenes).
- **Stage 2 (Segmenter Training):** Loads `table_candidate_B_fold_{fold}.pt` (16 training scenes).
- **Stage 3 (OOF Evaluation):** Loads `table_candidate_B_fold_{fold}.pt` (16 training scenes) to evaluate the 4 held-out validation scenes.
- **Stage 4 (Dev Inference):** Loads `tbl19_20sc_scale_consistent.pt` (all 20 training scenes) for the 5-fold ensemble deployed to the unlabeled Dev test set.
- **Logit Calibration (`calibration.py`):** Operates on `nested_val_scenes` within the training split only, with zero access to held-out validation or Dev scenes.

---

## 3. Out-Of-Fold (OOF) Scoring Protocol Audit

### 3.1 Global Confusion Matrix Aggregation (No Score Averaging)
In `tools/v18/run_v18_pipeline.py` lines 594–598:
```python
evaluator = V18OOFEvaluator(num_classes=5)
for f in folds:
    evaluate_oof_fold(f, config, device, evaluator, checkpoint_dir, use_mock=use_mock)

results = evaluator.compute(q_nr_provisional=0.70)
```
- A single `V18OOFEvaluator` instance is created before the fold loop.
- Each fold updates the shared confusion matrix (`evaluator.conf_mat += c_mat`) and metric lists.
- Only after all 5 folds finish (20 scenes, 4,000 images) is `compute()` executed.
- This guarantees **strict global accumulation** across all 4,000 images, adhering to official challenge protocol and preventing Simpson's paradox or fold-averaging skew.

### 3.2 All-Scene Representation in 20-Scene OOF
Because the 20 EURS scenes collectively include all 5 classes (Background, Coral, Fish, Shark/Ray, Macro-Fauna), $N_{\text{gt}, c} > 0$ for all $c \in \{0, 1, 2, 3, 4\}$ in the full OOF run. Consequently:
$$\text{mIoU}_{\text{present}} \equiv \text{mIoU}_{5\text{cls}} \quad \text{and} \quad \text{Macro F1}_{\text{present}} \equiv \text{Macro F1}_{5\text{cls}}$$
The absent-class split issue exists exclusively during isolated single-fold validation.

### 3.3 Server-Side Metric Isolation
- **$Q_{\text{NR}}$ (10% of Official Score):** Requires server-side UIQM, UCIQE, and CCF algorithms with unpublished challenge normalization constants. Locally, $S_{\text{exact\_90pct}} = 0.60 Q_{\text{seg}} + 0.30 Q_{\text{FR}}$ isolates verified local components. A sensitivity bracket is reported for $Q_{\text{NR}} \in [0.40, 0.80]$.
- **Coverage Penalty:** Evaluated locally as $\prod_{c \in C_{\text{present}}} \mathbb{I}(\text{TP}_c > 0)$. Marked as `UNVERIFIED_PROVISIONAL` to reflect that server-side behavior regarding IoU thresholds is proprietary.

---

## 4. Forensic Audit of What the Pilot Actually Trained

### 4.1 Checkpoint Artifact Provenance
Inspection of `checkpoints/v18/pilot/fold_0/`:
1. **Refiner Checkpoint (`refiner_best.pt`):**
   - **Size:** 5,174,549 bytes (4.93 MB)
   - **Architecture:** `ConditionedRestorationRefinerV18` (181 parameter tensors)
   - **Epochs / Steps:** 1 epoch, 800 optimizer steps (batch size 4, 3,200 training images)
   - **Loss & Validation PSNR:** Train Loss $0.0917$, Val PSNR $26.71$ dB
2. **Segmenter Checkpoint (`segmenter_best.pt`):**
   - **Size:** 1,729,008,927 bytes (1.61 GB)
   - **Architecture:** `EURSAPEXV18Segmenter` (820 parameter tensors, 401.60 million parameters)
   - **Backbone Verification:**
     - ViT: `vit_large_patch16_dinov3.lvd1689m` (patch embed `[1024, 3, 16, 16]`, 24 Transformer blocks)
     - ConvNeXt: `convnext_base.dinov3_lvd1689m` (stem + 4 stages)
     - Loaded genuine pretrained weights offline from local cache.
   - **Loss Convergence:** Cross-entropy + Lovász + Boundary loss decreased from $0.5252$ to $0.2793$ across 800 steps.
   - **Embedded Provenance:** Records `fold: 0`, `train_scenes` (16 scenes), `val_scenes` (4 scenes), `seed: 42`, `config_hash`, and full raw confusion matrix.

### 4.2 Hardware Utilization & Full Training Projection
- **GPU:** NVIDIA GeForce RTX 5090 (32 GB)
- **Refiner Training:** 178.0s train, 32.9s val (total 210.9s / epoch). Throughput: 18.0 img/s. Peak VRAM: 8.23 GB.
- **Segmenter Training:** 423.5s train, 62.6s val (total 486.1s / epoch). Throughput: 7.6 img/s. Peak VRAM: 16.70 GB.
- **VRAM Headroom:** **15.14 GB** unused headroom (peak allocation was 52.5% of capacity).
- **Full 5-Fold 15-Epoch Training Projection:**
  - Refiner (Stage 1): $5 \times 15 \times 210.9\,\text{s} = 4.39\,\text{hours}$
  - Segmenter (Stage 2): $5 \times 15 \times 486.1\,\text{s} = 10.13\,\text{hours}$
  - **Total Projected Wall-Clock Time:** **$14.52\,\text{hours}$** (fully feasible within a single continuous run).

---

## 5. Byte-for-Byte Baseline Preservation Audit

All baseline checkpoints, manifests, and official submissions were verified against reference hashes:
| File | Status | SHA256 Hash |
| :--- | :---: | :--- |
| `deploy/submission_v16_apex_sota.zip` | INTACT | `943369FCDFF398C1C18C9B43D4AA792BAA1E343914A3DC6F186CA52BB8CED6DF` |
| `deploy/submission_v17_dual_backbone_sota.zip` | INTACT | `50F85EC2111AD6D62A03D718FB8FE09FE4E75201D7CDF359C9A81BAAC138CC4E` |
| `splits/manifest_5fold.json` | INTACT | `CFBA12CAAABC79BB092F7DFA88405482E2C62C0AF2F4ED296B33F766EAEBD179` |
| `checkpoints/track1_screening/tbl19_20sc_clean.pt` | INTACT | `2C35DF3B0F5D3317C3672B2867E0059B893ACECD8699C5A2271383A1179D9562` |
| `checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt` | INTACT | `097CC9380FE11CA672F712481EC8187BBCE2E3EE84F1B17D2DC2288793B0E9A4` |

---

## 6. Pre-Training Gate Conclusion

| Audit Item | Criterion | Verification Evidence | Status |
| :--- | :--- | :--- | :---: |
| **Absent-Class Metrics** | Eliminate artificial 1.0; report dual means & pixel counts | `tools/v18/evaluator.py` updated, 13/13 tests pass, pilot reproduced | **PASS** |
| **Restoration Table** | Zero validation scene targets in fold tables | 16-scene fitting verified in `exp4_1`, SHA256 verified distinct | **PASS** |
| **OOF Protocol** | Single global 5x5 confusion matrix aggregation | Single evaluator passed across folds in `run_v18_pipeline.py` | **PASS** |
| **Model Backbones** | Real DINOv3 ViT-Large/16 + ConvNeXt-Base (no mocks) | 401.6M parameters in checkpoint, 24 Transformer blocks | **PASS** |
| **Throughput & VRAM** | Finite convergence, <32 GB VRAM on RTX 5090 | Refiner 8.23 GB, Segmenter 16.70 GB, 15.14 GB headroom | **PASS** |
| **Baselines** | Zero mutation of V16/V17 baselines | All 5 baseline SHA256 hashes byte-for-byte identical | **PASS** |

### Final Gate Verdict: **`PASS`**
The V18 Track 1 codebase is validated, sealed, and ready for full 5-fold training execution whenever authorized.
