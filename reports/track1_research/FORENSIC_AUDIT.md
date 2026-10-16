# SAT-EURS Track 1 — Comprehensive Forensic Repository & Method Audit

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)  
**Status:** Complete Read-Only Audit (Phase A)

---

## 1. Executive Summary & Ranked Issue Register

This audit investigates the twelve specific vulnerability areas outlined in Phase A against the actual working tree, training scripts, data loaders, checkpoints, and submission generators. Every claim is classified into: **VERIFIED FACT**, **OBSERVED RESULT**, **HYPOTHESIS**, **ESTIMATED QUANTITY**, or **UNKNOWN**.

### Summary of Ranked Vulnerabilities

| # | Vulnerability Area | File / Location | Severity | Classification | Status & Proposed Fix |
| :-: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Sampling with Replacement in `__getitem__`** | `tools/train_segformer_baseline.py:108`<br>`tools/train_dinov2_l.py:157` | **HIGH** | **VERIFIED FACT** | `random.choice(self.items)` ignores DataLoader indices, drawing samples with replacement. Fix: Use deterministic indexed access `self.items[idx]` and seedable samplers. |
| **2** | **In-Sample Data Leakage in Validation Baseline** | `checkpoints/v4_full20/best_v4_checkpoint.pt` | **CRITICAL** | **VERIFIED FACT** | Trained on all 20 scenes, reporting an inflated $Q_{\text{seg}} = 0.7078$ on the 4 "val" scenes. Truly unseen Dev performance is $Q_{\text{seg}} = 0.4240$. Fix: Strict scene-disjoint 5-fold CV. |
| **3** | **RSUIGM Table Fitted with Held-Out Clean Scenes** | `deploy/export_v5_and_track1.py:96` | **HIGH** | **VERIFIED FACT** | `scenes = sorted([...])[::2]` included validation scenes (`fish1`, `fish12`, `seahorse1`) when fitting the 200 $(wt, \text{depth})$ operators. Fix: Fit strictly on fold training scenes. |
| **4** | **Diagnostic Evaluation on 10% Stride Sample** | `tools/evaluate_track1_ablations.py:97` | **HIGH** | **VERIFIED FACT** | `sampled = deg_files[::10]` evaluated only 80 images instead of all 800 validation images. Fix: Full-set evaluation for all promotion gates. |
| **5** | **DeepLabV3 Input Normalization Bug** | `deploy/generate_verified_track1_submission.py:224`<br>`tools/evaluate_track1_ablations.py:224` | **CRITICAL** | **VERIFIED FACT** | DeepLabV3 was trained with $(x - \text{mean}) / \text{std}$ but evaluated on unnormalized $[0, 1]$ inputs, collapsing standalone $Q_{\text{seg}}$ to 0.0745. Fix: Enforce ImageNet normalization contract. |
| **6** | **Model Architecture Mislabeling** | `tools/evaluate_track1_ablations.py:71`<br>`reports/FINAL_RESULTS.md` | **MEDIUM** | **VERIFIED FACT** | Model called "SegFormer MiT-B3" was actually `smp.DeepLabV3Plus(encoder_name="mit_b3")` (DeepLabV3+ decoder with ASPP, not SegFormer MLP decoder). Fix: Exact naming. |
| **7** | **Hardcoded Train/Val Scene Lists** | `tools/train_segformer_baseline.py:44`<br>`tools/train_dinov2_l.py:55` | **MEDIUM** | **VERIFIED FACT** | Global static scene lists bypass dynamic fold manifests. Fix: Load splits dynamically from canonical fold manifest. |
| **8** | **Restored Image Aspect Ratio Distortion** | `tools/build_cached_dataset.py:29`<br>`data/cached_1024` | **HIGH** | **VERIFIED FACT** | Default mode was `"stretch"`, distorting non-square underwater scenes (e.g., 4:3, 16:9) into 1:1 squares. Fix: Preserve aspect ratio or pad consistently. |
| **9** | **Provisional Q_NR Treated as Official Score** | `metrics/nr_metric.py`<br>`reports/SCORE_DISCREPANCY_AUDIT.md` | **MEDIUM** | **VERIFIED FACT** | Local $Q_{\text{NR}}$ was an uncalibrated contrast surrogate ($\text{std} \times 2.5$). Official Q_NR normalizations are unknown. Fix: Gate promotion on verified $Q_{\text{seg}}$ and raw restoration metrics. |
| **10** | **Submission Archive Exceeding Size Limit** | `deploy/eurs_track1_dev_submission.zip` | **HIGH** | **VERIFIED FACT** | Archive was 1,240.59 MB (> 1024 MB platform limit) because restored images were uncompressed PNGs. Fix: Use JPEG quality 98 (subsampling=0) to achieve 500-750 MB. |
| **11** | **Heuristic Post-Processing in SOTA Scripts** | `deploy/generate_track1_submission_sota.py`<br>`deploy/generate_sota_track1_submission.py` | **MEDIUM** | **VERIFIED FACT** | Scripts used arbitrary manual logit adjustments (`logits[4] += 1.8`, `logits[3] -= 4.0`) rather than principled probability calibration. Fix: Calibrate thresholds on OOF validation. |
| **12** | **Missing Referenced Scripts** | `models/v5_9_ensemble.py` | **LOW** | **VERIFIED FACT** | File referenced in earlier prompts does not exist in the working tree. Status: Non-reproducible from snapshot; marked as missing. |

---

## 2. Detailed Forensic Analysis by Investigation Area

### 2.1 Hidden Sampling with Replacement in `Dataset.__getitem__`
- **Location:** `tools/train_segformer_baseline.py` (lines 107–108), `tools/train_dinov2_l.py` (lines 156–157).
- **Exact Code Found:**
  ```python
  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
      if self.is_train:
          item = random.choice(self.items)
      else:
          item = self.items[idx]
  ```
- **Consequence:** `DataLoader(shuffle=True)` generated shuffled integer indices, but `__getitem__` immediately discarded `idx` and sampled uniformly at random with replacement from `self.items` using Python's global RNG. Across 3,200 training draws per epoch, on average $(1 - 1/3200)^{3200} \approx 36.8\%$ of available variations were completely skipped in any given epoch, while others were drawn 2–5 times.
- **Proposed Correction:** Use deterministic indexing: `item = self.items[idx]`.
- **Regression Test:** `tests/test_dataset_contracts.py`: Verify that for an epoch of length $N$, all $N$ indices are sampled exactly once when `shuffle=False`, and a deterministic permutation is applied when `shuffle=True`.

### 2.2 In-Sample Data Leakage & Root Cause of the 0.50 Score Ceiling
- **Location:** `checkpoints/v4_full20/best_v4_checkpoint.pt`.
- **Evidence:** Inspection of `manifest_full.json` vs historical training logs confirms EURSNet-V4 full20 was trained on **all 20 scenes**.
- **Consequence:** Evaluating on `['fish1', 'fish12', 'seahorse1', 'sharkFish']` yielded an in-sample $Q_{\text{seg}} = 0.7078$. When submitted to the official Dev set containing 6 truly unseen scenes (`fish10`, `fish19`, `fish3`, `fish6`, `sharRayFish`, `starfish1`), the ensemble generalization dropped to $Q_{\text{seg}} = 0.4240$.
- **Proposed Correction:** Strict 5-fold cross-validation where validation scenes are 100% held out from training.

### 2.3 RSUIGM Table Fitting Contamination
- **Location:** `deploy/export_v5_and_track1.py` (lines 94–98).
- **Exact Code Found:**
  ```python
  scenes = sorted([os.path.splitext(f)[0] for f in os.listdir(clean_dir) if f.endswith(".png")])[::2]
  ```
- **Consequence:** `scenes[::2]` evaluated to `['fish1', 'fish12', 'fish14', 'fish17', 'fish2', 'fish21', 'fish5', 'fish9', 'seahorse1', 'starfish2']`. This included `fish1`, `fish12`, and `seahorse1`—three of the four validation scenes in `manifest_full.json`.
- **Proposed Correction:** In all cross-validation workflows, fit RSUIGM polynomial lookup tables strictly on `train_scenes`.
- **Regression Test:** `tests/test_rsuigm_disjointness.py`: Assert `set(table_scenes).isdisjoint(set(val_scenes))`.

### 2.4 DeepLabV3 Normalization Bug
- **Location:** `deploy/generate_verified_track1_submission.py` (line 224) and `tools/evaluate_track1_ablations.py` (line 224).
- **Consequence:** DeepLabV3-ResNet101 was trained on ImageNet-normalized inputs `(x - mean) / std`, but during evaluation, it was fed `rest_phys` in range $[0, 1]$ directly. This domain shift caused DeepLabV3's predictions to collapse to $Q_{\text{seg}} = 0.0745$, corrupting ensemble voting.
- **Proposed Correction:** Apply ImageNet standardization whenever DeepLabV3 is evaluated.

### 2.5 Architecture Naming Parity
- **Location:** `tools/evaluate_track1_ablations.py` (line 71).
- **Evidence:** `model_sf = smp.DeepLabV3Plus(encoder_name="mit_b3", ...)`.
- **Finding:** The model reported as "SegFormer MiT-B3" is structurally DeepLabV3+ with an MiT-B3 transformer backbone and an ASPP decoder, rather than the original SegFormer MLP decoder architecture.
- **Proposed Correction:** Label accurately in experiment ledgers as `smp-deeplabv3plus-mit-b3`.
