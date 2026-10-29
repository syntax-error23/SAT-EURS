# SAT-EURS Track 2 — Scoring Objective, Dataset, and Validation Protocol Audit

**Auditor:** Independent Senior Computer Vision Researcher & Competition Auditor  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Audited Code:** `metrics/segmentation_metrics.py`, `metrics/restoration_metrics.py`, `metrics/composite_score.py`, `splits/manifest_5fold.json`, `audits/class_distribution_stats.json`, `tools/audit_folds_and_support.py`  

---

## 1. Official Competition Scoring Objective Reconstruction

The official composite score for the MaCVi @ WACV 2027 EURS Challenge (Track 1 and Track 2) is formulated as:

$$S_{\text{composite}} = 0.60 \cdot Q_{\text{seg}} + 0.30 \cdot Q_{\text{FR}} + 0.10 \cdot Q_{\text{NR}}$$

### 1.1 Structural Metric Properties: Additive vs. Multiplicative
- **High-Level Weighting:** Additive combination ($60\%$ Segmentation, $30\%$ Full-Reference Restoration, $10\%$ No-Reference Quality).
- **Segmentation Sub-Objective ($Q_{\text{seg}}$):** **Strictly Multiplicative** with a Hard Veto:
  $$Q_{\text{seg}} = \text{mIoU} \cdot \text{Macro\_F1} \cdot P_{\text{cov}}$$
  If a model fails to detect even a single pixel of any ground-truth class present in the evaluation split ($\text{TP}_c = 0$), $P_{\text{cov}} = 0.0$, which **instantly zeroes the entire segmentation score**, reducing the maximum achievable composite score from $1.000$ down to $0.400$ ($0.30 \cdot Q_{\text{FR}} + 0.10 \cdot Q_{\text{NR}}$).
- **Restoration Sub-Objective ($Q_{\text{FR}}$):** Additive combination of normalized fidelity metrics:
  $$Q_{\text{FR}} = 0.45 \cdot \text{PSNR}_{\text{norm}} + 0.45 \cdot \text{SSIM} + 0.10 \cdot \text{Sharpness}$$

---

## 2. Component-by-Component Formula Audit

### 2.1 Segmentation Metric Implementation (`metrics/segmentation_metrics.py`)
1. **Per-Class IoU and F1:**
   $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c + \epsilon}, \quad F_{1,c} = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c + \epsilon}$$
2. **Mean IoU & Macro F1:**
   $$\text{mIoU} = \frac{1}{|C|} \sum_{c \in C} \text{IoU}_c, \quad \text{Macro\_F1} = \frac{1}{|C|} \sum_{c \in C} F_{1,c}$$
3. **Coverage Penalty ($P_{\text{cov}}$):**
   $$P_{\text{cov}} = \begin{cases} 1.0 & \text{if } \forall c \in C_{\text{present}}: \text{TP}_c > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
4. **The "Absent Class" Anomaly Discovered in Local Code:**
   In `metrics/segmentation_metrics.py` (lines 55–60):
   ```python
   denominator_iou = tp + fp + fn
   per_class_iou = np.where(denominator_iou > 0, tp / (denominator_iou + eps), 1.0)
   denominator_f1 = 2 * tp + fp + fn
   per_class_f1 = np.where(denominator_f1 > 0, (2 * tp) / (denominator_f1 + eps), 1.0)
   ```
   > [!CAUTION]
   > **Critical Forensic Discovery:** When a class is absent from the ground truth ($\text{TP}_c + \text{FN}_c = 0$) AND the model predicts zero pixels for that class ($\text{FP}_c = 0$), the denominator is exactly zero. The code assigns $\text{IoU}_c = 1.0$ and $F_{1,c} = 1.0$ (vacuous truth).
   > When evaluating on a split where a class is missing (e.g. Fold 0 missing Class 3), suppressing that class completely gives an unearned 1.0, artificially inflating the 5-class average mIoU by up to $+0.20$!

### 2.2 Full-Reference Restoration Implementation (`metrics/restoration_metrics.py`)
1. **PSNR Normalization Window:**
   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$
   $$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0}, 0.0, 1.0\right)$$
   - Below 15.0 dB $\implies \text{PSNR}_{\text{norm}} = 0.0$.
   - Above 35.0 dB $\implies \text{PSNR}_{\text{norm}} = 1.0$.
   - Each $1.0\text{ dB}$ increase between $15$ and $35\text{ dB}$ contributes:
     $$\Delta Q_{\text{FR}} = 0.45 \cdot \frac{1.0}{20.0} = +0.0225 \implies \Delta S_{\text{composite}} = 0.30 \cdot 0.0225 = \mathbf{+0.00675}$$
2. **SSIM:** Mean SSIM across R, G, B channels with window size 11, $\sigma=1.5$, $C_1 = 0.01^2, C_2 = 0.03^2$.
3. **Sharpness Ratio:**
   $$\text{Sharpness} = \text{clip}\left(\frac{\text{Var}(\text{Laplacian}(I_{\text{pred\_gray}}))}{\text{Var}(\text{Laplacian}(I_{\text{clean\_gray}}))}, 0.0, 1.0\right)$$
   Protected against zero denominator ($\text{Var}_{\text{clean}} < 10^{-7}$).

### 2.3 No-Reference Quality Metric ($Q_{\text{NR}}$)
- The official challenge uses a server-side ensemble of UIQM, UCIQE, and CCF.
- Local evaluation does not possess the proprietary official server evaluator.
- Local evaluation uses a constant stub $Q_{\text{NR}} = 0.5000$, and projects server composite scores using the official server baseline average $Q_{\text{NR}} = 0.7180$.
- Because $Q_{\text{NR}}$ is weighted at $10\%$, a $0.20$ swing in $Q_{\text{NR}}$ represents $\pm 0.020$ on the final composite score.

---

## 3. Dataset Census and Semantic Class Distributions

The EURS dataset comprises 20 unique clean scenes, each synthetically degraded across 10 Jerlov water types and 20 depths ($5\text{m} \dots 100\text{m}$ in $5\text{m}$ increments), yielding exactly 200 degraded variants per scene (4,000 total degraded images).

### 3.1 Class Census & Distribution (`audits/class_distribution_stats.json`)

| Class ID | Semantic Class Name | Total Pixels (4,000 imgs) | Pixel Percentage | Relative Frequency | Scene Support (Scenes with Class > 0) |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Background** (Water / Seabed) | 199,628,984 | **55.31%** | 1.00x | 18 of 20 scenes (absent in `fish18`, `fish20`) |
| **1** | **Coral / Substrate** | 121,751,549 | **33.73%** | 0.61x | 16 of 20 scenes (absent in `fish15`, `fish21`, `fish7`, `starfish2`, `turtles1`) |
| **2** | **Fish** | 28,166,499 | **7.80%** | 0.14x | 16 of 20 scenes (absent in `ray1`, `seahorse1`, `starfish2`, `turtles1`) |
| **3** | **Elasmobranch** (Shark / Ray) | 3,389,841 | **0.94%** | 0.017x | **Only 2 scenes:** `ray1` (113k px), `sharkFish` (48k px) |
| **4** | **Other** (Invertebrates / Fauna)| 7,977,590 | **2.21%** | 0.040x | **Only 3 scenes:** `seahorse1` (318k px), `starfish2` (14k px), `turtles1` (277k px) |

> [!IMPORTANT]
> **Extreme Class Imbalance:** Classes 3 and 4 combined represent only **3.15%** of all pixels in the dataset, but contribute **40.0%** of the mIoU and Macro F1 metrics (2 out of 5 classes).
> Furthermore, Class 3 is present in ONLY two scenes in the entire dataset, and Class 4 in ONLY three scenes.

---

## 4. The 5-Fold Validation Split Integrity Audit

The 5-fold cross-validation partition is specified in `splits/manifest_5fold.json`. We audited the per-fold class presence using `tools/audit_folds_and_support.py`:

| Fold ID | Validation Scenes | Clean Images | Degraded Images | Class 0 Px | Class 1 Px | Class 2 Px | Class 3 Px (Shark/Ray) | Class 4 Px (Macro-fauna) | Rare-Class Integrity Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fold 0** | `fish1, fish11, fish12, seahorse1` | 4 | 800 | 1,725,359 | 1,971,563 | 178,681 | **0 (ABSENT)** | 318,701 | **DEFECTIVE (Missing Class 3)** |
| **Fold 1** | `fish13, fish14, fish15, sharkFish`| 4 | 800 | 2,874,630 | 1,049,630 | 221,756 | 48,288 | **0 (ABSENT)** | **DEFECTIVE (Missing Class 4)** |
| **Fold 2** | `fish17, fish18, fish2, ray1` | 4 | 800 | 1,212,737 | 2,725,326 | 143,091 | 113,150 | **0 (ABSENT)** | **DEFECTIVE (Missing Class 4)** |
| **Fold 3** | `fish20, fish21, fish4, starfish2` | 4 | 800 | 2,232,318 | 1,408,283 | 539,652 | **0 (ABSENT)** | 14,051 | **DEFECTIVE (Missing Class 3)** |
| **Fold 4** | `fish5, fish7, fish9, turtles1` | 4 | 800 | 2,114,352 | 1,641,835 | 161,020 | **0 (ABSENT)** | 277,097 | **DEFECTIVE (Missing Class 3)** |

### 4.1 Root Cause of Previous Validation Failures
1. **Single-Fold Blindness:** Every single fold in the 5-fold partition contains only 4 scenes, and by combinatorial constraint, **no single fold contains all 5 classes**.
2. **Prior Track 2 Flaw:** The previous Track 2 evaluation scripts (`tools/evaluate_track2_onnx.py` and `tools/train_track2_student.py`) evaluated exclusively on Fold 0.
3. **The Artificial Calibration Bias:** In `tools/train_track2_student.py` lines 481–501, the author swept Class 3 logit offsets on Fold 0 and selected $-4.0$. Because Fold 0 has 0 pixels of Class 3, the $-4.0$ offset caused the model to predict 0 pixels of Class 3.
4. **The Inflated Result:** Because $0/0 \implies 1.0$, Class 3 received a perfect $1.000$ IoU and $1.000$ F1, artificially driving reported mIoU to $0.5389$ and $Q_{\text{seg}}$ to $0.3314$.
5. **The Real-World Catastrophe:** If this model is evaluated on Fold 1, Fold 2, or the official test set (where Class 3 is present), the $-4.0$ offset suppresses true Class 3 detections, driving $\text{TP}_3 \to 0$. This triggers $P_{\text{cov}} = 0.0$, **reducing $Q_{\text{seg}}$ to exactly 0.0000**!

---

## 5. Corrected Validation Protocol: Deterministic Fold Pairing

To guarantee that **all 5 classes have positive support in both training and validation**, validation must never be run on a single 4-scene fold.

Using `tools/audit_folds_and_support.py`, all 10 possible 2-fold combinations ($S_{\text{folds}}$) were evaluated. The mathematically optimal partition is:

### Primary Development & Validation Split ($S_{\text{folds}} = \{0, 1\}$ vs. $C_{\text{folds}} = \{2, 3, 4\}$):
- **Validation Group ($S_{\text{folds}} = \{0, 1\}$, 8 Scenes, 1,600 Images):**
  - Scenes: `fish1`, `fish11`, `fish12`, `seahorse1`, `fish13`, `fish14`, `fish15`, `sharkFish`.
  - Class 0 Px: 4,599,989 | Class 1 Px: 3,021,193 | Class 2 Px: 400,437
  - **Class 3 Px (Shark/Ray): 48,288 (Positive Support, 1 scene)**
  - **Class 4 Px (Other): 318,701 (Positive Support, 1 scene)**
  - All 5 classes are strictly present: $\text{GT}_c > 0$ for all $c \in \{0, 1, 2, 3, 4\}$.
- **Training Group ($C_{\text{folds}} = \{2, 3, 4\}$, 12 Scenes, 2,400 Images):**
  - Scenes: `fish17, fish18, fish2, ray1, fish20, fish21, fish4, starfish2, fish5, fish7, fish9, turtles1`.
  - Class 3 Px: 113,150 | Class 4 Px: 291,148.
  - All 5 classes are strictly present.

### Validation Integrity Rules:
1. **Never evaluate Track 2 on Fold 0 alone.** All future student evaluations must evaluate on the 8-scene combined split $S_{\text{folds}} = \{0, 1\}$ (1,600 images) or full 5-fold Out-Of-Fold cross-validation (4,000 images).
2. **Eliminate the vacuous 1.0 assignment.** Evaluation scripts must calculate mIoU strictly over classes where $\text{GT}_c > 0$, and separately report the full 5-class average with zero-credit for unpredicted classes.
3. **Reject any negative logit bias that suppresses rare classes.** Logit offsets must be bounded and validated across scenes where the target class is present.
