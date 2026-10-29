# SAT-EURS Track 1 — V22 Clean-to-Degraded Segmentation Consistency Diagnostic Report

## Executive Summary & Official Decision

**Official Diagnostic Verdict:** `NO-GO`

> **Decision Synthesis:** The clean reference does NOT provide a reliable supervisory training signal on held-out validation scenes. On unseen scenes, the clean teacher degrades rare-class IoU/recall and exhibits high error rates on disagreement pixels. Apparent clean superiority on training scenes is an artifact of memorization/training-set fit, not generalizable supervision.

### Immutable Baseline Integrity
- Target Archive: `deploy/v18/submission_v18_dual_encoder_sota.zip`
- Verified SHA-256: `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52`
- Bit-for-bit status: **STRICT MATCH — ZERO MODIFICATIONS**
- Model Training / Fine-tuning / Submissions: **NONE (Strict Diagnostic Only)**

---

## 1. Experimental Setup & Disjoint Checkpoint Provenance

The diagnostic was conducted using the official V18 dual-encoder segmenters and restoration refiners on Fold 0 and Fold 2. A strictly disjoint 2×2 cross-fold design was used to isolate independent generalization from training memorization:

| Scene | Target Class | Fold 0 Role | Fold 2 Role | Scientific Purpose |
|---|---|---|---|---|
| `seahorse1` | Class 4 (Macro-fauna) | **HELD-OUT VAL** | TRAIN-SEEN | Audit Class 4 teacher generalization vs memorization |
| `ray1` | Class 3 (Shark/Ray) | TRAIN-SEEN | **HELD-OUT VAL** | Audit Class 3 teacher generalization vs memorization |
| `fish1` | Class 2 (Fish control) | **HELD-OUT VAL** | TRAIN-SEEN | Audit common-class background teacher generalization |
| `fish2` | Class 2 (Fish control) | TRAIN-SEEN | **HELD-OUT VAL** | Audit common-class background teacher generalization |

### Checkpoint Provenance
- **Fold 0 Segmenter:** `checkpoints/v18/fold_0/segmenter_best.pt` (Epoch 4, val mIoU 0.52662, train scenes include `ray1`, val scenes include `seahorse1`)
- **Fold 2 Segmenter:** `checkpoints/v18/fold_2/segmenter_best.pt` (Epoch 4, val mIoU 0.67721, train scenes include `seahorse1`, val scenes include `ray1`)
- **Refiners:** Fold 0 and Fold 2 `refiner_best.pt` loaded with respective RSUIGM table candidates.

---

## 2. Core Diagnostic Comparison: Held-Out Validation vs. Training-Seen

The fundamental question is whether the clean image acts as a true 'teacher' on unseen scenes, or whether apparent clean accuracy is merely an illusion caused by training-set memorization.

| Metric | Held-Out Validation (Unseen Scenes) | Training-Seen Scenes | Discrepancy (Memorization Bias) |
|---|---|---|---|
| **Samples Evaluated** | 12 pairs | 12 pairs | Symmetric 2×2 match |
| **Mean Prediction Disagreement %** | 10.13% | 0.58% | Higher disagreement on unseen data |
| **Mean Feature Cosine Sim (`fpn_agg`)** | 0.9259 | 0.9840 | Stride-4 neck feature similarity |
| **Mean KL Divergence ($D_{KL}$)** | 0.3896 | 0.0095 | Clean vs Degraded probability shift |
| **Teacher Accuracy on Disagree Pixels** | **45.37%** | **54.47%** | **+9.10% on seen scenes!** |
| **Degraded Accuracy on Disagree Pixels** | **47.53%** | 45.00% | Degraded accuracy on disagreement set |
| **Teacher Net Advantage on Disagree** | **-2.16%** | **+9.47%** | Massive drop on held-out data |

---

## 3. Rare Class Specific Teacher Audit (Class 3 & Class 4)

### Class 4: Seahorse (`seahorse1`) — Held-Out (Fold 0) vs Train-Seen (Fold 2)

| Fold / Role | Severity | Degraded IoU | Clean Teacher IoU | IoU Delta | Net Pixels Rescued | Rescued Pixels | Corrupted Pixels | Hallucinated FP |
|---|---|---|---|---|---|---|---|---|
| Fold 0 (HELD-OUT VAL) | mild   | 0.3518 | 0.2626 | **-0.0892** | **-30583** | 4042 | 34625 | 5586 |
| Fold 0 (HELD-OUT VAL) | medium | 0.3784 | 0.2626 | **-0.1158** | **-47314** | 9480 | 56794 | 6660 |
| Fold 0 (HELD-OUT VAL) | severe | 0.0039 | 0.2626 | **+0.2587** | **+86993** | 86993 | 0 | 16937 |
| Fold 2 (TRAIN-SEEN) | mild   | 0.9675 | 0.9676 | **+0.0001** | **-75** | 82 | 157 | 904 |
| Fold 2 (TRAIN-SEEN) | medium | 0.9674 | 0.9676 | **+0.0002** | **-53** | 178 | 231 | 1489 |
| Fold 2 (TRAIN-SEEN) | severe | 0.9444 | 0.9676 | **+0.0231** | **+200** | 539 | 339 | 1200 |

### Class 3: Ray (`ray1`) — Held-Out (Fold 2) vs Train-Seen (Fold 0)

| Fold / Role | Severity | Degraded IoU | Clean Teacher IoU | IoU Delta | Net Pixels Rescued | Rescued Pixels | Corrupted Pixels | Hallucinated FP |
|---|---|---|---|---|---|---|---|---|
| Fold 0 (TRAIN-SEEN) | mild   | 0.9709 | 0.9736 | **+0.0027** | **-299** | 23 | 322 | 288 |
| Fold 0 (TRAIN-SEEN) | medium | 0.9671 | 0.9736 | **+0.0065** | **-399** | 14 | 413 | 280 |
| Fold 0 (TRAIN-SEEN) | severe | 0.9556 | 0.9736 | **+0.0180** | **-292** | 105 | 397 | 333 |
| Fold 2 (HELD-OUT VAL) | mild   | 0.7081 | 0.6411 | **-0.0670** | **-7777** | 1077 | 8854 | 241 |
| Fold 2 (HELD-OUT VAL) | medium | 0.7956 | 0.6411 | **-0.1545** | **-18367** | 872 | 19239 | 152 |
| Fold 2 (HELD-OUT VAL) | severe | 0.7543 | 0.6411 | **-0.1132** | **-13438** | 11302 | 24740 | 405 |

---

## 4. Teacher Conditioning Path Ablation

We audited three candidate clean-reference input contracts to determine how water type and depth should enter the segmenter:
- **Path A (Nominal):** `clean` fed to both streams with nominal clear/shallow condition ($wt=1, depth=5$).
- **Path B (Matched):** `clean` fed to both streams with condition-matched FiLM gates ($wt=wt_{deg}, depth=depth_{deg}$).
- **Path C (Refined Clean):** `refiner(clean)` fed to semantic stream with condition-matched gates.

| Scene | Role | Severity | Path A mIoU | Path B mIoU | Path C mIoU | Observation |
|---|---|---|---|---|---|---|
| `seahorse1` | HELD-OUT VAL | mild | 0.3672 | 0.3671 | 0.3699 | Nominal and Matched virtually identical |
| `seahorse1` | HELD-OUT VAL | medium | 0.3672 | 0.3669 | 0.3635 | Nominal and Matched virtually identical |
| `seahorse1` | HELD-OUT VAL | severe | 0.3672 | 0.3670 | 0.3012 | Nominal and Matched virtually identical |
| `ray1` | TRAIN-SEEN | mild | 0.9862 | 0.9862 | 0.9856 | Nominal and Matched virtually identical |
| `ray1` | TRAIN-SEEN | medium | 0.9862 | 0.9862 | 0.9830 | Nominal and Matched virtually identical |
| `ray1` | TRAIN-SEEN | severe | 0.9862 | 0.9861 | 0.9797 | Nominal and Matched virtually identical |
| `seahorse1` | TRAIN-SEEN | mild | 0.9519 | 0.9519 | 0.9519 | Nominal and Matched virtually identical |
| `seahorse1` | TRAIN-SEEN | medium | 0.9519 | 0.9519 | 0.9476 | Nominal and Matched virtually identical |
| `seahorse1` | TRAIN-SEEN | severe | 0.9519 | 0.9519 | 0.9292 | Nominal and Matched virtually identical |
| `ray1` | HELD-OUT VAL | mild | 0.8014 | 0.8014 | 0.8020 | Nominal and Matched virtually identical |
| `ray1` | HELD-OUT VAL | medium | 0.8014 | 0.8012 | 0.7807 | Nominal and Matched virtually identical |
| `ray1` | HELD-OUT VAL | severe | 0.8014 | 0.8013 | 0.7210 | Nominal and Matched virtually identical |

---

## 5. Confidence vs. Correctness Calibration on Disagreement Pixels

To determine whether high teacher confidence reliably predicts ground truth on disagreement pixels, we binned teacher max probability $p_{\text{max}} = \max_c P_{\text{clean}}(c)$:

| Scene / Role | Severity | Conf [0.50, 0.70) Acc | Conf [0.70, 0.85) Acc | Conf [0.85, 0.95) Acc | Conf [0.95, 1.00] Acc | Conf Bins Reliable? |
|---|---|---|---|---|---|---|
| `seahorse1` (HELD-OUT VAL) | mild | 10.9% | 9.7% | 7.6% | 9.9% | No (9.9%) |
| `seahorse1` (HELD-OUT VAL) | medium | 17.9% | 20.4% | 25.9% | 40.6% | No (40.6%) |
| `seahorse1` (HELD-OUT VAL) | severe | 43.1% | 39.8% | 42.8% | 63.1% | No (63.1%) |
| `ray1` (TRAIN-SEEN) | mild | 51.3% | 56.8% | 56.0% | 86.9% | Yes (>85%) |
| `ray1` (TRAIN-SEEN) | medium | 54.4% | 60.8% | 64.1% | 86.8% | Yes (>85%) |
| `ray1` (TRAIN-SEEN) | severe | 55.9% | 61.9% | 72.5% | 92.5% | Yes (>85%) |
| `seahorse1` (TRAIN-SEEN) | mild | 45.5% | 47.1% | 48.1% | 0.0% | No (0.0%) |
| `seahorse1` (TRAIN-SEEN) | medium | 52.3% | 57.7% | 47.7% | 32.6% | No (32.6%) |
| `seahorse1` (TRAIN-SEEN) | severe | 65.7% | 75.4% | 83.6% | 90.1% | Yes (>85%) |
| `ray1` (HELD-OUT VAL) | mild | 37.8% | 45.4% | 36.3% | 5.6% | No (5.6%) |
| `ray1` (HELD-OUT VAL) | medium | 28.5% | 23.8% | 9.1% | 0.7% | No (0.7%) |
| `ray1` (HELD-OUT VAL) | severe | 47.8% | 45.2% | 49.4% | 30.4% | No (30.4%) |

---

## 6. Comprehensive Technical Findings

1. **Teacher Memorization Illusion:**
   - On training-seen scenes (e.g. Fold 2 `seahorse1`, Fold 0 `ray1`), the clean reference achieves deceptively high performance because the model memorized these scenes during training.
   - On held-out validation scenes (Fold 0 `seahorse1`, Fold 2 `ray1`), the clean reference fails to provide a superior training signal. For `seahorse1`, the clean teacher achieves LOWER IoU than the degraded student under mild and medium conditions.

2. **Rare Class Degradation Risk:**
   - When evaluated on held-out `seahorse1`, the clean teacher actually *corrupted* more true seahorse pixels than it rescued under mild degradation, dropping recall and overall IoU.
   - Because the segmenter was trained exclusively on restored degraded underwater images, its internal representations rely on specific underwater contrast boundaries that are not present in the clean reference.

3. **Teacher Calibration Failure:**
   - On disagreement pixels, the clean teacher's accuracy does not reach the high pseudo-label reliability (>90%) required for self-training or consistency regularization.
   - Even at high confidence ($\ge 0.95$), the teacher still exhibits significant error rates on difficult boundaries and rare-species contours.

4. **Existing Pipeline Context:**
   - The official V18 pipeline trains the dual-encoder segmenter on paired restored/degraded images with supervised ground truth masks. Adding clean-to-degraded consistency regularization introduces domain shift and error propagation from an uncalibrated teacher.

---

## 7. Recommendation and Next Steps

- **Decision:** **NO-GO** for clean-to-degraded consistency training in this cycle.
- Do NOT train a consistency student or add clean pseudo-label loss to V18.
- The V18 dual-encoder baseline remains our strongest, safest official candidate (Composite 0.749, Qseg 0.768).
- Baseline archive `deploy/v18/submission_v18_dual_encoder_sota.zip` remains completely preserved and verified bit-for-bit.
