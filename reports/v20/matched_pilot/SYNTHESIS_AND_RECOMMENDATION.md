# SAT-EURS Track 1 — V20 Matched Loss Ablation Benchmark: Synthesis & Decision

**Benchmark Status:** COMPLETE  
**Execution Date:** October 18, 2026 (Completed 14:39:46 UTC)  
**Execution Platform:** NVIDIA GeForce RTX 5090 (32 GB VRAM, PyTorch 2.x, BF16 AMP)  
**Immutable Baseline Archive:** `deploy/v18/submission_v18_dual_encoder_sota.zip`  
**Baseline Cryptographic Hash (SHA-256):** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52` (Verified 100% Intact Pre- and Post-Run)  

---

## 1. Executive Summary & Final Recommendation

| Benchmark Dimension | Measured Result | Evaluation Gate | Status |
| :--- | :--- | :--- | :---: |
| **Gate 1: Repeatable Improvement over Matched Control** | Fold 0: $\Delta Q_{\text{seg}} = +0.05726$; Fold 2: $\Delta Q_{\text{seg}} = -0.18705$ (Mean: **$-0.06490$**) | Positive $\Delta Q_{\text{seg}}$ on both test folds | **FAILED** |
| **Gate 2: Class-4 Gain Without Majority Class Degradation** | Fold 0 Class 4 IoU: $+0.16315$; but Fold 2 Class 2 IoU: $-0.20240$, Class 0: $-0.06324$ | Preserved majority class IoUs | **FAILED** |
| **Gate 3: Credible Class-3 Gain on Fold 2 (`ray1`)** | Control Class 3 IoU: **0.53320** vs Treatment: **0.29840** ($\Delta = \mathbf{-0.23480}$) | Treatment exceeds Control on Class 3 | **FAILED** |
| **Gate 4: Elimination of Late-Epoch Boundary Collapse** | Fold 0 Ep 4 $\to$ Ep 9: 0.3960 $\to$ 0.1374; Fold 2 Ep 4 $\to$ Ep 9: 0.2984 $\to$ 0.0994 | Monotonic / stable late convergence | **FAILED** |
| **Baseline Archive Integrity** | Pre-run SHA256: `e82cbea...`; Post-run SHA256: `e82cbea...` | Byte-for-byte identical | **PASSED** |

### **FINAL EXECUTIVE VERDICT: STRICT NO-GO FOR FULL FIVE-FOLD V20 TRAINING**

> **DO NOT LAUNCH FULL FIVE-FOLD V20 TRAINING WITH THIS RECIPE.**  
> Under a strictly controlled, matched A/B protocol where both arms start from the identical initial model state and share the exact same architecture, data sampler, and 9-epoch schedule:
> 1. The V20 class-weighted loss recipe produced an **isolated gain on Fold 0** ($\Delta Q_{\text{seg}} = +0.05726$, driven by Class 4), but suffered a **catastrophic regression on Fold 2** ($\Delta Q_{\text{seg}} = -0.18705$, $\Delta \text{Class 3} = -0.23480$, $\Delta \text{Class 2} = -0.20240$).
> 2. On Fold 2, the **Control Arm (V18 original loss recipe)** achieved an outstanding $Q_{\text{seg}} = \mathbf{0.60655}$ with Class 3 IoU of **0.53320** (recall 53.67%, precision 98.80%), whereas the Treatment Arm reached only $Q_{\text{seg}} = 0.41950$ and Class 3 IoU of 0.29840.
> 3. The class-weighted Lovász loss causes severe late-epoch feature collapse as the learning rate decays, collapsing rare-class recall by over $65\%$ between Epoch 4 and Epoch 9.

---

## 2. Quantitative Comparative Tables

### 2.1 Fold 0 Paired Results (`val: ['fish1', 'fish11', 'fish12', 'seahorse1']`)
*Ground Truth Classes Present: 0 (Background), 1 (Coral Reef), 2 (Fish), 4 (Macro-Fauna / Seahorse). Class 3 is absent.*

| Metric Dimension | Control Arm (V18 Recipe) | Treatment Arm (V20 Recipe) | Absolute Delta ($\Delta$) | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Best Checkpoint Epoch** | Epoch 5 | Epoch 4 | $-1$ Epoch | — |
| **Val $Q_{\text{seg}}$ (Present)** | **0.34837** | **0.40563** | **+0.05726** | **WIN** |
| **Val mIoU (Present)** | 0.52672 | 0.56956 | **+0.04284** | **WIN** |
| **Val Macro F1 (Present)** | 0.66141 | 0.71218 | **+0.05077** | **WIN** |
| **Coverage Penalty** | 1.0000 | 1.0000 | 0.0000 | MATCH |
| Class 0 (Background) IoU | **0.70704** | 0.68708 | $-0.01996$ | PASS ($-2\%$) |
| Class 1 (Coral Reef) IoU | 0.76708 | **0.76947** | $+0.00239$ | PASS |
| Class 2 (Fish) IoU | 0.39996 | **0.42573** | $+0.02577$ | PASS |
| Class 3 (Elasmobranch) IoU | 0.00000 (Absent) | 0.00000 (Absent) | $0.00000$ | N/A |
| **Class 4 (Macro-Fauna) IoU** | **0.23280** | **0.39595** | **+0.16315** | **MASSIVE GAIN** |
| Class 4 Precision | **0.79567** | 0.66167 | $-0.13400$ | Lower |
| Class 4 Recall | 0.24760 | **0.49646** | **+0.24886** | **2x RECALL** |
| Class 4 True Positives (pixels) | 3,946,608 | **7,913,383** | **+3,966,775** | **+100.5%** |
| Strict 5-Class $Q_{\text{seg}}$ (0.0 for absent) | 0.22296 | 0.25960 | $+0.03664$ | WIN |

---

### 2.2 Fold 2 Paired Results (`val: ['fish17', 'fish18', 'fish2', 'ray1']`)
*Ground Truth Classes Present: 0 (Background), 1 (Coral Reef), 2 (Fish), 3 (Elasmobranch / Ray). Class 4 is absent.*

| Metric Dimension | Control Arm (V18 Recipe) | Treatment Arm (V20 Recipe) | Absolute Delta ($\Delta$) | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Best Checkpoint Epoch** | **Epoch 9** | Epoch 4 | $+5$ Epochs | Control Kept Learning |
| **Val $Q_{\text{seg}}$ (Present)** | **0.60655** | **0.41950** | **-0.18705** | **SEVERE LOSS** |
| **Val mIoU (Present)** | **0.73060** | 0.59332 | **-0.13728** | **SEVERE LOSS** |
| **Val Macro F1 (Present)** | **0.83021** | 0.70703 | **-0.12318** | **SEVERE LOSS** |
| **Coverage Penalty** | 1.0000 | 1.0000 | 0.0000 | MATCH |
| Class 0 (Background) IoU | **0.89218** | 0.82894 | **-0.06324** | LOSS |
| Class 1 (Coral Reef) IoU | **0.94762** | 0.89894 | **-0.04868** | LOSS |
| Class 2 (Fish) IoU | **0.54941** | 0.34701 | **-0.20240** | **SEVERE LOSS** |
| **Class 3 (Elasmobranch) IoU** | **0.53320** | **0.29840** | **-0.23480** | **CATASTROPHIC LOSS** |
| Class 3 Precision | **0.98795** | 0.98350 | $-0.00445$ | Parity |
| Class 3 Recall | **0.53669** | 0.29990 | **-0.23679** | **$-44.1\%$ Recall** |
| Class 3 True Positives (pixels) | **3,036,166** | 1,696,592 | **-1,339,574** | **$-44.1\%$** |
| Class 4 Hallucinated Pixels (FP) | **191,357** | 511,904 | $+320,547$ | **+167% False Positives** |
| Strict 5-Class $Q_{\text{seg}}$ (0.0 for absent) | **0.38819** | 0.26848 | **-0.11971** | SEVERE LOSS |

---

## 3. Key Technical Takeaways

1. **Unweighted Lovász is a vital structural regularizer:**
   In V18's original loss, Lovász was unweighted ($\frac{1}{|C|} \sum L_c$). Because Lovász gradient operates on sorted permutation errors, adding high multipliers ($w_c = 3.5$) creates severe non-local gradient shocks on rare scenes (`sharkFish`) that destabilize feature representations.
2. **The "Majority Scene Drift" Phenomenon:**
   Because rare classes appear in only 1 or 2 scenes out of 20, 15 out of 16 training scenes in a fold contain 0 rare-class pixels. Unweighted Lovász allows the optimizer to converge smoothly without massive oscillation. Class-weighted Lovász forces large updates on rare crops, followed by catastrophic forgetting on majority crops.
3. **Control Arm SOTA on Fold 2:**
   Under identical 9-epoch training, the original V18 loss recipe achieved **$Q_{\text{seg}} = 0.60655$** and **Class 3 IoU = $0.53320$** at Epoch 9. This demonstrates that the V18 baseline architecture and loss are fundamentally capable of high-performance rare-class segmentation without the instability introduced by V20.

---

## 4. Cryptographic Proof of Baseline Integrity

```
File: deploy/v18/submission_v18_dual_encoder_sota.zip
Expected SHA256: e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52
Pre-Run SHA256:  e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52 [MATCH]
Post-Run SHA256: e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52 [MATCH]
```
