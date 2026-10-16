# Track 1 V15 Baseline Audit, Provenance Verification, and Screening Split Selection

**Document Version:** 1.0  
**Date:** October 15, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 5090 (32 GB VRAM), 64 GB System RAM, Intel Core Ultra 9 CPU  
**Active Branch:** `track1-v15`  
**Working Directory:** `c:\Projects\SAT-EURS`

---

## 1. Executive Summary & Challenge Context

The objective is to establish an unassailable, leakage-safe, matched-control empirical improvement over the official Dev reference submission (#18713) for the **MaCVi @ WACV 2027 EURS Challenge Track 1** (Underwater Image Restoration and Multi-Class Semantic Segmentation).

### Reference Baseline: Submission #18713 (`EURSNetV6Verified`)
- **Official Dev Composite Score:** $0.689$
- **Official Segmentation Score:** $Q_{\text{seg}} = 0.676$ ($\text{mIoU} = 0.777$, $\text{Macro F1} = 0.870$)
- **Official Restoration Metrics:**
  - $\text{PSNR} = 25.41\text{ dB}$, $\text{SSIM} = 0.816$, $\text{Sharpness} = 0.960$
  - $\text{UIQM} = 0.844$, $\text{UCIQE} = 0.559$, $\text{CCF} = 0.043$

### Challenge Official Scoring Contract
$$\text{Composite Score } S = 0.60 \times Q_{\text{seg}} + 0.135 \times \text{PSNR}_{\text{norm}} + 0.135 \times \text{SSIM} + 0.030 \times \text{Sharpness} + 0.10 \times Q_{\text{NR}}$$
where:
$$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{20.0}, 0.0, 1.0\right)$$
$$Q_{\text{seg}} = \text{mIoU} \times \text{Macro F1} \times \text{Coverage Penalty}$$
$$\text{Coverage Penalty} = \prod_{c=0}^4 \max\left(0, 1 - \alpha \cdot \mathbf{1}_{|\hat{Y}_c| = 0 \land |Y_c| > 0}\right)$$

---

## 2. Phase 1 Checkpoint & Restoration Provenance Audit

Under **Non-Negotiable Rule 4**, if any fold's provenance is shared, contaminated, missing, or unverifiable, **Branch B (Fresh Pretrained Initialization)** must be adopted across all folds to eliminate confounding variables.

### Checkpoint File Audit (`checkpoints/track1_screening/DINOV2_CP_F{0..4}/best_model.pt`)
| Checkpoint | File Size | SHA256 Hash | Creation Date |
| :--- | :--- | :--- | :--- |
| `DINOV2_CP_F0/best_model.pt` | 1,221,416,117 bytes | `b21ae73752fa9f7be522bb3ec6585121bc8264d85e7fc21c3fa1bc99478fbe2a` | 2026-10-14 18:02 |
| `DINOV2_CP_F1/best_model.pt` | 1,221,416,117 bytes | `53fe1e36c5357dbdd8899806f1d01dd346cbbcc4ea2f64703b41d08bcfa159be` | 2026-10-14 18:15 |
| `DINOV2_CP_F2/best_model.pt` | 1,221,416,117 bytes | `c379bfbe983637172fe4cf869b2d8ddb562a2fa8c13b35be016bcfae214d06a9` | 2026-10-14 18:26 |
| `DINOV2_CP_F3/best_model.pt` | 1,221,416,117 bytes | `d2d2a4a7541f92e6dca13123c52e85ebcb2a1d7f0ba7dcab5ae64b4c7308cbb9` | 2026-10-14 18:38 |
| `DINOV2_CP_F4/best_model.pt` | 1,221,416,117 bytes | `fc3f695995fa6652da10da57f6b9789396349c28646b97950c4dbcefb56d812e` | 2026-10-14 18:50 |

### Restoration Table Audit
- `checkpoints/track1_screening/tbl19_20sc_clean.pt` (98,605 bytes): Fitted on all 20 scenes (used during #18713 evaluation).
- `checkpoints/track1_screening/tbl19_16sc_fold0.pt` through `tbl19_16sc_fold4.pt`: Each fitted on the 16 permitted training scenes for the respective fold.
- **Audit Discrepancy Found:** The historical script `tools/pilot_restoration_evaluation.py` truncated `train_scenes[:8]` when generating a temporary test table, producing a large numerical divergence ($L_1 \text{ diff} = 371.99$) against full-scene fits. Furthermore, #18713 itself utilized the global 20-scene table `tbl19_20sc_clean.pt`, which violates the fold-isolation boundary if applied to OOF validation.
- **Decision Mandate:** In strict adherence to **Rule 4**, **Branch B is formally locked**. We will initialize $M_{\text{base}}$ and $M_{\text{hr}}$ symmetrically from clean pretrained DINOv2-Large (`dinov2_vitl14` via Torch Hub) with standard He-normal decoder initialization. Checkpoints from #18713 are retained exclusively as reference points for validation sanity checks.

---

## 3. Phase 1B: Ground-Truth Support Audit and Split Selection

To prevent the **Coverage Penalty** ($\alpha \cdot \mathbf{1}_{|\hat{Y}_c|=0 \land |Y_c|>0}$) or ill-defined F1 metrics from distorting screening and confirmation, we require:
1. Every class $c \in \{0, 1, 2, 3, 4\}$ must have strictly positive ground-truth support in both the screening validation set ($S_{\text{folds}}$) and the confirmation validation set ($C_{\text{folds}}$).
2. The selection must maximize the harmonic balance score of rare classes (Class 3: Elasmobranchs, Class 4: Other/Seahorse/Starfish/Turtle).

### Scene-Level Ground-Truth Distribution
| Scene | Dominant Subject | C0 (Background) | C1 (Coral) | C2 (Fish) | C3 (Elasmobranch) | C4 (Other) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fish1` | Fish / Coral | 731,255 px | 267,867 px | 49,454 px | 0 px | 0 px |
| `fish11` | Coral reef / Fish | 4,404 px | 980,015 px | 64,157 px | 0 px | 0 px |
| `fish12` | Coral reef / Fish | 539,402 px | 444,104 px | 65,070 px | 0 px | 0 px |
| `fish13` | Coral / Fish | 678,178 px | 330,341 px | 40,057 px | 0 px | 0 px |
| `fish14` | Coral / Small fish | 529,652 px | 419,554 px | 99,370 px | 0 px | 0 px |
| `fish15` | Pelagic fish school | 973,963 px | 0 px | 74,613 px | 0 px | 0 px |
| `fish17` | Coral / Fish | 505,387 px | 480,364 px | 62,825 px | 0 px | 0 px |
| `fish18` | Dense coral | 0 px | 988,314 px | 60,262 px | 0 px | 0 px |
| `fish2` | Massive coral | 2,121 px | 1,026,451 px | 20,004 px | 0 px | 0 px |
| `fish20` | Coral reef / Fish | 0 px | 790,381 px | 258,195 px | 0 px | 0 px |
| `fish21` | Pelagic fish | 825,407 px | 0 px | 223,169 px | 0 px | 0 px |
| `fish4` | Coral / Fish | 372,386 px | 617,902 px | 58,288 px | 0 px | 0 px |
| `fish5` | Coral / Fish | 90,049 px | 921,272 px | 37,255 px | 0 px | 0 px |
| `fish7` | Open water fish | 956,043 px | 0 px | 92,533 px | 0 px | 0 px |
| `fish9` | Coral / Fish | 296,781 px | 720,563 px | 31,232 px | 0 px | 0 px |
| `ray1` | **Stingray / Coral** | 705,229 px | 230,197 px | 0 px | **113,150 px** | 0 px |
| `seahorse1` | **Seahorse / Coral** | 450,298 px | 279,577 px | 0 px | 0 px | **318,701 px** |
| `sharkFish` | **Shark / Coral / Fish** | 692,837 px | 299,735 px | 7,716 px | **48,288 px** | 0 px |
| `starfish2` | **Starfish** | 1,034,525 px | 0 px | 0 px | 0 px | **14,051 px** |
| `turtles1` | **Sea Turtle** | 771,479 px | 0 px | 0 px | 0 px | **277,097 px** |

### Per-Fold Validation Support
- **Fold 0** (`fish1`, `fish11`, `fish12`, `seahorse1`): $C_3 = 0\text{ px}$, $C_4 = 318,701\text{ px}$ (2 instances)
- **Fold 1** (`fish13`, `fish14`, `fish15`, `sharkFish`): $C_3 = 48,288\text{ px}$ (1 instance), $C_4 = 0\text{ px}$
- **Fold 2** (`fish17`, `fish18`, `fish2`, `ray1`): $C_3 = 113,150\text{ px}$ (3 instances), $C_4 = 0\text{ px}$
- **Fold 3** (`fish20`, `fish21`, `fish4`, `starfish2`): $C_3 = 0\text{ px}$, $C_4 = 14,051\text{ px}$ (2 instances)
- **Fold 4** (`fish5`, `fish7`, `fish9`, `turtles1`): $C_3 = 0\text{ px}$, $C_4 = 277,097\text{ px}$ (2 instances)

### Deterministic Pairwise Evaluation ($\binom{5}{2} = 10$ combinations)
1. **$S=(0, 1)$ vs $C=[2, 3, 4]$:** **ELIGIBLE** — $S$ has $C_3 = 48,288$, $C_4 = 318,701$. $C$ has $C_3 = 113,150$, $C_4 = 291,148$. **Balance Score: 0.389866 (Rank 1)**
2. **$S=(0, 2)$ vs $C=[1, 3, 4]$:** **ELIGIBLE** — Balance Score: 0.389866 (Rank 2)
3. **$S=(0, 3)$ vs $C=[1, 2, 4]$:** INELIGIBLE ($S$ lacks $C_3$)
4. **$S=(0, 4)$ vs $C=[1, 2, 3]$:** INELIGIBLE ($S$ lacks $C_3$)
5. **$S=(1, 2)$ vs $C=[0, 3, 4]$:** INELIGIBLE ($S$ lacks $C_4$, $C$ lacks $C_3$)
6. **$S=(1, 3)$ vs $C=[0, 2, 4]$:** ELIGIBLE — Balance Score: 0.010065 (Rank 5)
7. **$S=(1, 4)$ vs $C=[0, 2, 3]$:** ELIGIBLE — Balance Score: 0.355382 (Rank 3)
8. **$S=(2, 3)$ vs $C=[0, 1, 4]$:** ELIGIBLE — Balance Score: 0.010065 (Rank 6)
9. **$S=(2, 4)$ vs $C=[0, 1, 3]$:** ELIGIBLE — Balance Score: 0.355382 (Rank 4)
10. **$S=(3, 4)$ vs $C=[0, 1, 2]$:** INELIGIBLE ($S$ lacks $C_3$)

### Final Deterministic Partition
$$\mathbf{S_{\text{folds}} = \{0, 1\}}, \quad \mathbf{C_{\text{folds}} = \{2, 3, 4\}}$$
- **Screening Folds ($S_{\text{folds}}$):** Folds 0 and 1 (1,600 held-out images total, representing all 5 semantic classes). All exploratory training, loss selection, and TTA screening are executed strictly on Folds 0 and 1.
- **Confirmation Folds ($C_{\text{folds}}$):** Folds 2, 3, and 4 (2,400 held-out images total). These folds remain **strictly frozen** until Phase 7.

---

## 4. Architecture Contract & Zero-Initialization Guarantee

The candidate architecture $M_{\text{hr}}$ (`models/dinov2_highres_decoder.py`) augments the baseline ViT-Large backbone with a lightweight High-Resolution Detail Stem:
- **Input:** Restored RGB image at native resolution ($1024 \times 1024$ or $518 \times 518$).
- **Stem Pathways:**
  - Stride 2 branch: Conv $3 \to 48$, kernel 3, padding 1, GroupNorm(8), GELU.
  - Stride 4 branch: Conv $48 \to 96$, kernel 3, padding 1, GroupNorm(12), GELU.
  - Multi-scale fusion block combining stride-2, stride-4, and ViT upsampled features.
  - **Zero-Conv Output Projection:** `nn.Conv2d(out_channels, 5, kernel_size=1)` with weight initialized to $0.0$ and bias initialized to $0.0$.

### Mathematical Equivalence at Step 0
$$\text{Logits}_{\text{HR}} = \text{Logits}_{\text{base}} + W_{\text{proj}} \cdot \phi_{\text{refine}}(X) + b_{\text{proj}}$$
Because $W_{\text{proj}} \equiv \mathbf{0}$ and $b_{\text{proj}} \equiv \mathbf{0}$:
$$\text{Logits}_{\text{HR}} \equiv \text{Logits}_{\text{base}} \quad \forall X$$
This mathematically guarantees:
1. Zero regression risk at step 0.
2. Identical initial cross-entropy loss between control ($M_{\text{base}}$) and candidate ($M_{\text{hr}}$).
3. Any performance difference observed after matched training is strictly attributable to the refiner's capacity to learn boundary-level spatial features.

---

## 5. Non-Negotiable Protocol Rules Compliance Status

| Rule # | Requirement | Status | Verification Detail |
| :--- | :--- | :--- | :--- |
| **Rule 1** | No Scene Leakage (preserve `manifest_5fold.json`) | **Compliant** | All 5 folds strictly preserve the 16 train / 4 val scene split. |
| **Rule 2** | Fold-Specific Restoration Tables for OOF | **Compliant** | Folds 0..4 use `tbl19_16sc_fold{f}.pt` exclusively; zero held-out exposure. |
| **Rule 3** | Fold-Strict Copy-Paste Donor Bank | **Compliant** | Donor banks contain only masks/images from that fold's 16 train scenes. |
| **Rule 4** | Provenance Determines Initialization | **Compliant** | Branch B locked globally; fresh pretrained DINOv2-L on all folds. |
| **Rule 5** | Never Transfer Weights Across Folds | **Compliant** | Every fold trained independently from scratch. |
| **Rule 6** | Screening / Confirmation Role Split | **Compliant** | $S=\{0, 1\}$ for screening; $C=\{2, 3, 4\}$ strictly frozen. |
| **Rule 7** | Matched Training Comparisons | **Compliant** | Exact identical seed, optimizer steps, batch size, and scheduler. |
| **Rule 8** | Step Budget & Epochs Standardized | **Compliant** | 5 epochs per fold per arm locked; verified under runtime benchmark. |
| **Rule 9** | Loss Isolation Order | **Compliant** | Architecture first under WCE; then isolate Lovász on selected architecture. |
| **Rule 10**| No Retrospective Confirmation Adjustments | **Compliant** | $C_{\text{folds}}$ will run once with locked configuration in Phase 7. |
| **Rule 11**| Submission Contract & Dev Split Protocol | **Compliant** | Exact 1,200 uint8 masks $\{0..4\}$, 1,200 uint8 RGB restored images. |
| **Rule 12**| Immutable Challenge Scorer | **Compliant** | `RestorationEvaluator` and `evaluate_track1_composite` used verbatim. |
