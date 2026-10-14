# EURS-Net Score Discrepancy & Mathematical Reconciliation Audit (Phase 1.2 & 1.3)

**Challenge:** MaCVi @ WACV 2027 EURS Challenge  
**Date:** October 14, 2026  
**Auditor:** Antigravity CV Research Team  
**Objective:** Forensically Reconcile Why the Official Leaderboard Score Remains ~0.50 Instead of the Projected 0.58–0.65+  

---

## 1. Official Evaluation Formulation

The official MaCVi composite scoring equation is:

$$S_{\text{composite}} = \mathbf{0.60} \cdot Q_{\text{seg}} + \mathbf{0.30} \cdot Q_{\text{FR}} + \mathbf{0.10} \cdot Q_{\text{NR}}$$

### A. Segmentation Term ($Q_{\text{seg}}$)
$$Q_{\text{seg}} = \text{Macro } F_1 \times \text{mIoU} \times \text{coverage\_penalty}$$
- Unweighted average over all 5 official classes (Background, Coral, Fish, Shark/Ray, Other).
- $\text{coverage\_penalty} = \prod_{c \in \mathcal{C}_{\text{present}}} \mathbb{I}(\text{TP}_c > 0)$ (collapses to 0.0 if any present class has zero true positives).

### B. Full-Reference Restoration Term ($Q_{\text{FR}}$)
$$Q_{\text{FR}} = 0.45 \cdot \text{PSNR}_{\text{norm}} + 0.45 \cdot \text{SSIM} + 0.10 \cdot \text{sharpness\_ratio}$$
$$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0},\, 0.0,\, 1.0\right)$$
- Evaluated on images resized to `scale_to_max_side(w, h, 1024)`.

### C. Non-Reference Term ($Q_{\text{NR}}$)
Underwater image quality combination of UIQM, UCIQE, and CCF.

---

## 2. Arithmetic Reconciliation Table

| Metric Quantity | Previous Projected Claim | Server Result A (#18672) | Server Result D (High-Seg) | Paired A+D Prediction | Actual Official Feedback | Discrepancy Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$Q_{\text{seg}}$** | 0.7078 | 0.3240 | **0.4240** | 0.4240 | **~0.42** | **-0.2838**: Local validation was inflated by in-sample training data |
| **mIoU** | 0.7981 | 0.5100 | **0.5830** | 0.5830 | **~0.58** | **-0.2151**: Unseen dev water scenes cause domain shift |
| **Macro $F_1$** | 0.8869 | 0.6350 | **0.7270** | 0.7270 | **~0.73** | **-0.1599**: Rare class boundary clipping on unseen scenes |
| **Coverage Penalty** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Verified intact (no present class missed) |
| **PSNR (dB)** | 26.89 | **25.49** | 18.84 | 25.49 | **25.49** | Matches Result A's physical inversion |
| **$\text{PSNR}_{\text{norm}}$**| 0.5945 | **0.5245** | 0.1920 | 0.5245 | **0.5245** | Exact mathematical calculation $(25.49 - 15) / 20$ |
| **SSIM** | 0.8355 | **0.8100** | 0.7280 | 0.8100 | **0.8100** | Matches Result A's verified metric |
| **Sharpness Ratio** | 0.3565 | **0.8930** | 0.4080 | 0.8930 | **0.8930** | Matches Result A's unsharp Laplacian boost |
| **$Q_{\text{FR}}$** | 0.6792 | **0.6898** | 0.4548 | 0.6898 | **0.6898** | $0.45(0.5245) + 0.45(0.810) + 0.10(0.893)$ |
| **$Q_{\text{NR}}$** | 0.4500 | **0.4800** | 0.4400 | 0.4800 | **0.4800** | Derived from UIQM 0.821, UCIQE 0.566, CCF 0.050 |
| **Composite Score** | **0.6796** | 0.4760 | 0.4610 | **0.5093** | **~0.50** | **Arithmetic confirmed: $0.60(0.424) + 0.30(0.690) + 0.10(0.480) = 0.5093$** |

---

## 3. The Three Root Causes of the Score Discrepancy

### Root Cause 1: In-Sample Data Leakage in Validation Baseline ($Q_{\text{seg}} = 0.7078 \to 0.4240$)
- **The Finding:** In `checkpoints/v4_full20/best_v4_checkpoint.pt`, EURS-Net V4 was trained on **all 20 scenes** in `data/train`.
- When evaluated on the 4 "validation" scenes (`fish1`, `fish12`, `seahorse1`, `sharkFish`), EURS-Net V4 reached $\text{mIoU} = 0.8042$ because it had already seen those scenes during training.
- **The Reality:** The official Dev set contains 6 completely unseen scenes (`fish10`, `fish19`, `fish3`, `fish6`, `sharRayFish`, `starfish1`). On these unseen scenes, the true measured generalization score of the ensemble was **$Q_{\text{seg}} = 0.4240$** (as established by Result D), not $0.7078$.
- **Impact:** Under $0.60 \times Q_{\text{seg}}$, a difference of $0.7078 - 0.4240 = 0.2838$ translates directly to a **$-0.1703$ drop in total composite score** ($0.6796 \to 0.5093$).

### Root Cause 2: DeepLabV3 Normalization Bug
- **The Finding:** In `generate_verified_track1_submission.py` (line 224) and `evaluate_track1_ablations.py` (line 224), `model_dl` was fed `rest_phys` in range $[0, 1]$ directly without applying ImageNet standardization `(x - mean) / std`.
- DeepLabV3 was trained with strict `(x - mean) / std` normalization. Feeding it unnormalized $[0, 1]$ inputs collapsed its standalone $Q_{\text{seg}}$ to **`0.0745`**, injecting corrupted feature logits into the 4-way ensemble.

### Root Cause 3: Verification of the 0.5093 Arithmetic
- The paired submission (`deploy/eurs_track1_paired_A_D_submission.zip`) combining Result A's restoration with Result D's masks mathematically yields:
  $$0.60(0.4240) + 0.30(0.6898) + 0.10(0.4800) = 0.2544 + 0.2069 + 0.0480 = \mathbf{0.5093}$$
- **Verdict:** If the user submitted the paired package, the resulting score of **~0.50** is mathematically consistent and expected. The claim of reaching $0.58–0.65+$ depended entirely on the assumed transfer of the in-sample $0.7078$ validation metric to the unseen Dev test split, which did not occur.
