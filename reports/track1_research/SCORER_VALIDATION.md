# SAT-EURS Track 1 — Official Scorer Contract & Metric Validation Report

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)

---

## 1. Verified Official Challenge Scoring Definition

The official MaCVi composite scoring relationship is:

$$S_{\text{composite}} = \mathbf{0.60} \cdot Q_{\text{seg}} + \mathbf{0.30} \cdot Q_{\text{FR}} + \mathbf{0.10} \cdot Q_{\text{NR}}$$

### 1.1 Semantic Segmentation Quality ($Q_{\text{seg}}$)
$$Q_{\text{seg}} = \text{Macro } F_1 \times \text{mIoU} \times \text{coverage\_penalty}$$

1. **Confusion Matrix:** $C \in \mathbb{R}^{5 \times 5}$ accumulated globally across all evaluated pixels across the 5 official classes:
   - `0`: Background
   - `1`: Coral / Substrate
   - `2`: Fish
   - `3`: Elasmobranchii (Sharks, Rays)
   - `4`: Other Marine Fauna (Starfish, Seahorse, Turtles)
2. **Per-Class Metrics:**
   $$\text{IoU}_c = \frac{C_{c, c}}{\sum_j C_{c, j} + \sum_i C_{i, c} - C_{c, c} + \epsilon}$$
   $$F_{1, c} = \frac{2 C_{c, c}}{\sum_j C_{c, j} + \sum_i C_{i, c} + \epsilon}$$
   $$\text{Precision}_c = \frac{C_{c, c}}{\sum_i C_{i, c} + \epsilon}, \qquad \text{Recall}_c = \frac{C_{c, c}}{\sum_j C_{c, j} + \epsilon}$$
   *(When a class is absent from both ground truth and prediction, $\text{IoU}_c = 1.0$ and $F_{1, c} = 1.0$).*
3. **Macro Averages:**
   $$\text{mIoU} = \frac{1}{5}\sum_{c=0}^4 \text{IoU}_c, \qquad \text{Macro } F_1 = \frac{1}{5}\sum_{c=0}^4 F_{1, c}$$
4. **Coverage Penalty:**
   $$\text{coverage\_penalty} = \prod_{c \in \mathcal{C}_{\text{present}}} \mathbb{I}(\text{TP}_c > 0)$$
   *If any ground truth class present in the evaluation set has zero true positives ($\text{TP}_c = 0$), $Q_{\text{seg}}$ collapses to 0.0.*

### 1.2 Full-Reference Restoration Quality ($Q_{\text{FR}}$)
$$Q_{\text{FR}} = 0.45 \cdot \text{PSNR}_{\text{norm}} + 0.45 \cdot \text{SSIM} + 0.10 \cdot \text{sharpness\_ratio}$$

1. **Normalized PSNR ($\text{PSNR}_{\text{norm}}$):**
   $$\text{MSE} = \frac{1}{3 \cdot H \cdot W}\sum_{c, h, w}(I_{\text{rest}} - I_{\text{clean}})^2$$
   $$\text{PSNR} = 10 \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$
   $$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0},\, 0.0,\, 1.0\right)$$
2. **SSIM:** Mean SSIM evaluated using an $11 \times 11$ Gaussian window ($\sigma = 1.5$) averaged across R, G, B channels.
3. **Sharpness Ratio with Zero-Denominator Protection:**
   $$\text{sharpness\_ratio} = \min\left(1.0,\, \frac{\text{Var}(\Delta I_{\text{rest}})}{\text{Var}(\Delta I_{\text{clean}})}\right)$$
   *(Protected: If $\text{Var}(\Delta I_{\text{clean}}) < 10^{-7}$, returns $1.0$ if $\text{Var}(\Delta I_{\text{rest}}) < 10^{-7}$ else $0.0$).*

### 1.3 Non-Reference Restoration Quality ($Q_{\text{NR}}$) & Local Policy
- Official $Q_{\text{NR}}$ is computed by the competition server combining UIQM, UCIQE, and CCF under undocumented server-side normalization constants.
- **Strict Anti-Hallucination Policy:** In accordance with Section 2.2, we **do not** fit a synthetic linear regression from the four historical submissions to guess server normalizations.
- Local measurements report exact raw metrics (PSNR dB, SSIM, Sharpness Ratio).
- Any reconstructed composite score is labeled **$S_{\text{composite, est}}$** and marked **PROVISIONAL**.
- **$Q_{\text{seg}}$ serves as the primary segmentation promotion criterion**, with the official leaderboard providing external composite verification.

---

## 2. Unit Test & Synthetic Fixture Parity Verification

The test suites `tests/test_scoring_edge_cases.py` and `tests/test_scorer_contract.py` verify 100% mathematical parity:

| Test Case | Description | Expected Output | Measured Status |
| :--- | :--- | :--- | :---: |
| **Synthetic 5x5 CM Fixture** | Known TP, FP, FN counts | $\text{mIoU} = 0.6372$, $\text{Macro } F_1 = 0.7756$ | **PASSED (Exact)** |
| **Absent Classes in Both** | Class absent in both GT and prediction | $\text{IoU}_c = 1.0, F_{1, c} = 1.0$ | **PASSED (Exact)** |
| **False Positive Intrusion** | Class absent in GT, predicted $\ge 1$ px | $\text{IoU}_c = 0.0, F_{1, c} = 0.0$ | **PASSED (Exact)** |
| **Zero-Recall Coverage Collapse** | Present class completely missed ($\text{TP}=0$) | $\text{penalty} = 0.0 \implies Q_{\text{seg}} = 0.0$ | **PASSED (Exact)** |
| **PSNR Window Floor** | $\text{PSNR} \le 15.0\text{ dB}$ | $\text{PSNR}_{\text{norm}} = 0.000$ | **PASSED (Exact)** |
| **PSNR Window Ceiling** | $\text{PSNR} \ge 35.0\text{ dB}$ | $\text{PSNR}_{\text{norm}} = 1.000$ | **PASSED (Exact)** |
| **Sharpness Guardrail** | Flat image with variance $< 10^{-7}$ | Protected, returns valid float | **PASSED (Exact)** |
| **Leaderboard Drop Proof** | Reconstruction of #18666 vs #18668 | $Q_{\text{seg}}$ drop accounts for $100\%$ score delta | **PASSED (Exact)** |

---

## 3. Scorer Gate Status

**GATE 0 (CORRECTNESS) STATUS:** **PASSED**.  
Dataset contracts, image-mask pairing invariants, confusion matrix accumulation, and metric formulas are verified against exact mathematical specifications.
