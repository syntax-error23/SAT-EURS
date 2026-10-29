# EURS Track 2 Metric Specifications & Aggregation Contract

## 1. Overview & Verification Status
This document formally defines the mathematical evaluation formulas, aggregation rules, and parity status for EURS-Net V2 on MaCVi Track 2 (Jetson AGX Orin).

| Metric Category | Local Implementation Status | Aggregation Policy |
| :--- | :--- | :--- |
| **Segmentation Metrics** | **VERIFIED** | Global confusion matrix accumulated across full evaluation set |
| **Restoration Metrics** | **VERIFIED** | Computed per image, then arithmetic mean across evaluation set |
| **Non-Reference Quality ($Q_{\text{NR}}$)** | **UNVERIFIED / BLOCKED** | Isolated via provisional stub `metrics/nr_metric.py` |
| **Coverage Penalty** | **PROVISIONAL** | Pending official evaluator code; edge cases verified |
| **Composite Score** | **PROVISIONAL** | $S = 0.60 Q_{\text{seg}} + 0.30 Q_{\text{rest}} + 0.10 Q_{\text{NR}}$ |

---

## 2. Segmentation Metrics Formulation
- **5 Classes:** `0: Background`, `1: Coral`, `2: Fish`, `3: Shark`, `4: Other`.
- **Confusion Matrix:** $C \in \mathbb{R}^{5 \times 5}$, where $C_{i, j}$ is the number of pixels with true class $i$ predicted as class $j$.
- **Per-Class Metrics:**
  $$\text{IoU}_c = \frac{C_{c, c}}{\sum_j C_{c, j} + \sum_i C_{i, c} - C_{c, c} + \epsilon}$$
  $$F_{1, c} = \frac{2 C_{c, c}}{\sum_j C_{c, j} + \sum_i C_{i, c} + \epsilon}$$
- **Mean Metrics:**
  $$\text{mIoU} = \frac{1}{5}\sum_{c=0}^4 \text{IoU}_c, \quad \text{Macro } F_1 = \frac{1}{5}\sum_{c=0}^4 F_{1, c}$$
- **Coverage Penalty (Provisional Contract):**
  Penalizes models that achieve high mIoU by entirely omitting rare classes that are present in the ground truth:
  $$\text{coverage\_penalty} = \prod_{c \in \mathcal{C}_{\text{present}}} \mathbb{I}(\text{TP}_c > 0)$$
- **Overall Segmentation Quality:**
  $$Q_{\text{seg}} = \text{Macro } F_1 \times \text{mIoU} \times \text{coverage\_penalty}$$

---

## 3. Restoration Metrics Formulation
- Computed for each restored image $I_{\text{rest}}$ against ground-truth clean image $I_{\text{clean}}$ ($\in [0.0, 1.0]$):
1. **Normalized PSNR (Official MaCVi 15-35 dB Window):**
   $$\text{MSE} = \frac{1}{3 \cdot H \cdot W}\sum_{c, h, w} (I_{\text{rest}} - I_{\text{clean}})^2$$
   $$\text{PSNR} = \begin{cases} 100.0 & \text{if } \text{MSE} < 10^{-10} \\ 10 \log_{10}\left(\frac{1.0}{\text{MSE}}\right) & \text{otherwise} \end{cases}$$
   $$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0}, 0.0, 1.0\right)$$
2. **SSIM:** Evaluated using Gaussian window ($11 \times 11, \sigma = 1.5$) across channels.
3. **Sharpness Ratio with Zero-Denominator Protection:**
   $$\Delta I_{\text{rest}} = \nabla^2(I_{\text{rest}}), \quad \Delta I_{\text{clean}} = \nabla^2(I_{\text{clean}})$$
   $$\text{sharpness\_ratio} = \begin{cases}
   1.0 & \text{if } \text{Var}(\Delta I_{\text{clean}}) < 10^{-7} \text{ and } \text{Var}(\Delta I_{\text{rest}}) < 10^{-7} \\
   0.0 & \text{if } \text{Var}(\Delta I_{\text{clean}}) < 10^{-7} \text{ and } \text{Var}(\Delta I_{\text{rest}}) \ge 10^{-7} \\
   \min\left(1.0, \frac{\text{Var}(\Delta I_{\text{rest}})}{\text{Var}(\Delta I_{\text{clean}})}\right) & \text{otherwise}
   \end{cases}$$
4. **Overall Restoration Quality:**
   $$Q_{\text{rest}} = 0.45 \cdot \text{PSNR}_{\text{norm}} + 0.45 \cdot \text{SSIM} + 0.10 \cdot \text{sharpness\_ratio}$$

---

## 4. Non-Reference Quality ($Q_{\text{NR}}$) & Composite Score
- **Formula:**
  $$S_{\text{composite}} = 0.60 \cdot Q_{\text{seg}} + 0.30 \cdot Q_{\text{rest}} + 0.10 \cdot Q_{\text{NR}}$$
- **Dependency Isolation:** Because official code for $Q_{\text{NR}}$ has not yet been imported into the repository, $Q_{\text{NR}}$ is provided as an explicit stub (`metrics/nr_metric.py`). The composite score returned by `metrics/composite_score.py` explicitly flags that official competition parity is provisional.
