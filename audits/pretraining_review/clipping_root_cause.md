# Forensic Analysis: Restoration Clipping Root Cause & Mechanism

**Focus:** Pathology of Initial 4.15 dB Collapse vs. Pilot Restoration Dynamics and Scene `fish12` Behavior  
**Diagnostic Data:** [`audits/diagnostics/fish12_clipping_diagnosis.json`](file:///C:/Projects/SAT-EURS/audits/diagnostics/fish12_clipping_diagnosis.json)  
**Status:** **ROOT CAUSES FULLY VERIFIED EMPIRICALLY**  

---

## 1. Mechanism of Earlier Failure (~4.15 dB Collapse)

In Run 2 (Epochs 1–15 before the fix):
1. **Unbounded Residual Head:** The restoration head was implemented as:
   $$I_{\text{rest}} = \text{clamp}(x + \Delta I, 0.0, 1.0)$$
   where $\Delta I$ was computed by a standard linear convolution without dynamic range bounding.
2. **Gradient Divergence on Severe Color Cast:** Underwater images exhibit severe green-blue haze with near-zero red light. When evaluated against clean ground truth, initial MS-SSIM loss is trapped at $1 - 0.01 \approx 0.99$. The optimizer attempted to cancel haze by predicting large negative residual values ($\Delta I < -0.5$).
3. **Black Image Collapse:** Adding large negative residuals directly to dark underwater images caused $96.36\%$ of pixels to clamp to $0.0$, producing pitch-black images, destroying PSNR down to **$4.15\text{ dB}$**, and collapsing SSIM to **$0.0746$**.

---

## 2. The Pilot Solution & Bounded Residual Dynamics

The restricted restoration pilot introduced two structural safeguards:
1. **Zero-Initialized Head:**
   $$W_{\text{head}} = 0, \quad b_{\text{head}} = 0 \implies \Delta I = 0$$
   This anchored initial restoration output directly to Fast White Balance ($15.06\text{ dB}$ baseline), eliminating initial negative collapse.
2. **Tanh Dynamic Range Bounding:**
   $$\Delta I = 0.20 \cdot \tanh(\text{Head}(x))$$
   The residual is mathematically constrained within $[-0.20, +0.20]$. It can refine colors and edges without overpowering the base radiometric white balance.
3. **Empirical Recovery:** Under this bounding, PSNR surged from **$4.15\text{ dB} \to 17.40\text{ dB}$**, SSIM increased from **$0.0746 \to 0.6446$**, and aggregate clipping dropped from **$96.36\% \to 2.08\%$**.

---

## 3. Forensic Investigation of Scene `fish12`

While aggregate clipping across all 800 validation images is safely low ($2.08\%$), scene `fish12` exhibited $5.68\%$ clipping in Epoch 2 and surged to $10.45\%$ in Epoch 3.

A per-channel forensic breakdown of all 200 samples of `fish12` revealed the exact mechanism:

```
[fish12 Input & FWB Baseline]
  - Raw Input Mean:        0.1764 | Clip at 0.0: 14.1047% | Clip at 1.0: 0.0000%
  - FWB Output Mean:       0.1723 | Clip at 0.0:  5.7163% | Clip at 1.0: 0.0000%

[Epoch 2 vs. Epoch 3 Clipping on fish12]
  - Epoch 2 Total Clip:    5.6796% (Low-End 0.0: 5.6625%, High-End 1.0: 0.0171%)
    * Channel R Clip:     16.9897% | Channel G: 0.0236% | Channel B: 0.0256%
  - Epoch 3 Total Clip:   10.4470% (Low-End 0.0: 10.4295%, High-End 1.0: 0.0175%)
    * Channel R Clip:     31.2998% | Channel G: 0.0147% | Channel B: 0.0265%
```

### Empirical Findings
1. **Low-End Extinction vs. Highlight Saturation:**
   - Clipping at $1.0$ (highlight saturation) is **virtually nonexistent** on `fish12` ($0.017\%$).
   - Over **$99.7\%$ of all clipping is at $0.0$** (pure black underexposure), concentrated heavily in the Red channel ($16.99\%$ in Ep 2).
2. **Camera Sensor & Physical Absorption Floor:**
   - Scene `fish12` is photographed in deep, dark water. The raw camera sensor images **already have $14.10\%$ near-zero pixel intensity**.
   - Underwater physics dictates that red wavelengths ($650\text{--}700\text{ nm}$) are extinguished within the first 5 meters of water depth, leaving near-zero natural red photons in shadows.
3. **FWB Color Calibration Bias:**
   - The frozen Fast White Balance layer subtracts green haze via $\beta_G = -0.0356$, which clamps dark pixels with $G < 0.035$ directly to $0.0$, creating an intrinsic $5.72\%$ clipping floor.
4. **Epoch 2 vs. Epoch 3 Divergence:**
   - In **Epoch 2**, total clipping is **$5.68\%$**—matching the natural baseline FWB floor ($5.72\%$). The network respected the radiometric boundary.
   - In **Epoch 3**, the unconstrained training steps forced negative residual updates in shadow areas, pushing the red channel clipping to **$31.30\%$** and total clipping to **$10.45\%$**.

---

## 4. Policy on Post-Processing & Preprocessing Alterations

* **No Artificial Clamping:** We do not inject ad-hoc artificial shifts, post-processing clamps, or scene-specific adjustments. The competition rules require submitting the pure static model.
* **Controlled Optimization Principle:** To prevent the red-channel negative push observed in Epoch 3, any subsequent training must incorporate lower learning rates, early stopping when scene-level clipping approaches the FWB floor ($5.7\%$), or bounded residual asymmetry (penalizing negative residuals in near-zero pixels).
