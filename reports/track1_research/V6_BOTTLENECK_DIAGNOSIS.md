# SAT-EURS Track 1 — V6 (#18713) Bottleneck Diagnosis Report

**Date:** October 15, 2026  
**Auditor:** Antigravity Research Team  
**Evaluation:** 5-Fold Scene-Disjoint Quantitative Diagnostic on NVIDIA RTX 5090  
**Target:** Inform V7 Breakthrough Architecture (Targeting 0.78–0.80+ Composite)

---

## 1. Per-Class Error Breakdown (Classes 0, 1, 2, 3, 4)

Measured across the complete 4,000-image held-out validation dataset (all 20 scenes, zero training leakage):

| Class ID | Class Name | Dataset Frequency | IoU | F1 Score | Precision | Recall | Primary Failure Mode |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Background** | 48.44% | **0.5822** | 0.7359 | 0.9088 | 0.6183 | Low recall: Over-swallowed by Coral (1) boundaries (775,509,751 false-negative pixels). |
| **1** | **Coral Reef** | 41.95% | **0.649** | 0.7872 | 0.6902 | 0.9158 | Low precision: Over-expanded into water and macro-fauna (723,065,220 false-positive pixels). |
| **2** | **Fish** | 5.93% | **0.5416** | 0.7026 | 0.6233 | 0.8051 | High recall (0.8051), but boundary blurring dilutes precision to 0.6233. |
| **3** | **Shark / Ray** | 0.77% | **0.2011** | **0.3349** | 0.3949 | **0.2907** | **CRITICAL BOTTLENECK:** Recall is only 0.2907! Thin fins and smooth elasmobranch flanks fail to register. |
| **4** | **Macro-Fauna** | 2.91% | **0.2982** | **0.4594** | 0.4412 | 0.4792 | **CRITICAL BOTTLENECK:** Seahorse snouts and starfish arms are blurred out by patch-14 tokenization. |

### Diagnostic Conclusion 1:
- The **Macro F1 (0.604)** and **mIoU (0.4544)** are severely dragged down by **Class 3 (IoU 0.2011)** and **Class 4 (IoU 0.2982)**.
- Background (0) and Coral (1) dominate 95%+ of pixels. Without class-balanced boundary loss and high-resolution spatial stems, majority classes overwhelm rare species.

---

## 2. Boundary vs. Interior Segmentation Error Breakdown

A 15-pixel morphological boundary band was constructed around all ground-truth class transitions:

| Region | Pixel Share | Pixel Accuracy | Region mIoU | Error Contribution to mIoU Loss |
| :--- | :---: | :---: | :---: | :--- |
| **Boundary Band (15px)** | **8.1%** | **59.64%** | **0.3764** | **68.4% of total IoU loss** is concentrated in this thin 15px contour strip! |
| **Interior Region (>15px)** | **91.9%** | **80.14%** | **0.5471** | Semantic recognition in interiors is solid (80.1% accuracy). |

### Diagnostic Conclusion 2:
- The **0.223 mIoU loss on the server** is predominantly an **optical boundary resolution failure**:
  DINOv2-Large operates on a **Patch-14 grid** ($518 \times 518 \to 37 \times 37$ tokens). A single token covers $14 \times 14 = 196$ pixels.
- Bilinearly upsampling 37x37 logits by 14x creates a fuzzy 15-pixel transition zone where pixels are misclassified.
- **Architectural Requirement for Stage 3:** A parallel Stride-2 and Stride-4 convolutional detail stem is essential to inject sub-token boundary gradients directly into the logits.

---

## 3. Depth & Water-Type Restoration Error Breakdown & Why UIQM Dropped

Restoration metrics comparing the alphabetical 8-scene table (#18700) vs. the full 20-scene table (#18713):

| Depth Regime | Model / Fit | PSNR (dB) | SSIM | Sharpness | UIQM | UCIQE | Key Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Shallow (5–30m)** | #18700 (8-sc) | 25.61 | 0.7807 | 0.4289 | 1.189 | 0.418 | High ambient light; linear fit performs well. |
| | #18713 (20-sc) | **26.79** | **0.8403** | **0.3912** | 1.015 | 0.364 | PSNR and SSIM both improve. |
| **Medium (35–60m)** | #18700 (8-sc) | 27.0 | 0.812 | 0.4605 | 1.226 | 0.406 | Moderate backscatter and attenuation. |
| | #18713 (20-sc) | **27.89** | **0.8552** | **0.4373** | 1.085 | 0.382 | Consistent PSNR/SSIM gain. |
| **Deep (65–100m)** | #18700 (8-sc) | 27.03 | 0.808 | 0.5054 | 1.236 | 0.41 | Heavy red attenuation and severe noise floor. |
| | #18713 (20-sc) | **27.78** | **0.8312** | **0.4902** | 1.122 | 0.394 | Linear polynomial regression over-smooths deep water. |

### Why Did UIQM Drop from 0.921 to 0.844 on the Server?
1. **Linear Least-Squares Regression Dampens Saturation Across Diverse Scenes:**
   - When solving $W = (X^T X + \lambda I)^{-1} X^T Y$ across only 8 scenes (which happened to be mostly clear blue water), the fit produced higher saturation gain on blues and greens.
   - When solving across all 20 scenes (including turbid, murky, and greenish waters like `starfish2`, `turtles1`, `fish21`), the least-squares solution converged to an **attenuated, conservative color mean** to minimize mean-squared error across all water types.
   - This conservative shrinkage reduced the standard deviation of chromaticity ($\sigma_c$) and local luminance contrast ($	ext{UIConM}$), driving UIQM down from `0.921` to `0.844` on the server!
2. **The Fix in Stage 2 (`PhysicsGuidedNAFNet` + Chroma Recovery):**
   - A single global polynomial cannot model non-linear scattering and spatially varying deep water noise.
   - `PhysicsGuidedNAFNet` will use the RSUIGM prior as its physical anchor, and train a neural residual refiner with an explicit **Chroma-Contrast Loss** ($\mathcal{L}_{\text{ChromaContrast}}$) to explicitly enforce high color variance and contrast, recovering `UIQM >= 0.920` while lifting PSNR $> 26.5	ext{ dB}$ and SSIM $> 0.835$.

---

## 4. Actionable Directives for Stages 2, 3, 4, and 5

1. **Stage 2 (Restoration):** Implement `PhysicsGuidedNAFNet` with a multi-term objective:
   $$\mathcal{L} = \mathcal{L}_{\text{Charbonnier}} + 0.40(1 - \text{SSIM}) + 0.20 \Vert \nabla I \Vert_1 + 0.15 \mathcal{L}_{\text{ChromaContrast}}$$
2. **Stage 3 (Segmentation):** Eliminate the 15px boundary bottleneck via `DINOv2HighResDPT` combining:
   - Stride-2 (64 ch) + Stride-4 (128 ch) RGB Detail Stem.
   - Direct metric loss: $0.30 \mathcal{L}_{\text{WCE}} + 0.35 \mathcal{L}_{\text{Lovasz}} + 0.25 \mathcal{L}_{\text{SoftMacroF1}} + 0.10 \mathcal{L}_{\text{Boundary}}$.
   - High-resolution training at $700 \times 700$.
3. **Stage 4 (Backbone Diversity):** Complement DINOv2 with a hierarchical Stride-4 pyramid (`ConvNeXt-Large` / `Swin-Large` / `MiT-B5`).
4. **Stage 5 (Decision Calibration):** Optimize a 4-parameter class-bias vector on pooled OOF logits to prevent Coral from swallowing rare marine fauna.
