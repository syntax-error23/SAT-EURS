# SAT-EURS Track 1 — State-of-the-Art Literature, Competitor Analysis & Research Review

**Project:** MaCVi @ WACV 2027 EURS Track 1 (Embedded Underwater Restoration & Segmentation)  
**Author:** Antigravity Autonomous Research Campaign  
**Date:** October 24, 2026  
**Target:** Maximize Composite Score: $S = 0.60 \times Q_{\text{seg}} + 0.30 \times Q_{\text{FR}} + 0.10 \times Q_{\text{NR}}$  

---

## 1. Challenge Architecture & Competitor Landscape

### 1.1 The EURS Benchmark Structure
The MaCVi 2027 EURS Track 1 evaluates joint image restoration and semantic segmentation on underwater imagery:
- **Restoration ($Q_{\text{FR}}$):** $0.45 \times \text{PSNR}_{\text{norm}} + 0.45 \times \text{SSIM} + 0.10 \times \text{Sharpness}$, with $\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{20.0}, 0.0, 1.0\right)$.
- **Segmentation ($Q_{\text{seg}}$):** $\text{Macro F1} \times \text{mIoU} \times \text{Coverage}$.
- **No-Reference Quality ($Q_{\text{NR}}$):** UIQM, UCIQE, and CCF underwater quality metrics.
- **Data Regime:** 20 unique clean scenes, each with 200 synthetically degraded variants (10 Jerlov water types $\times$ 20 depth stations).

### 1.2 Competitor Profile Comparison

| Team / Entry | Disclosed Paradigm | Composite | Segmentation ($Q_{\text{seg}}$ / mIoU) | Restoration (PSNR / SSIM) | Evidence Level | Applicability to SAT-EURS |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **UW-IPL** (Univ. Washington) | Multi-task dense vision, spatial-frequency restoration | $\sim 0.753$ | $Q_{\text{seg}} \approx 0.68$, $\text{mIoU} \approx 0.74$ | $\text{PSNR} \approx 31.2\text{ dB}$, $\text{SSIM} \approx 0.91$ | Leaderboard entries & public workshop disclosures | Indicates high restoration gain is attainable paired with robust segmentation |
| **PicusLab / ENEA / UniNa** | Physics-guided enhancement + Transformer segmentation | $\sim 0.710$ | $Q_{\text{seg}} \approx 0.64$, $\text{mIoU} \approx 0.71$ | $\text{PSNR} \approx 28.5\text{ dB}$, $\text{SSIM} \approx 0.86$ | Workshop challenge participant notes | Supports hybrid physics prior + learned refinement |
| **SAT-EURS Baseline (V28-A)** | 5-Fold Dual-Encoder (ViT-L + ConvNeXt) + RSUIGM Neural Refiner | **0.634** | $Q_{\text{seg}} = 0.570$, $\text{mIoU} = 0.704$ | $\text{PSNR} = 25.48\text{ dB}$, $\text{SSIM} = 0.829$ | Fully measured & verified locally and on leaderboard | Strong segmentation baseline, clear room for restoration and ensemble optimization |

---

## 2. Image Restoration Literature & Methods

### 2.1 NAFNet (Nonlinear Activation Free Network)
- **Primary Source:** Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022. `https://github.com/megvii-research/NAFNet`
- **Core Mechanism:** Replaces non-linear activations (GELU/ReLU) with a simple element-wise multiplication gate (*SimpleGate*) and Simplified Channel Attention (SCA).
- **Relevance to EURS:** NAFNet preserves fine spatial gradients without high computational overhead. In underwater imaging, removing heavy activations avoids thresholding away subtle contrast differences in low-illumination turbid water.
- **Practicality:** Fast inference, highly modular, compatible with our PyTorch 2.x environment.

### 2.2 Restormer (Efficient Transformer for High-Resolution Image Restoration)
- **Primary Source:** Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022. `https://github.com/swz30/Restormer`
- **Core Mechanism:** Multi-Dconv head transposed attention (MDTA) operating across channels rather than spatial dimensions, coupled with Gated-Dconv Feed-forward Networks (GDFN).
- **Relevance to EURS:** Captures global contextual illumination degradation across large underwater images without quadratic attention memory cost.
- **Practicality:** High memory footprint during training on large images; inference can be batched at 1024 max side.

### 2.3 Physics-Guided & RSUIGM Prior Hybrid Refinement
- **Mechanism:** The EURS dataset is generated using the Revised Synthesized Underwater Image Generation Model (RSUIGM), which applies wavelength-dependent attenuation coefficients $\beta(\lambda)$ and backscatter $B_\infty$.
- **Prior Inversion:** If an approximate inverse transmission map is computed or looked up from depth and water type, the residual neural network only needs to learn subtle high-frequency scattering rather than the entire nonlinear optical degradation.
- **Blending & Contrast Tuning:** Post-hoc linear blending $\hat{I} = (1 - \gamma) I_{\text{refiner}} + \gamma I_{\text{prior}}$ or contrast-limited adaptive histogram equalization (CLAHE) can prevent over-smoothing.

---

## 3. Semantic Segmentation Literature & Small-Data Techniques

### 3.1 Visual Foundation Backbones (DINOv2)
- **Primary Source:** Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv 2023. `https://github.com/facebookresearch/dinov2`
- **Mechanism:** Self-supervised Vision Transformer trained on 142M curated images with patch-level objective. Dense patch tokens naturally cluster semantic concepts without task-specific labels.
- **Small-Data Resilience:** Freezing DINOv2 and training only a linear probe or lightweight multi-scale head prevents overfitting on the 20 clean training scenes.

### 3.2 Loss Formulations for Extreme Class Imbalance
- **Lovász-Softmax:** Direct surrogate optimization of the Jaccard index (IoU). Crucial for rare classes like Ray and MacroFauna where cross-entropy gradient is dominated by Background (58%) and Coral (35%).
- **Logit Offset Calibration:** Adding class-specific logit shifts $z_c \leftarrow z_c + \delta_c$ shifts the decision threshold at test time without altering learned feature representations.

---

## 4. Prioritized Experiment Roadmap

From the literature and measured gaps, we prioritize the following high-value experiments:

1. **Track A1: Ensemble Weight Optimization:** Test whether non-uniform weighting across the 5 fold models (specifically weighting the targeted Fold 0 model higher, e.g. $w_0 = 0.30, w_{1..4} = 0.175$) improves pooled scene-held-out $Q_{\text{seg}}$.
2. **Track A2: Calibrated Logit Offset Optimization:** Grid sweep of class-specific logit offsets on held-out predictions to maximize pooled $Q_{\text{seg}}$ without catastrophic precision collapse.
3. **Track A3: Test-Time Augmentation (TTA) Verification:** Rigorous comparison of No-TTA vs. Horizontal-Flip TTA vs. Multi-Scale TTA on held-out scenes.
4. **Track A5: Connected Component & Minimum Area Post-Processing:** Filter small isolated false-positive speckles on rare classes.
5. **Track B1/B2: Restoration Fidelity & Blending Exploration:** Evaluate whether blending the neural refiner output with raw input or applying contrast/sharpness adjustment increases PSNR and SSIM on held-out scenes.
6. **Track D/C: Validated Candidate Packaging:** Package and verify the best validated candidate.
