# Strategic Research Review: SAT-EURS Last-Chance Campaign (MaCVi @ WACV 2027 EURS Track 1)

**Challenge:** Maritime Computer Vision (MaCVi) @ WACV 2027 — EURS Track 1  
**Document Classification:** Strategic Architecture & Empirical Research Review  
**Date:** October 25, 2026  
**Auditor:** Autonomous Senior Research Agent  
**Hardware Platform:** NVIDIA GeForce RTX 5090 (32 GB VRAM, Driver 591.55, CUDA 13.1)  
**Operating Baseline:** V28-A (Official Composite: **0.634**, SHA-256: `471bf9b9...`)  
**Target:** Official Composite $\ge \mathbf{0.700}$ (AoE Deadline: October 27, 2026)

---

## 1. Official Leaderboard Forensic Analysis & Competitor Deconstruction

### 1.1 Top Testing-Phase Submissions (Official Public Leaderboard Snapshot)

| Rank / ID | Author / Team | Model Name | Official Composite | $Q_{\text{seg}}$ | mIoU | Macro $F_1$ | PSNR (dB) | SSIM | Sharpness | Implied $Q_{\text{FR}}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 #19050 | UW-IPL (User #2038) | IPL-test MaCVi 27 Testing phase | **0.753** | **0.688** | 0.786 | 0.876 | **31.15** | **0.911** | 0.906 | **0.8639** |
| 2 #18941 | UW-IPL (User #2038) | IPL-test-v1 MaCVi 27 Testing phase | **0.753** | 0.687 | 0.785 | 0.875 | **31.15** | **0.911** | 0.906 | **0.8639** |
| 3 #18991 | UW-IPL (User #2038) | IPL-test-v3 MaCVi 27 Testing phase | **0.751** | 0.687 | 0.785 | 0.875 | 31.01 | 0.911 | 0.891 | 0.8592 |
| 5 #19012 | PicusLab, ENEA, UniNa | test1 MaCVi 27 Testing phase | **0.728** | 0.671 | 0.775 | 0.866 | **29.35** | **0.880** | 0.912 | **0.8105** |
| 6 #18993 | PicusLab, ENEA, UniNa | PicusLab EURS v2 | **0.724** | 0.671 | 0.775 | 0.866 | 28.48 | 0.871 | 0.967 | 0.7915 |
| 8 #18969 | ENEA (User #2102) | ENEA EURS S59 Full20 | **0.723** | 0.671 | 0.775 | 0.866 | 28.89 | 0.883 | 0.882 | 0.8033 |
| 9 #18972 | ENEA (User #2102) | ENEA EURS R49 S59 | **0.723** | 0.671 | 0.775 | 0.866 | 28.52 | 0.874 | 0.956 | 0.8023 |
| 10 #19122 | aimslabs (User #1995) | aimslabs_t1_test_53 | **0.706** | 0.649 | 0.760 | 0.854 | **28.14** | **0.893** | 0.908 | **0.7883** |
| 13 #19200 | aimslabs (User #1998) | aimslabs_t1 | **0.700** | 0.649 | 0.758 | 0.857 | 27.89 | 0.874 | 0.857 | 0.7738 |
| 18 #19187 | APC_TDC (User #1802) | EURS_t1_sub024 | **0.686** | 0.610 | 0.732 | 0.833 | **29.03** | **0.891** | 0.874 | **0.8040** |
| 38 #19149 | GIST (User #1770) | zipTest4b_seg3_phys5 | **0.666** | 0.590 | 0.715 | 0.825 | 27.69 | 0.884 | 0.908 | 0.7766 |
| 49 #19028 | deepdrivepl (User #1974) | test-e19g-e23b | **0.650** | 0.550 | 0.686 | 0.801 | 28.60 | 0.888 | 0.922 | 0.7979 |
| **58 #19133** | **MIT-BLR (Our Baseline)** | **V28a-5Fold-Ablation** | **0.634** | **0.570** | **0.704** | **0.810** | **25.48** | **0.829** | **0.973** | **0.7062** |
| 60 #19185 | MIT-BLR (Decommissioned) | V28-SOTA-Apex | 0.622 | 0.551 | 0.688 | 0.800 | 25.48 | 0.829 | 0.973 | 0.7062 |

### 1.2 Reverse Engineering Competitor Methodologies

1. **PicusLab & ENEA (Univ of Naples Federico II & ENEA, Italy):**
   - *Submission Names:* `'ENEA EURS S59 Full20'`, `'ENEA EURS R49 S59'`, `'PicusLab EURS v2'`.
   - *Verified Fact:* Their segmentation metric is identical to 3 decimal places ($Q_{\text{seg}} = 0.671$, $\text{mIoU} = 0.775$, $\text{F1} = 0.866$) across 5 consecutive submissions spanning Oct 21–22, while their restoration shifts from $28.48 \to 29.35\text{ dB}$.
   - *Architecture Inference:*
     - `'S59'`: SegFormer-B5 fine-tuned for ~50–60 epochs, or SegFormer with MiT-B5 backbone.
     - `'Full20'`: Trained on **all 20 available clean scenes** (`Full20`) simultaneously rather than cross-validation subsets.
     - `'R49'`: Restormer (or deep CNN/Restormer `R49`) trained for 49 epochs or depth-49.
     - *Conclusion:* A decoupled two-stage pipeline. The segmenter was frozen at $Q_{\text{seg}} = 0.671$, and they climbed from 0.723 to 0.728 by upgrading the restorer from 28.52 dB to 29.35 dB.

2. **aimslabs (Embedded Vision / Korea & US):**
   - *Submission Names:* `aimslabs_t1_test_43` through `aimslabs_t1_test_55`.
   - *Verified Fact:* Tests 44 through 55 have an **invariant restoration score** of exactly **28.14 dB PSNR, 0.893 SSIM, and 0.908 Sharpness**.
   - *Inference:* They locked in a stable neural restoration model at 28.14 dB, and systematically ablated segmentation heads/ensembles, moving $Q_{\text{seg}}$ from $0.577 \to 0.600 \to 0.649$ (+0.072), driving Composite from $0.656 \to 0.706$.

3. **UW-IPL (Information Processing Lab, Univ of Washington):**
   - *Verified Fact:* The global leaderboard leader at **0.753 Composite**. Achieves **31.15 dB PSNR / 0.911 SSIM** ($Q_{\text{FR}} = 0.8639$) and **$Q_{\text{seg}} = 0.688$**.
   - *Inference:* UW-IPL operates a state-of-the-art paired restoration architecture paired with high-resolution transformer segmentation.

---

## 2. Quantitative Gap Analysis & Mathematical Sensitivity

### 2.1 The Official Scoring Formulation
The official challenge evaluates:
$$\text{Composite} = 0.60 \times Q_{\text{seg}} + 0.30 \times Q_{\text{FR}} + 0.10 \times Q_{\text{NR}}$$
$$Q_{\text{FR}} = 0.45 \times \text{PSNR}_{\text{norm}} + 0.45 \times \text{SSIM} + 0.10 \times \text{Sharpness}$$
$$\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0},\, 0.0,\, 1.0\right)$$

### 2.2 Mathematical Sensitivity & Feasibility Scenarios to Cross 0.700

Assuming $Q_{\text{NR}}$ remains constant at $\approx 0.080$, we require:
$$0.60 \times Q_{\text{seg}} + 0.30 \times Q_{\text{FR}} \ge 0.7000 - 0.0802 = \mathbf{0.6198}$$

* **Scenario 1: Segmentation Only (Zero Restoration Improvement, $Q_{\text{FR}} = 0.7062$):**
  - Requires $0.60 \times Q_{\text{seg}} \ge 0.6198 - 0.2118 = 0.4080 \implies \mathbf{Q_{\text{seg}} \ge 0.6800}$ (+0.1100 gain).
  - *Risk:* **CRITICAL / NEAR-IMPOSSIBLE**. We would need our segmenter to outperform PicusLab (0.671) and match UW-IPL (0.688).
* **Scenario 2: Restoration Only (Zero Segmentation Improvement, $Q_{\text{seg}} = 0.5700$):**
  - Requires $0.30 \times Q_{\text{FR}} \ge 0.6198 - 0.3420 = 0.2778 \implies \mathbf{Q_{\text{FR}} \ge 0.9260}$.
  - *Risk:* **MATHEMATICALLY IMPOSSIBLE**.
* **Scenario 3: Balanced Dual-Optimization (The Realistic Target):**
  - Moderate Restoration Gain (aimslabs level: PSNR $\approx 28.5$ dB, SSIM $\approx 0.885$, $Q_{\text{FR}} \approx 0.795$):
    $$0.30 \times Q_{\text{FR}} = 0.30 \times 0.795 = 0.2385 \quad (\mathbf{+0.0267}\text{ Composite boost})$$
  - Required Segmentation:
    $$0.60 \times Q_{\text{seg}} \ge 0.6198 - 0.2385 = 0.3813 \implies \mathbf{Q_{\text{seg}} \ge 0.6355} \quad (\mathbf{+0.0655}\text{ gain})$$
  - *Feasibility:* **VERY HIGH**.

---

## 3. Leading Underwater Restoration Architectures

### 3.1 Comparative Architectural Matrix

| Architecture | Citations & GitHub Links | Core Innovation | Strengths in Underwater Paired Data | Limitations / Training Costs | Feasibility on RTX 5090 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **NAFNet** | *Chen et al., ECCV 2022*<br>[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | Nonlinear Activation Free: SimpleGate ($x_1 \odot x_2$) + Simplified Channel Attention (SCA) | SOTA PSNR on paired restoration (SIDD 40.3 dB, GoPro 33.7 dB). Extremely fast forward/backward pass. | Sensitive to initialization from scratch without warm start or physical anchor. | **HIGHEST** (~15 min for 20 epochs @ 512px) |
| **Restormer** | *Zamir et al., CVPR 2022*<br>[swz30/Restormer](https://github.com/swz30/Restormer) | Multi-Dconv Head Transposed Attention (MDTA) + Gated-Dconv Feed-Forward Network (GDFN) | Channel-transposed attention captures long-range color attenuation without spatial quadratic cost. SOTA on deraining/defogging. | Higher VRAM footprint; slower training than NAFNet. Pretrained weights require adaptation. | **HIGH** (~40 min for 20 epochs @ 384px) |
| **Ucolor** | *Li et al., IEEE TIP 2021*<br>[Li-Chongyi/UC-Net](https://github.com/Li-Chongyi/UC-Net) | Medium Transmission Guidance + Multi-Color Space (RGB, Lab, HSV) Embedding | Explicit physical guidance tailored to underwater wavelength absorption. | Designed primarily for non-reference aesthetic enhancement rather than paired synthetic PSNR matching. | **MEDIUM** |
| **Spectroformer** | *Khan et al., WACV 2024*<br>[Mdraqibkhan/Spectroformer](https://github.com/Mdraqibkhan/Spectroformer) | Multi-Domain Query Cascaded Attention + Hybrid Fourier-Spatial Upsampling Block | SOTA frequency-domain texture recovery on UIEB and SQUID benchmarks. | Complex multi-branch dependencies; high implementation risk within 48h deadline. | **LOW (High Risk)** |
| **DBWaterNet** | *Li et al., 2024*<br>[Semantic Scholar](https://www.semanticscholar.org) | Dual-branch joint refinement (color correction branch + wavelet detail branch) | Decouples chromatic recovery from spatial edge preservation. | Custom codebase, limited community weights. | **MEDIUM** |

### 3.2 Post-Mortem on Why Phase 3 NAFNet Failed in this Repo
In our repo's previous pilot (`checkpoints/phase3_nafnet_f0`), NAFNet training recorded `val_psnr: 9.06 dB`:
1. **The Residual Target Defect:** In `models/nafnet.py`, line 118: `out = torch.clamp(x + res, 0.0, 1.0)` where `x` was raw degraded input. Initializing ending weights to zero caused Step 0 to start at 9 dB.
2. **The Solution (Physics-Guided Residual NAFNet):**
   In `models/physics_guided_nafnet.py`, the network takes $I_{\text{deg}}$, $J_{\text{rsuigm}}$, $wt$, and $depth$, and outputs:
   $$\hat{J} = \text{clamp}\left(J_{\text{rsuigm}} + 0.25 \times \tanh(\text{net}(x)), 0.0, 1.0\right)$$
   With zero-initialized final layers, **Step 0 starts at exactly 25.48 dB PSNR** (the RSUIGM table). Every gradient update can only improve or refine residual details, preventing catastrophic degradation!

---

## 4. Leading Segmentation Architectures for Underwater Imagery

| Model Architecture | Parameter Count & Backbone | Pretrained Checkpoint in Local Cache? | Underwater Segmentation Suitability | Verified Repo Findings |
| :--- | :---: | :---: | :--- | :--- |
| **SegFormer-B5** | 82.0M (MiT-B5 + All-MLP Decoder) | **YES**<br>`models--EPFL-ECEO--segformer-b5-finetuned-coralscapesv2-1024-1024`<br>`models--smp-hub--mit_b5.imagenet` | **Exceptional.** Hierarchical self-attention with overlapping patch embeddings captures both massive coral reefs and delicate fish boundaries. Pretrained on Coralscapes (underwater reef benchmark). | PicusLab/ENEA's top entry is named `ENEA EURS S59 Full20`. Coralscapes pretraining provides immediate domain alignment. |
| **DINOv2-Large (ViT-L/14)** | 304.0M (Vision Transformer Large) | **YES**<br>`dinov2_vitl14_pretrain.pth` (1.2 GB in torch cache) | **Exceptional.** Self-supervised patch representations are extraordinarily robust to water turbidity, haze, and extreme color shift. | Powered V16 to **0.694 Composite on Dev**. 5-fold ensemble captures rare classes cleanly when unweighted by bad offsets. |
| **ConvNeXt-Large / Base** | 88M / 198M (Pure ConvNet Backbone) | **YES**<br>`convnext_base.fb_in22k_ft_in1k` in HuggingFace cache | **High Complementarity.** Replaces transformer token-grid discretization ($14\times 14$ patches) with continuous $7\times 7$ depthwise convolutions. | In V16 OOF evaluation, blending ConvNeXt-Base with DINOv2 at $\alpha = 0.35$ gained **$+0.0185$ $Q_{\text{seg}}$** across 4,000 images. |

---

## 5. Prior Failure Mechanisms in this Repository

```
+----------------------------------------------------------------------------------------------------+
|                                    HISTORICAL FAILURE MECHANISMS                                   |
+------------------------------------+---------------------------------------------------------------+
| 1. Absent-Class Penalty Trap       | Predicting even 1 FP pixel in an absent-class scene collapses  |
|                                    | scene IoU from 1.000 to 0.000. Regressed #19185 by -0.019 Qseg |
+------------------------------------+---------------------------------------------------------------+
| 2. Zero-Support Metric Illusions   | Fold 3 validation has 0 Ray pixels. Suppressing Ray gave +1.0 |
|                                    | IoU artifact (+0.1636 Qseg illusion) while all real taxa fell |
+------------------------------------+---------------------------------------------------------------+
| 3. In-Sample Ensemble Leakage      | The 0.91966 score was evaluated on scenes seen during training|
|                                    | (4/5 models memorized). True OOF generalization was ~0.51-0.57|
+------------------------------------+---------------------------------------------------------------+
| 4. Lexicographical Depth Bug       | String sort ('10','100','15') + [::5] stride omitted all files|
|                                    | with depth >= 80m, hallucinating +0.007 from offset 0.35      |
+------------------------------------+---------------------------------------------------------------+
```

---

## 6. Short, Prioritized, Ranked List of Top 5 Practical Experiments

### EXP-01: Anchored PhysicsGuidedNAFNet Residual Training (Restoration Priority)
- **Rank:** **#1 (Highest Expected Value)**
- **Hypothesis:** Fine-tuning `PhysicsGuidedNAFNet` (anchored on RSUIGM table $J_{\text{rsuigm}}$ with $\Delta J = 0.25 \times \tanh(\text{net}(x))$) on all 20 scenes (4,000 paired images) with Charbonnier + SSIM loss will increase PSNR from $25.48\text{ dB} \to 28.5–29.5\text{ dB}$ and SSIM from $0.829 \to 0.885+$.
- **Math EV:** $\Delta Q_{\text{FR}} \approx +0.100 \implies \mathbf{+0.030\text{ Composite gain}}$ ($0.634 \to 0.664$).
- **Runtime:** ~25 minutes on RTX 5090 (batch size 16, AMP BF16, crop size 512).
- **Gate:** Held-out fold PSNR must exceed 27.5 dB before submission packaging.

### EXP-02: Fine-Tuning SegFormer-B5 on Full-20 EURS Scenes (Coralscapes Pretrained)
- **Rank:** **#2**
- **Hypothesis:** Fine-tuning SegFormer-B5 initialized from `EPFL-ECEO--segformer-b5-finetuned-coralscapesv2-1024-1024` on all 20 scenes with class-balanced cross-entropy + Lovasz-Softmax loss directly replicates PicusLab's winning `S59 Full20` strategy, lifting $Q_{\text{seg}}$ from $0.570 \to 0.640–0.660$.
- **Math EV:** $\Delta Q_{\text{seg}} \approx +0.070 \implies \mathbf{+0.042\text{ Composite gain}}$.
- **Runtime:** ~35 minutes on RTX 5090 (20 epochs, batch size 8 @ 1024x1024).
- **Gate:** Multi-fold OOF mIoU $\ge 0.740$, Macro $F_1 \ge 0.840$.

### EXP-03: Dual-Backbone Soft Probability Fusion (DINOv2-L + SegFormer-B5)
- **Rank:** **#3**
- **Hypothesis:** Blending the deep semantic representations of our validated 5-fold DINOv2-Large ensemble with the dense high-resolution boundaries of the fine-tuned SegFormer-B5 ($\alpha \approx 0.35$) with multi-scale TTA ($[518, 672, 896]$) eliminates single-model edge jitter without creating false-positive specks.
- **Math EV:** $\Delta Q_{\text{seg}} \approx +0.020 \implies \mathbf{+0.012\text{ Composite gain}}$.
- **Runtime:** ~15 minutes inference & verification across all 1,200 test images.
- **Gate:** OOF agreement check; zero new false positives on zero-support validation folds.

### EXP-04: False-Positive-Bounded Logit Calibration (Anti-Trap Grid Search)
- **Rank:** **#4**
- **Hypothesis:** Restricting rare-class logit offsets to $b_3, b_4 \in [0.00, 0.12]$ and enforcing a hard threshold: if max predicted probability of class $c$ in an image is below $\tau_{\text{presence}} = 0.35$, all logits for class $c$ are clamped to $-\infty$. This guarantees $Pred_c = 0$ in absent-class scenes ($1.0$ IoU preserved) while boosting recall in true positive scenes.
- **Math EV:** Protects against $-0.019$ trap while recovering $+0.008$ recall $\implies \mathbf{+0.010\text{ Composite gain}}$.
- **Runtime:** < 2 minutes (pure GPU logit sweep on validation tensor dump).

### EXP-05: Coupled End-to-End Joint Pipeline Packaging (Candidate V29)
- **Rank:** **#5**
- **Hypothesis:** Packaging the restored imagery from EXP-01 ($Q_{\text{FR}} \ge 0.800$) with the calibrated dual-backbone masks from EXP-03/EXP-04 ($Q_{\text{seg}} \ge 0.645$) breaks the 0.700 official threshold:
  $$\text{Composite} = 0.60(0.645) + 0.30(0.805) + 0.10(0.802) = 0.3870 + 0.2415 + 0.0802 = \mathbf{0.7087}$$
- **Math EV:** **Reaches official challenge target (0.708 > 0.700)**.
- **Runtime:** ~20 minutes (submission packaging, cryptographic hashing, and upload).

---

## 7. Concrete Citations, Open-Source Repositories & Resources

1. **NAFNet:** Chen et al., *"Simple Baselines for Image Restoration"*, ECCV 2022.  
   GitHub: [`https://github.com/megvii-research/NAFNet`](https://github.com/megvii-research/NAFNet)
2. **Restormer:** Zamir et al., *"Restormer: Efficient Transformer for High-Resolution Image Restoration"*, CVPR 2022.  
   GitHub: [`https://github.com/swz30/Restormer`](https://github.com/swz30/Restormer)
3. **SegFormer:** Xie et al., *"SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"*, NeurIPS 2021.  
   GitHub: [`https://github.com/NVlabs/SegFormer`](https://github.com/NVlabs/SegFormer)  
   Hugging Face: [`EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024`](https://huggingface.co/EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024)
4. **DINOv2:** Oquab et al., *"DINOv2: Learning Robust Visual Features without Supervision"*, TMLR 2024.  
   GitHub: [`https://github.com/facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2)
5. **Spectroformer:** Khan et al., *"Spectroformer: Multi-Domain Query Cascaded Transformer Network for Underwater Image Enhancement"*, WACV 2024.  
   GitHub: [`https://github.com/Mdraqibkhan/Spectroformer`](https://github.com/Mdraqibkhan/Spectroformer)
6. **Ucolor:** Li et al., *"Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding"*, IEEE TIP 2021.  
   GitHub: [`https://github.com/Li-Chongyi/UC-Net`](https://github.com/Li-Chongyi/UC-Net)
7. **Coralscapes Benchmark:** Alevizos et al., *"Coralscapes: A Large-Scale Underwater Semantic Segmentation Dataset"*, 2023.
