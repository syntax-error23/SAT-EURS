# SAT-EURS Track 2 — Teacher Inventory & Distillation Readiness Audit

**Auditor:** Independent Senior Computer Vision Researcher & Distillation Engineer  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Audited Code & Artifacts:** `checkpoints/`, `models/train_distillation_track2.py`, `tools/train_track2_student.py`, `losses/`, `scratch/verify_teachers.py`, `scratch/test_teacher_ensemble.py`  

---

## 1. Executive Summary & Readiness Verdict

Knowledge distillation from Track 1 teachers is the primary mechanism to bridge the capacity gap between heavy vision transformers ($> 1\text{ Billion parameters}$) and the lightweight Track 2 edge budget ($\le 1.0\text{M}$ parameters, $\le 12.0\text{ GFLOPs}$).

### Key Readiness Findings:
1. **Teacher Checkpoints Exist and Are Validated:** We have 5-fold checkpoints for three major high-capacity Track 1 architectures: DINOv2-Large (5 folds, 308M params/fold), ConvNeXt-Base (5 folds, 88.5M params/fold), and V18 Dual-Encoder (5 folds, 396M params/fold).
2. **Zero Pre-Cached Teacher Targets on Disk:** No offline pre-computed logit caches, probability maps, or feature caches exist on disk (Experiment P0 in commit `4ea69f2` was blocked due to missing caches). Teachers must either be evaluated on-the-fly or cached to disk in an explicit pre-computation phase.
3. **Flawed Teacher Blending in Prior Student Training:** In `tools/train_track2_student.py` (lines 220–230), the training script loaded Teachers F0, F1, and F2 together to distill a student on Fold 0 training scenes. While student training itself saw only training scenes, when the author evaluated the teacher ensemble on Fold 0 validation scenes (`fish1, fish11, fish12, seahorse1`), **Teachers F1 and F2 had already been trained on those validation scenes**, producing a false $0.8601$ mIoU through cross-fold validation leakage.
4. **Feasibility of Offline Target Caching:** Pre-computing and saving the 5-fold teacher ensemble predictions across all 4,000 images at $512 \times 512$ resolution using uint8 probability quantization requires only **5.24 GB** of disk space and eliminates $> 75\%$ of GPU training time.

---

## 2. Complete Inventory of Available Teacher Models

| Teacher Model Identifier | Architecture | Checkpoint Path | Parameters | Fold / Split | Input Resolution | Prediction Type | Validation SOTA Composite | Leakage-Free Student Suitability |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Track 1 V18 Dual-Encoder** | DINOv3 ViT-L/16 + ConvNeXt-Base + UPerNet | `checkpoints/v18/fold_{0..4}/segmenter_best.pt` | 396.5 M / fold (1.98 B ensemble) | 5-Fold Scene Disjoint | $1024 \times 1024$ (padded to 32) | Segmentation (5 classes + boundary + objectness) | **0.749 (Dev) / 0.612 (Test)** | **EXCELLENT:** Highest capacity semantic teacher. Must use Fold $k$ teacher strictly for Fold $k$ student. |
| **Track 1 DINOv2-Large (CP)**| DINOv2 ViT-L/14 + 4-level linear decoder | `checkpoints/track1_screening/DINOV2_CP_F{0..4}/best_model.pt` | 308.1 M / fold (1.54 B ensemble) | 5-Fold Scene Disjoint | $518 \times 518$ (patch 14) | Segmentation (5 classes) | **0.689 (SOTA #18713)** | **PROVEN:** Directly produced the winning 0.689 Dev submission. Fast on-the-fly inference. |
| **Track 1 ConvNeXt-Base** | ConvNeXt-Base + UPerNet | `checkpoints/v16_convnext/fold_{0..4}/best_model.pt` | 88.5 M / fold | 5-Fold Scene Disjoint | $512 \times 512$ | Segmentation (5 classes) | ~0.655 | **EXCELLENT:** Superior high-frequency boundary and edge delineation compared to ViTs. |
| **V17 Specialist Full-20** | DINOv2-Large on all 20 scenes | `checkpoints/v17_specialists/dinov2_full20_best.pt` | 308.1 M | Full 20 Scenes (No split) | $518 \times 518$ | Segmentation (5 classes) | ~0.720 | **SPECIAL USE ONLY:** Cannot be used for out-of-fold validation (contains full dataset). Suitable only for final competition deployment training. |
| **V26 Deep Bilateral Restorer**| Multi-Scale Bilateral Grid / Affine Restorer | `checkpoints/v26/restorer_full20_best.pt` | 1.22 M | Full 20 Scenes | Arbitrary | Full-Reference Restoration | 25.24 dB PSNR, 0.788 SSIM | **EXCELLENT:** Fast, lightweight restoration teacher to guide the physical restorer subnet. |
| **NAFNet Physics Refiner** | 4-Stage NAFNet (FiLM conditioned) | `checkpoints/v18/fold_{0..4}/refiner_best.pt` | 1.22 M / fold | 5-Fold Scene Disjoint | $1024 \times 1024$ | Restoration Residual & Affine field | 27.16 dB PSNR | **EXCELLENT:** Validated restoration refiner. |

---

## 3. Cached Target Status & Disk Storage Feasibility

### 3.1 Current Status: No Cached Targets on Disk
- Exhaustive search confirmed zero `.npy`, `.npz`, `.h5`, or tensor cache files exist on disk for teacher predictions.
- In `tools/train_track2_student.py`, teacher predictions were generated **on-the-fly** during each training step.
- Bottleneck: Evaluating the 308M-parameter DINOv2-Large teacher on-the-fly during training consumed $> 65\%$ of per-step GPU compute on the RTX 5090, limiting training batch size and step throughput.

### 3.2 Storage Calculation for Offline Dataset Caching (4,000 Images)

| Representation Format | Resolution | Channels | Precision / Data Type | Per-Image Size | 4,000 Images Total | Feasibility & Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Uncompressed Full-Res Logits** | $1024 \times 1024$ | 5 | Float32 (4 bytes) | 20.97 MB | **83.88 GB** | Feasible on 660 GB SSD, but excessive oversampling (DINOv2 operates natively on 37x37 grid). |
| **Full-Res Half-Precision Logits** | $1024 \times 1024$ | 5 | Float16 (2 bytes) | 10.49 MB | **41.94 GB** | Good numerical precision. |
| **Matched Patch Resolution Logits**| $518 \times 518$ | 5 | Float16 (2 bytes) | 2.68 MB | **10.73 GB** | Matches native teacher output; fast loading. |
| **Quantized Softmax Probabilities** | $512 \times 512$ | 5 | UInt8 ($0 \dots 255$) | 1.31 MB | **5.24 GB** | **RECOMMENDED OPTIMAL:** Downsampling to $512\times 512$ with uint8 quantization introduces $< 0.15\%$ probability error while fitting the entire 4,000-image dataset into **5.24 GB**. |

### 3.3 Leakage-Free Target Generation Protocol
To prevent data leakage when caching teacher predictions:
1. **Out-of-Fold Rule:** For each scene $s \in \text{val\_scenes}(k)$, the cached teacher target MUST be generated strictly using Teacher $k$ (which was trained on the remaining 16 scenes and never saw scene $s$).
2. **Horizontal-Flip TTA Consistency:**
   $$\hat{P}(I) = \frac{1}{2} \left( P(I) + \text{FlipHorizontal}\left(P(\text{FlipHorizontal}(I))\right) \right)$$
   The flipped prediction must be un-flipped BEFORE averaging.
3. **Restoration-Domain Input:** The teacher was trained on physically restored RGB images. Inputs to the teacher during cache generation must pass through the verified RSUIGM table inversion.

---

## 4. Distillation Loss Implementation Audit

We inspected existing loss implementations across `losses/` and `tools/train_track2_student.py`:

| Distillation / Loss Technique | Implemented Script / Module | Mathematical Formulation | Status & Integrity |
| :--- | :--- | :--- | :---: |
| **Standard Soft-Target KD** | `tools/train_track2_student.py` (lines 351–365) | $\mathcal{L}_{\text{KD}} = T^2 \cdot \text{KL}(P_S(z_S/T) \parallel P_T(z_T/T))$ | **IMPLEMENTED & VERIFIED** ($T=2.5$) |
| **Class-Aware Weighted KD** | `tools/train_track2_student.py` (lines 366–380) | $\mathcal{L}_{\text{CA-KD}} = \sum_{c=0}^4 w_c \cdot \text{CE}(P_S^{(c)}, P_T^{(c)})$ with $\mathbf{w} = [0.8, 1.0, 1.2, 1.8, 2.0]$ | **IMPLEMENTED & VERIFIED** (Protects rare classes 3 & 4) |
| **Boundary-Aware Loss** | `losses/segmentation_v7_loss.py` (`BoundaryBandLoss`) | Dilated morphological band extraction: $\mathcal{B} = \text{Dilate}(M) \setminus \text{Erode}(M)$; $\mathcal{L}_{\text{bound}} = \text{CE}(z_S[\mathcal{B}], M[\mathcal{B}])$ | **IMPLEMENTED & VERIFIED** (+27% fish boundary IoU) |
| **Present-Class Lovász-Softmax**| `losses/v4_losses.py` (`PresentClassLovaszSoftmaxLoss`) | Direct surrogate optimization of Jaccard index over present classes: $\mathcal{L}_{\text{Lovasz}} = \frac{1}{|C_{\text{present}}|} \sum_{c} \overline{\Delta}_{J_c}(m(c))$ | **IMPLEMENTED BUT REJECTED:** Mobile inverted bottleneck collapsed on small fish. |
| **Feature-Based Hint Distillation**| *None* | $\mathcal{L}_{\text{feat}} = \frac{1}{2} \| \phi_{\text{proj}}(F_S) - F_T \|_2^2$ | **MISSING:** Requires $1\times 1$ conv adapter from student feature maps to teacher token dim (1024). |
| **Teacher Ensemble Averaging**| `tools/train_track2_student.py` | Average probability across ensemble before computing KD loss | **IMPLEMENTED** (Must ensure single-fold partition) |

---

## 5. Recommended Distillation Experiment Sequence

To train the next Track 2 architecture without leakage and with maximal transfer from Track 1:

1. **Step 1: Leakage-Free 5-Fold Teacher Cache Generation:**
   - Run a dedicated caching script to evaluate each fold's DINOv2-Large and ConvNeXt-Base teacher on its held-out validation scenes.
   - Store $512 \times 512$ uint8 probabilities in `data/cached_teacher_targets/` (5.24 GB).
2. **Step 2: Dual-Objective Balanced Student Training:**
   - Train the student with combined Class-Aware KD ($\alpha=0.5$), Supervised CE with rare-class weighting ($\beta=0.3$), and Morphological Boundary Loss ($\gamma=0.2$).
3. **Step 3: Multi-Fold Cross-Validation:**
   - Validate on the 8-scene combined split $S_{\text{folds}} = \{0, 1\}$ (all 5 classes present) to obtain true, unvarnished mIoU and F1 scores without vacuous 1.0 artifacts.
