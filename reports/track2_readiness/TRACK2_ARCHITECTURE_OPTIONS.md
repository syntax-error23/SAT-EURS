# SAT-EURS Track 2 — Proposed Architecture Directions & Evaluation Options

**Author:** Senior Computer Vision Researcher & Model Optimization Specialist  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Target Hardware:** NVIDIA Jetson AGX Orin (Embedded Edge)  
**Complexity Caps:** Total Parameters $\le 1,000,000$ ($\le 1.0\text{M}$), Total Static Compute $\le 12.0\text{ GFLOPs}$ at $1024 \times 1024$  

---

## 1. Executive Summary: The Three Architecture Candidates

Based on the evidence collected during our repository, scoring, and data audits, we propose three distinct, viable architectural directions for the next phase of Track 2 development. None of these candidates is prematurely crowned as the final winner; each targets a specific trade-off between parameter utilization, optical restoration fidelity, and rare-class semantic segmentation.

```
+----------------------------------------------------------------------------------------------------+
|                                    CANDIDATE COMPARISON MATRIX                                     |
+------------------------------+---------------------------+-----------------------+-----------------+
| Architectural Dimension      | Candidate A: EdgeSOTA-V2c | Candidate B: Scaled-V3| Candidate C: DP-Edge|
+------------------------------+---------------------------+-----------------------+-----------------+
| Paradigm                     | Cascaded Lightweight      | Cascaded Capacity-Max | Shared Dual-Path|
| Target Parameter Budget      | ~0.354 M (35.4% cap)      | ~0.880 M (88.0% cap)  | ~0.950 M (95.0% cap)
| Target Compute Budget        | ~2.25 GFLOPs (18.8% cap)  | ~7.80 GFLOPs (65.0%)  | ~9.50 GFLOPs (79.2%)
| Restorer Subnet              | 19-term RSUIGM + Router   | 19-term RSUIGM + Fast | Shared Stem +   |
|                              | (26k params, 0.20 GFLOPs) | (26k params, 0.20 GF) | Bilateral Slice |
| Segmenter Backbone           | MobileNetV3-Small         | MobileNetV3-Large     | Shared Inverted |
|                              | (Pretrained ImageNet)     | (Pretrained ImageNet) | Bottleneck (MB2)|
| Multi-Scale Decoder          | 4-level Pyramid Fusion    | LR-ASPP + Stride 4/8  | Asymmetric Dual |
|                              | (Progressive DW-PW)       | Boundary Fusion       | Neck Heads      |
| Host CPU Latency (Est.)      | ~35 ms (28 FPS)           | ~65 ms (15 FPS)       | ~80 ms (12 FPS) |
| Projected Orin Latency (FP16)| ~10 - 12 ms (>80 FPS)     | ~18 - 25 ms (>40 FPS) | ~25 - 32 ms (>30 FPS)
| Primary Strategic Focus      | Fast Recovery / Fix Bug   | Maximize mIoU & Rare  | Zero Redundant  |
|                              | (Eliminate -4.0 C3 Trap)  | Species Representation| Feature Compute |
+------------------------------+---------------------------+-----------------------+-----------------+
```

---

## 2. Candidate A: Corrected EURS-EdgeSOTA-V2 (V2c — Recovery Baseline)

### 2.1 Concept & Rationale
Candidate A is the minimal-risk correction of the existing `EURS-EdgeSOTA-V2` candidate. It preserves the fast, proven 0.354M parameter pipeline, but completely excises the flawed $-4.0$ Class 3 logit suppression and absent-class validation bias, replacing it with balanced, multi-fold validated calibration.

### 2.2 Inference Data Flow
```
INPUT [1, 3, 1024, 1024]
  │
  ├──► Downsample to [1, 3, 128, 128] ──► 3-stage Router CNN ──► Condition Logits [1, 200]
  │                                                                 │
  │                                                      Softmax(T=0.1) ──► Effective W [1, 19, 3]
  │                                                                 │
  ├──► 19-term RSUIGM Feature Construction [1, 19, 1024, 1024] ─────┴──► Linear Projection
  │                                                                         │
  │                                                                Gaussian Unsharp (alpha=0.15)
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┼──► OUTPUT 1: 'restored_image'
  │                                                                         │
  │                                                                         ▼
  └──────────────────────────────────────────────────────────► MobileNetV3-Small Encoder
                                                                 │
                                                                 ├── mb_256 [1, 16, 256, 256] (Stride 4)
                                                                 ├── mb_128 [1, 24, 128, 128] (Stride 8)
                                                                 ├── mb_64  [1, 48,  64,  64] (Stride 16)
                                                                 └── mb_32  [1, 96,  32,  32] (Stride 32)
                                                                 │
                                                               4-Level Progressive Pyramid Decoder
                                                                 │
                                                               Calibrated Head: Conv2d 32->5 (Bias: [0, 0, 0, +0.10, +0.25])
                                                                 │
                                                               Bilinear Interpolation (Stride 4 -> 1024x1024)
                                                                 │
                                                                 └──► OUTPUT 2: 'seg_logits' [1, 5, 1024, 1024]
```

### 2.3 Resource & Complexity Budget
- **Restoration Subnet:** 26,427 parameters, 0.2016 GFLOPs.
- **Segmentation Subnet:** 327,273 parameters, 2.0480 GFLOPs.
- **Total Parameters:** **0.3537 M** ($\le 1.0\text{M}$ cap, **64.6% headroom**).
- **Total Compute:** **2.2496 GFLOPs** ($\le 12.0\text{ GFLOPs}$ cap, **81.3% headroom**).

### 2.4 Teacher & Distillation Strategy
- **Teachers:** 5-Fold DINOv2-Large Ensemble (`checkpoints/track1_screening/DINOV2_CP_F{0..4}`).
- **Losses:**
  $$\mathcal{L} = 0.50 \cdot \mathcal{L}_{\text{CA-KD}}(T=2.5) + 0.30 \cdot \mathcal{L}_{\text{CE}}(\mathbf{w}=[0.8, 1.0, 1.2, 1.8, 2.0]) + 0.20 \cdot \mathcal{L}_{\text{Boundary}}$$
- **Calibration Vector:** Baked vector $\mathbf{b} = [0.0, 0.0, 0.0, +0.10, +0.25]$ (strictly non-negative for rare classes, preventing false-negative erasure).

### 2.5 Advantages, Risks, and Rejection Criteria
- **Advantages:** Immediate turn-around; zero architecture search required; verified ONNX export and parity already working.
- **Risks:** Severely under-utilizes the 1.0M parameter budget (64% wasted), limiting its ultimate ceiling on rare species representation.
- **Rejection Criteria:** Rejected if genuine 2-fold cross-validation ($S_{\text{folds}}=\{0,1\}$) yields $Q_{\text{seg}} < 0.280$ or if Class 3 IoU collapses to $< 0.15$.

---

## 3. Candidate B: Restoration-Guided Capacity-Scaled Hybrid (Scaled-V3 — High-Performance)

### 3.1 Concept & Rationale
Candidate B addresses the root performance bottleneck: **capacity starvation of the segmentation student**. By expanding the student backbone to **ImageNet-pretrained MobileNetV3-Large** and widening the decoder feature channels, Candidate B fully exploits the competition budget (using ~0.88M parameters and ~7.8 GFLOPs), substantially boosting the representation capacity for rare classes and boundary contours.

### 3.2 Inference Data Flow
```
INPUT [1, 3, 1024, 1024]
  │
  ├──► Fast RSUIGM Inversion Subnet (0.20 GFLOPs, 26k params) ──► OUTPUT 1: 'restored_image'
  │                                                                     │
  │                                                                     ▼
  └──────────────────────────────────────────────────────────► MobileNetV3-Large Backbone
                                                                 │
                                                                 ├── Stage 1: [1,  24, 256, 256] (Stride 4)
                                                                 ├── Stage 2: [1,  40, 128, 128] (Stride 8)
                                                                 ├── Stage 3: [1,  80,  64,  64] (Stride 16)
                                                                 └── Stage 4: [1, 160,  32,  32] (Stride 32, Dilated)
                                                                 │
                                                               Dual LR-ASPP Context Aggregator (in=160, out=96)
                                                                 │
                                                               Multi-Scale Boundary Skip Fusion:
                                                                 ├── Lateral 1: Conv 40->32 on Stride 8
                                                                 └── Lateral 2: Conv 24->24 on Stride 4
                                                                 │
                                                               Progressive Refinement:
                                                                 ├── 32x32 -> 128x128 Fusion (96 + 32 -> 64)
                                                                 └── 128x128 -> 256x256 Fusion (64 + 24 -> 48)
                                                                 │
                                                               High-Res Classification Head (48 -> 5)
                                                                 │
                                                               Bilinear Interpolation (256x256 -> 1024x1024)
                                                                 │
                                                                 └──► OUTPUT 2: 'seg_logits' [1, 5, 1024, 1024]
```

### 3.3 Resource & Complexity Budget
- **Restoration Subnet:** 26,427 parameters, 0.2016 GFLOPs.
- **MobileNetV3-Large Backbone:** ~685,000 parameters, ~5.45 GFLOPs.
- **Multi-Scale Context Decoder:** ~168,000 parameters, ~2.15 GFLOPs.
- **Total Parameters:** **~0.879 M** ($\le 1.0\text{M}$ cap, **12.1% safety margin**).
- **Total Compute:** **~7.80 GFLOPs** ($\le 12.0\text{ GFLOPs}$ cap, **35.0% safety margin**).
- **Calculation Methodology:** Verified via static analytical formula matching `deploy/check_submission_flops.py`.

### 3.4 Teacher & Distillation Strategy
- **Dual-Teacher Blending:** Distilled simultaneously from Track 1 V18 ViT-L/16 (semantic context) AND Track 1 ConvNeXt-Base (boundary contours) using pre-cached offline targets.
- **Losses:**
  $$\mathcal{L} = 0.40 \cdot \mathcal{L}_{\text{KD}}(T=2.0) + 0.30 \cdot \mathcal{L}_{\text{CE}}(\mathbf{w}_{\text{rare}}) + 0.20 \cdot \mathcal{L}_{\text{Bound}} + 0.10 \cdot \mathcal{L}_{\text{Dice}}$$

### 3.5 Advantages, Risks, and Rejection Criteria
- **Advantages:** 2.5x more parameter capacity for segmentation; higher representational fidelity for small fish and macro-fauna; utilizes proven ImageNet pretrained weights.
- **Risks:** Higher compute (7.8 vs 2.2 GFLOPs), though still comfortably within the 12 GFLOPs cap; requires careful memory budgeting during distillation.
- **Rejection Criteria:** Rejected if ONNX export exceeds 1.000M parameters or if latency on host CPU exceeds 90 ms.

---

## 4. Candidate C: Dual-Path Asymmetric Joint Network (DP-Edge — Efficiency Alternative)

### 4.1 Concept & Rationale
Candidate C eliminates redundant feature extraction between restoration and segmentation. Instead of running a standalone restoration subnet and then feeding RGB into a separate segmentation backbone, Candidate C uses a **shared lightweight convolutional stem**, branching into:
1. A **downsampled Bilateral Grid Slice head** that generates spatially varying affine transformation matrices for fast full-resolution restoration.
2. A **deep semantic segmentation branch** operating directly on the shared multi-scale feature representations.

### 4.2 Inference Data Flow
```
INPUT [1, 3, 1024, 1024]
  │
  ▼
Shared Low-Level Convolutional Stem (Stride 2 & 4, channels 16, 32)
  │
  ├──► Feature Map F_stride4 [1, 32, 256, 256]
  │       │
  │       ├──► RESTORATION PATH:
  │       │    Downsample to 64x64 ──► Bilateral Grid Head (channels 64) ──► 12-dim Affine Grid [1, 12, 16, 16]
  │       │                                                                  │
  │       │    Bilateral Slice & Apply on Input RGB [1, 3, 1024, 1024] ◄─────┘
  │       │      │
  │       │      └──► OUTPUT 1: 'restored_image' [1, 3, 1024, 1024]
  │       │
  │       └──► SEGMENTATION PATH:
  │            Deep Inverted Bottleneck Stages (Channels: 48, 96, 160)
  │              │
  │            ASPP Context Aggregator (Stride 32)
  │              │
  │            Skip Fusion with F_stride4
  │              │
  │            Segmentation Head (48 -> 5)
  │              │
  │              └──► OUTPUT 2: 'seg_logits' [1, 5, 1024, 1024]
```

### 4.3 Resource & Complexity Budget
- **Shared Stem:** 4,800 parameters, 0.45 GFLOPs.
- **Bilateral Grid Restoration Head:** 78,000 parameters, 0.65 GFLOPs.
- **Segmentation Branch & Decoder:** ~865,000 parameters, ~8.40 GFLOPs.
- **Total Parameters:** **~0.948 M** ($\le 1.0\text{M}$ cap, **5.2% headroom**).
- **Total Compute:** **~9.50 GFLOPs** ($\le 12.0\text{ GFLOPs}$ cap, **20.8% headroom**).

### 4.4 Advantages, Risks, and Rejection Criteria
- **Advantages:** Eliminates discrete condition router errors entirely; bilateral grid guarantees spatially smooth restoration without boundary artifacts; maximum theoretical parameter efficiency.
- **Risks:** Bilateral grid slice operator in ONNX requires custom ONNX sub-graph tracing (Gather/GridSample) which can face compatibility hurdles with TensorRT on older JetPack versions.
- **Rejection Criteria:** Rejected if ONNX export introduces unsupported TensorRT operators or if joint training causes negative task transfer (restoration gradient fighting segmentation gradient).

---

## 5. Provisional Evidence-Based Ranking

| Rank | Candidate | Projected Composite | Technical Readiness | Confidence Level | Primary Rationale |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Candidate B (Scaled-V3)** | **0.420 – 0.450** | High | **High (85%)** | Directly addresses the confirmed root cause: parameter starvation of the segmentation student. Pretrained MobileNetV3-Large is proven and reliable. |
| **2** | **Candidate A (V2c)** | **0.370 – 0.395** | Immediate | **Very High (95%)** | Simplest path to an immediate, trustworthy baseline. Code and weights already exist; requires only validation re-calibration. |
| **3** | **Candidate C (DP-Edge)** | **0.410 – 0.440** | Medium | **Medium (60%)** | Architecturally superior for joint learning, but carries higher ONNX export and TensorRT deployment risk. |

### What Could Invert This Ranking:
- If Candidate B's larger backbone exceeds latency budgets on Orin hardware ($> 33\text{ ms}$), Candidate A automatically becomes the primary deployment candidate.
- If the RSUIGM router in Candidate B shows visible color shifts on unseen benthic test scenes, Candidate C's continuous bilateral grid becomes essential.
