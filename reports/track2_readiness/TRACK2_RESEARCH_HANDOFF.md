# SAT-EURS Track 2 — Executive Technical Handoff & Research Roadmap

**Author:** Senior Computer Vision Researcher & Competition Auditor  
**Date:** October 23, 2026  
**Repository Branch:** `track1-v18-dual-encoder-sota` (Commit `cd677b2`)  
**Phase:** Pre-Architecture Research, Repository Audit, and Evidence Collection (Read-Only Phase Complete)  
**Deliverable Directory:** `reports/track2_readiness/`  

---

## 1. Executive Summary & Repository Preservation State

This document provides the formal technical handoff for the next researcher tasked with designing, training, and deploying the winning Track 2 architecture for the MaCVi @ WACV 2027 EURS Challenge.

### 1.1 Preservation Guarantee
- **Repository State:** Branch `track1-v18-dual-encoder-sota`, commit hash `cd677b2eaf60298a2938d4ac90dd46470d78e2f8`.
- **Integrity Verified:** Zero files modified or overwritten in tracked repository trees (`git diff` and `git diff --staged` are completely empty).
- **V26 Integrity:** Candidate archive `deploy/v26/submission_v26_test_apex_sota.zip` (SHA-256: `f7af242f806dd64c3daca6f6d2119dfa1074c77c89f847068691dfe1f6b5a70f`) is **100% bit-for-bit preserved and untouched**.
- **Protected V18 Baseline:** SHA-256 `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52` remains immutable.

---

## 2. Definitive Answers to the Five Core Mission Questions

### Question 1: What exactly did we submit as V26, and what evidence supports its design?
- **Answer:** V26 is **exclusively a Track 1 submission** (`deploy/v26/submission_v26_test_apex_sota.zip`, 498.00 MB), consisting of 2,400 files (1,200 restored JPEGs + 1,200 PNG masks) covering the 6 hidden test scenes (`test04`, `test09`, `test10`, `test11`, `test19`, `test20`).
- **Architecture:** An unconstrained 10-model ensemble:
  - 5-Fold NAFNet Restoration Refiners with 19-term RSUIGM analytical priors and $\alpha=0.10$ unsharp enhancement.
  - 5-Fold Dual-Encoder Segmenters (Meta DINOv3 ViT-L/16 on restored RGB + DINOv3 ConvNeXt-Base on raw RGB + UPerNet neck with FiLM water-type/depth gating).
  - Horizontal-flip TTA and logit calibration vector $[0.0, 0.0, 0.0, 0.10, 0.25]$.
- **Evidence:** V26 eliminates V25's destructive double-unsharp filter (recovering $+0.29\text{ dB}$ PSNR) and moderates macro-fauna bias (protecting Class 2 fish recall). Flip TTA delivers $+0.0371\text{ } Q_{\text{seg}}$ (+15.3% relative).
- **Complexity:** ~1.98 Billion parameters, > 1,996 GFLOPs. **It cannot run under Track 2 constraints.**

### Question 2: What is the strongest genuinely validated Track 2 baseline?
- **Answer:** **EURS-Net V5 Calibrated (`#18699`)** (`deploy/eurs_net_v5_calibrated_track2.onnx`).
- **Official Score:** **0.3380 Composite** ($Q_{\text{seg}} = 0.2450$, mIoU = 0.4260, Macro F1 = 0.5760, PSNR = 17.10 dB, SSIM = 0.8210, Sharpness = 0.3190).
- **Compliance:** 0.7683M parameters ($\le 1.0\text{M}$), 9.0500 GFLOPs ($\le 12.0\text{ GFLOPs}$).
- **Status:** This is the **only Track 2 model with a verified score on the official leaderboard**. Newer candidates like `EURSEdgeSOTA-V2` claim 0.4074 locally, but this was an artifact of flawed single-fold validation and has never been submitted.

### Question 3: Why is Track 2 behind, and which problems matter most to the official score?
- **Root Cause 1 (Absent Class Validation Artifact):** Previous Track 2 research evaluated models only on Fold 0. In Fold 0, Class 3 is absent. A $-4.0$ offset baked into Class 3 caused the model to predict 0 pixels of Class 3. The local evaluator assigned $0/0 \implies 1.0$ IoU, creating a false impression of $0.5389$ mIoU and $0.3314$ $Q_{\text{seg}}$. On real test scenes containing sharks/rays, this offset triggers the **zero-coverage penalty ($P_{\text{cov}} = 0.0$)**, which completely destroys $Q_{\text{seg}}$!
- **Root Cause 2 (Severe Under-Utilization of Budget):** Candidate `EURSEdgeSOTA-V2` used only 0.354M of the 1.0M parameters (64.6% unused) and 2.25 of the 12.0 GFLOPs (81.3% unused). The student was severely starved of capacity to represent complex coral textures and small marine fauna simultaneously.
- **Root Cause 3 (Blind Restoration Penalty):** Track 1 achieved 25.41 dB using oracle filename lookups. Track 2 models must infer the condition blindly from RGB, dropping blind RSUIGM restoration to ~20.62 dB.

### Question 4: What training data, teachers, checkpoints, infrastructure, and evaluation tools are actually available?
- **Data:** 20 clean scenes, 4,000 degraded images ($1024 \times 1024$ cached anti-aliased in `data/cached_1024/`).
- **Teachers Available:**
  - 5-Fold DINOv2-Large Ensemble (`checkpoints/track1_screening/DINOV2_CP_F{0..4}`, 308M params/fold).
  - 5-Fold ConvNeXt-Base Ensemble (`checkpoints/v16_convnext/fold_{0..4}`, 88.5M params/fold).
  - 5-Fold V18 Dual-Encoder Ensemble (`checkpoints/v18/fold_{0..4}`, 396M params/fold).
- **Cached Logits:** **Zero cached targets exist on disk.** Must be generated.
- **Hardware Tools:** Official complexity checker (`deploy/check_submission_flops.py`) verified and fully functional. Host RTX 5090 (32.6 GB VRAM) available for sub-hour training runs.

### Question 5: What architecture and experiments should be considered next?
- **Candidate Direction:** **Candidate B (Restoration-Guided Capacity-Scaled Hybrid — Scaled-V3)**.
- **Key Strategy:** Keep the lightweight RSUIGM restorer (0.20 GFLOPs, 26k params), upgrade the student to **ImageNet-pretrained MobileNetV3-Large** (~685k params) with an **LR-ASPP multi-scale boundary fusion neck** (~168k params), totaling **~0.88M parameters and ~7.8 GFLOPs**.
- **Distillation:** Distill from pre-cached targets of the 5-fold DINOv2-Large and ConvNeXt-Base teachers using balanced class weighting ($\mathbf{w}_{\text{rare}} \ge 1.0$).
- **Validation:** Validate strictly on the 8-scene combined split $S_{\text{folds}} = \{0, 1\}$ (where all 5 classes are strictly present), reporting genuine, unvarnished metrics.

---

## 3. High-Impact Uncertainties & Missing Information

1. **Physical Jetson AGX Orin Execution:** No physical benchmarks have been performed on real Orin hardware. The procedure script (`deploy/benchmark_orin.sh`) is complete, but awaits execution on physical hardware.
2. **Proprietary No-Reference Server Evaluator ($Q_{\text{NR}}$):** The exact weights and algorithm for the official server-side UIQM/UCIQE/CCF metric are withheld by the challenge organizers. Local evaluation must continue using the established $Q_{\text{NR}} = 0.50$ baseline / $0.718$ server projection.
3. **Pending V26 Test Leaderboard Result:** Official test set scores for submission V26 remain pending from the competition server.

---

## 4. The Exact Next Experiment to Run: EXP-TRACK2-SCALED-V3

### Experiment Specification:
- **Phase:** Implementation & Verification
- **Prerequisite 1:** Execute a target caching script to generate $512 \times 512$ uint8 teacher probability targets for all 4,000 images (`data/cached_teacher_targets/`, ~5.2 GB, ~18 minutes on RTX 5090).
- **Prerequisite 2:** Implement `Candidate B` in `models/track2_edgesota_v3.py` (MobileNetV3-Large backbone + LR-ASPP + Stride 4 boundary skip fusion).
- **Training Configuration:**
  - Optimizer: AdamW, differential learning rate ($1.5 \times 10^{-4}$ backbone, $3.0 \times 10^{-4}$ decoder), cosine annealing, 30 epochs (~45 minutes on RTX 5090 with cached targets).
  - Loss: $0.40 \cdot \mathcal{L}_{\text{KD}} + 0.30 \cdot \mathcal{L}_{\text{CE}}(\mathbf{w}) + 0.20 \cdot \mathcal{L}_{\text{Bound}} + 0.10 \cdot \mathcal{L}_{\text{Dice}}$.
  - Calibration: Multi-fold validated positive offsets $[0.0, 0.0, 0.0, +0.10, +0.25]$.
- **Validation Gate:** Evaluate on $S_{\text{folds}} = \{0, 1\}$ (8 scenes, 1,600 images). Model promoted only if:
  - $\text{Total Parameters} \le 1.0\text{M}$ and $\text{Compute} \le 12.0\text{ GFLOPs}$.
  - $P_{\text{cov}} = 1.000$ (strictly zero class erasure).
  - Genuine $Q_{\text{seg}} \ge 0.3000$ (beating Baseline #18699 at 0.2450 by $\ge +22\%$).
  - Full Composite Score $\ge 0.3800$ (beating Baseline #18699 at 0.3380 by $\ge +0.042$).

---

## 5. Audit Deliverables Index

All detailed investigation reports have been written to `reports/track2_readiness/` and are available for deep technical inspection:

1. [`reports/track2_readiness/V26_SUBMISSION_AUDIT.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/V26_SUBMISSION_AUDIT.md)  
   *Exact V26 artifacts, 10-model architecture, cryptographic checksums, OOF ablation evidence, and testing-phase defect post-mortem.*
2. [`reports/track2_readiness/TRACK2_REPOSITORY_INVENTORY.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/TRACK2_REPOSITORY_INVENTORY.md)  
   *Classification of all Track 2 candidates (`VALIDATED`, `TRAINED_UNVERIFIED`, `SUPERSEDED`, `REJECTED`), checkpoints, teacher assets, and tooling.*
3. [`reports/track2_readiness/TRACK2_SCORING_AND_DATA_AUDIT.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/TRACK2_SCORING_AND_DATA_AUDIT.md)  
   *Mathematical formula analysis, coverage penalty veto, class distribution census, 5-fold split defect post-mortem, and the deterministic 2-fold pairing solution.*
4. [`reports/track2_readiness/TRACK2_DISTILLATION_READINESS.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/TRACK2_DISTILLATION_READINESS.md)  
   *Inventory of high-capacity teachers (V18, DINOv2, ConvNeXt), target cache storage analysis, implemented loss suites, and leakage-free distillation protocol.*
5. [`reports/track2_readiness/TRACK2_DEPLOYMENT_AUDIT.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/TRACK2_DEPLOYMENT_AUDIT.md)  
   *Official complexity checker rules, parameter/compute headroom breakdown, host CPU vs. Jetson Orin measurement segregation, and input/output contract verification.*
6. [`reports/track2_readiness/TRACK2_ARCHITECTURE_OPTIONS.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/TRACK2_ARCHITECTURE_OPTIONS.md)  
   *Comprehensive specification of Candidate A (V2c), Candidate B (Scaled-V3), and Candidate C (DP-Edge), resource budgets, data flows, and evidence-based rankings.*
7. [`reports/track2_readiness/TRACK2_RESEARCH_HANDOFF.md`](file:///c:/Projects/SAT-EURS/reports/track2_readiness/TRACK2_RESEARCH_HANDOFF.md)  
   *Executive summary, answers to core mission questions, primary uncertainties, and the exact next experiment to run.*
