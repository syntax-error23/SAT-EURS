# SAT-EURS Track 1 — V18 Implementation & No-GPU Verification Summary

**Date:** 2026-10-17  
**Auditor:** Antigravity CV Research Team  
**System Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)  
**Status:** All V18 Modules Implemented, Synthetically Verified on CPU, Zero GPU Compute Run  

---

## 1. Code Checks & Synthetic Verification Summary

All modules and contracts were verified via CPU-only unit tests (`tests/v18/test_v18_synthetic.py` and `tools/v18/run_v18_pipeline.py --dry_run`):

| Test Suite / Component | Verification Focus | Expected Output / Contract | Measured Result | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Zero-Init Identity** | `ConditionedRestorationRefinerV18` | $\|I_{\text{refined}} - I_{\text{base}}\|_{\infty} < 10^{-6}$ | **0.00000000 (Exact)** | **PASSED** |
| **Restoration Loss** | `RestorationV18Loss` | Charbonnier + MS-SSIM + Lap + Affine + Res | **Finite (0.5266)** | **PASSED** |
| **Aspect Padding Invariance** | `V18CanonicalPreprocessor` | Aspect-preserving scale + pad/unpad to 32 | **Exact target shape restored** | **PASSED** |
| **Mask Categorical Invariants** | `V18CanonicalPreprocessor` | Classes strictly in $\{0, 1, 2, 3, 4\}$ | **Preserved (0 invalid IDs)** | **PASSED** |
| **Condition Parser Validation** | `parse_scene_wt_depth` | Fails loudly on invalid $wt$ or depth | **ValueError raised on invalid** | **PASSED** |
| **Ignore Index Masking** | `SegmentationV18Loss` | All-255 target produces 0.0 main loss (no nan) | **l_ce = 0.0, l_lovasz = 0.0** | **PASSED** |
| **Dual-Encoder Neck Shapes** | `EURSAPEXV18Segmenter` | Logits: `[B, 5, H, W]`, Bnd: `[B, 1, H, W]`, Obj: `[B, 1, H, W]` | **[1, 5, 64, 64], [1, 1, 64, 64], [1, 1, 64, 64]** | **PASSED** |
| **Learned Gating Maps** | `LearnedSpatialGate` | Initialized to balanced $0.50$ gate | **Sigmoid(0) = 0.50 exact** | **PASSED** |
| **Fold Manifest Disjointness** | `splits/manifest_5fold.json` | 5 folds, 20 unique scenes, 0 variant leakage | **Verified 100% disjoint** | **PASSED** |
| **Pipeline CLI Safe Default** | `tools/v18/run_v18_pipeline.py` | CPU dry-run without writing submissions or training | **Exits 0 in 0.12s on CPU** | **PASSED** |

---

## 2. Unresolved Weight & Dataset Prerequisites

In strict adherence to the **no-unauthorized-download rule**, zero large weights or external datasets were downloaded or installed during this task. Below are the prerequisites required before executing a full training run:

### A. Pretrained Model Weights (DINOv3 on LVD-1689M)
1. **DINOv3 ViT-L/16**:
   - **Model Identifier:** `vit_large_patch16_dinov3`
   - **HuggingFace Hub ID:** `timm/vit_large_patch16_dinov3.lvd1689m`
   - **License:** Meta DINOv3 Research License (`dinov3-license`)
   - **Expected Download Size:** ~1.2 GB
   - **Status:** Architecture supported in installed `timm` (1.0.30). Weights not downloaded.
2. **DINOv3 ConvNeXt-Base**:
   - **Model Identifier:** `convnext_base.dinov3_lvd1689m`
   - **HuggingFace Hub ID:** `timm/convnext_base.dinov3_lvd1689m`
   - **License:** Meta DINOv3 Research License (`dinov3-license`)
   - **Expected Download Size:** ~350 MB
   - **Status:** Architecture supported in installed `timm`. Weights not downloaded.
3. **Fallback Option (Ready Immediately without Downloads):**
   - The environment already contains cached `dinov2_vitl14_pretrain.pth` (1.2 GB) and `convnext_base.fb_in22k_ft_in1k` in `C:\Users\HP\.cache\torch\hub\checkpoints\` and HuggingFace hub.
   - The V18 codebase supports initializing with these cached weights as an immediate offline alternative.

### B. External Partial-Label Datasets
If external representation pretraining is enabled:
1. **Coralscapes:** ~18.5 GB (CC BY-NC 4.0, coral category maps to EURS Class 1).
2. **SUIM:** ~2.1 GB (MIT License, coarse groups map to binary marine objectness).
3. **UIIS10K:** ~12.0 GB (Academic Research License, echinoderms map to EURS Class 4).
4. **USIS10K:** ~8.4 GB (Open Academic Use, salient masks map to binary objectness).
- **Total External Storage Requirement:** ~41.0 GB.
- **Status:** Adapters and taxonomy manifests (`configs/v18/external_taxonomy_manifest.json`) are fully implemented and unit-tested; no datasets have been downloaded.

---

## 3. Preserved Artifacts & Cryptographic Verification

All baseline models, manifests, and submissions from V15, V16, and V17 remain untouched byte-for-byte:

| Artifact Path | Target Description | Verified Status | SHA256 Hash |
| :--- | :--- | :---: | :--- |
| `checkpoints/track1_screening/tbl19_20sc_clean.pt` | Original 20-Scene Table | **UNMODIFIED** | `2c35df3b0f5d3317c3672b2867e0059b893acecd8699c5a2271383a1179d9562` |
| `checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt` | Verified R1 Table | **UNMODIFIED** | `0412854ddf038c1a63cbb03024c0d02ee073b609939e144a37f59d57b1f3c32e` |
| `deploy/submission_v16_apex_sota.zip` | Official Dev Record (0.694) | **UNMODIFIED** | `943369fcdff398c1c18c9b43d4aa792baa1e343914a3dc6f186ca52bb8ced6df` |
| `deploy/submission_v17_dual_backbone_sota.zip` | V17 SOTA Candidate | **UNMODIFIED** | `50f85ec2111ad6d62a03d718fb8fe09fe4e75201d7cdf359c9a81baac138cc4e` |
| `deploy/FROZEN_CANDIDATE_V17_MANIFEST.json` | V17 Provenance Manifest | **UNMODIFIED** | Verified Compliant |
| `splits/manifest_5fold.json` | Official 5-Fold Split | **UNMODIFIED** | `d4e56598c1052ea63a232768565a4e40e6c60dc327f55b91b97a2245b98f2445` |

---

## 4. Planned Command Line for Eventual Full Run (Pending Approval)

When explicit authorization is granted, the full V18 pipeline can be executed via the unified CLI:

```bash
# Planned Full 5-Fold Training, Evaluation, and Packaging Command:
python tools/v18/run_v18_pipeline.py --stage full --folds 0 1 2 3 4 --config_dir configs/v18 --checkpoint_dir checkpoints/v18 --report_dir reports/v18
```

### Configurable Resource & Time Budget (RTX 5090, 32 GB VRAM):
- **Phase 1: Restoration Refiner Training:**
  - 15 epochs per fold $\times$ 5 folds
  - With in-memory RAM caching: ~2.5 mins per fold $\implies$ **~12.5 mins total**.
- **Phase 2: Segmenter Training (DINOv3 Dual-Encoder):**
  - 15 epochs per fold $\times$ 5 folds (batch size 16 with accum steps 4)
  - With frozen early blocks and BF16 AMP: ~9.0 mins per fold $\implies$ **~45.0 mins total**.
- **Phase 3: Strict 5-Fold OOF Evaluation & Calibration:**
  - All 4,000 images evaluated paired $\implies$ **~5.0 mins**.
- **Phase 4: Dev Inference (1,200 images) & Packaging:**
  - Full-resolution TTA inference $\implies$ **~6.0 mins**.
- **Total Projected Runtime:** **~68.5 minutes** on the RTX 5090.

---

## 5. Stop Condition

In accordance with the prompt's mandatory rules:
- **Zero GPU training was launched.**
- **Zero GPU benchmarks or smoke tests were executed.**
- **Zero external weights or datasets were downloaded.**
- **No submission package was generated or uploaded.**
- All code, configs, adapters, and tests are implemented and ready for execution upon user approval.
