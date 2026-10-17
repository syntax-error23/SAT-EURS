# SAT-EURS Track 1 — V18 Architecture Implementation Report

**Challenge:** MaCVi @ WACV 2027 EURS Track 1 (Underwater Restoration & Semantic Segmentation)  
**Date:** 2026-10-17  
**Author:** Antigravity CV Research Team  
**Design Contract:** `SAT_EURS_V18_Architecture_Blueprint.md`  
**Execution Status:** Full Implementation Completed (Implementation-Only, Zero GPU Compute Used)  

---

## 1. Executive Architectural Overview

V18 implements a unified, score-targeted system addressing both $Q_{\text{seg}}$ (60% weight in composite) and full-reference restoration fidelity $Q_{\text{FR}}$ (30% weight in composite) to bridge the gap from V16's official 0.694 record toward 0.80+.

```
                             [ Raw Degraded Underwater Input Image ]
                                                |
                       +------------------------+------------------------+
                       |                                                 |
                       v                                                 v
        [ Verified Scale-Consistent RSUIGM ]               [ Raw Complementary Pathway ]
        - Table: tbl19_20sc_scale_consistent.pt            - Official DINOv3 ConvNeXt-Base
        - 19-term basis @ 1024x1024 native scale           - Strides: 4, 8, 16, 32
        - Gaussian Unsharp (1.34 / -0.34, sigma=2.0)       - Two-Stage Raw Detail Stem (Strides 2, 4)
                       |                                                 |
                       v                                                 |
        [ Conditioned Restoration Refiner V18 ]                          |
        - Concatenated [Degraded, Base] (6ch)                            |
        - FiLM Condition Injection (Water Type + Depth)                  |
        - Multi-Scale NAFNet (32, 64, 128, 192)                          |
        - Dual Heads:                                                    |
          * Low-res Local Affine (Identity init)                         |
          * Full-res RGB Residual (Zero init)                            |
        - Guaranteed Step 0 Identity: J_refined = J_base                 |
                       |                                                 |
                       v                                                 |
        [ Main Semantic Pathway (DINOv3 ViT-L/16) ]                      |
        - Restored RGB normalized                                        |
        - Dense tokens tapped from blocks 5, 11, 17, 23 (Stride 16)      |
        - First 18 blocks frozen; final 6 blocks adapted                 |
                       |                                                 |
                       +------------------------+------------------------+
                                                |
                                                v
                              [ V18 UPerNet / FPN Fusion Neck ]
                              - Multi-scale projections (256 channels)
                              - Dynamic Learned Spatial Gating at Strides 16 & 4
                              - Bounded FiLM Condition Modulation in Decoder
                              - Aspect-Preserving Reflection Padding (Multiple of 32)
                                                |
                       +------------------------+------------------------+
                       |                        |                        |
                       v                        v                        v
            [ Main Five-Class Head ]    [ Boundary Aux Head ]   [ Objectness Aux Head ]
            - Classes {0, 1, 2, 3, 4}   - 1-channel Logits      - 1-channel Binary Logits
            - Unpadded to target size   - 15px Boundary Band    - Marine vs. Background
                       |
                       v
            [ Nested Logit Calibration ]
            - Offsets (b3=+0.15, b4=+0.15)
            - 100% Coverage Preserved
```

---

## 2. Inventory of Implemented V18 Modules

### A. Preprocessing & Geometry Contracts
- **Module:** [`data/v18/preprocessing.py`](file:///c:/Projects/SAT-EURS/data/v18/preprocessing.py)
  - `V18CanonicalPreprocessor`: Scales images to `max_side=1024` preserving native aspect ratio.
  - Pads internally to nearest multiple of 32 using reflection padding.
  - Strictly unpads/crops outputs back to `(target_h, target_w)` before serialization.
  - `parse_scene_wt_depth`: Enforces strict valid ranges: $wt \in \{1..10\}$, $depth \in \{5..100 \text{ step } 5\}$.
  - `create_run_manifest`: Generates cryptographic provenance manifests with checkpoint SHA256 hashes.
  - `validate_fold_manifest_integrity`: Enforces disjoint 5-fold split integrity with zero variant leakage.

### B. Restoration Refiner
- **Module:** [`models/v18/restoration_refiner.py`](file:///c:/Projects/SAT-EURS/models/v18/restoration_refiner.py)
  - `ConditionedRestorationRefinerV18`: Consumes concatenated 6-channel input `[degraded_rgb, base_restored_rgb]`.
  - Consumes post-unsharp base output $I_{\text{base}}$ to preserve the proven 25.55 dB SOTA floor at step 0.
  - `ConditionEmbedding`: Maps discrete water type and depth into per-channel scale ($\gamma$) and shift ($\beta$) vectors.
  - Multi-scale NAFNet encoder-decoder with channels `[32, 64, 128, 192]`.
  - Head A (Local Affine): Generates 12-channel $3 \times 3$ transform + bias at $1/8$ resolution; initialized to identity ($\mathbf{A} = \mathbf{I}_3, \mathbf{b} = \mathbf{0}$).
  - Head B (RGB Residual): Generates $3$-channel residual; strictly zero-initialized (weights $=0$, bias $=0$).
  - Mathematical Guarantee: Step 0 prediction satisfies $\|I_{\text{refined}} - I_{\text{base}}\|_{\infty} = 0.00000000$.

- **Module:** [`losses/v18/restoration_losses.py`](file:///c:/Projects/SAT-EURS/losses/v18/restoration_losses.py)
  - `RestorationV18Loss`:
    $$\mathcal{L}_{\text{rest}} = 0.50 \cdot \mathcal{L}_{\text{Charb}} + 0.30 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.10 \cdot \mathcal{L}_{\text{Lap}} + 0.05 \cdot \mathcal{L}_{\text{affine}} + 0.05 \cdot \mathcal{L}_{\text{res}}$$
  - `AffineRegularizationLoss`: Penalizes $\|\mathbf{A} - \mathbf{I}\|_F^2$, $\|\mathbf{b}\|_2^2$, and Total Variation smoothness.
  - `ResidualMagnitudeLoss`: Penalizes $\|\Delta_{\text{res}}\|_2^2$ to prevent representational drift.

### C. DINOv3 Dual-Encoder Semantic Segmenter
- **Module:** [`models/v18/encoder_dinov3.py`](file:///c:/Projects/SAT-EURS/models/v18/encoder_dinov3.py)
  - `DINOv3ViTBackbone`: Wraps Meta DINOv3 ViT-L/16 (`timm/vit_large_patch16_dinov3.lvd1689m`). Taps blocks 5, 11, 17, 23. Drops 5 prefix tokens ($1$ CLS + $4$ registers). Features: $4 \times [B, 1024, H/16, W/16]$. Freezes patch embed and blocks 0..17.
  - `DINOv3ConvNeXtBackbone`: Wraps official DINOv3 ConvNeXt-Base (`timm/convnext_base.dinov3_lvd1689m`, `features_only=True`). Outputs strides 4 (128ch), 8 (256ch), 16 (512ch), 32 (1024ch). Freezes early stages.
  - Includes safe mock backbones for offline CPU testing without downloaded weights.

- **Module:** [`models/v18/neck_upernet.py`](file:///c:/Projects/SAT-EURS/models/v18/neck_upernet.py)
  - `V18DetailStem`: Two-stage convolutional stem on raw degraded RGB at strides 2 (64ch) and 4 (128ch).
  - `LearnedSpatialGate`: Dynamically fuses semantic transformer features and raw convolutional features via sigmoid gate $g \in [0, 1]$.
  - `DecoderFiLMGate`: Bounded FiLM condition modulation ($(1 + 0.2\tanh(\gamma)) \cdot f + 0.2\tanh(\beta)$).
  - Multi-Head Decoders: Five-class main logits + 1-channel boundary logits + 1-channel binary objectness logits.

- **Module:** [`models/v18/segmenter.py`](file:///c:/Projects/SAT-EURS/models/v18/segmenter.py)
  - `EURSAPEXV18Segmenter`: Unifies restored-image ViT path, raw-image ConvNeXt path, detail stem, condition encoder, and UPerNet neck.

- **Module:** [`losses/v18/segmentation_losses.py`](file:///c:/Projects/SAT-EURS/losses/v18/segmentation_losses.py)
  - `SegmentationV18Loss`:
    $$\mathcal{L}_{\text{seg}} = 0.40 \cdot \mathcal{L}_{\text{WCE}} + 0.30 \cdot \mathcal{L}_{\text{Lovasz}} + 0.20 \cdot \mathcal{L}_{\text{Dice}} + 0.05 \cdot \mathcal{L}_{\text{Boundary}} + 0.05 \cdot \mathcal{L}_{\text{Objectness}}$$
  - Capped class weights: $\text{clip}(\text{median}(f) / f_c,\, 0.5,\, 4.0)$ to prevent numerical destabilization.
  - Full `ignore_index = 255` masking: when an external sample has 0 main-class labels, main losses cleanly return 0.0 without generating `nan`s.

### D. Dataset & External Adapters
- **Module:** [`data/v18/dataset.py`](file:///c:/Projects/SAT-EURS/data/v18/dataset.py)
  - `V18EURSInvarianceDataset`: Emits paired degraded and clean views with aligned masks.
  - Rare-class object-centered crop (60% probability for classes 3 and 4).
  - `V18SceneBalancedSampler`: Enforces balanced scene sampling per batch to prevent rare-class starvation.
- **Module:** [`data/v18/external_adapters.py`](file:///c:/Projects/SAT-EURS/data/v18/external_adapters.py)
  - Strict partial labeling adapters for Coralscapes, SUIM, UIIS10K, USIS10K.
  - Ambiguous categories map strictly to `255`, never background (`0`).
  - `export_external_taxonomy_manifest`: Exports versioned dataset specification JSON.
  - `generate_contact_sheet_overlay`: Renders color-coded QA verification overlays.

### E. Calibration, Evaluation & Packaging
- **Module:** [`tools/v18/calibration.py`](file:///c:/Projects/SAT-EURS/tools/v18/calibration.py)
  - `V18LogitCalibrator`: Nested cross-scene logit offset search on training subscenes. Enforces 100% coverage penalty preservation. Never uses validation or Dev ground truth.
- **Module:** [`tools/v18/evaluator.py`](file:///c:/Projects/SAT-EURS/tools/v18/evaluator.py)
  - `V18OOFEvaluator`: Strict 5-fold OOF evaluation. Accumulates confusion matrix globally across 4,000 images. Reports official $Q_{\text{FR}}$, $Q_{\text{seg}}$, and flags estimated composite as PROVISIONAL.
- **Module:** [`tools/v18/package_submission.py`](file:///c:/Projects/SAT-EURS/tools/v18/package_submission.py)
  - `V18SubmissionPackager`: Verifies 7-point pre-submission checklist, unpads to native dimensions, and writes atomic ZIP package.
- **Module:** [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py)
  - Unified CLI entrypoint with safe CPU dry-run default (`--dry_run`).

---

## 3. Configuration Inventory (`configs/v18/`)

1. [`configs/v18/restoration_config.json`](file:///c:/Projects/SAT-EURS/configs/v18/restoration_config.json):
   - Table ID: `checkpoints/exp4_1_scale_consistent/tbl19_20sc_scale_consistent.pt`
   - Channels: `[32, 64, 128, 192]`, Crop sizes: `[512, 768]`, Loss weights: `[0.50, 0.30, 0.10, 0.05, 0.05]`.
2. [`configs/v18/segmentation_config.json`](file:///c:/Projects/SAT-EURS/configs/v18/segmentation_config.json):
   - Encoders: `vit_large_patch16_dinov3.lvd1689m` + `convnext_base.dinov3_lvd1689m`.
   - Neck dim: 256, Loss weights: `[0.40, 0.30, 0.20, 0.05, 0.05]`, Rare crop prob: 0.60.
3. [`configs/v18/external_taxonomy_manifest.json`](file:///c:/Projects/SAT-EURS/configs/v18/external_taxonomy_manifest.json):
   - Formal licenses, source URLs, expected storage footprints, and category mappings for external datasets.
4. [`configs/v18/inference_config.json`](file:///c:/Projects/SAT-EURS/configs/v18/inference_config.json):
   - Output paths, unpadding geometry, TTA configuration, calibration offsets `[0, 0, 0, 0.15, 0.15]`.

---

## 4. Design Decisions & Reconciled Discrepancies

1. **Base Restoration Input (Pre- vs. Post-Unsharp):**
   - *Decision:* `ConditionedRestorationRefinerV18` consumes **post-unsharp** output $I_{\text{base}}$.
   - *Rationale:* The post-unsharp output achieved the highest measured sharpness ratio ($0.960$ on Dev) and PSNR ($25.55\text{ dB}$). Initializing the neural refiner on post-unsharp base ensures that the zero-residual start reproduces the SOTA restoration score on day 1.
2. **Intermediate Aspect Ratio Squashing:**
   - *Decision:* Completely eliminated.
   - *Rationale:* V18 replaces intermediate square resizing ($1024 \times 1024$ and $518 \times 518$) with aspect-preserving resize + internal reflection padding to nearest multiple of 32. Output serialization unpads cleanly back to target dimensions.
3. **Fixed vs. Learned Alpha Fusion:**
   - *Decision:* Replaced fixed heuristic $\alpha = 0.35$ with `LearnedSpatialGate`.
   - *Rationale:* The gate computes per-pixel attention between transformer semantics and convolutional textures, initialized to balanced $0.5$ fusion and regularized during training.
4. **All-Ignore Mask Guard:**
   - *Decision:* Added `has_valid_main` check in `SegmentationV18Loss`.
   - *Rationale:* Standard PyTorch `CrossEntropyLoss` returns `nan` when all pixels are 255 (as in external datasets providing only objectness masks). The guard returns $0.0$ cleanly, allowing auxiliary heads to train without destabilizing main-head gradients.
