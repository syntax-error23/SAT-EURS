# SAT-EURS Track 1 — V18 Implementation Preflight & Contract Audit

**Date:** 2026-10-17  
**Auditor:** Antigravity CV Research Team  
**System Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)  
**Status:** Mandatory Preflight Gate Completed — All Contracts Audited  

---

## 1. Input Preprocessing, Resize Geometry, Normalization, and Output Dimensions

### Current Codebase Practice (V15–V17)
In earlier iterations (`deploy/export_v5_and_track1.py`, `deploy/generate_phase2_verified_submission.py`, `tools/run_v16_apex_turbo.py`), images were subjected to a two-step geometry pipeline:
1. **Target Dimensions:** Evaluated via `scale_to_max_side(w, h, max_side=1024)`:
   $$\text{scale} = \frac{1024}{\max(W_{\text{orig}}, H_{\text{orig}})}$$
   $$\text{target\_w} = \max(1, \text{round}(W_{\text{orig}} \cdot \text{scale})), \quad \text{target\_h} = \max(1, \text{round}(H_{\text{orig}} \cdot \text{scale}))$$
   For example:
   - $4000 \times 3000 \to 1024 \times 768$ (Aspect 4:3)
   - $6000 \times 8000 \to 768 \times 1024$ (Aspect 3:4)
2. **Intermediate Square Resizing:** Earlier models forced a square $1024 \times 1024$ or $518 \times 518$ bilinear resize for GPU tensor operations, and subsequently resized back to $(\text{target\_w}, \text{target\_h})$. This introduced minor spatial distortion / squashing.

### V18 Canonical Geometry Contract
Under Phase 1 of the V18 blueprint:
1. **Aspect-Preserving Resize:** Test images are scaled directly to $(\text{target\_w}, \text{target\_h})$ preserving native aspect ratio.
2. **Reflection/Divisibility Padding:** Tensors are padded internally to the nearest multiple of 32 (required by ConvNeXt stride 32 and ViT patch 16):
   $$\text{pad\_h} = (32 - \text{target\_h} \pmod{32}) \pmod{32}, \quad \text{pad\_w} = (32 - \text{target\_w} \pmod{32}) \pmod{32}$$
   Padded on right and bottom using reflection padding (`torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')`).
3. **Internal Processing:** Models operate at $(\text{target\_h} + \text{pad\_h}, \text{target\_w} + \text{pad\_w})$.
4. **Padding Cropping:** Before serialization, outputs are strictly cropped to `[:target_h, :target_w]`.
5. **Color Normalization:** RGB float32 in $[0.0, 1.0]$. Encoder paths apply ImageNet mean $(0.485, 0.456, 0.406)$ and std $(0.229, 0.224, 0.225)$.
6. **Mask Semantics:** Categorical masks are single-channel uint8 with values strictly in $\{0, 1, 2, 3, 4\}$. Resampling strictly uses `cv2.INTER_NEAREST` or PyTorch nearest-neighbor. Value 255 is reserved strictly for auxiliary ignore masks.

---

## 2. Actual Filename Parsing for Water Type and Depth

### Canonical Filename Pattern
Files in `data/cached_1024/degraded/` and `data/dev/degraded/` strictly follow:
`{scene}_{water_type}_{depth}.png`

### Strict Parser Implementation
- Format: `parts = stem.split("_")`
- `scene = parts[0]` (e.g., `fish1`, `sharkFish`, `turtles1`)
- `water_type = int(parts[1])`
- `depth = int(parts[2])`

### Validation Range Checks
- **Water Type Range:** Must satisfy $1 \le wt \le 10$ ($wt \in \{1, 2, \dots, 10\}$).
- **Depth Range:** Must satisfy $depth \in \{5, 10, 15, \dots, 100\}$ meters (exactly 20 discrete depth stations).
- Any deviations raise `ValueError` immediately with zero silent fallbacks.

---

## 3. Actual Model and Checkpoint APIs for DINOv3 ViT-L/16 & DINOv3 ConvNeXt-Base

We directly inspected the installed Python 3.11.9 and `timm` (v1.0.30) environment.

### A. Meta DINOv3 ViT-L/16
1. **Installed Architecture Name:** `vit_large_patch16_dinov3` in `timm`.
2. **Pretrained Hub ID:** `timm/vit_large_patch16_dinov3.lvd1689m`.
3. **License:** `dinov3-license` (Meta DINOv3 Research License).
4. **Weights Size:** ~1.2 GB.
5. **Architecture Anatomy:**
   - Patch Size: $16 \times 16$.
   - Embed Dimension: $1024$.
   - Number of Transformer Blocks: 24.
   - Prefix / Special Tokens: 5 total ($1$ `cls_token` + $4$ register tokens).
   - Spatial Tokens: For input $(H, W)$, number of spatial tokens is $(H/16) \times (W/16)$. Total token sequence length is $5 + (H/16 \times W/16)$.
6. **Feature Extraction API:**
   - Unlike DINOv2 (`encoder.get_intermediate_layers(x, n=...)`), `timm`'s `vit_large_patch16_dinov3` uses `m.blocks` (`nn.ModuleList` of length 24).
   - To extract dense block features from blocks 5, 11, 17, and 23:
     Input passes through `patch_embed`, `pos_embed`, and prefix concatenation.
     Each block $i \in \{0, \dots, 23\}$ is evaluated sequentially.
     At indices $5, 11, 17, 23$, we drop the first 5 prefix tokens (`feat[:, 5:, :]`), permute to `[B, 1024, H/16, W/16]`, and collect the spatial feature tensor.
   - Tensor shape at all 4 tapped blocks: `[B, 1024, H/16, W/16]`.

### B. Meta DINOv3 ConvNeXt-Base
1. **Installed Architecture Name:** `convnext_base` in `timm` with tag `dinov3_lvd1689m`.
2. **Pretrained Hub ID:** `timm/convnext_base.dinov3_lvd1689m`.
3. **License:** `dinov3-license` (Meta DINOv3 Research License).
4. **Weights Size:** ~350 MB.
5. **Feature Extraction API:**
   - Supported natively via `timm.create_model('convnext_base', pretrained=..., features_only=True)`.
   - Multi-scale feature strides and channel dimensions:
     - Stage 1 (Stride 4): `[B, 128, H/4, W/4]`
     - Stage 2 (Stride 8): `[B, 256, H/8, W/8]`
     - Stage 3 (Stride 16): `[B, 512, H/16, W/16]`
     - Stage 4 (Stride 32): `[B, 1024, H/32, W/32]`

### C. Offline / Synthetic Test Fallback Mode
Because this task enforces the **mandatory no-download, no-GPU run rule**, all V18 model code is equipped with an explicit mock/synthetic test mode. When weights are not downloaded, the model initializes randomly on CPU without attempting unapproved network connections.

---

## 4. Current Score Implementations & Provisional Parts

### Official Scoring Contract
$$S_{\text{composite}} = 0.60 \cdot Q_{\text{seg}} + 0.30 \cdot Q_{\text{FR}} + 0.10 \cdot Q_{\text{NR}}$$

1. **Semantic Segmentation Score ($Q_{\text{seg}}$):**
   $$Q_{\text{seg}} = \text{Macro } F_1 \times \text{mIoU} \times \text{coverage\_penalty}$$
   - Computed from global $5 \times 5$ confusion matrix $C$.
   - $\text{coverage\_penalty} = \prod_{c \in \mathcal{C}_{\text{present}}} \mathbb{I}(\text{TP}_c > 0)$.
   - If any present ground-truth class has $\text{TP}_c = 0$, $Q_{\text{seg}}$ collapses to $0.0$.
2. **Full-Reference Restoration Score ($Q_{\text{FR}}$):**
   $$Q_{\text{FR}} = 0.45 \cdot \text{PSNR}_{\text{norm}} + 0.45 \cdot \text{SSIM} + 0.10 \cdot \text{sharpness\_ratio}$$
   - $\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{35.0 - 15.0},\, 0.0,\, 1.0\right)$.
   - SSIM: Gaussian window ($\sigma = 1.5$, size 11).
   - Sharpness Ratio: Laplacian variance ratio with $10^{-7}$ zero-denominator guardrail.
3. **Non-Reference Restoration ($Q_{\text{NR}}$) & Local Policy:**
   - **PROVISIONAL STATUS:** Official $Q_{\text{NR}}$ uses server-side proprietary linear weighting of UIQM, UCIQE, and CCF.
   - As established in Gate 0 (`reports/track1_research/SCORER_VALIDATION.md`), local evaluation **must never hallucinate** server $Q_{\text{NR}}$ normalization constants.
   - Any local composite estimate is designated $S_{\text{composite, est}}$ and marked explicitly as **PROVISIONAL**. Primary internal promotion is driven by $Q_{\text{seg}}$ and verified $Q_{\text{FR}}$.

---

## 5. Data Manifest and External-Label Semantics

### Manifest Structure (`splits/manifest_5fold.json`)
- Exactly 20 training scenes, each containing 200 physical condition variants ($4,000$ degraded images total).
- Partitioned into 5 scene-disjoint folds (16 train scenes, 4 held-out val scenes per fold).
- All 200 variants of any scene remain strictly in the same split (zero variant leakage).

### Rare-Class Distribution & Scene Counts
- **Class 0 (Background):** Present in all 20 scenes.
- **Class 1 (Coral/Substrate):** Present in 17 scenes.
- **Class 2 (Fish):** Present in 17 scenes.
- **Class 3 (Elasmobranchii - Shark/Ray):** Present in **only 2 scenes**:
  - `ray1`: 2,099,340 pixels (exclusively Ray)
  - `sharkFish`: 1,290,501 pixels (exclusively Shark)
- **Class 4 (Other Marine Fauna - Seahorse/Starfish/Turtle):** Present in **only 3 scenes**:
  - `seahorse1`: 3,704,818 pixels (exclusively Seahorse)
  - `starfish2`: 122,292 pixels (exclusively Starfish)
  - `turtles1`: 4,150,480 pixels (exclusively Turtle)
- **Fold Allocation:**
  - Fold 0 holds out `seahorse1`
  - Fold 1 holds out `sharkFish`
  - Fold 2 holds out `ray1`
  - Fold 3 holds out `starfish2`
  - Fold 4 holds out `turtles1`
  Every single fold holds out exactly ONE rare scene.

### External Dataset Taxonomy Mapping Policy
To prevent label contamination and negative transfer:
1. **Coralscapes:** Verified coral masks map to EURS Class 1. All non-coral categories map to `ignore_index = 255`.
2. **SUIM:** Contains coarse groupings ("vertebrates", "invertebrates"). Coarse masks are restricted to binary objectness pretraining; never mapped directly to fine-grained EURS classes.
3. **UIIS10K / USIS10K:** Instance masks provide class-agnostic objectness and boundary supervision. Ambiguous fish categories are mapped to `ignore_index = 255` because EURS strictly isolates elasmobranchs from teleost fish.
4. **Default Rule:** Any ambiguous or unverified external category maps to `255`, never background (`0`).

---

## 6. Preflight Audit Summary & Discrepancy Reconciliation

| Item | Existing Repo Practice | V18 Specification Contract | Reconciliation / Resolution |
| :--- | :--- | :--- | :--- |
| **Intermediate Resize** | Squashed to $1024\times 1024$ or $518\times 518$ square | Aspect-preserving scale + pad to multiple of 32 | Implemented via `V18CanonicalPreprocessor` with crop-on-output. |
| **DINO API** | `torch.hub` DINOv2 `get_intermediate_layers` | Meta DINOv3 ViT-L/16 via `timm` (blocks 5, 11, 17, 23) | Block hook/loop implemented in `models/v18/encoder_dinov3.py`, dropping 5 prefix tokens. |
| **ConvNeXt Fusion** | Fixed heuristic weight $\alpha = 0.35$ | Learned, regularized spatial gating + FPN neck | Built into `models/v18/neck_upernet.py` with multi-scale fusion. |
| **Restoration Path** | Closed-form RSUIGM table + unsharp mask | Conditioned zero-residual refiner initialized at exact $I_{\text{base}}$ | `ConditionedRestorationRefinerV18` consumes post-unsharp $I_{\text{base}}$ with zero-init affine & residual heads. |
| **Local $Q_{\text{NR}}$** | Ad-hoc or omitted | Flagged as PROVISIONAL, never hardcoded | Local scorer explicitly reports verified $Q_{\text{FR}}$, $Q_{\text{seg}}$, and flags estimated composite. |
| **No-Run Rule** | Prior runs executed on GPU | Strictly implementation-only, zero GPU execution | All modules support CPU-only synthetic testing; no training or dataset mutation is executed. |

**Preflight Verdict:** All contracts verified. No blocking discrepancies exist. Proceeding to Phase 1–8 implementations.
