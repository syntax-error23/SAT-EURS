# SAT-EURS Track 1 — V20 Phase 1: Experiment Setup & Loss Audit Report

**Auditor:** Empirical ML Specialist & Senior Computer Vision Engineer  
**Date:** October 18, 2026  
**Artifact Directory:** [`reports/v20/matched_pilot/`](file:///c:/Projects/SAT-EURS/reports/v20/matched_pilot/)  

---

## 1. Audit of Original V18 vs V20 Pilot Setup

A rigorous parameter-by-parameter audit was conducted between the production V18 training pipeline ([`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py)) and the V20 pilot script ([`tools/v20/run_v20_loss_pilot.py`](file:///c:/Projects/SAT-EURS/tools/v20/run_v20_loss_pilot.py)):

| Component | V18 Production Setting | V20 Pilot Setting | Audit Parity Status |
| :--- | :--- | :--- | :---: |
| **Architecture** | `EURSAPEXV18Segmenter` (ViT-L + ConvNeXt-B + UPerNet) | `EURSAPEXV18Segmenter` | **IDENTICAL** (401M params) |
| **Pretrained Weights** | `vit_large_patch16_dinov3.lvd1689m` + `convnext_base.dinov3_lvd1689m` | Same timm weights | **IDENTICAL** |
| **Refiner Stage** | `ConditionedRestorationRefinerV18` (Frozen in eval mode) | `ConditionedRestorationRefinerV18` | **IDENTICAL** |
| **Restoration Table** | `table_candidate_B_fold_{fold}.pt` | Same table | **IDENTICAL** |
| **Data Sampler** | `V18SceneBalancedSampler` (4 distinct scenes/batch) | `V18SceneBalancedSampler` | **IDENTICAL** |
| **Crop Size & Geometry** | $512 \times 512$ square crops | $512 \times 512$ square crops | **IDENTICAL** |
| **Rare-Class Centering** | $60\%$ probability centered on Class 3 or 4 | $60\%$ probability | **IDENTICAL** |
| **Spatial Augmentation** | Synchronous horizontal flip ($50\%$) | Synchronous horizontal flip ($50\%$) | **IDENTICAL** |
| **Optimizer** | AdamW (`weight_decay = 1e-4`) | AdamW (`weight_decay = 1e-4`) | **IDENTICAL** |
| **Parameter Groups** | ViT $0.10\times$, ConvNeXt $0.20\times$, Neck/Stem $1.0\times$ | Same differential groups | **IDENTICAL** |
| **Base Learning Rate** | $1.0 \times 10^{-4}$ | $1.0 \times 10^{-4}$ | **IDENTICAL** |
| **Batch Size & Accum** | `batch_size = 4`, `accum_steps = 4` (effective batch 16) | `batch_size = 4`, `accum_steps = 4` | **IDENTICAL** (200 updates/ep) |
| **Evaluator** | `V18OOFEvaluator(num_classes=5)` | `V18OOFEvaluator(num_classes=5)` | **IDENTICAL** |
| **Epoch Budget** | **9 Epochs** (`configs/v18/segmentation_config.json`) | **2 Epochs** (Pilot limitation) | **DIFFERENCE (Fixed in Phase 2)** |
| **Scheduler Horizon** | `CosineAnnealingLR(T_max=9, eta_min=1e-6)` | `CosineAnnealingLR(T_max=2, eta_min=1e-6)` | **DIFFERENCE (Fixed in Phase 2)** |
| **Initial Model State** | Historical random init (`seed=42+fold`) | Fresh random init (`seed=42+fold`) | **DIFFERENCE (Fixed in Phase 2)** |

---

## 2. Verification of Actual V18 Segmentation Loss

Inspection of [`losses/v18/segmentation_losses.py`](file:///c:/Projects/SAT-EURS/losses/v18/segmentation_losses.py) and [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py#L396-L408) confirms the original V18 recipe:

$$\mathcal{L}_{\text{V18}} = 0.40 \cdot \mathcal{L}_{\text{CE}} + 0.30 \cdot \mathcal{L}_{\text{Lovasz}} + 0.20 \cdot \mathcal{L}_{\text{Dice}} + 0.05 \cdot \mathcal{L}_{\text{Bnd}} + 0.05 \cdot \mathcal{L}_{\text{Obj}}$$

1. **Cross-Entropy Class Weights:**
   - In V18, class weights were computed by:
     ```python
     class_pixel_counts = [1000000, 50000, 30000, 5000, 2000] # Hardcoded mock counts
     class_weights = compute_capped_class_weights(class_pixel_counts, min_weight=0.50, max_weight=4.00)
     ```
   - This produced:
     $$\mathbf{w}_{\text{V18}} = [0.50, \; 0.60, \; 1.00, \; 4.00, \; 4.00]$$
   - **Flaw:** Both rare classes were clamped at the exact same cap ($4.00$), while Coral ($0.60$) received higher weight than Background ($0.50$) based on fictitious counts.
2. **Lovász-Softmax Loss:**
   - In V18, [`losses/v18/segmentation_losses.py`](file:///c:/Projects/SAT-EURS/losses/v18/segmentation_losses.py#L88):
     ```python
     return torch.stack(losses).mean()
     ```
   - This is an **unweighted arithmetic mean** across whatever classes are present in the batch.
   - If a batch contains Background, Coral, Fish, and Macro-Fauna, each class receives exactly $25\%$ of the Lovász gradient. The rare class receives zero relative boost in the IoU surrogate loss.
3. **Soft Dice Loss:**
   - In V18, [`losses/v18/segmentation_losses.py`](file:///c:/Projects/SAT-EURS/losses/v18/segmentation_losses.py#L142):
     ```python
     return 1.0 - torch.mean(dice)
     ```
   - Also an unweighted arithmetic mean across all 5 classes.

---

## 3. Evidence-Based Audit of V20 Pilot Epoch-2 Class-4 Collapse

The prompt correctly notes that V20 Epoch 1 achieved strong performance ($\text{IoU}_4 = 0.40291, \text{Recall}_4 = 0.57674$), but was followed at Epoch 2 by a regression ($\text{IoU}_4 = 0.13070, \text{Recall}_4 = 0.13589$).

An audit of the primary confusion matrices from [`reports/v20/pilot/PILOT_BENCHMARK_REPORT.json`](file:///c:/Projects/SAT-EURS/reports/v20/pilot/PILOT_BENCHMARK_REPORT.json) reveals:

### A. The Prediction Shift Matrix ($\Delta \text{CM} = \text{CM}_{\text{Ep2}} - \text{CM}_{\text{Ep1}}$):
```
                       Pred Cls 0    Pred Cls 1    Pred Cls 2    Pred Cls 3    Pred Cls 4
GT 0 (Background)   :  -2,951,463    +6,955,537      +271,661       -41,851    -4,233,884
GT 1 (Coral Reef)   :    +924,492      +626,852      +273,128        -1,470    -1,823,002
GT 2 (Fish)         :     +13,519      -147,458      +320,709             0      -186,770
GT 3 (Elasmobranch) :           0             0             0             0             0
GT 4 (Macro-Fauna)  :    +517,039      +807,481    +5,702,871          -336    -7,027,055
```

### B. Empirical Findings:
1. **Decision Boundary Shrinkage:**
   - In Epoch 1, the model predicted Class 4 for $16,069,792$ pixels, correctly capturing $9,193,043$ TP pixels with $6,876,749$ FP (Precision: $57.2\%$).
   - In Epoch 2, predicted Class 4 pixels collapsed by $82.6\%$ to only $2,799,081$ pixels.
   - Of the $7,027,055$ Class-4 pixels that ceased being predicted as Class 4, **$5,702,871$ pixels ($81.2\%$) were classified as Class 2 (Fish)**, $807,481$ as Coral, and $517,039$ as Background.
2. **False Positive Elimination vs False Negative Surge:**
   - The model eliminated almost all Class-4 false positives: FP on Background dropped from $4.86\text{M} \to 0.63\text{M}$ ($-87.0\%$) and on Coral dropped from $1.82\text{M} \to 1.88\text{k}$ ($-99.9\%$).
   - Consequently, **Class 4 Precision jumped from $57.2\% \to 77.4\%$**.
   - However, because the decision threshold tightened so aggressively, Class 4 Recall dropped from $57.7\% \to 13.6\%$.
3. **Root Cause — Premature Learning-Rate Decay under $T_{\text{max}} = 2$:**
   - Because `epochs = 2` was passed, the cosine scheduler used $T_{\text{max}} = 2$:
     $$\text{Epoch 1 LR} = 1.00 \times 10^{-4} \xrightarrow{\text{step}} 5.05 \times 10^{-5} \xrightarrow{\text{step}} 1.00 \times 10^{-6}$$
   - The learning rate was cut by $50\%$ after only 200 optimizer updates, and plummeted to near zero ($\eta_{\text{min}}$) during Epoch 2.
   - The model over-optimized on the easy background/coral loss (training loss dropped $40\%$ from $0.3179 \to 0.1906$), causing the classifier head to suppress the rare class in ambiguous underwater boundary regions.
   - In full V18 training ($T_{\text{max}} = 9$), LR at Epoch 2 remains high at $8.84 \times 10^{-5}$, allowing the network to explore representations before settling.

---

## 4. Preservation of Completed V20 Pilot Artifacts

All initial pilot deliverables remain preserved and unmodified in their dedicated directory:
- [`reports/v20/pilot/PILOT_BENCHMARK_REPORT.json`](file:///c:/Projects/SAT-EURS/reports/v20/pilot/PILOT_BENCHMARK_REPORT.json)
- [`reports/v20/pilot/PILOT_BENCHMARK_REPORT.md`](file:///c:/Projects/SAT-EURS/reports/v20/pilot/PILOT_BENCHMARK_REPORT.md)
- [`checkpoints/v20/pilot/fold_0/segmenter_best.pt`](file:///c:/Projects/SAT-EURS/checkpoints/v20/pilot/fold_0/segmenter_best.pt)

New matched-pilot runs will be strictly isolated in `experiments/v20/matched_pilot/`, `checkpoints/v20/matched_pilot/`, and `reports/v20/matched_pilot/`.
