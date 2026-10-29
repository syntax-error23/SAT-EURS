# SAT-EURS Track 2 — Restoration Diagnostics & Class 3 Bias Resolution

**Document Version:** 1.0.0  
**Audit Timestamp:** 2026-10-24T02:35:30+05:30  
**Author:** Senior CV Researcher & Model Optimization Engineer  
**Scope:** Root-cause analysis of EURS-EdgeSOTA-V2 defects, Class 3 bias offset, and physical restoration routing.

---

## 1. Class 3 Bias Diagnosis & The Absent-Class Evaluator Trap

### 1.1 Root-Cause Identification
In the existing Track 2 codebase, a severe anomaly was identified where predictions on Class 3 (Sharks / Rays) were suppressed by a large negative logit bias.

We inspected the actual model checkpoint and exported ONNX graph:
- In checkpoint `checkpoints/EXP-PRETRAINED-30EP_best.pt`, the learned classification head bias was `cls_head.bias[3] = -0.1782`.
- In the exported ONNX model `deploy/eurs_edgesota_v2_final.onnx` and evaluation scripts (`tools/train_track2_student.py`), an artificial **`-3.0` logit suppression offset** had been hardcoded, setting the static bias to `-3.1782`.

### 1.2 The Local Evaluator Bug ($0/0 \implies 1.0$)
To determine why this artificial offset was introduced, we created an isolated diagnostic script (`scratch/diagnose_class3_bias.py`) and evaluated both variants:

```python
# metrics/segmentation_metrics.py (defect identified):
if target_count == 0 and pred_count == 0:
    iou_class = 1.0  # FATAL BUG: Rewards total prediction failure with perfect score!
```

In the 5-fold cross-validation split (`splits/manifest_5fold.json`):
- **Fold 0 validation scenes** (`fish1`, `fish11`, `fish12`, `seahorse1`) contain Ground Truth for Classes 0, 1, 2, and 4, but **zero ground truth pixels for Class 3**.
- Applying `-3.0` suppression suppressed all Class 3 predictions on Fold 0 to 0 pixels.
- The defective evaluator calculated $\frac{0}{0} = 1.0$, awarding a **fake 1.0000 IoU** to the absent class, falsely boosting local Fold 0 mIoU from **0.3436 up to 0.5436 (+0.20 artificial inflation)**!

### 1.3 Catastrophic Real-World Generalization Failure
When evaluated on scenes that actually contain Class 3 ground truth (`ray1` and `sharkFish`):
- Setting the offset to `-3.0` caused Class 3 recall to plummet from **97.5% down to 64.2%**, erasing over **764,000 true positive pixels**.
- On any official competition evaluation containing sharks or rays, this artificial bias causes severe metric collapse and risks violating the non-zero recall requirement ($P_{\text{cov}} = 1.0$).

### 1.4 Resolution for SAT-EURS-LiteV3
1. **Zero Artificial Offsets:** No hardcoded suppression offsets are applied in `SAT-EURS-LiteV3`. Class boundary decisions are learned naturally via class-weighted loss and teacher distillation.
2. **Evaluation Protocol Correction:** Single-fold evaluations that suffer from absent classes are strictly prohibited from serving as final verification benchmarks. The official validation standard pools out-of-sample predictions across $S_{\text{folds}} = \{0, 1\}$ where all 5 classes have substantial positive ground truth support.

---

## 2. Restoration Matrix Routing & Polynomial Basis Resolution

### 2.1 The 11-Term / $K=10$ Proposal Audit
Previous preliminary proposals suggested replacing the Track 1 optical basis with an "11-term polynomial basis over $K=10$ degradation clusters".

An exhaustive repository audit established:
- **No 11-term basis exists** anywhere in the repository.
- **No $K=10$ condition bank exists** or was ever fitted.
- Truncating the 19-term basis to 11 terms would destroy the underlying physical optical model (which maps direct transmission, forward scatter, backscatter, and spatial illumination decay).

All verified tables on disk (`tbl19_20sc_clean.pt`, `tbl19_16sc_fold0.pt`, `tbl19_16sc_fold1.pt`) are **19-term RSUIGM bases over 200 degradation conditions** (10 water types $\times$ 20 depth levels). The condition bank tensor shape is `[200, 57]`.

### 2.2 Empirical Restoration Candidate Comparison
We executed an empirical restoration benchmark across held-out underwater scenes (`scratch/audit_restoration_candidates.py`):

| Restoration Mechanism | PSNR (dB) | SSIM | Sharpness Ratio | Inference Complexity | Feasibility |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Oracle Ground Truth Condition** | **24.71** | **0.8173** | 0.6171 | N/A (Requires GT metadata) | Theoretical Ceiling |
| **Discrete / Sharp Routing ($T=0.10$)** | **18.43** | **0.7109** | **0.6455** | **0.20 GFLOPs, 26k params** | **PASSED & VERIFIED** |
| **Softmax Mixture Routing ($T=1.0$)** | 12.61 | 0.5420 | 0.4812 | 0.20 GFLOPs, 26k params | FAILED (Convex collapse) |
| **Image-Space Convex Blending** | 14.12 | 0.5931 | 0.5104 | 0.60 GFLOPs, High VRAM | FAILED (Severe blur) |

### 2.3 Physical Root Cause of Convex Blending Failure
In underwater optics, attenuation coefficients $\beta(\lambda)$ and backscatter $B_\infty$ interact non-linearly across spectral channels and depth. Convex linear blending of incompatible condition matrices or restored images produces grey-fogging and washed-out contrast, dropping PSNR by nearly 6 dB.

In contrast, **sharp temperature routing ($T=0.10$)** acts as a differentiable approximation to discrete argmax selection, activating the single best physical condition while maintaining full differentiability during ONNX export.

### 2.4 Final Architectural Selection for LiteV3
- **Basis:** 19-term physical RSUIGM basis with multi-scale spatial pooling ($16\times 16$ and $64\times 64$).
- **Condition Bank:** 200 degradation conditions loaded as an immutable static buffer `[200, 57]`.
- **Router:** Lightweight 3-layer CNN + FC head (26,056 parameters, 0.20 GFLOPs at $384\times 768$).
- **Sharpening:** Onnx-safe depthwise Gaussian unsharp filtering with factor $\gamma = 1.20$.
- **Status:** Validated, compliant, and integrated into `SATEURSLiteV3`.
