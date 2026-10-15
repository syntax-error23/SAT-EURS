# Track 2 Controlled Experiment & Progressive Ablation Plan

**Target:** MaCVi @ WACV 2027 Challenge — Track 2 (Jetson AGX Orin)  
**Execution Environment:** NVIDIA RTX 5090 (32 GB VRAM), 64 GB RAM, Intel Core Ultra 9  
**Cross-Validation Protocol:** Strict 5-Fold Scene-Disjoint Cross-Validation (`splits/manifest_5fold.json`)  
**Evaluation Script:** `tools/evaluate_track2_onnx.py`  

---

## 1. Experimental Hypotheses & Rationale

| ID | Hypothesis | Success Criterion | Rationale |
| :--- | :--- | :--- | :--- |
| **H1** | Cascaded restoration (feeding physically restored RGB to the segmentation student) outperforms raw degraded RGB input. | $\Delta Q_{\text{seg}} \ge +0.050$ | Restoring optical spectrum recovers contrast and color channels that terrestrial backbones need. |
| **H2** | Distilling from the full 5-fold DINOv2-Large ensemble outperforms ground-truth CE alone. | $\Delta \text{mIoU} \ge +0.030$ | Transfers dark-knowledge and soft uncertainty boundaries from the 1.54B parameter Track 1 ensemble. |
| **H3** | Class-frequency weighted KL divergence prevents recall collapse on rare marine classes (Classes 3 & 4). | Class 3 & 4 $\text{IoU} \ge 0.450$ | Standard uniform distillation washes out rare class signals that represent $< 1\%$ of pixels. |
| **H4** | Dedicated morphological boundary loss ($\mathcal{L}_{\text{Boundary}}$) resolves the 15px boundary deficit. | Boundary $\text{mIoU} \ge 0.450$ (up from 0.376) | Direct contour penalization prevents coarse feature smearing at object transitions. |
| **H5** | Present-class Lovász-Softmax loss directly optimizes mIoU and composite $Q_{\text{seg}}$. | $\Delta Q_{\text{seg}} \ge +0.020$ | Lovász extension directly optimizes the non-smooth Jaccard metric instead of surrogate pixel log-loss. |
| **H6** | Logit-level decision boundary calibration ($\Delta_3 = -1.5, \Delta_4 = +0.8$) optimizes official $Q_{\text{seg}}$ on held-out scenes. | $\text{Coverage Penalty} = 1.0$, $Q_{\text{seg}} \ge 0.600$ | Balances false-positive noise against false-negative dropouts on rare elasmobranchs. |

---

## 2. 9-Step Progressive Ablation Matrix

We formulate a 9-step progressive ablation sequence where each step isolates a single architectural or algorithmic decision:

```
[Step 1: Baseline Reproduction]
       |
       v
[Step 2: Restoration Engine Isolation (Oracle vs Router vs Unsharp)]
       |
       v
[Step 3: Supervised-Only Segmentation Student (No Distillation)]
       |
       v
[Step 4: Input Domain Comparison (Raw Degraded vs Restored RGB)]
       |
       v
[Step 5: Multi-Teacher Soft Logit Distillation (DINOv2 Ensemble)]
       |
       v
[Step 6: Boundary Refinement Loss (15px Morphological Band)]
       |
       v
[Step 7: Direct Metric Optimization (Lovász-Softmax Loss)]
       |
       v
[Step 8: Class Bias Calibration Transfer (Delta_3, Delta_4)]
       |
       v
[Step 9: Complete End-to-End ONNX Export & Hardware Benchmark]
```

### Progressive Ablation Progression (Fixed Architecture & Leak-Free Splits):

| Run ID | Name / Purpose | Seg Input | Loss Formulation | Teacher Supervision | Primary Metric Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-SUPERVISED** | Supervised-Only Baseline | Restored RGB | Standard Cross-Entropy ($\mathcal{L}_{\text{CE}}$) | None (Ground Truth Only) | Reference Baseline ($Q_{\text{seg\_sup}}$) |
| **EXP-STANDARD-KD**| Stage 1: Standard KD + CE | Restored RGB | $\alpha \mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{KD\_std}}$ ($T=2.5$) | 5-Fold DINOv2 Ensemble | Does KD improve $Q_{\text{seg}}$ over EXP-SUPERVISED? ($\Delta \ge +0.02$) |
| **EXP-CLASS-AWARE**| Stage 2: Class-Aware KD | Restored RGB | $\alpha \mathcal{L}_{\text{CE}} + \beta \sum w_c \mathcal{L}_{\text{KD\_c}}$ | 5-Fold DINOv2 Ensemble | Does Class 3 & 4 IoU improve without false-positive collapse? |
| **EXP-BOUNDARY**   | Stage 3: Boundary-Aware Loss | Restored RGB | $\mathcal{L}_{\text{Stage2}} + \gamma \mathcal{L}_{\text{Boundary}}$ | 5-Fold DINOv2 Ensemble | Does 15px boundary-band mIoU improve while overall stays strong? |
| **EXP-LOVASZ**     | Stage 4: Lovász-Softmax | Restored RGB | $\mathcal{L}_{\text{Stage3}} + \delta \mathcal{L}_{\text{Lovasz}}$ | 5-Fold DINOv2 Ensemble | Does direct region-overlap optimization improve official $Q_{\text{seg}}$? |
| **EXP-COMBINED**   | Stage 5: Combined Best Config | Restored RGB | Retain ONLY verified winning components | 5-Fold DINOv2 Ensemble | Measure complete end-to-end official composite score |

---

## 3. Scene-Disjoint Validation Protocol & Metric Guardrails

With only 20 unique scenes in the dataset, cross-validation discipline is paramount to prevent overfitting:
1. **Scene-Disjoint Folds:**
   - Evaluated strictly using `splits/manifest_5fold.json`.
   - No clean reference image, degraded variant, mask, or feature statistic may cross between training and validation.
2. **Official Composite Scoring:**
   $$S = 0.60 \, Q_{\text{seg}} + 0.135 \, \text{PSNR}_{\text{norm}} + 0.135 \, \text{SSIM} + 0.030 \, \text{Sharpness} + 0.10 \, Q_{\text{NR}}$$
   where $\text{PSNR}_{\text{norm}} = \text{clip}\left(\frac{\text{PSNR} - 15.0}{20.0}, 0.0, 1.0\right)$ and $Q_{\text{seg}} = \text{mIoU} \times \text{Macro F1} \times \text{Coverage Penalty}$.
3. **Hard Promotion Criteria:**
   - **Parameter Count:** $\le 1,000,000$ (must pass `deploy/check_submission_flops.py`).
   - **Compute Cap:** $\le 12.0\text{ GFLOPs}$ (must pass `deploy/check_submission_flops.py`).
   - **Numerical Parity:** $\text{MAE} < 10^{-4}$ between PyTorch and ONNX Runtime.
   - **Coverage Floor:** Hard Coverage Penalty MUST equal $1.0000$ (all present ground-truth classes must have $\text{TP} > 0$).
   - **Composite Floor:** Must exceed prior Track 2 baseline ($S > 0.5196$).

---

## 4. Compute Resource Allocation & GPU Budget

Training on the NVIDIA RTX 5090 (32 GB VRAM):
- **Pre-Caching Teacher Soft Targets:**
  - 4,000 images evaluated with 5-fold DINOv2-Large with horizontal-flip TTA.
  - Runtime: ~8-12 minutes total.
  - Storage: ~5.2 GB on NVMe SSD (quantized uint8 probabilities at $512 \times 512$).
- **Student Distillation Training (`EURS-EdgeSOTA-V2`):**
  - Batch size: 16 on RTX 5090 (bfloat16 AMP).
  - Throughput: ~280 samples/second.
  - Steps per epoch (over 3,200 training images): 200 steps.
  - Time per epoch: ~12 seconds.
  - 15-Epoch Training Run: **~3 minutes total**!
- **ONNX Export and Official Evaluation:**
  - Runtime: ~50 seconds per full evaluation.
