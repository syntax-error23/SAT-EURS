# EURS Challenge: Team Experiment Tracker & Leaderboard
**Challenge:** Embedded Underwater Restoration and Segmentation (EURS) @ MaCVi, WACV 2027  
**Team Role (Tooling & Logging):** Arin  
**Model & Training Lead:** Tanmay  
**Data & Deployment Lead:** Shubham  

---

## 1. Challenge Evaluation Constraints & Composite Metric

| Metric / Constraint | Official Limit / Specification | EURS-Net V2 Target | Verification Tool |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | $\le 1,000,000$ ($1.00\text{ M}$) | $\approx 0.49 - 0.52\text{ M}$ | `check_submission_flops.py` |
| **Compute Complexity** | $\le 12.00\text{ GFLOPs}$ ($1 \times 3 \times 1024 \times 1024$) | $\approx 7.44 - 8.64\text{ GFLOPs}$ | `check_submission_flops.py` |
| **Target Deployment** | NVIDIA Jetson AGX Orin (30W mode, 16GB) | TensorRT 8.6/10.0 Static Engine | `deploy/export_onnx.py` |
| **Composite Metric** | $\mathbf{0.60 \cdot Q_{\text{seg}} + 0.30 \cdot Q_{\text{rest}} + 0.10 \cdot Q_{\text{NR}}}$ | Maximize ($> 0.75$) | Track 1 Public Dev Split |

* **$Q_{\text{seg}}$ (60%)**: 5-class Semantic Segmentation ($\text{mIoU}$, class-level F1, with coverage penalty for missing rare classes like fish & elasmobranchs).
* **$Q_{\text{rest}}$ (30%)**: Paired Restoration Quality ($\text{PSNR} \uparrow$, $\text{SSIM} \uparrow$, Laplacian Sharpness Ratio).
* **$Q_{\text{NR}}$ (10%)**: Non-Reference Underwater Visual Quality Metric.

---

## 2. Master Experiment Run Tracker

> **Logging Protocol:**
> 1. Run local evaluation via `python evaluate.py` to record `Local mIoU` and `Local PSNR`.
> 2. Verify static budget via `python deploy/check_flops.py` before submission.
> 3. Generate zipped archive via `python tools/make_submission.py --pred-dir <dir>` and upload to Track 1.
> 4. Record the official `Track 1 Composite Score` once the server finishes evaluation.

| Experiment ID | Date | Model Variant | Parameters (M) | GFLOPs | Local mIoU (%) | Local PSNR (dB) | Track 1 Composite Score | Git Commit / Checkpoint | Notes / Ablation Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BUDGET_CAP** | — | *Official Track 2 Limit* | $\le 1.000\text{ M}$ | $\le 12.00\text{ G}$ | — | — | — | — | *Hard constraint enforced statically* |
| **EXP-000** | 2026-10-05 | *Baseline V1 (MobileNetV2 + FPN)* | $0.462\text{ M}$ | $6.92\text{ G}$ | $52.40\%$ | $24.12\text{ dB}$ | *Pending* | `init-baseline` | Initial baseline; standard CE + L1 loss |
| **EXP-001** | *Upcoming* | *EURS-Net V2 (Stem + MBConv)* | $0.485\text{ M}$ | $7.35\text{ G}$ | — | — | — | `arch-v2-stem` | Add Fast White-Balance Residual Pre-Net |
| **EXP-002** | *Upcoming* | *EURS-Net V2 (+ CoordAttn)* | $0.505\text{ M}$ | $7.98\text{ G}$ | — | — | — | `arch-v2-ca` | Coordinate Attention in Stages 2 & 3 |
| **EXP-003** | *Upcoming* | *EURS-Net V2 (+ Focal/Dice Loss)* | $0.505\text{ M}$ | $7.98\text{ G}$ | — | — | — | `loss-weighted` | Class weights from `tools/eda_stats.py` |
| **EXP-004** | *Upcoming* | *EURS-Net V2 (+ Distillation)* | $0.518\text{ M}$ | $8.45\text{ G}$ | — | — | — | `distill-swin` | Teacher-student guidance via Swin-L |
| **EXP-FINAL** | *Milestone 5* | *EURS-Net V2 Standalone ONNX* | $\le 0.520\text{ M}$ | $\le 8.64\text{ G}$ | — | — | — | `release-track2` | Static TensorRT FP16 for Jetson AGX Orin |

---

## 3. Class Index Mapping Reference

As determined by challenge guidelines and verified by `tools/eda_stats.py`:

| Class ID | Target Entity | Palette Preview | Loss Imbalance Severity |
| :---: | :--- | :---: | :--- |
| **0** | Background | `[15, 23, 42]` | High dominance (~70–85% of total pixels) |
| **1** | Coral | `[249, 115, 22]` | Moderate frequency; fine spatial boundaries |
| **2** | Fish | `[234, 179, 8]` | Rare; small-scale objects subject to coverage penalty |
| **3** | Elasmobranch (Sharks & Rays) | `[6, 182, 212]` | Highly sparse; critical class for composite segment score |
| **4** | Other (Marine Life / Debris) | `[168, 85, 247]` | Sparse contextual objects |

---

## 4. Submission Checklist (Pre-Flight Verification)

Before submitting any candidate model to Track 1 or Track 2:
- [ ] Model parameter count verified $\le 1.0\text{ M}$ via `deploy/check_flops.py`.
- [ ] GFLOPs verified $\le 12.0\text{ G}$ on input shape $[1, 3, 1024, 1024]$.
- [ ] Zero dynamic operators in ONNX graph (`deploy/export_onnx.py`).
- [ ] `tools/visualizer.py` tri-panel inspection generated for qualitative verification (`experiments/plots/`).
- [ ] `tools/make_submission.py` executed without errors, producing `submission.zip`.
- [ ] `submission.zip` contains valid `restored/` (RGB PNG) and `seg_masks/` (single-channel uint8 PNG, values $0..4$).
- [ ] Log entry created in Section 2 with Experiment ID, local scores, and commit hash.
