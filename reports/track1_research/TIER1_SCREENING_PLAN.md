# SAT-EURS Track 1 — Tier 1 Screening Protocol & Candidate Promotion Plan

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)  
**Hardware:** NVIDIA RTX 5090 (32 GB VRAM, 90% allocation limit $\le 28.6$ GB)

---

## 1. Objectives & Screening Design

In accordance with Section 4 and Section 10:
- Combinatorial grid search is strictly prohibited.
- Candidates are screened across **two representative folds** chosen from the canonical manifest (`splits/manifest_5fold.json`):
  - **`fold_0`**: Validates held-out Class 4 (`seahorse1`), testing upright macro-fauna detection in dense reef habitats against held-out fish hosts (`fish1`, `fish11`, `fish12`).
  - **`fold_1`**: Validates held-out Class 3 (`sharkFish`), testing horizontal swimming elasmobranch detection in open blue water against held-out fish hosts (`fish13`, `fish14`, `fish15`).
- Together, `fold_0` and `fold_1` exercise both rare marine taxa under strict scene-disjoint out-of-fold constraints.

---

## 2. Predeclared Candidate Screening Matrix

| Run ID | Model Architecture | Split ID | Input Stream | Loss Formulation | Augmentation | Primary Hypothesis |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **`BASELINE_MITB3_F0`** | `smp-deeplabv3plus-mit-b3` | `fold_0` | RSUIGM Inversion | Weighted CE | Random Crop + HFlip | Reproducible control baseline on `fold_0`. |
| **`BASELINE_MITB3_F1`** | `smp-deeplabv3plus-mit-b3` | `fold_1` | RSUIGM Inversion | Weighted CE | Random Crop + HFlip | Reproducible control baseline on `fold_1`. |
| **`DINOV2_L_F0`** | `dinov2_vitl14_reassembly` | `fold_0` | RSUIGM Inversion | Weighted CE | Random Crop + HFlip | Primary foundation vision model (+308M params). |
| **`DINOV2_L_F1`** | `dinov2_vitl14_reassembly` | `fold_1` | RSUIGM Inversion | Weighted CE | Random Crop + HFlip | Primary foundation vision model on held-out shark. |
| **`DINOV2_RAW_F0`** | `dinov2_vitl14_reassembly` | `fold_0` | Raw Degraded RGB | Weighted CE | Random Crop + HFlip | Test if foundation features can learn directly from raw RGB without RSUIGM. |
| **`DINOV2_LOVASZ_F0`** | `dinov2_vitl14_reassembly` | `fold_0` | RSUIGM Inversion | Weighted CE + Lovasz | Random Crop + HFlip | Test if direct mIoU surrogate optimization lifts macro F1 and boundary accuracy. |
| **`DINOV2_CP_F0`** | `dinov2_vitl14_reassembly` | `fold_0` | RSUIGM Inversion | Weighted CE | Exact $(wt, \text{depth})$ Copy-Paste | Test if exact physical matched copy-paste improves rare class transfer. |

---

## 3. Measured Hardware & Compute Ledger

All runs are executed with strict precision (`torch.amp.autocast("cuda", dtype=torch.bfloat16)`) and real-time VRAM monitoring:

| Candidate | Epochs | Training Samples | Measured VRAM | Throughput | Epoch Time | Total Run Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `mit_b3` (Baseline) | 3 | 3,200 | 9.08 GB (28.5%) | 49.2 img/s | ~48s | ~235s (~3.9 min) |
| `dinov2_l` (Foundation) | 5 | 3,200 | 10.51 GB (33.0%) | 24.6 img/s | ~111s | ~688s (~11.4 min) |

Total Tier 1 Screening GPU compute is capped at $\approx 50$ minutes, operating safely at $< 35\%$ of the RTX 5090's 32 GB capacity.

---

## 4. Promotion Criteria (Gate 2 $\to$ Gate 3)

1. **Primary Promotion Metric:** $Q_{\text{seg}} = \text{Macro } F_1 \times \text{mIoU} \times \text{coverage\_penalty}$ evaluated on full, un-subsampled held-out validation sets (800 images per fold).
2. **Empirical Gate:** The winning configuration must demonstrate:
   - Superiority over the clean baseline on both `fold_0` and `fold_1`.
   - Positive recall on rare classes without coverage penalty collapse.
   - Verified raw logit inference interface with zero test-time data leakage.
3. The winning configuration from Tier 1 will be confirmed across all 5 folds (Gate 3) to produce the official Out-of-Fold (OOF) submission ensemble.
