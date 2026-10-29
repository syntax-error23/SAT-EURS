# SAT-EURS Track 2 — Sprint Phase 2 Report: 5-Fold Scene-Disjoint Validation

**Date:** 2026-10-24  
**Author:** Antigravity Senior Computer Vision Research & Optimization Team  
**Sprint Status:** PHASE 2 COMPLETE — SCENE-DISJOINT 5-FOLD BASELINE ESTABLISHED  
**Validation Inventory:** 20 scenes, 800 out-of-fold images (40 per scene, step 5)  
**Zero-Leakage Guarantee:** Every scene evaluated exclusively by a model trained without that scene or any of its degraded variants.

---

## 1. Overall Pooled 5-Fold Out-of-Sample Performance

| Metric | Measured Value | Leaderboard Scaling Factor | Weighted Score |
| :--- | :---: | :---: | :---: |
| **mIoU (5-Class)** | **0.3582** | — | — |
| **Macro-F1 (5-Class)** | **0.4527** | — | — |
| **Coverage Penalty ($P_{\text{cov}}$)** | **1.0** (All classes present have $\text{TP} > 0$) | — | — |
| **$Q_{\text{seg}}$** | **0.1621** | 0.60 | **0.0973** |
| **PSNR** | **18.90 dB** ($\text{PSNR}_{\text{norm}} = 0.1949$) | 0.45 of $Q_{\text{FR}}$ | — |
| **SSIM** | **0.6730** | 0.45 of $Q_{\text{FR}}$ | — |
| **Sharpness Ratio (vs Clean)** | **0.0840** | 0.10 of $Q_{\text{FR}}$ | — |
| **$Q_{\text{FR}}$** | **0.3989** | 0.30 | **0.1197** |
| **$Q_{\text{NR}}$ (Leaderboard Calibrated)** | **0.7430** | 0.10 | **0.0743** |
| **Official Composite ($S_{\text{composite}}$)** | **0.2912** | 1.00 | **0.2912** |

---

## 2. Per-Class Out-of-Fold Performance Across All 20 Scenes

| Class ID & Category | Class IoU | Class F1 | Frequency in Dataset | Primary Failure Mode |
| :---: | :---: | :---: | :---: | :--- |
| **0: Background** | **0.7083** | **0.8293** | 55.13% | High fidelity water-column separation |
| **1: Coral** | **0.6888** | **0.8157** | 37.54% | Robust substrate segmentation |
| **2: Fish** | **0.3290** | **0.4951** | 2.17% | Good contour capture, small fish missed |
| **3: Elasmobranch** | **0.0556** | **0.1054** | 2.83% | **Severe out-of-fold generalization drop** |
| **4: Other** | **0.0091** | **0.0181** | 2.33% | **Near-zero detection on unseen species** |

---

## 3. Critical Rare-Class Scene Deep Dive

| Scene Name | Held-Out Fold | Present Classes | Scene mIoU | Scene F1 | Class 3 IoU | Class 4 IoU | Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `sharkFish` | Fold 1 | 0, 1, 2, 3 | 0.2502 | 0.3493 | **0.2550** | — | When trained on `ray1`, model detects shark body reasonably |
| `ray1` | Fold 2 | 0, 1, 3 | 0.1868 | 0.2365 | **0.0001** | — | When trained on `sharkFish`, flat seabed ray is completely missed |
| `seahorse1` | Fold 0 | 0, 1, 4 | 0.0942 | 0.1427 | — | **0.0281** | Seahorse body largely confused with coral branches |
| `starfish2` | Fold 3 | 0, 4 | 0.1600 | 0.1788 | — | **0.0035** | Small starfish on seabed predicted as background |
| `turtles1` | Fold 4 | 0, 4 | 0.1453 | 0.1685 | — | **0.0008** | Pelagic sea turtle completely missed |

---

## 4. Key Takeaways for Phase 3

1. **Why Track 2 Composite is Depressed:**
   - Background (Class 0) and Coral (Class 1) achieve strong performance (IoUs ~0.70).
   - Fish (Class 2) achieves modest performance (IoU ~0.33).
   - But **Class 3 and Class 4 collapse out-of-fold** (IoUs of 0.0556 and 0.0091), dragging down arithmetic 5-class $\text{mIoU}$ (0.3582) and $\text{Macro-F1}$ (0.4527), driving multiplicative $Q_{\text{seg}} = 0.3582 \times 0.4527 = \mathbf{0.1621}$.
2. **Phase 3 Focus:**
   - Isolated rare-class interventions (scene-balanced sampling, class weighting, present-class Dice) must directly lift Class 3 and Class 4 recall to break through the 0.20+ $Q_{\text{seg}}$ ceiling.
