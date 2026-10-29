# SAT-EURS Track 1  Validation Set & Scene-Level Leakage Audit

## 1. Scene-Level Data Hierarchy & Split Protocol

- The EURS training dataset consists of **20 unique clean scenes**.
- Each scene has 1 clean ground-truth image + mask, and **200 synthetically degraded variants** (10 Jerlov water types x 20 depth stations).
- **Critical Rule:** Validation must always be **scene-disjoint**. Evaluating a model on degraded variants of a scene seen during training constitutes severe data leakage.

## 2. 5-Fold Scene-Disjoint Split Definition

| Fold ID | Validation Scenes (4 Scenes) | Training Scenes (16 Scenes) | Absent Classes in Val ($GT=0$) | Present Classes in Val ($GT>0$) |
| :---: | :--- | :--- | :--- | :--- |
| **Fold 0** | `fish1, fish11, fish12, seahorse1` | 16 scenes | **Ray** | Background, Coral, Fish, MacroFauna |
| **Fold 1** | `fish13, fish14, fish15, sharkFish` | 16 scenes | **MacroFauna** | Background, Coral, Fish, Ray |
| **Fold 2** | `fish17, fish18, fish2, ray1` | 16 scenes | **MacroFauna** | Background, Coral, Fish, Ray |
| **Fold 3** | `fish20, fish21, fish4, starfish2` | 16 scenes | **Ray** | Background, Coral, Fish, MacroFauna |
| **Fold 4** | `fish5, fish7, fish9, turtles1` | 16 scenes | **Ray** | Background, Coral, Fish, MacroFauna |

## 3. Per-Scene Geometric & Class Prevalence Census

| Scene ID | Dimensions ($H \times W$) | Total Pixels | Background (%) | Coral (%) | Fish (%) | Ray (%) | MacroFauna (%) | Classes Present |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `fish1` | 3000x4000 | 12,000,000 | 69.67% | 25.61% | 4.72% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish11` | 3000x4000 | 12,000,000 | 0.41% | 93.47% | 6.12% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish12` | 3000x4000 | 12,000,000 | 51.43% | 42.37% | 6.21% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish13` | 3000x4000 | 12,000,000 | 64.64% | 31.54% | 3.82% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish14` | 3000x4000 | 12,000,000 | 50.52% | 39.99% | 9.49% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish15` | 3000x4000 | 12,000,000 | 92.89% | 0.00% | 7.11% | 0.00% | 0.00% | Background, Fish |
| `fish17` | 4000x3000 | 12,000,000 | 48.21% | 45.80% | 5.99% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish18` | 3000x4000 | 12,000,000 | 0.00% | 94.26% | 5.74% | 0.00% | 0.00% | Coral, Fish |
| `fish2` | 3000x4000 | 12,000,000 | 0.20% | 97.90% | 1.91% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish20` | 4567x6870 | 31,375,290 | 0.00% | 75.38% | 24.62% | 0.00% | 0.00% | Coral, Fish |
| `fish21` | 4480x6720 | 30,105,600 | 78.73% | 0.00% | 21.27% | 0.00% | 0.00% | Background, Fish |
| `fish4` | 4000x3000 | 12,000,000 | 35.43% | 59.01% | 5.56% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish5` | 4032x3024 | 12,192,768 | 8.61% | 87.84% | 3.55% | 0.00% | 0.00% | Background, Coral, Fish |
| `fish7` | 7283x9710 | 70,717,930 | 91.18% | 0.00% | 8.82% | 0.00% | 0.00% | Background, Fish |
| `fish9` | 3000x4000 | 12,000,000 | 28.30% | 68.72% | 2.98% | 0.00% | 0.00% | Background, Coral, Fish |
| `ray1` | 3136x6204 | 19,455,744 | 67.23% | 21.98% | 0.00% | 10.79% | 0.00% | Background, Coral, Ray |
| `seahorse1` | 3024x4032 | 12,192,768 | 42.99% | 26.63% | 0.00% | 0.00% | 30.39% | Background, Coral, MacroFauna |
| `sharkFish` | 6484x4323 | 28,030,332 | 66.01% | 28.65% | 0.74% | 4.60% | 0.00% | Background, Coral, Fish, Ray |
| `starfish2` | 4032x2268 | 9,144,576 | 98.66% | 0.00% | 0.00% | 0.00% | 1.34% | Background, MacroFauna |
| `turtles1` | 3235x4853 | 15,699,455 | 73.56% | 0.00% | 0.00% | 0.00% | 26.44% | Background, MacroFauna |

## 4. Key Diagnostic Insights on Rare Classes

1. **Class 3 (Ray / Elasmobranch):**
   - Present in ONLY 2 training scenes: `ray1` (Fold 2) and `sharkFish` (Fold 1).
   - Completely absent ($GT = 0$) in Fold 0, Fold 3, and Fold 4 validation splits.
2. **Class 4 (MacroFauna - Seahorse, Starfish, Turtle):**
   - Present in ONLY 3 training scenes: `seahorse1` (Fold 0), `starfish2` (Fold 3), and `turtles1` (Fold 4).
   - Completely absent ($GT = 0$) in Fold 1 and Fold 2 validation splits.
3. **The Absent-Class Vacuous Truth Artifact:**
   - In single-fold splits where Ray or MacroFauna is absent, a model that predicts 0 pixels gets $IoU = 1.0$ (vacuous truth).
   - Predicting even 1 false-positive pixel drops IoU from $1.0$ to $0.0$, penalizing single-fold mIoU by $-0.20$.
   - However, on the 1,200 hidden test images, all 5 classes are present. Therefore, pooled 5-fold evaluation across all 20 scenes (where all 5 classes have $GT > 0$) is the only uncorrupted local validation benchmark.
