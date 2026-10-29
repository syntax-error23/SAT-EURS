# SAT-EURS Track 1 — Phase 1 Forensic Diagnostic: Validation Integrity & Leakage Audit

**Campaign:** SAT-EURS Track 1 Phase 1 Forensic Diagnostic  
**Date:** October 24, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** MaCVi @ WACV 2027 EURS Track 1 Testing Phase  

---

## 1. Executive Summary

This audit evaluates the integrity of the 5-fold cross-validation protocol, scene partitioning, cross-contamination, and the discrepancy between in-sample training performance and out-of-fold generalization.

Key audit findings:
1. **Zero Scene Leakage:** Verified 100% scene-disjoint partitioning across all 5 folds. The intersection between training scenes and validation scenes is strictly empty ($\emptyset$).
2. **Severe Class Starvation in Individual Folds:**
   - Class 3 (Ray / Elasmobranch) exists in **only two clean scenes** (`ray1`, `sharkFish`) across the entire dataset.
   - Class 4 (Macro-Fauna) exists in **only three clean scenes** (`seahorse1`, `starfish2`, `turtles1`).
   - Consequently, **every single 5-fold validation split is missing at least one class**: Folds 0, 3, and 4 completely lack Class 3 (Ray) in validation; Folds 1 and 2 completely lack Class 4 (Macro-Fauna).
3. **The Generalization Cliff:**
   - In-Sample (Full-20 trained on clean, evaluated on same clean scenes): $Q_{\text{seg}} = 0.94 - 0.98$.
   - Out-of-Fold (Fold models evaluated on unseen validation scenes): $Q_{\text{seg}} \approx 0.458$ (mIoU 0.614, Macro $F_1$ 0.747).
   - Hidden Test Phase Leaderboard (Official submission #19096): $Q_{\text{seg}} \approx 0.438$ (Composite $0.633$).

---

## 2. Dataset-Wide Class Occurrence Census (All 20 Training Scenes)

Programmatic inspection of all 20 clean masks in `data/masks/`:

| Scene ID | Present Classes | Total Pixels | Background (0) | Coral (1) | Fish (2) | Ray (3) | Macro-Fauna (4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fish1` | {0, 1, 2} | 12.00M | 8.36M | 3.07M | 566K | 0 | 0 |
| `fish11` | {0, 1, 2} | 12.00M | 49K | 11.22M | 734K | 0 | 0 |
| `fish12` | {0, 1, 2} | 12.00M | 6.17M | 5.08M | 745K | 0 | 0 |
| `fish13` | {0, 1, 2} | 12.00M | 7.76M | 3.78M | 459K | 0 | 0 |
| `fish14` | {0, 1, 2} | 12.00M | 6.06M | 4.80M | 1.14M | 0 | 0 |
| `fish15` | {0, 2} | 12.00M | 11.15M | 0 | 853K | 0 | 0 |
| `fish17` | {0, 1, 2} | 12.00M | 5.79M | 5.50M | 719K | 0 | 0 |
| `fish18` | {1, 2} | 12.00M | 0 | 11.31M | 689K | 0 | 0 |
| `fish2` | {0, 1, 2} | 12.00M | 24K | 11.75M | 229K | 0 | 0 |
| `fish20` | {1, 2} | 31.38M | 0 | 23.65M | 7.73M | 0 | 0 |
| `fish21` | {0, 2} | 30.11M | 23.70M | 0 | 6.40M | 0 | 0 |
| `fish4` | {0, 1, 2} | 12.00M | 4.25M | 7.08M | 667K | 0 | 0 |
| `fish5` | {0, 1, 2} | 12.00M | 1.05M | 10.71M | 433K | 0 | 0 |
| `fish7` | {0, 2} | 70.72M | 64.48M | 0 | 6.24M | 0 | 0 |
| `fish9` | {0, 1, 2} | 12.00M | 3.40M | 8.25M | 358K | 0 | 0 |
| `ray1` | **{0, 1, 3}** | 19.46M | 13.08M | 4.28M | **0** | **2.10M** | 0 |
| `seahorse1`| **{0, 1, 4}** | 12.19M | 5.24M | 3.25M | **0** | 0 | **3.70M** |
| `sharkFish`| **{0, 1, 2, 3}**| 28.02M | 18.50M | 8.03M | 207K | **1.29M** | 0 |
| `starfish2`| **{0, 4}** | 9.14M | 9.02M | 0 | **0** | 0 | **122K** |
| `turtles1` | **{0, 4}** | 15.70M | 11.55M | 0 | **0** | 0 | **4.15M** |

### Structural Vulnerability Analysis:
1. **Fish (Class 2) is Absent in 4 Scenes:** `ray1`, `seahorse1`, `starfish2`, and `turtles1` have 0 fish pixels.
2. **Ray (Class 3) Exists in Only 2 Scenes:** `ray1` (2.10M px) and `sharkFish` (1.29M px).
3. **Macro-Fauna (Class 4) Exists in Only 3 Scenes:** `seahorse1` (3.70M px), `turtles1` (4.15M px), and `starfish2` (122K px).

---

## 3. Official 5-Fold Partitioning & Class Coverage Audit

Partition mapping from [`data/splits.json`](file:///c:/Projects/SAT-EURS/data/splits.json) and [`tools/v18/run_v18_pipeline.py`](file:///c:/Projects/SAT-EURS/tools/v18/run_v18_pipeline.py):

| Fold | Training Scenes (16) | Validation Scenes (4) | Train Classes | Validation Classes | Missing in Val | Leakage Check |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 0** | 16 scenes (inc. `ray1`, `sharkFish`, `starfish2`, `turtles1`) | `fish1`, `fish11`, `fish12`, `seahorse1` | {0, 1, 2, 3, 4} | {0, 1, 2, 4} | **Class 3 (Ray)** | **$\emptyset$ (Clean)** |
| **Fold 1** | 16 scenes (inc. `ray1`, `seahorse1`, `starfish2`, `turtles1`) | `fish13`, `fish14`, `fish15`, `sharkFish`| {0, 1, 2, 3, 4} | {0, 1, 2, 3} | **Class 4 (MacroFauna)**| **$\emptyset$ (Clean)** |
| **Fold 2** | 16 scenes (inc. `sharkFish`, `seahorse1`, `starfish2`, `turtles1`) | `fish17`, `fish18`, `fish2`, `ray1` | {0, 1, 2, 3, 4} | {0, 1, 2, 3} | **Class 4 (MacroFauna)**| **$\emptyset$ (Clean)** |
| **Fold 3** | 16 scenes (inc. `ray1`, `sharkFish`, `seahorse1`, `turtles1`) | `fish20`, `fish21`, `fish4`, `starfish2` | {0, 1, 2, 3, 4} | {0, 1, 2, 4} | **Class 3 (Ray)** | **$\emptyset$ (Clean)** |
| **Fold 4** | 16 scenes (inc. `ray1`, `sharkFish`, `seahorse1`, `starfish2`) | `fish5`, `fish7`, `fish9`, `turtles1` | {0, 1, 2, 3, 4} | {0, 1, 2, 4} | **Class 3 (Ray)** | **$\emptyset$ (Clean)** |

---

## 4. Methodological Implications for Phase 2

1. **Why Single-Fold Validation Flaws Arise:**
   Because every fold is starved of either Class 3 or Class 4, validating on a single fold (e.g. Fold 0 alone) gives an incomplete picture of model performance. In Fold 0, Class 3 has 0 ground-truth pixels, so ray false-positive tendencies can never be measured.
2. **The Full-20 Necessity:**
   Because only 2 scenes contain rays and only 3 contain macro-fauna, training on a 16-scene fold starves the model of 33%–50% of the entire challenge taxonomy for those rare classes!
   This explains why training on **all 20 scenes** is structurally necessary to achieve high recall on the hidden test set.
3. **The Clean vs Degraded Domain Shift:**
   The Full-20 model was trained exclusively on clean EURS scenes (using synthetic crops), whereas the test set contains severe Jerlov water types ($WT \ge 8, Depth \ge 60\text{m}$).
   The model must be exposed to degraded underwater conditions during training to prevent confidence collapse on low-contrast test scenes.
