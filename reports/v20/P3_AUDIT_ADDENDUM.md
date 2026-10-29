# SAT-EURS Track 1 — V20 Audit Addendum: Resolution of P3 Reporting Inconsistencies

**Author:** Empirical ML Specialist & Computer Vision Research Engineer  
**Date:** October 18, 2026  
**Target Reference:** [`reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.md`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.md), [`CLASS4_ROUTING_DIAGNOSTIC.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.json)  

---

## 1. Resolution 1: The Exact Mathematical Formula for $Q_{\text{seg}}$

### A. The Discrepancy
In the narrative synthesis report [`V19_P3_CLASS4_ROUTING_AND_POSTHOC_EXHAUSTION_REPORT.md`](file:///C:/Users/HP/.gemini/antigravity/brain/696a0e0c-40de-42f0-bb83-99167e290f61/V19_P3_CLASS4_ROUTING_AND_POSTHOC_EXHAUSTION_REPORT.md), Section 2 stated:
$$Q_{\text{seg}} = \sqrt{\text{mIoU} \times \text{Macro } F_1} \times \text{Coverage Penalty} \quad \text{[REPORT TYPO]}$$
However, the reported numerical values throughout the project (e.g., $0.47274$, $0.62167$, $0.76044$) do not match a geometric mean:
$$\sqrt{0.62167 \times 0.76044} = 0.68756 \ne 0.47274$$
Instead, they match the arithmetic product:
$$0.62167 \times 0.76044 \times 1.0 = \mathbf{0.47274}$$

### B. Implementation Audit
Inspection of the actual codebase confirms that **every evaluator implementation** computes the product:
1. [`metrics/segmentation_metrics.py`](file:///c:/Projects/SAT-EURS/metrics/segmentation_metrics.py#L80-L81):
   ```python
   official_q_seg = macro_f1 * miou
   q_seg = official_q_seg * hard_coverage_penalty
   ```
2. [`tools/v19/run_class4_routing_diagnostic.py`](file:///c:/Projects/SAT-EURS/tools/v19/run_class4_routing_diagnostic.py#L108-L109):
   ```python
   q_seg_pres = miou_pres * macro_f1_pres * cov_penalty
   q_seg_5cls = miou_5cls * macro_f1_5cls * cov_penalty
   ```
3. [`reports/v19/exp_p3_class4_gate/reconciliation_data.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p3_class4_gate/reconciliation_data.json#L20):
   ```json
   "Q_seg": "mIoU * Macro_F1 * coverage_penalty"
   ```

### C. Correction
The square root in the markdown report text was a typographical error in the KaTeX formatting. The actual mathematical formula used across all evaluation scripts, audits, and ledgers is strictly:
$$\mathbf{Q_{\text{seg}} = \text{mIoU} \times \text{Macro } F_1 \times \text{Coverage Penalty}}$$
No scoring code is changed.

---

## 2. Resolution 2: P3 Fold-Level Delta Recomputation & 3/5 Wins Explanation

### A. The Discrepancy
In [`reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.md`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.md), Section 6 displayed a table:
```
Fold ID | Val Scenes Included                   | Selected Setting  | Inner Q_seg | Outer Q_seg | C0_cal Q_seg | Delta Q_seg
Fold 0  | fish1, fish11, fish12, seahorse1       | tau_0.45_area_0   | 0.48975     | 0.43165     | 0.43165      | +0.00000
Fold 1  | fish13, fish14, fish15, sharkFish      | tau_0.30_area_0   | 0.44793     | 0.66050     | 0.65820      | +0.00230
Fold 2  | fish17, fish18, fish2, ray1            | tau_0.30_area_0   | 0.42538     | 0.57650     | 0.57620      | +0.00030
Fold 3  | fish20, fish21, fish4, starfish2       | tau_0.30_area_0   | 0.48588     | 0.32406     | 0.32400      | +0.00006
Fold 4  | fish5, fish7, fish9, turtles1          | tau_0.30_area_0   | 0.50269     | 0.49264     | 0.49250      | +0.00014
```
This displayed table appeared to show four positive deltas (Folds 1, 2, 3, 4) and one zero delta (Fold 0), which would suggest 4 wins. However, the diagnostic script and JSON report strictly recorded:
$$\text{Gate 5: Positive Improvement on } \ge 4/5 \text{ Folds: } \mathbf{False} \text{ (Wins: } \mathbf{3/5}\text{)}$$

### B. Root Cause Analysis
Audit of the true control values stored in [`reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p2_ensemble_benchmark/ENSEMBLE_BENCHMARK.json) and evaluated inside [`tools/v19/run_class4_routing_diagnostic.py`](file:///c:/Projects/SAT-EURS/tools/v19/run_class4_routing_diagnostic.py) reveals that the markdown table contained manually transcribed, erroneous numbers for `C0_cal Q_seg`.

The true unrounded metrics from the primary confusion matrices are:

| Fold ID | Validation Scenes | Selected Setting | LOFO Outer $Q_{\text{seg}}$ | Verified `C0_cal` $Q_{\text{seg}}$ | True Unrounded Delta $\Delta Q_{\text{seg}}$ | Script Evaluation (`q_lofo > q_c0`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | `fish1`, `fish11`, `fish12`, `seahorse1` | `tau_0.45_area_0` | **0.43165** | **0.42951** | **+0.00214** | **WIN (1/5)** |
| **Fold 1** | `fish13`, `fish14`, `fish15`, `sharkFish` | `tau_0.30_area_0` | **0.66050** | **0.66050** | **+0.00000** | **NO WIN (Tie)** |
| **Fold 2** | `fish17`, `fish18`, `fish2`, `ray1` | `tau_0.30_area_0` | **0.57650** | **0.57646** | **+0.00004** | **WIN (2/5)** |
| **Fold 3** | `fish20`, `fish21`, `fish4`, `starfish2` | `tau_0.30_area_0` | **0.32406** | **0.32500** | **-0.00094** | **NO WIN (Regression)** |
| **Fold 4** | `fish5`, `fish7`, `fish9`, `turtles1` | `tau_0.30_area_0` | **0.49264** | **0.49263** | **+0.00001** | **WIN (3/5)** |

### C. Resolution
- **Actual Wins:** Exactly **3 out of 5 folds** (Folds 0, 2, 4).
- Fold 1 was an exact tie ($+0.00000$), not a win.
- Fold 3 was a slight regression ($-0.00094$), not a win.
- The script evaluation logic (`if q_lofo_k > q_c0_k: fold_wins += 1`) evaluated strictly to $3$, correctly failing Gate 5.

---

## 3. Resolution 3: Verification of Actual Class-3 & Class-4 Metrics

Direct extraction from [`reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.json`](file:///c:/Projects/SAT-EURS/reports/v19/exp_p3_class4_gate/CLASS4_ROUTING_DIAGNOSTIC.json) confirms:

### Class 3 (Elasmobranchii) — 100% Invariant:
- Ground Truth: $32,287,600$ pixels
- Predicted Pixels: $38,627,715 \to 38,627,715$ ($\Delta = 0$)
- True Positives ($\text{TP}_3$): $26,133,051 \to 26,133,051$ ($\Delta = 0$)
- False Positives ($\text{FP}_3$): $12,494,664 \to 12,494,664$ ($\Delta = 0$)
- False Negatives ($\text{FN}_3$): $6,154,549 \to 6,154,549$ ($\Delta = 0$)
- Precision: $0.67654 \to 0.67654$ ($\Delta = 0.00000$)
- Recall: $0.80938 \to 0.80938$ ($\Delta = 0.00000$)
- F1 Score: $0.73702 \to 0.73702$ ($\Delta = 0.00000$)
- IoU: $0.58356 \to 0.58356$ ($\Delta = 0.00000$)
- Overrides: Exactly 0 pixels reassigned from Class 3.

### Class 4 (Macro-Fauna) — Precision-Positive but Volume-Constrained:
- Ground Truth: $121,969,800$ pixels
- Predicted Pixels: $101,282,896 \to 101,921,241$ ($+638,345$)
- True Positives ($\text{TP}_4$): $67,516,233 \to 67,893,927$ ($+377,694$)
- False Positives ($\text{FP}_4$): $33,766,663 \to 34,027,314$ ($+260,651$)
- False Negatives ($\text{FN}_4$): $54,453,567 \to 54,075,873$ ($-377,694$)
- Precision: $0.66661 \to 0.66614$ ($-0.00047$)
- Recall: $0.55355 \to 0.55665$ ($+0.00310$)
- F1 Score: $0.60484 \to 0.60649$ ($+0.00165$)
- IoU: $0.43353 \to 0.43523$ ($+0.00170$)

All 638,345 reassigned pixels came from non-Class-3 origins: Class 0 (383,914), Class 1 (68,663), and Class 2 (185,768).
