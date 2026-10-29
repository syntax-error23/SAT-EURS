# SAT-EURS Track 1 — V20 Matched Loss Ablation Benchmark Report

**Date:** 2026-10-18 14:39:46 UTC  
**Benchmark Type:** Strictly Matched Paired Pilot (Identical Initial State, Sampler, 9-Epoch Schedule)  
**Baseline SHA256:** `e82cbea22ed068a3bfd01d7111ec412391df0fdc66e4ad2d954589d9efc81a52` (100% Intact)  

---

## Fold 0 Paired Results (`val: ['fish1', 'fish11', 'fish12', 'seahorse1']`)

- **Shared Init State Hash:** `e3c397a02c1cbdbdbc3e085e3723b334c59ac1f40fee13e8dc8ea4dd0563cc88`
- **Control Arm (V18 Recipe):** Best Epoch 5, Val $Q_{\text{seg}}$ = 0.34837
- **Treatment Arm (V20 Recipe):** Best Epoch 4, Val $Q_{\text{seg}}$ = 0.40563
- **Net Improvement ($\Delta Q_{\text{seg}}$):** **+0.05726**

### Detailed Comparative Metrics Table:

| Metric Dimension | Control Arm (V18 Loss) | Treatment Arm (V20 Loss) | Absolute Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Val $Q_{\text{seg}}$ (Present)** | 0.34837 | 0.40563 | **+0.05726** | GAIN |
| **Val mIoU (Present)** | 0.52672 | 0.56956 | **+0.04284** | GAIN |
| **Val Macro F1 (Present)** | 0.66141 | 0.71218 | **+0.05077** | GAIN |
| **Coverage Penalty** | 1.0 | 1.0 | Exact | MATCH |
| Class 0 (Background) IoU | 0.7070 | 0.6871 | -0.0200 | REGR |
| Class 1 (Coral Reef) IoU | 0.7671 | 0.7695 | +0.0024 | PASS |
| Class 2 (Fish) IoU | 0.4000 | 0.4257 | +0.0258 | PASS |
| Class 3 (Elasmobranch) IoU | 0.0000 | 0.0000 | **+0.0000** | N/A |
| Class 4 (Macro-Fauna) IoU | 0.2328 | 0.3960 | **+0.1631** | GAIN |

### Training Loss & Validation Traces Across 9 Epochs:

| Epoch | Control Loss | Control $Q_{\text{seg}}$ | Treatment Loss | Treatment $Q_{\text{seg}}$ | LR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3175 | 0.2978 | 0.3114 | 0.2383 | 0.000100 |
| 2 | 0.1999 | 0.1820 | 0.1968 | 0.2005 | 0.000097 |
| 3 | 0.1568 | 0.0000 | 0.1595 | 0.3376 | 0.000088 |
| 4 | 0.1378 | 0.2440 | 0.1393 | 0.4056 | 0.000075 |
| 5 | 0.1298 | 0.3484 | 0.1297 | 0.3252 | 0.000059 |
| 6 | 0.1202 | 0.3105 | 0.1209 | 0.2752 | 0.000042 |
| 7 | 0.1137 | 0.3220 | 0.1155 | 0.2414 | 0.000026 |
| 8 | 0.1092 | 0.3196 | 0.1099 | 0.2640 | 0.000013 |
| 9 | 0.1053 | 0.2973 | 0.1061 | 0.2708 | 0.000004 |

---

## Fold 2 Paired Results (`val: ['fish17', 'fish18', 'fish2', 'ray1']`)

- **Shared Init State Hash:** `7787e16b6cdb2f7400383af555c6caf6367afcc7d7f08ca9e9c87fdf80215afc`
- **Control Arm (V18 Recipe):** Best Epoch 9, Val $Q_{\text{seg}}$ = 0.60655
- **Treatment Arm (V20 Recipe):** Best Epoch 4, Val $Q_{\text{seg}}$ = 0.41950
- **Net Improvement ($\Delta Q_{\text{seg}}$):** **-0.18705**

### Detailed Comparative Metrics Table:

| Metric Dimension | Control Arm (V18 Loss) | Treatment Arm (V20 Loss) | Absolute Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Val $Q_{\text{seg}}$ (Present)** | 0.60655 | 0.41950 | **-0.18705** | REGR |
| **Val mIoU (Present)** | 0.73060 | 0.59332 | **-0.13728** | REGR |
| **Val Macro F1 (Present)** | 0.83021 | 0.70703 | **-0.12318** | REGR |
| **Coverage Penalty** | 1.0 | 1.0 | Exact | MATCH |
| Class 0 (Background) IoU | 0.8922 | 0.8289 | -0.0632 | REGR |
| Class 1 (Coral Reef) IoU | 0.9476 | 0.8989 | -0.0487 | REGR |
| Class 2 (Fish) IoU | 0.5494 | 0.3470 | -0.2024 | REGR |
| Class 3 (Elasmobranch) IoU | 0.5332 | 0.2984 | **-0.2348** | REGR |
| Class 4 (Macro-Fauna) IoU | 0.0000 | 0.0000 | **+0.0000** | N/A |

### Training Loss & Validation Traces Across 9 Epochs:

| Epoch | Control Loss | Control $Q_{\text{seg}}$ | Treatment Loss | Treatment $Q_{\text{seg}}$ | LR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.3278 | 0.3984 | 0.3349 | 0.3599 | 0.000100 |
| 2 | 0.1991 | 0.3174 | 0.2015 | 0.3343 | 0.000097 |
| 3 | 0.1660 | 0.3811 | 0.1679 | 0.3375 | 0.000088 |
| 4 | 0.1408 | 0.4925 | 0.1439 | 0.4195 | 0.000075 |
| 5 | 0.1298 | 0.3686 | 0.1297 | 0.3407 | 0.000059 |
| 6 | 0.1201 | 0.5258 | 0.1227 | 0.3903 | 0.000042 |
| 7 | 0.1151 | 0.3940 | 0.1164 | 0.3834 | 0.000026 |
| 8 | 0.1120 | 0.5791 | 0.1127 | 0.3699 | 0.000013 |
| 9 | 0.1088 | 0.6066 | 0.1109 | 0.4034 | 0.000004 |

---

