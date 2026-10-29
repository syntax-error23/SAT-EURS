# SAT-EURS Track 1 — V26 Experiment 4: Extreme Degradation Analysis & Safeguard Audit

## Executive Summary

As diagnosed in the V25 post-mortem, extreme combinations of oceanic water types (Jerlov type 9 and 10) and high water depths ($\ge 80$m) undergo strong optical extinction, yielding near-black or strictly zero-valued RGB inputs.

This empirical audit comprehensively analyzed all **4,000 training images** (across 20 scenes) and all **1,200 testing images** (across 6 test scenes) to evaluate the prevalence of extinction and test potential low-information safeguards.

---

## 1. Extinction Census Across Train and Test Splits

| Dataset Partition | Total Images | Pure-Zero Images ($[0,0,0]$) | Near-Zero Images ($\max < 5$) |
| :--- | :---: | :---: | :---: |
| **Training Set (20 scenes)** | 4,000 | **61 (1.52%)** | 254 (6.35%) |
| **Testing Set (6 scenes)** | 1,200 | **14 (1.17%)** | 72 (6.00%) |

### Extinction Conditions Identified:
- **Training Set:** `wt10` at depths $80, 85, 90, 95, 100$m, and `wt9` at depth $100$m.
- **Testing Set:** `wt10` at depths $85, 90, 95, 100$m.

In both sets, pure-zero images represent **less than 1.6%** of total evaluation volume. Over **98.4%** of images contain active optical signal.

---

## 2. Ground-Truth Distribution of Pure-Zero Images

A naive heuristic proposed in some prior post-mortems was an unconditional "predict background (Class 0)" gate on zero-valued images. To test the scientific validity of this safeguard, we evaluated the exact ground-truth class distribution of all 61 zero-valued training images:

| Semantic Class | Ground Truth Pixels | % of Zero-Valued Pixel Area |
| :--- | :---: | :---: |
| **Class 0: Background** | 31,348,357 | **49.01%** |
| **Class 1: Coral** | 26,299,022 | **41.12%** |
| **Class 2: Fish** | 3,871,083 | **6.05%** |
| **Class 3: Shark/Ray** | 629,178 | **0.98%** |
| **Class 4: Macro-fauna** | 1,815,496 | **2.84%** |
| **Total Foreground (1–4)** | **32,614,779** | **50.99%** |

### Critical Finding:
**50.99% of ground-truth pixels in pure-zero images are FOREGROUND.**
Corals, fish, sharks, and macro-fauna are actively present in these scenes despite total optical extinction in the simulated RGB sensor.

---

## 3. Decision on Low-Information Safeguards

1. **Rejection of Unconditional Background Gating:**
   An unconditional background gate ($\text{if } \max = 0 \implies \text{pred} = 0$) would cause **false negative erasure on 51% of ground truth foreground pixels**. Furthermore, forcing Class 0 on these images would trigger severe recall drops for rare classes (Class 3 and Class 4) and risk coverage penalty degradation.
2. **Adoption of Condition-Aware Neural Priors:**
   The `EURSAPEXV18Segmenter` incorporates a dedicated `ConditionEncoder(wt, depth)` that embeds water type and depth. Under zero optical input, the model relies on the learned geometric and class priors conditioned on the depth/water-type embedding, predicting spatial layout based on scene topology rather than collapsing all pixels to background.
3. **Restoration Protocol Compliance:**
   For pure-zero inputs, physical refiner scaling can multiply zero without inventing false signal. The refiner correctly preserves zero-bounded stability without exploding numerical gradients.
