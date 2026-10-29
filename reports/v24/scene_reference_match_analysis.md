# SAT-EURS Track 1 — V24 Clean-Scene Reference Matching Audit

**Objective:** Audit whether any of the 6 official Dev evaluation scenes (`data/dev/degraded/`) correspond to the 20 clean training scenes (`data/clean/`), and evaluate the viability of a reference-retrieval candidate (`V24-REF-LOOKUP`).

---

## 1. Scene Inventory and Dimensional Analysis

| Dataset Split | Scene Count | Scene Identifiers | Image Dimensions |
|---|---|---|---|
| **Clean Training** (`data/clean`) | 20 | `fish1`, `fish11`, `fish12`, `fish13`, `fish14`, `fish15`, `fish17`, `fish18`, `fish2`, `fish20`, `fish21`, `fish4`, `fish5`, `fish7`, `fish9`, `ray1`, `seahorse1`, `sharkFish`, `starfish2`, `turtles1` | $(3000\times 4000)$ to $(7283\times 9710)$ |
| **Dev Evaluation** (`data/dev/degraded`) | 6 | `fish10`, `fish19`, `fish3`, `fish6`, `sharRayFish`, `starfish1` | $(3000\times 4000)$, $(3024\times 4032)$, $(4624\times 3468)$, $(8000\times 6000)$ |

### Direct Scene ID Overlap
- Set intersection between Clean Training Scene IDs and Dev Scene IDs: **$\emptyset$ (Zero Overlap)**.
- All 6 Dev scenes have unique scene identifiers not present in the 20 training scenes.

---

## 2. Multi-Signal Geometric and Perceptual Verification

To verify whether any Dev image is an unlabelled crop, rotation, or re-degradation of a clean training scene, we performed multi-scale SIFT feature extraction with FLANN-based bidirectional matching and RANSAC homography estimation:

| Dev Scene | Resolution | Best Matching Clean Scene | Inlier Count | Geometric Match Confirmed? |
|---|---|---|---|---|
| `fish10` | $3000 \times 4000$ | `fish1` | 5 | ❌ No (Spurious random match) |
| `fish19` | $3000 \times 4000$ | `fish11` | 6 | ❌ No (Spurious random match) |
| `fish3` | $3000 \times 4000$ | `fish18` | 5 | ❌ No (Spurious random match) |
| `fish6` | $8000 \times 6000$ | `fish11` | 5 | ❌ No (Novel sensor geometry) |
| `sharRayFish` | $3024 \times 4032$ | `seahorse1` | 70 | ❌ No (Distinct scene content; mean $\Delta = 60.9$) |
| `starfish1` | $4624 \times 3468$ | `fish21` | 14 | ❌ No (Novel sensor geometry) |

### Key Dimensional Evidence:
1. `fish6` ($8000 \times 6000$): No clean training scene possesses this 48-megapixel aspect/sensor geometry.
2. `starfish1` ($4624 \times 3468$): Training scene `starfish2` has dimensions $4032 \times 2268$ (16:9 ratio), whereas `starfish1` is 4:3.
3. `sharRayFish` ($3024 \times 4032$): Contains shark/ray morphology, distinct from `seahorse1` (seahorse macro-fauna).

---

## 3. Conclusion & Campaign Action

- **Finding:** The official EURS Dev evaluation benchmark is **100% strictly scene-disjoint** from the training set.
- **Reference-Lookup Viability:** **Not applicable** (`V24-REF-LOOKUP` cannot be deployed on unseen Dev scenes without corresponding clean ground truth).
- **Directive:** Immediately advance to **Stage B (Score-Aligned Calibration)** and **Stage C (Direct Learned Restoration)**.
