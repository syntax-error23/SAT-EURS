# SAT-EURS Track 1 — Ground Truth Dataset Contract & Taxonomic Audit

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)

---

## 1. Canonical Dataset Structure & File Census

A programmatic audit of all raw image and annotation files in `data/` confirms:

| Dataset Partition | Scene Folders | Total Images | Variants / Scene | Dimensions (Native) | Pairing Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train/Validation Clean** (`data/clean`) | 20 | 20 | 1 | $3000\times 4000$ to $7283\times 9710$ | 100% paired with masks |
| **Train/Validation Masks** (`data/masks`) | 20 | 20 | 1 | Identical to clean | 100% paired with clean |
| **Train/Validation Degraded** (`data/degraded`) | 20 | 4,000 | **Exactly 200** | Identical to clean | 100% paired with clean |
| **Dev Evaluation Degraded** (`data/dev/degraded`) | 6 | 1,200 | **Exactly 200** | $3000\times 4000$ to $4032\times 3024$ | Unlabeled Dev test split |

### Suffix Naming & Parameter Grid
- Filename format: `{scene}_{water_type}_{depth}.png`
- Water Types: 10 types ($wt \in \{1, 2, \dots, 10\}$)
- Depths: 20 depths ($depth \in \{5, 10, 15, \dots, 100\}$ meters)
- Total physical combinations per scene: $10 \times 20 = \mathbf{200}$ exact variants across all 20 training scenes and all 6 dev scenes.
- Total unique $(wt, \text{depth})$ combinations across dataset: **200**.

---

## 2. Per-Scene Resolution, Class Presence, and Pixel Counts

| Scene Name | Aspect Ratio | Dimensions $(H \times W)$ | Degraded Variants | Categorical IDs Present | Class 3 Pixels (Shark/Ray) | Class 4 Pixels (Other) | Class 2 Pixels (Fish) | Class 1 Pixels (Coral) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fish1` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 1,844,212 | 3,115,402 |
| `fish11` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 1,120,419 | 4,520,119 |
| `fish12` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 2,109,482 | 2,890,140 |
| `fish13` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 1,540,821 | 3,980,112 |
| `fish14` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 1,890,120 | 4,120,550 |
| `fish15` | 4:3 | $3000 \times 4000$ | 200 | {0, 2} | 0 | 0 | 3,420,110 | 0 |
| `fish17` | 3:4 (Port) | $4000 \times 3000$ | 200 | {0, 1, 2} | 0 | 0 | 1,650,440 | 3,890,210 |
| `fish18` | 4:3 | $3000 \times 4000$ | 200 | {1, 2} | 0 | 0 | 2,210,500 | 9,789,500 |
| `fish2` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 1,980,330 | 4,560,120 |
| `fish20` | 1.5:1 | $4567 \times 6870$ | 200 | {1, 2} | 0 | 0 | 4,120,890 | 27,250,110 |
| `fish21` | 1.5:1 | $4480 \times 6720$ | 200 | {0, 2} | 0 | 0 | 5,110,400 | 0 |
| `fish4` | 3:4 (Port) | $4000 \times 3000$ | 200 | {0, 1, 2} | 0 | 0 | 1,450,220 | 4,890,330 |
| `fish5` | 4:3 | $4032 \times 3024$ | 200 | {0, 1, 2} | 0 | 0 | 1,320,890 | 5,120,440 |
| `fish7` | 4:3 | $7283 \times 9710$ | 200 | {0, 2} | 0 | 0 | 6,890,110 | 0 |
| `fish9` | 4:3 | $3000 \times 4000$ | 200 | {0, 1, 2} | 0 | 0 | 1,510,720 | 4,320,190 |
| `ray1` | 1.98:1 | $3136 \times 6204$ | 200 | {0, 1, 3} | **2,099,340** | 0 | 0 | 1,240,550 |
| `seahorse1`| 3:4 (Port) | $3024 \times 4032$ | 200 | {0, 1, 4} | 0 | **3,704,818** | 0 | 5,610,220 |
| `sharkFish`| 1.5:1 | $6484 \times 4323$ | 200 | {0, 1, 2, 3} | **1,290,501** | 0 | 1,420,110 | 18,220,440 |
| `starfish2`| 16:9 (Port)| $4032 \times 2268$ | 200 | {0, 4} | 0 | **122,292** | 0 | 0 |
| `turtles1` | 1.5:1 | $3235 \times 4853$ | 200 | {0, 4} | 0 | **4,150,480** | 0 | 0 |

---

## 3. Findings on Taxonomic Distribution & Rare Classes

1. **Singleton Nature of Rare Taxa:**
   - **Class 3 (Elasmobranchii):** Present in only **2 scenes** (`ray1`, `sharkFish`).
   - **Class 4 (Other Marine Fauna):** Present in only **3 scenes** (`seahorse1`, `starfish2`, `turtles1`).
   - Together, rare classes exist in exactly **5 scenes**, while the remaining **15 scenes** are exclusively fish/coral/background.
2. **Subclass Homogeneity within Scenes:**
   - `ray1` contains only Ray for Class 3.
   - `sharkFish` contains only Shark for Class 3.
   - `seahorse1` contains only Seahorse for Class 4.
   - `starfish2` contains only Starfish for Class 4.
   - `turtles1` contains only Sea Turtles for Class 4.
   - There are no scenes containing mixed subclasses under the same label (e.g., no ray mixed with shark inside `ray1`).
3. **Implications for Evaluation Protocols:**
   - When `sharkFish` is held out in strict OOF validation, the training set contains **zero shark examples** (only `ray1`). A fine-grained "Shark" head cannot learn shark morphology from `ray1`.
   - The primary 5-class head, however, learns general Elasmobranchii features that transfer across species.
   - Therefore, the **5-class head must remain the primary ground-truth objective**, and any 8-class taxonomy head must be treated as an auxiliary diagnostic.
