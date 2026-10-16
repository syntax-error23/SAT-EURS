# SAT-EURS Track 1 — Split Validation & Cross-Validation Protocol Specification

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)  
**Manifest File:** [`splits/manifest_5fold.json`](file:///c:/Projects/SAT-EURS/splits/manifest_5fold.json)  
**Manifest SHA256:** `b14f238c0be820ae225ea058f72e128bb96e35760f7341022692d3a08e24dfd6`

---

## 1. Dual-Protocol Validation Design

Because rare taxa in EURS are singletons at the source-scene level (Class 3 in `ray1` and `sharkFish`; Class 4 in `seahorse1`, `starfish2`, `turtles1`), evaluating a model under only one question produces misleading conclusions:
- If a strict OOF fold holds out `sharkFish`, the training data contains zero visual examples of sharks. Demanding the model predict a shark without having seen one tests pure zero-shot transfer, not learned representation.
- Conversely, evaluating only on scenes seen during training introduces in-sample leakage and inflates $Q_{\text{seg}}$ (as happened with the $0.7078$ baseline).

To resolve this, we implement **two mathematically separated evaluation protocols** supported by an identical set of 5 fold models:

### Protocol 1: Strict Scene-Disjoint 5-Fold OOF Evaluation (Generalization Standard)
- All 20 scenes are partitioned into 5 mutually exclusive, collectively exhaustive folds.
- Every source scene and its 200 degradation variants appears in the validation set of **exactly one fold**.
- All other 16 scenes form that fold's training set ($\text{Train} \cap \text{Val} = \emptyset$).
- Predictions for each validation scene are made **exclusively** by the model trained on that fold (which never saw that scene).
- The 5 validation sets are aggregated into the canonical Out-Of-Fold (OOF) dataset of all 4,000 images, yielding the official strict OOF $Q_{\text{seg}}$, mIoU, and Macro $F_1$.

### Protocol 2: Rare-Seen / Unseen-Host Evaluation (Domain Transfer Diagnostic)
- Evaluates: *Does a model trained with known rare species transfer those taxa onto completely unseen background scenes?*
- In Protocol 1, every fold holds out **1 rare scene + 3 fish scenes** in validation, while keeping the **other 4 rare scenes in training**.
- Therefore, the 3 held-out fish scenes in each fold serve as genuine, unseen underwater host environments.
- By utilizing the exact same fold checkpoints, Protocol 2 evaluates the detection and boundary delineation of training-seen rare taxa when transferred into unseen host scenes with zero extra training compute.

### Protocol 2 Supplementary: Synthetic Unseen-Host Probe
- Takes foreground instances from the training-seen rare donor scenes of the active fold (`train_scenes`).
- Pastes them into the fold's held-out fish host images (`val_fish_hosts`) matching the exact physical $(wt, \text{depth})$ degradation suffix.
- Evaluates whether the trained model detects the known rare object under domain shift (new background, lighting, and occlusions).
- **Guardrail:** Labeled strictly `synthetic_unseen_host_probe`; never substituted for official natural OOF metrics.

---

## 2. Canonical 5-Fold Scene Partitioning Table

| Fold ID | Val Rare Scene | Val Fish Host Scenes | Validation Set (4 scenes / 800 imgs) | Training Set (16 scenes / 3,200 imgs) | Train Classes Present |
| :---: | :---: | :---: | :--- | :--- | :---: |
| **`fold_0`** | `seahorse1` (Cls 4) | `fish1`, `fish11`, `fish12` | `fish1`, `fish11`, `fish12`, `seahorse1` | 12 fish + `ray1`, `sharkFish`, `starfish2`, `turtles1` | **All {0, 1, 2, 3, 4}** |
| **`fold_1`** | `sharkFish` (Cls 3) | `fish13`, `fish14`, `fish15` | `fish13`, `fish14`, `fish15`, `sharkFish` | 12 fish + `ray1`, `seahorse1`, `starfish2`, `turtles1` | **All {0, 1, 2, 3, 4}** |
| **`fold_2`** | `ray1` (Cls 3) | `fish17`, `fish18`, `fish2` | `fish17`, `fish18`, `fish2`, `ray1` | 12 fish + `seahorse1`, `sharkFish`, `starfish2`, `turtles1` | **All {0, 1, 2, 3, 4}** |
| **`fold_3`** | `starfish2` (Cls 4) | `fish20`, `fish21`, `fish4` | `fish20`, `fish21`, `fish4`, `starfish2` | 12 fish + `ray1`, `seahorse1`, `sharkFish`, `turtles1` | **All {0, 1, 2, 3, 4}** |
| **`fold_4`** | `turtles1` (Cls 4) | `fish5`, `fish7`, `fish9` | `fish5`, `fish7`, `fish9`, `turtles1` | 12 fish + `ray1`, `seahorse1`, `sharkFish`, `starfish2` | **All {0, 1, 2, 3, 4}** |

### Mathematical Validation Invariants
1. **Scene Disjointness:** $\forall i \in \{0..4\}, \quad \text{Train}_i \cap \text{Val}_i = \emptyset$.
2. **Exhaustive Partition:** $\bigcup_{i=0}^4 \text{Val}_i = \text{All 20 Scenes}$.
3. **No Overlap Across Folds:** $\forall i \neq j, \quad \text{Val}_i \cap \text{Val}_j = \emptyset$.
4. **Balanced Class Representation:** Every fold contains positive training pixels for all 5 semantic classes.
5. **Screening Strategy (Tier 1):** `fold_0` (evaluating held-out Class 4 `seahorse1`) and `fold_1` (evaluating held-out Class 3 `sharkFish`) provide comprehensive coverage of all rare classes for Tier 1 screening.
