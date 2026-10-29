# SAT-EURS Track 1 — Phase 1 Forensic Diagnostic: Test Scene Diagnostics

**Campaign:** SAT-EURS Track 1 Phase 1 Forensic Diagnostic  
**Date:** October 24, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** MaCVi @ WACV 2027 EURS Track 1 Testing Phase  
**Artifact Manifest:** [`reports/v28/phase1_diagnostic/test_manifest.json`](file:///c:/Projects/SAT-EURS/reports/v28/phase1_diagnostic/test_manifest.json) (1,200 items)  
**Scene Summary:** [`reports/v28/phase1_diagnostic/scene_summary.json`](file:///c:/Projects/SAT-EURS/reports/v28/phase1_diagnostic/scene_summary.json)  
**Contact Sheets:** [`reports/v28/phase1_diagnostic/contact_sheets/`](file:///c:/Projects/SAT-EURS/reports/v28/phase1_diagnostic/contact_sheets/) (30 4-panel diagnostic sheets)

---

## 1. Executive Summary

This diagnostic establishes the exact structural, geometric, and biological composition of the 6 official hidden test scenes (`test04`, `test09`, `test10`, `test11`, `test19`, `test20`) totaling 1,200 test images (200 degradation variants per scene).

Programmatic enumeration over the production V27 predictions has revealed a striking **scene-level ecological partition** that explains why naive globalensembling and uniform thresholding underperformed.

---

## 2. Test Scene Inventory & Geometric Specifications

Each test scene contains exactly 200 degradation variants: 10 Jerlov water types ($WT \in \{1 \dots 10\}$) crossed with 20 discrete depth stations ($Depth \in \{5 \dots 100\text{m}, \text{step } 5\text{m}\}$).

| Scene ID | Native Resolution | Scaled Resolution | Aspect Ratio | Total Images | Total Pixels (at Scaled Res) | Primary Environment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `test04` | $2160 \times 3840$ | $576 \times 1024$ | 16:9 Landscape | 200 | 117,964,800 px | Coral Benthos + Benthic Macro-Fauna |
| `test09` | $4000 \times 3000$ | $1024 \times 768$ | 4:3 Portrait | 200 | 157,286,400 px | Dense Complex Coral Reef + Schooling Fish |
| `test10` | $4000 \times 3000$ | $1024 \times 768$ | 4:3 Portrait | 200 | 157,286,400 px | Turbid Coral Reef + Small Cryptic Fish |
| `test11` | $3000 \times 4000$ | $768 \times 1024$ | 3:4 Landscape | 200 | 157,286,400 px | Open Coral Reef + Fast Pelagic Fish |
| `test19` | $6484 \times 4323$ | $1024 \times 683$ | 3:2 Landscape | 200 | 139,878,400 px | Pelagic Open Water / Sandy Seafloor + Eagle Ray |
| `test20` | $7203 \times 4804$ | $1024 \times 683$ | 3:2 Landscape | 200 | 139,878,400 px | Deep Pelagic Blue Water + Massive Stingray / Shark |
| **Total** | — | — | — | **1,200** | **869,580,800 px** | — |

---

## 3. Ecological Specialization Census (V27 Predictions)

Programmatic census of all 869.58 million predicted pixels in the official V27 submission:

| Scene | Class 0: Background | Class 1: Coral | Class 2: Fish | Class 3: Ray | Class 4: MacroFauna | Dominant Fauna |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `test04` | 39.09% (46.1M) | 44.65% (52.7M) | 0.18% (0.21M) | 0.00% (0 px) | **16.09% (18.98M)** | **100% of all Class 4 Macro-Fauna** |
| `test09` | 37.35% (58.7M) | **59.39% (93.4M)**| **3.26% (5.12M)** | 0.00% (0 px) | 0.00% (0 px) | Coral Reef + Dense Fish Schools |
| `test10` | 39.84% (62.7M) | **58.97% (92.7M)**| 1.19% (1.87M) | 0.00% (0 px) | 0.00% (0 px) | Turbid Coral Reef + Solitary Fish |
| `test11` | 54.84% (86.3M) | 42.16% (66.3M) | **3.00% (4.72M)** | 0.00% (81 px)*| 0.00% (0 px) | Mixed Reef + Swimming Fish |
| `test19` | **93.98% (131.5M)**| 0.63% (0.88M) | 0.20% (0.28M) | **5.19% (7.26M)** | 0.00% (0 px) | **Pelagic Ray (Manta/Eagle)** |
| `test20` | **85.80% (120.0M)**| 0.98% (1.36M) | 2.23% (3.12M) | **10.99% (15.37M)**| 0.00% (0 px) | **Massive Stingray / Shark** |
| **Total** | **58.10% (505.3M)**| **35.35% (307.4M)**| **1.76% (15.33M)**| **2.60% (22.63M)** | **2.18% (18.98M)**| — |

*\*81 pixels is isolated boundary noise.*

---

## 4. Key Forensic Findings

### Finding 1: Extreme Ecological Disjointness
- **Class 4 (Macro-Fauna) is 100% localized to `test04`:** Across all 1,200 test images, Class 4 appears **only** in `test04`. In the other five scenes, Class 4 pixel count is strictly 0.
- **Class 3 (Ray) is 99.999% localized to `test19` and `test20`:** In scenes `test04`, `test09`, and `test10`, Class 3 is strictly 0. In `test11`, it is 81 pixels (noise).
- **The "Reef vs Pelagic" Bimodal Split:**
  - Reef scenes (`test04`, `test09`, `test10`, `test11`): Background is only 37%–54%; Coral dominates (42%–59%).
  - Pelagic scenes (`test19`, `test20`): Background dominates overwhelmingly (86%–94%); Coral is negligible (< 1%).

### Finding 2: The Severe Resolution Shrinkage on Pelagic Scenes (`test19` and `test20`)
- Native resolutions for `test19` ($6484 \times 4323$) and `test20` ($7203 \times 4804$) are **31.0 to 34.6 Megapixels**.
- When scaled to $\max(W, H) = 1024$, they are downsampled by **$6.33\times$ to $7.03\times$ linear factor** (a **$44\times$ to $49\times$ pixel area reduction**).
- Small fish that were $150 \times 80$ pixels in the native 7K image shrink to tiny $21 \times 11$ pixel blurs at 1024 resolution.
- Under severe water degradation ($WT \ge 8, Depth \ge 60\text{m}$), these tiny objects lose all high-frequency cues and are absorbed into the background.

### Finding 3: Visual Contact Sheet Failure Modes (Inspected across 30 Diagnostic Panels)
1. **Turbidity Mask Erosion ($WT \ge 8, Depth \ge 60\text{m}$):**
   - In `test10_8_40` and `test10_10_90`, heavy green cast and backscatter completely mask coral textures.
   - The segmenter predicts false background "holes" inside large coral structures where contrast drops.
2. **Ray Boundary Bleed in Pelagic Scenes (`test19_10_90`, `test20_10_90`):**
   - In deep pelagic scenes, the faint wingtips of rays blend into the scattering background. The global offset $b_3 = 0.10$ preserves the body core but leaves fuzzy, inconsistent margins.
3. **Restoration Ringing vs Blur:**
   - The $\alpha = 0.10$ unsharp filter sharpens coral edges cleanly in mild conditions ($WT=1, D=10$). However, in deep conditions ($WT=10, D=90$), it amplifies high-frequency water speckle and sensor noise, degrading structural sharpness.

---

## 5. Stratified Diagnostic Sheets Catalog

30 individual 4-panel diagnostic sheets generated in `reports/v28/phase1_diagnostic/contact_sheets/`:

- **Scene `test04` (Macro-Fauna & Coral):**
  - Mild: `test04_1_10_mild_contact.jpg`
  - Mid: `test04_3_30_mid_contact.jpg`
  - Deep: `test04_5_60_deep_contact.jpg`
  - Turbid: `test04_8_40_turbid_contact.jpg`
  - Extreme: `test04_10_90_extreme_contact.jpg`
- **Scene `test09` (Complex Reef & Fish):**
  - Mild: `test09_1_10_mild_contact.jpg`
  - Mid: `test09_3_30_mid_contact.jpg`
  - Deep: `test09_5_60_deep_contact.jpg`
  - Turbid: `test09_8_40_turbid_contact.jpg`
  - Extreme: `test09_10_90_extreme_contact.jpg`
- **Scene `test10` (Turbid Reef):**
  - Mild: `test10_1_10_mild_contact.jpg`
  - Mid: `test10_3_30_mid_contact.jpg`
  - Deep: `test10_5_60_deep_contact.jpg`
  - Turbid: `test10_8_40_turbid_contact.jpg`
  - Extreme: `test10_10_90_extreme_contact.jpg`
- **Scene `test11` (Mixed Reef & Fish):**
  - Mild: `test11_1_10_mild_contact.jpg`
  - Mid: `test11_3_30_mid_contact.jpg`
  - Deep: `test11_5_60_deep_contact.jpg`
  - Turbid: `test11_8_40_turbid_contact.jpg`
  - Extreme: `test11_10_90_extreme_contact.jpg`
- **Scene `test19` (Eagle Ray & Open Water):**
  - Mild: `test19_1_10_mild_contact.jpg`
  - Mid: `test19_3_30_mid_contact.jpg`
  - Deep: `test19_5_60_deep_contact.jpg`
  - Turbid: `test19_8_40_turbid_contact.jpg`
  - Extreme: `test19_10_90_extreme_contact.jpg`
- **Scene `test20` (Stingray & Deep Water):**
  - Mild: `test20_1_10_mild_contact.jpg`
  - Mid: `test20_3_30_mid_contact.jpg`
  - Deep: `test20_5_60_deep_contact.jpg`
  - Turbid: `test20_8_40_turbid_contact.jpg`
  - Extreme: `test20_10_90_extreme_contact.jpg`
