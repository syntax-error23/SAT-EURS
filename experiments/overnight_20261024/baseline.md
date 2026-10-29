# SAT-EURS Track 1 — Production Baseline & Archive Verification

## 1. Cryptographic Hash Verification

| Version | Archive Path | Size (MB) | Expected SHA-256 | Actual SHA-256 | Status | Official Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **V18** | `deploy/v18/submission_v18_dual_encoder_sota.zip` | 540.26 | `e82cbea22ed068a3...` | `e82cbea22ed068a3...` | **EXACT MATCH** | **0.749** (Development Phase) |
| **V27** | `deploy/v27/submission_v27_test_hybrid_sota.zip` | 498.36 | `722b106ae817e4ef...` | `722b106ae817e4ef...` | **EXACT MATCH** | **0.633** (Testing Phase Entry #19096) |
| **V28-A** | `deploy/v28/submission_v28a_fivefold_ablation.zip` | 497.98 | `471bf9b9710e2922...` | `471bf9b9710e2922...` | **EXACT MATCH** | **0.634** (Testing Phase Official Entry) |

## 2. Archive Structural Layout

| Version | Total Files | Top-level Directories | `restored/` Count | `seg_masks/` Count | Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V18** | 2400 | `restored, seg_masks` | 1200 | 1200 | **100% COMPLIANT** |
| **V27** | 2400 | `restored, seg_masks` | 1200 | 1200 | **100% COMPLIANT** |
| **V28-A** | 2400 | `restored, seg_masks` | 1200 | 1200 | **100% COMPLIANT** |

## 3. Official Metric Context (V28-A: Official Composite = 0.634)

| Component / Metric | Official V27 (#19096) | Official V28-A (Verified) | Delta (V28-A vs V27) | Scoring Weight |
| :--- | :---: | :---: | :---: | :---: |
| **Composite Score ($S$)** | **0.633** | **0.634** | **+0.001** | **100%** |
| **Segmentation Quality ($Q_{\text{seg}}$)** | $0.568$ | **0.570** | $+0.002$ | $60\%$ |
| **Segmentation mIoU** | $0.700$ | **0.704** | $+0.004$ | — |
| **Segmentation Macro F1** | $0.810$ | **0.810** | $+0.000$ | — |
| **Restoration PSNR (dB)** | $25.48$ | $25.48$ | $+0.00$ (Byte-identical) | — |
| **Restoration SSIM** | $0.829$ | $0.829$ | $+0.000$ (Byte-identical) | — |
| **Restoration Sharpness Ratio** | $0.973$ | $0.973$ | $+0.000$ (Byte-identical) | — |
| **No-Reference UIQM** | $0.644$ | $0.644$ | $+0.000$ (Byte-identical) | — |
| **No-Reference UCIQE** | $0.592$ | $0.592$ | $+0.000$ (Byte-identical) | — |
| **No-Reference CCF** | $0.090$ | $0.090$ | $+0.000$ (Byte-identical) | — |

### Key Observation:
- V28-A removed the Full-20 segmentation blend ($w=0.50 \to 0.00$), which lifted official $Q_{\text{seg}}$ from $0.568$ to $0.570$ (+0.002) and mIoU from $0.700$ to $0.704$ (+0.004), boosting composite to **0.634**.
- However, +0.001 is a modest gain. To make progress toward the competitive frontier (current testing leaderboard reference ~0.753), we must explore both segmentation and restoration improvements systematically.
