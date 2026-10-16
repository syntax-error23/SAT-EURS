# SAT-EURS Track 1 — Historical Submission Provenance & Integrity Ledger

**Date:** October 14, 2026  
**Auditor / Agent:** Antigravity Research Team  
**Evaluation Target:** Track 1 Unconstrained Leaderboard (MaCVi @ WACV 2027)

---

## 1. Archived Submission ZIP Integrity & Cryptographic Hashes

All historical submission packages located in `deploy/` have been audited. In compliance with Rule 2 and Rule 3, all historical ZIP files and their hashes are strictly preserved and remain unaltered:

| Submission Archive | Size (MB) | SHA256 Checksum | Verified Output Layout | Status |
| :--- | :---: | :--- | :---: | :---: |
| `deploy/eurs_track1_compliant_candidate_submission.zip` | 833.93 | `0072b017616c5a91ab093111ca3c853bdcd405ca137488b5d3f111623fd38168` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_compliant_recovery_submission.zip` | 901.82 | `85108f3f2c06a5faf023a17feda4f3bcc1e8666720774689e03fdec640bd988f` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_dev_compact.zip` | 735.84 | `0ef1a7f120852824b580cfaf712612cdf3a5f7ae16fbeb3bb7a9aa68ae4b2173` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_dev_submission.zip` | 1240.59 | `65d8cae3f31587ea7995441cc753747c8efe52140f274b6d1aa7500743c58e65` | 1,200 restored + 1,200 masks | **PRESERVED** (Exceeds 1 GB size limit) |
| `deploy/eurs_track1_dryrun_candidate.zip` | 1520.74 | `dc5fea7959a60bb1eba718733ea59a559e8866926c0f9e4898f1d49a67e7ff2f` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_dryrun_compact_candidate.zip` | 564.91 | `d6f4b9dfae2a084154d4820722fea48276fe1d5ae8a115557ea36417c3f7d2d6` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_paired_A_D_submission.zip` | 574.97 | `3daf756f635c366a443e59ea8d9e4e96ed2ea0718f11ec4e3427fb6137c39d9c` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_ultimate_recovery_submission.zip` | 571.64 | `2fec31f54ee03afcebb5f949b642fc9f9752b4d2c4cd7e759857c7aaddf444cf` | 1,200 restored + 1,200 masks | **PRESERVED** |
| `deploy/eurs_track1_verified_submission.zip` | 902.06 | `6b1b36ea81972405c0e1321f09e86663e364c00be24de37ba39126dbaba1740f` | 1,200 restored + 1,200 masks | **PRESERVED** |

---

## 2. Official Leaderboard Historical Results & Script Provenance

| Sub ID | Method Name | Date | S_composite | Q_seg | mIoU | F1 | PSNR (dB) | SSIM | Sharpness | Implied Q_NR | Generating Script |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#18666** | EURS-Net V5-5 | 13-Oct-2026 | **0.473** | 0.347 | 0.521 | 0.666 | 24.50 | 0.821 | 0.611 | 0.715 | `deploy/generate_track1_submission_sota.py` |
| **#18668** | EURS-Net V5-6 | 13-Oct-2026 | **0.467** | 0.336 | 0.523 | 0.643 | 24.50 | 0.821 | 0.611 | 0.721 | `deploy/generate_sota_track1_submission.py` |
| **#18672** | Server Result A | 13-Oct-2026 | **0.476** | 0.324 | 0.510 | 0.635 | **25.49** | **0.810** | **0.893** | 0.480 | `deploy/track1_dev_v5/restored` |
| **#18688** | Paired A+D | 14-Oct-2026 | **0.509** | **0.424** | **0.583** | **0.727** | **25.49** | **0.810** | **0.893** | 0.480 | Restored: Result A; Masks: Result D (`deploy/track1_dev_submission_v6`) |
| **#18698** | **EURS-Net-V6-DINOv2-SOTA** | **14-Oct-2026** | **0.678** | **0.659** | **0.765** | **0.861** | **25.44** | **0.809** | **0.955** | **0.709** | `tools/evaluate_5fold_oof_and_package.py` (DINOv2-L + RSUIGM + Exact-Depth CP) |
| **Pending** | **EURS-Net-V6-MultiScale-DINOv2** | **14-Oct-2026** | **0.70+ (est)** | **0.69+ (est)** | **0.78+ (est)** | **0.88+ (est)** | **25.45+** | **0.810+** | **0.955+** | **0.709** | `tools/generate_track1_multiscale_sota.py` (Multi-Scale TTA [448,518,588] + 20-Scene RSUIGM) |

---

## 3. Discrepancy & Provenance Findings

### Finding 1: Mathematical Reconstruction of Historical Scores
- **Evidence:** 
  For #18666: $Q_{\text{seg}} = 0.521 \times 0.666 = 0.34699 \approx 0.347$.
  $Q_{\text{FR}} = 0.45 \times \frac{24.50 - 15}{20} + 0.45(0.821) + 0.10(0.611) = 0.6443$.
  With $S_{\text{composite}} = 0.60(0.347) + 0.30(0.6443) + 0.10(Q_{\text{NR}}) = 0.473$, the implied $Q_{\text{NR}} \approx 0.715$.
  For Paired A+D (#18688): $0.60(0.4240) + 0.30(0.6898) + 0.10(0.4800) = 0.2544 + 0.2069 + 0.0480 = \mathbf{0.5093}$.
- **Status:** **VERIFIED FACT**. The scoring formulas are mathematically confirmed.

### Finding 2: Root Cause of Prior ~0.50 Ceiling
- **Evidence:** The prior team projected scores of $0.58 - 0.65+$ based on validation metrics of $Q_{\text{seg}} = 0.7078$. However, that metric was computed using `checkpoints/v4_full20/best_v4_checkpoint.pt`, which had been trained on **all 20 scenes**, causing in-sample evaluation leakage. On the truly unseen Dev test scenes (`fish10`, `fish19`, `fish3`, `fish6`, `sharRayFish`, `starfish1`), the ensemble achieved $Q_{\text{seg}} = 0.4240$, pulling the overall composite to $0.5093$.
- **Status:** **VERIFIED FACT**.

### Finding 3: Missing Scripts & Heuristic Discrepancies
- A file referenced in earlier prompts (`models/v5_9_ensemble.py`) is **not present** in the repository.
- Scripts `#18666` and `#18668` differed primarily by manual logit offsets (`logits[4] += 1.8` vs `0.8`) and negative class penalties (`logits[3] -= 4.0` vs `-5.0`), rather than principled probability calibration.
- **Status:** **VERIFIED FACT**.
